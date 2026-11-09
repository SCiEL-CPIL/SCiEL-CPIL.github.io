import time
import random
import numpy as np

N = 64
M = N * N
MAX_DIST = (N-1)**2 + (N-1)**2  # 8064 for N=64

def construct():
    start_time = time.time()
    deadline = start_time + 290
    
    # Precompute all points as a numpy array for fast distance calculations
    points = np.array([(x, y) for x in range(N) for y in range(N)], dtype=np.int32)
    
    # Precompute squared distances between all pairs of points
    # dist_matrix[i, j] = squared distance between point i and point j
    # Shape: (M, M)
    diffs = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dist_matrix = np.sum(diffs * diffs, axis=2)
    
    # For each point i, precompute which other points are at each distance
    # dist_to_points[i][d] = set of point indices j such that dist_matrix[i,j] == d
    # This is expensive to precompute fully, so we'll compute on demand
    
    best_size = 0
    best_S_list = []
    
    def try_add_bitset(S, dist_bitsets, p_idx):
        """
        Attempts to add point p_idx to set S using bitset representation.
        dist_bitsets: dict mapping point_idx -> bitset (int) where bit d is set if distance d is used.
        Returns True if successful.
        """
        if p_idx in S:
            return False
            
        if len(S) == 0:
            S.add(p_idx)
            dist_bitsets[p_idx] = 0
            return True
            
        # Compute distances from p_idx to all points in S
        S_list = list(S)
        dists = dist_matrix[p_idx, S_list]
        
        # Check for duplicate distances (p_idx as apex)
        if len(set(dists)) != len(dists):
            return False
            
        # Build bitset for p_idx's distances
        p_bitset = 0
        for d in dists:
            d_val = int(d)
            p_bitset |= (1 << d_val)
            
        # Check if any distance conflicts with existing points' distance sets
        for s_idx in S_list:
            if p_bitset & dist_bitsets[s_idx]:
                return False
                
        # Validity confirmed
        S.add(p_idx)
        dist_bitsets[p_idx] = p_bitset
        
        # Update distance bitsets for all existing points
        for i, s_idx in enumerate(S_list):
            d_val = int(dists[i])
            dist_bitsets[s_idx] |= (1 << d_val)
            
        return True

    def greedy_pass_bitset(order, time_limit):
        """
        Runs a greedy pass with the given order of points using bitset representation.
        """
        S = set()
        dist_bitsets = {}
        
        for p_idx in order:
            if time.time() > time_limit:
                break
            try_add_bitset(S, dist_bitsets, p_idx)
                
        return S, dist_bitsets

    def local_search_bitset(S, dist_bitsets, time_limit):
        """
        Local search using bitset representation.
        Tries to add points, and if stuck, tries swaps.
        """
        no_improve = 0
        max_no_improve = 50
        
        while time.time() < time_limit:
            improved = False
            
            # Try to add a random candidate
            candidates = [i for i in range(M) if i not in S]
            if candidates:
                random.shuffle(candidates)
                for p_idx in candidates[:100]:  # Limit candidates to avoid slow loops
                    if try_add_bitset(S, dist_bitsets, p_idx):
                        improved = True
                        break
            
            if improved:
                no_improve = 0
                continue
            
            no_improve += 1
            
            if no_improve > max_no_improve:
                # Try a swap: remove one point, try to add two
                if len(S) > 2:
                    s_list = list(S)
                    random.shuffle(s_list)
                    removed = None
                    added = 0
                    
                    for r_idx in s_list[:20]:
                        # Remove r_idx
                        S.discard(r_idx)
                        r_dist = dist_bitsets.pop(r_idx, 0)
                        
                        # Update dist_bitsets for remaining points
                        for s_idx in S:
                            d_val = int(dist_matrix[r_idx, s_idx])
                            dist_bitsets[s_idx] &= ~(1 << d_val)
                        
                        # Try to add up to 2 new points
                        candidates = [i for i in range(M) if i not in S]
                        random.shuffle(candidates)
                        for p_idx in candidates[:50]:
                            if try_add_bitset(S, dist_bitsets, p_idx):
                                added += 1
                                if added >= 2:
                                    break
                        
                        if added > 1:
                            improved = True
                            break
                        else:
                            # Revert: re-add r_idx
                            S.add(r_idx)
                            dist_bitsets[r_idx] = r_dist
                            for s_idx in S:
                                d_val = int(dist_matrix[r_idx, s_idx])
                                dist_bitsets[s_idx] |= (1 << d_val)
                    
                    if improved:
                        no_improve = 0
                        continue
                
                # If swap didn't help, do a larger perturbation
                if len(S) > 5:
                    k = random.randint(3, 6)
                    to_remove = random.sample(list(S), min(k, len(S)))
                    for r in to_remove:
                        S.discard(r)
                        r_dist = dist_bitsets.pop(r, 0)
                        for s_idx in S:
                            d_val = int(dist_matrix[r, s_idx])
                            dist_bitsets[s_idx] &= ~(1 << d_val)
                
                no_improve = 0

    # Phase 1: Random Greedy Restarts
    t_phase1_end = start_time + 45
    
    while time.time() < t_phase1_end:
        order = list(range(M))
        random.shuffle(order)
        
        S, dist_bitsets = greedy_pass_bitset(order, deadline)
        
        if len(S) > best_size:
            best_size = len(S)
            best_S_list = list(S)

    # Phase 2: Local Search on best solution
    if best_S_list:
        S = set(best_S_list)
        # Rebuild dist_bitsets from scratch
        dist_bitsets = {}
        for p_idx in S:
            dist_bitsets[p_idx] = 0
        
        S_list = list(S)
        for i in range(len(S_list)):
            for j in range(i+1, len(S_list)):
                d_val = int(dist_matrix[S_list[i], S_list[j]])
                dist_bitsets[S_list[i]] |= (1 << d_val)
                dist_bitsets[S_list[j]] |= (1 << d_val)
        
        local_search_bitset(S, dist_bitsets, deadline - 5)
        
        if len(S) > best_size:
            best_size = len(S)
            best_S_list = list(S)
    
    # Return the best solution found
    return [tuple(points[i]) for i in best_S_list]
