import time
import random
import numpy as np
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 280  # Safety margin

    N = 64
    M = N * N
    
    # Precompute all points
    points = [(x, y) for x in range(N) for y in range(N)]
    coords = np.array(points, dtype=np.int32)
    
    # Precompute squared distance matrix
    # dist[i][j] = squared distance between points[i] and points[j]
    # 4096x4096x4 bytes = ~67MB, acceptable
    diff = coords[:, None, :] - coords[None, :, :]
    dist = np.sum(diff**2, axis=-1)
    
    # For fast lookup of distances, we can also precompute for each point,
    # a mapping from distance value to list of points at that distance.
    # But that's M * (M-1) entries, which is 16M. Might be too much memory or slow to build.
    # Instead, we will use the dist matrix directly.
    
    # Optimization: Precompute for each point i, the list of distances to all other points.
    # And maybe a set for O(1) lookup?
    # Actually, for the try_add check:
    # 1. Check if p has duplicate distances to S.
    #    We can compute dist[p, S] and check for duplicates.
    # 2. Check if for any q in S, there exists r in S such that dist[q, r] == dist[p, q].
    #    This is equivalent to: for each q in S, is dist[p, q] in the set of distances from q to S?
    
    # Let's maintain for the current set S:
    # S_set: set of indices in S
    # dist_to_S: for each point i in S, a set of distances to other points in S.
    #   dist_to_S[i] = { dist[i, j] for j in S, j != i }
    
    # Updating dist_to_S when adding p:
    # For each q in S:
    #   dist_to_S[q].add(dist[p, q])
    #   dist_to_S[p].add(dist[p, q])  # dist[p, q] == dist[q, p]
    
    # Updating dist_to_S when removing p:
    # For each q in S:
    #   dist_to_S[q].remove(dist[p, q])
    #   # Note: dist_to_S[p] is discarded
    
    # This makes the check for adding p:
    # 1. Compute d_vals = [dist[p, q] for q in S]
    # 2. If len(set(d_vals)) < len(d_vals): return False
    # 3. For each q in S:
    #      d = dist[p, q]
    #      if d in dist_to_S[q]: return False
    # 4. If all pass, add p to S and update dist_to_S.
    
    # This is O(|S|) per candidate check (step 1 and 3).
    # Step 1 involves computing distances and hashing.
    # Step 3 involves set lookups.
    
    # Let's implement this.
    
    def try_add(S_set, dist_to_S, p):
        """
        Try to add point p to set S_set.
        S_set: set of indices in S
        dist_to_S: dict mapping index i in S to set of distances from i to other points in S
        Returns True if added successfully, False otherwise.
        """
        if p in S_set:
            return False
        
        # Step 1: Check for duplicate distances from p to S
        d_vals = []
        for q in S_set:
            d = dist[p, q]
            d_vals.append(d)
        
        if len(set(d_vals)) < len(d_vals):
            return False
        
        # Step 2: Check if any d in d_vals is already a distance from q to S
        # For each q in S, check if dist[p, q] is in dist_to_S[q]
        for q in S_set:
            d = dist[p, q]
            if d in dist_to_S[q]:
                return False
        
        # If we get here, p can be added
        return True

    def add_point(S_set, dist_to_S, p):
        """
        Add point p to S_set and update dist_to_S.
        Assumes try_add has already checked validity.
        """
        S_set.add(p)
        dist_to_S[p] = set()
        for q in S_set:
            if q == p:
                continue
            d = dist[p, q]
            dist_to_S[p].add(d)
            dist_to_S[q].add(d)

    def remove_point(S_set, dist_to_S, p):
        """
        Remove point p from S_set and update dist_to_S.
        """
        if p not in S_set:
            return
        for q in S_set:
            if q == p:
                continue
            d = dist[p, q]
            dist_to_S[q].remove(d)
        del dist_to_S[p]
        S_set.discard(p)

    def greedy_construct():
        S_set = set()
        dist_to_S = {}
        candidates = list(range(M))
        random.shuffle(candidates)
        
        for p in candidates:
            if time.time() > deadline:
                break
            if try_add(S_set, dist_to_S, p):
                add_point(S_set, dist_to_S, p)
        
        return S_set, dist_to_S

    def local_search(S_set, dist_to_S):
        S_set = set(S_set)
        dist_to_S = {k: set(v) for k, v in dist_to_S.items()}
        
        # Precompute candidate indices not in S for faster access
        all_indices = np.arange(M)
        
        improved = True
        while improved and time.time() < deadline - 10:
            improved = False
            S_list = list(S_set)
            random.shuffle(S_list)
            
            # Try adding points not in S - use numpy for faster filtering
            S_array = np.array(list(S_set), dtype=np.int32)
            mask = ~np.isin(all_indices, S_array)
            candidates = all_indices[mask]
            random.shuffle(candidates.tolist())
            candidates = candidates.tolist()
            
            for p in candidates:
                if time.time() > deadline - 10:
                    break
                if try_add(S_set, dist_to_S, p):
                    add_point(S_set, dist_to_S, p)
                    improved = True
                    break
            
            if improved:
                continue
            
            # If no addition, try a swap
            # Remove a random point, then try to add a different one
            if S_list:
                # Try a few swaps
                swaps_tried = 0
                max_swaps = 100
                while swaps_tried < max_swaps and time.time() < deadline - 10:
                    q_to_remove = random.choice(S_list)
                    remove_point(S_set, dist_to_S, q_to_remove)
                    
                    # Regenerate candidates after removal
                    S_array = np.array(list(S_set), dtype=np.int32)
                    mask = ~np.isin(all_indices, S_array)
                    candidates = all_indices[mask].tolist()
                    random.shuffle(candidates)
                    
                    added = False
                    for p in candidates:
                        if time.time() > deadline - 10:
                            break
                        if p == q_to_remove:
                            continue
                        if try_add(S_set, dist_to_S, p):
                            add_point(S_set, dist_to_S, p)
                            added = True
                            break
                    
                    if not added:
                        # Revert
                        add_point(S_set, dist_to_S, q_to_remove)
                        swaps_tried += 1
                    else:
                        improved = True
                        break
                if not improved:
                    break
            else:
                break
                
        return S_set, dist_to_S

    best_S_set = set()
    best_dist_to_S = {}
    
    # Multiple restarts with adaptive count based on remaining time
    num_restarts = 30
    for i in range(num_restarts):
        if time.time() > deadline - 30:
            break
        
        S_set, dist_to_S = greedy_construct()
        S_set, dist_to_S = local_search(S_set, dist_to_S)
        
        if len(S_set) > len(best_S_set):
            best_S_set = S_set.copy()
            best_dist_to_S = {k: set(v) for k, v in dist_to_S.items()}
    
    # Final polish on the best set found
    if time.time() < deadline - 5:
        best_S_set, best_dist_to_S = local_search(best_S_set, best_dist_to_S)

    return [points[i] for i in best_S_set]

# For testing
if __name__ == "__main__":
    res = construct()
    print(len(res))
