import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 275  # Safety margin

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
    
    # Strategy:
    # 1. Greedy construction with multiple random restarts.
    # 2. Local search (add and swap) to improve.
    # 
    # Optimization:
    # Maintain dist_to_S: for each point i in S, a set of distances to other points in S.
    # dist_to_S[i] = { dist[i, j] for j in S, j != i }
    #
    # Check for adding p:
    # 1. Compute d_vals = [dist[p, q] for q in S]
    # 2. If len(set(d_vals)) < len(d_vals): return False (duplicate distance from p)
    # 3. For each q in S: if dist[p, q] in dist_to_S[q]: return False
    #    (This checks if there exists r in S such that dist(q, r) == dist(p, q))
    
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
        # We need to check if any distance from p to q (q in S) is repeated.
        # Also, we need the list of distances for step 2.
        d_vals = []
        seen_dists = set()
        for q in S_set:
            d = dist[p, q]
            if d in seen_dists:
                return False
            seen_dists.add(d)
            d_vals.append((q, d))
        
        # Step 2: Check if any d in d_vals is already a distance from q to S
        # For each (q, d) in d_vals, check if d is in dist_to_S[q]
        for q, d in d_vals:
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
            dist_to_S[q].discard(d)
        dist_to_S.pop(p, None)
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
        
        improved = True
        while improved and time.time() < deadline - 10:
            improved = False
            S_list = list(S_set)
            random.shuffle(S_list)
            
            # Try adding points not in S
            # Optimization: Instead of creating a list of all non-S points (expensive),
            # we can just iterate over a shuffled range and check membership.
            # But checking membership in a set is O(1), so it's fine.
            # However, creating the list [i for i in range(M) if i not in S_set] is O(M).
            # Let's do that.
            candidates = [i for i in range(M) if i not in S_set]
            random.shuffle(candidates)
            
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
                max_swaps = 50
                while swaps_tried < max_swaps and time.time() < deadline - 10:
                    q_to_remove = random.choice(S_list)
                    remove_point(S_set, dist_to_S, q_to_remove)
                    
                    candidates = [i for i in range(M) if i not in S_set]
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
    
    # Multiple restarts
    # We have ~275s. Precomputation takes a few seconds.
    # Let's allow enough time for restarts.
    num_restarts = 20
    for i in range(num_restarts):
        if time.time() > deadline - 20:
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
