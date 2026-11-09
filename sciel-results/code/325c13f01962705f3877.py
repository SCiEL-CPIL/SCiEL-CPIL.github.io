import time
import random
import numpy as np

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
    
    # Helper to check if point p can be added to set S (list of indices)
    # S_set: set of indices for O(1) lookup
    # dist_to_S: dict mapping index i to set of distances from i to other points in S
    def try_add(S_set, dist_to_S, p):
        if p in S_set:
            return False
        
        # Get distances from p to all points in S
        # We need the list of points in S to index into dist
        s_list = list(S_set)
        if not s_list:
            return True
            
        d_vals = dist[p, s_list]
        
        # Check 1: Distances from p to S must be unique
        # Using a set is fast for small arrays
        if len(set(d_vals)) < len(d_vals):
            return False
            
        # Check 2: For each s in S, d(p,s) must not be in dist_to_S[s]
        # We iterate over s_list and check the corresponding distance
        for i, s in enumerate(s_list):
            if d_vals[i] in dist_to_S[s]:
                return False
                
        return True

    def add_point(S_set, dist_to_S, p):
        S_set.add(p)
        dist_to_S[p] = set()
        for q in S_set:
            if q == p:
                continue
            d = dist[p, q]
            dist_to_S[p].add(d)
            dist_to_S[q].add(d)

    def remove_point(S_set, dist_to_S, p):
        if p not in S_set:
            return
        for q in S_set:
            if q == p:
                continue
            d = dist[p, q]
            dist_to_S[q].discard(d)
        dist_to_S.pop(p, None)
        S_set.discard(p)

    def greedy_construct(order):
        S_set = set()
        dist_to_S = {}
        for p in order:
            if time.time() > deadline - 5:
                break
            if try_add(S_set, dist_to_S, p):
                add_point(S_set, dist_to_S, p)
        return S_set, dist_to_S

    def local_search(S_set, dist_to_S, max_iters=1000):
        S_set = set(S_set)
        dist_to_S = {k: set(v) for k, v in dist_to_S.items()}
        
        improved = True
        iter_count = 0
        while improved and time.time() < deadline - 10 and iter_count < max_iters:
            improved = False
            iter_count += 1
            
            # 1. Try to add any point
            # We sample a subset of candidates to speed up
            # If we check all M points, it's 4096 * O(|S|) checks.
            # Let's check all points, it's fast enough if |S| is small.
            candidates = [i for i in range(M) if i not in S_set]
            random.shuffle(candidates)
            
            added = False
            for p in candidates:
                if try_add(S_set, dist_to_S, p):
                    add_point(S_set, dist_to_S, p)
                    improved = True
                    added = True
                    break
            
            if added:
                continue
            
            # 2. If no addition, try swaps
            # Remove a random point, try to add a different one
            if not S_set:
                break
                
            s_list = list(S_set)
            random.shuffle(s_list)
            
            swap_success = False
            # Try a few swaps
            for r in s_list[:5]: # Try removing up to 5 random points
                remove_point(S_set, dist_to_S, r)
                
                candidates = [i for i in range(M) if i not in S_set]
                random.shuffle(candidates)
                
                for p in candidates:
                    if p == r:
                        continue
                    if try_add(S_set, dist_to_S, p):
                        add_point(S_set, dist_to_S, p)
                        # Size is same, but configuration changed
                        # We mark as improved to continue searching for additions
                        improved = True
                        swap_success = True
                        break
                
                if not swap_success:
                    # Revert
                    add_point(S_set, dist_to_S, r)
                else:
                    break
                    
            if not improved:
                # Stuck. Try a 2-swap? Or just break and rely on restarts.
                # Let's try a simple 2-swap: remove r1, r2, add p1, p2
                # This is expensive. We'll skip for now and rely on restarts.
                break
                
        return S_set, dist_to_S

    best_S_set = set()
    best_dist_to_S = {}
    
    # Multiple restarts with different orders
    # 1. Random
    # 2. Row-major
    # 3. Col-major
    # 4. Diagonal
    # 5. Center-out
    
    orders = []
    
    # Random orders
    for _ in range(10):
        o = list(range(M))
        random.shuffle(o)
        orders.append(o)
        
    # Deterministic orders
    # Row-major
    orders.append(list(range(M)))
    # Col-major
    orders.append([i for i in range(M) if i % N == 0]) # This is just first col, need proper col-major
    # Proper col-major: index = x + y*N
    # We want to iterate y then x? No, standard row-major is x fast, y slow.
    # Col-major is y fast, x slow.
    # Let's just generate a few specific patterns.
    
    # Center out
    center = N // 2
    center_idx = center * N + center
    # Sort by distance to center
    center_coords = coords[center_idx]
    d_center = np.sum((coords - center_coords)**2, axis=-1)
    orders.append(list(np.argsort(d_center)))
    
    # Run greedy + local search for each order
    for o in orders:
        if time.time() > deadline - 20:
            break
            
        S_set, dist_to_S = greedy_construct(o)
        S_set, dist_to_S = local_search(S_set, dist_to_S)
        
        if len(S_set) > len(best_S_set):
            best_S_set = S_set.copy()
            best_dist_to_S = {k: set(v) for k, v in dist_to_S.items()}
            
    # Final polish
    if time.time() < deadline - 5:
        best_S_set, best_dist_to_S = local_search(best_S_set, best_dist_to_S, max_iters=5000)

    return [points[i] for i in best_S_set]

# For testing
if __name__ == "__main__":
    res = construct()
    print(len(res))
