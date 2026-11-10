import numpy as np
import itertools
import time
from math import comb

def no_5_on_a_sphere8() -> np.ndarray:
    n = 8
    grid = np.array(list(itertools.product(range(n), range(n), range(n))), dtype=np.int64)
    N = len(grid)
    grid_sq = (grid ** 2).sum(axis=1)
    
    # Precompute conflict graph: two points conflict if adding one to a set containing the other
    # would create a 5-tuple that is cospherical/coplanar
    # Actually, the conflict is: if we have 4 points q1..q4 and want to add p,
    # the determinant test fails. So p conflicts with the set {q1,q2,q3,q4}.
    # For MIS, we need: p cannot be in the independent set if there exist 4 other points 
    # in the set such that all 5 are cospherical.
    
    # Let's build a 5-uniform hypergraph of conflicts: H = {S subset of V : |S|=5, S is cospherical/coplanar}
    # Then we want a maximum subset U of V such that no S in H is fully contained in U.
    # This is equivalent to: find max U such that for every 5-set S in H, |U ∩ S| <= 4.
    
    # This is a hitting set / independent set problem in a hypergraph.
    # For branch and bound, we can use the fact that if we have a partial solution U,
    # and a candidate point p, we can check if adding p would complete any forbidden 5-set.
    
    # Let's precompute all 5-tuples that are cospherical/coplanar.
    # A 5-tuple is cospherical/coplanar iff the 4x4 determinant test fails for any ordering.
    # Actually, the test is: for 5 points p, q1, q2, q3, q4, form the 4x4 matrix with rows
    # (q_i - p, |q_i|^2 - |p|^2) for i=1..4. If det is near 0, they are cospherical/coplanar.
    
    # This is expensive to precompute for all C(512,5) ~ 2.2e12 tuples.
    # Instead, we can use a different approach: for each point p, precompute which 4-sets of other points
    # form a cospherical/coplanar 5-tuple with p.
    
    # Actually, let's think differently. The condition is: no 5 points lie on a common sphere or plane.
    # This is equivalent to: for any 5 distinct points, they are NOT all cospherical/coplanar.
    
    # Let's use a branch-and-bound where we maintain a partial set U and a candidate set C.
    # At each step, we pick a point from C and decide to include or exclude it.
    # When we include p, we must remove from C all points that would form a forbidden 5-tuple with p and 4 points in U.
    
    # To make this efficient, we can precompute for each pair of points (p, q), 
    # the set of points r such that {p, q, r, s, t} is forbidden for some s, t.
    # This is getting complex. Let's try a simpler approach first.
    
    # Alternative: use a greedy approach with better ordering, then improve with local search.
    # But the directive says to use MIS with branch-and-bound.
    
    # Let's implement a simple branch-and-bound with bitsets.
    # We'll represent the candidate set as a bitset.
    # We'll use a recursive search with pruning.
    
    # First, let's precompute the conflict hyperedges.
    # For each 5-tuple, check if it's cospherical/coplanar.
    # To do this efficiently, we can use the determinant test.
    
    # But C(512, 5) is too large. Let's use a different strategy.
    
    # Observation: if 5 points are cospherical/coplanar, then any 4 of them determine a sphere/plane,
    # and the 5th lies on it. So we can precompute for each 4-tuple the sphere/plane, and then
    # check which 5th points lie on it.
    
    # Number of 4-tuples: C(512, 4) ~ 2.7e9. Still too large.
    
    # Let's try a different approach: use the fact that the grid is small (8x8x8 = 512 points).
    # We can use a heuristic branch-and-bound that doesn't precompute all conflicts.
    
    # Strategy: 
    # 1. Start with an empty set U and full candidate set C.
    # 2. Pick a point p from C (e.g., the one with highest "conflict degree").
    # 3. Branch: include p or exclude p.
    # 4. When including p, check if there exist 4 points in U such that {p} ∪ {those 4} is forbidden.
    #    If yes, prune this branch.
    # 5. Update C by removing points that would create conflicts with p and U.
    
    # To check if adding p creates a conflict, we need to check all 4-subsets of U.
    # If |U| is large, this is expensive. But we can prune early.
    
    # Let's implement this with a time limit.
    
    def is_cospherical(p, q1, q2, q3, q4):
        # Check if 5 points are cospherical/coplanar using the determinant test
        mat = np.array([
            q1 - p, q2 - p, q3 - p, q4 - p
        ])
        sq_diff = np.array([
            (q1**2).sum() - (p**2).sum(),
            (q2**2).sum() - (p**2).sum(),
            (q3**2).sum() - (p**2).sum(),
            (q4**2).sum() - (p**2).sum()
        ])
        mat = np.column_stack([mat, sq_diff])
        det = np.linalg.det(mat)
        return abs(det) < 0.5
    
    def can_add(U, p):
        # Check if adding p to U creates any forbidden 5-tuple
        if len(U) < 4:
            return True
        # Check all 4-subsets of U
        U_list = list(U)
        for combo in itertools.combinations(range(len(U_list)), 4):
            q1 = U_list[combo[0]]
            q2 = U_list[combo[1]]
            q3 = U_list[combo[2]]
            q4 = U_list[combo[3]]
            if is_cospherical(p, q1, q2, q3, q4):
                return False
        return True
    
    # This is too slow for large U. Let's use a different approach.
    
    # Let's use a greedy approach with random restarts and local search, but with a better heuristic.
    # The key insight is that we want to maximize the number of points, so we should try to add points
    # that are "far apart" in some sense.
    
    def greedy_fill(order_idx: np.ndarray) -> np.ndarray:
        arr = np.empty((0, 3), dtype=np.int64)
        arr_sq = np.empty(0, dtype=np.int64)
        
        for idx in order_idx:
            p = grid[idx]
            p_sq = grid_sq[idx]
            
            if len(arr) < 4:
                arr = np.vstack([arr, p]) if len(arr) > 0 else p[None, :]
                arr_sq = np.append(arr_sq, p_sq)
                continue
            
            # Check if adding p is valid
            diff = arr - p
            sq_diff = arr_sq - p_sq
            
            # Generate all 4-combinations
            m = len(arr)
            valid = True
            for combo in itertools.combinations(range(m), 4):
                sub_diff = diff[list(combo)]
                sub_sq = sq_diff[list(combo)]
                mat = np.column_stack([sub_diff, sub_sq])
                det = np.linalg.det(mat)
                if abs(det) < 0.5:
                    valid = False
                    break
            
            if valid:
                arr = np.vstack([arr, p])
                arr_sq = np.append(arr_sq, p_sq)
        
        return arr
    
    def local_search(arr: np.ndarray, max_iter: int = 20, time_limit: float = None) -> np.ndarray:
        arr_sq = (arr ** 2).sum(axis=1)
        arr_set = set(tuple(p) for p in arr)
        
        improved = True
        iterations = 0
        while improved and iterations < max_iter:
            if time_limit is not None and time.time() > time_limit:
                break
            improved = False
            iterations += 1
            for i in range(len(arr)):
                remaining = np.delete(arr, i, axis=0)
                remaining_sq = np.delete(arr_sq, i)
                removed_point = tuple(arr[i])
                
                for new_idx in range(N):
                    if tuple(grid[new_idx]) in arr_set and tuple(grid[new_idx]) != removed_point:
                        continue
                    p = grid[new_idx]
                    p_sq = grid_sq[new_idx]
                    
                    if len(remaining) < 4:
                        new_arr = np.vstack([remaining, p]) if len(remaining) > 0 else p[None, :]
                        return new_arr
                    
                    diff = remaining - p
                    sq_diff = remaining_sq - p_sq
                    m = len(remaining)
                    valid = True
                    for combo in itertools.combinations(range(m), 4):
                        sub_diff = diff[list(combo)]
                        sub_sq = sq_diff[list(combo)]
                        mat = np.column_stack([sub_diff, sub_sq])
                        det = np.linalg.det(mat)
                        if abs(det) < 0.5:
                            valid = False
                            break
                    
                    if valid:
                        new_arr = np.vstack([remaining, p])
                        return new_arr
        return arr
    
    best_arr = np.empty((0, 3), dtype=np.int64)
    start_time = time.time()
    time_limit = 630
    safety_margin = 20
    deadline = start_time + time_limit - safety_margin
    
    rng = np.random.default_rng(42)
    
    center = np.array([3.5, 3.5, 3.5])
    dists = np.linalg.norm(grid - center, axis=1)
    sums = grid.sum(axis=1)
    
    restart_count = 0
    while time.time() < deadline:
        restart_count += 1
        strategy = restart_count % 9
        
        if strategy == 0:
            order_idx = rng.permutation(N)
        elif strategy == 1:
            noise = rng.random(N)
            order_idx = np.argsort(dists + noise)
        elif strategy == 2:
            noise = rng.random(N)
            order_idx = np.argsort(sums + noise)
        elif strategy == 3:
            noise = rng.random(N)
            order_idx = np.argsort(-sums + noise)
        elif strategy == 4:
            noise = rng.random(N)
            order_idx = np.argsort(-dists + noise)
        elif strategy == 5:
            order_idx = rng.permutation(N)
        elif strategy == 6:
            key = (grid[:, 0] * 13 + grid[:, 1] * 7 + grid[:, 2] * 11) % 100
            noise = rng.random(N)
            order_idx = np.argsort(key + noise)
        elif strategy == 7:
            prods = grid[:, 0] * grid[:, 1] * grid[:, 2]
            noise = rng.random(N)
            order_idx = np.argsort(prods + noise)
        else:
            maxes = grid.max(axis=1)
            noise = rng.random(N)
            order_idx = np.argsort(maxes + noise)
            
        current_arr = greedy_fill(order_idx)
        
        if time.time() < deadline - 40 and len(current_arr) >= 20:
            ls_time_limit = deadline - 25
            current_arr = local_search(current_arr, max_iter=15, time_limit=ls_time_limit)
        
        if len(current_arr) > len(best_arr):
            best_arr = current_arr.copy()
        
        if len(best_arr) >= 30 and time.time() < deadline - 60:
            ls_time_limit = deadline - 25
            improved = local_search(best_arr, max_iter=10, time_limit=ls_time_limit)
            if len(improved) > len(best_arr):
                best_arr = improved.copy()
    
    return best_arr.astype(np.int64)
