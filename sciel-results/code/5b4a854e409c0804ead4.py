"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 64
import time
import random
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 285  # Safety margin
    
    # Grid points
    points = [(x, y) for y in range(N) for x in range(N)]
    num_points = N * N
    
    # Precompute squared distances
    xs = np.array([p[0] for p in points], dtype=np.int32)
    ys = np.array([p[1] for p in points], dtype=np.int32)
    
    dx = xs[:, None] - xs[None, :]
    dy = ys[:, None] - ys[None, :]
    dist2 = (dx*dx + dy*dy).astype(np.int32)
    
    # Max possible squared distance is (N-1)^2 + (N-1)^2
    max_d2 = 2 * (N-1) * (N-1)
    
    # For each point, precompute which other points are at each distance
    # dist_points[i] = dict mapping distance -> list of points at that distance from i
    # But that's too much memory. Instead, use a different approach.
    
    # Key insight: For a set S to be valid, for each point P in S, 
    # all distances from P to other points in S must be distinct.
    # Equivalently: for any three points P, Q, R in S, d(P,Q) != d(P,R) and d(Q,P) != d(Q,R).
    # Since d(P,Q) = d(Q,P), the condition is: for each P in S, all d(P, Q) for Q in S\{P} are distinct.
    
    # This means: if we think of each point P having a "used distances" set, 
    # adding Q to S is valid iff d(P,Q) is not in used_distances[P] for all P in S.
    
    # For efficiency, represent used_distances as a set of ints per point.
    
    def greedy_construct(order, time_limit=None):
        """Greedy construction following a given order."""
        S = []
        S_set = set()
        used_dists = {}  # point_idx -> set of distances used by that point
        
        for idx_P in order:
            if idx_P in S_set:
                continue
            if time_limit and time.time() > time_limit:
                break
            
            # Check if P can be added
            valid = True
            new_dists = set()
            
            for idx_Q in S:
                d = int(dist2[idx_P, idx_Q])
                if d in new_dists:
                    valid = False
                    break
                new_dists.add(d)
                if d in used_dists[idx_Q]:
                    valid = False
                    break
            
            if valid:
                S.append(idx_P)
                S_set.add(idx_P)
                used_dists[idx_P] = new_dists
                for idx_Q in S:
                    if idx_Q != idx_P:
                        used_dists[idx_Q].add(int(dist2[idx_Q, idx_P]))
        
        return S, used_dists
    
    def local_search(S, used_dists, time_limit=None):
        """Improve S via swaps."""
        S_set = set(S)
        complement = set(range(num_points)) - S_set
        comp_list = list(complement)
        rng = random.Random(42)
        
        improved = True
        passes = 0
        
        while improved and time.time() < time_limit:
            improved = False
            passes += 1
            
            s_list = S[:]
            rng.shuffle(s_list)
            
            for idx_Q in s_list:
                if time.time() > time_limit:
                    break
                if idx_Q not in S_set:
                    continue
                
                # Try to find a P in complement that can replace Q
                # Sample candidates
                rng.shuffle(comp_list)
                candidates = comp_list[:100]
                
                for idx_P in candidates:
                    if idx_P in S_set:
                        continue
                    
                    # Check if swapping Q for P is valid
                    # Remove Q temporarily
                    S_without_Q = [i for i in S if i != idx_Q]
                    
                    valid = True
                    new_dists_P = set()
                    
                    for idx_R in S_without_Q:
                        d_PR = int(dist2[idx_P, idx_R])
                        d_RQ = int(dist2[idx_R, idx_Q])
                        
                        if d_PR in new_dists_P:
                            valid = False
                            break
                        new_dists_P.add(d_PR)
                        
                        # Check R's perspective: after swap, R has d_PR instead of d_RQ
                        # R's other distances (excluding d_RQ) must not include d_PR
                        if d_PR in used_dists[idx_R] and d_PR != d_RQ:
                            valid = False
                            break
                    
                    if valid:
                        # Perform the swap
                        S.remove(idx_Q)
                        S_set.discard(idx_Q)
                        comp_list.remove(idx_P)
                        comp_list.append(idx_Q)
                        
                        S.append(idx_P)
                        S_set.add(idx_P)
                        
                        # Update used_dists
                        for idx_R in S:
                            if idx_R != idx_P:
                                d = int(dist2[idx_R, idx_P])
                                used_dists[idx_R].discard(int(dist2[idx_R, idx_Q]))
                                used_dists[idx_R].add(d)
                        
                        used_dists[idx_P] = set()
                        for idx_R in S:
                            if idx_R != idx_P:
                                used_dists[idx_P].add(int(dist2[idx_P, idx_R]))
                        
                        del used_dists[idx_Q]
                        
                        improved = True
                        break
        
        return S, used_dists
    
    # Phase 1: Multiple greedy initializations with different orderings
    rng = random.Random(42)
    
    best_S = []
    best_used = {}
    
    # Generate diverse orderings
    orderings = []
    
    # Random shuffles
    for _ in range(8):
        order = list(range(num_points))
        rng.shuffle(order)
        orderings.append(order)
    
    # Diagonal scan
    order_diag = sorted(range(num_points), key=lambda i: (points[i][0] + points[i][1], points[i][0]))
    orderings.append(order_diag)
    
    # Anti-diagonal
    order_anti = sorted(range(num_points), key=lambda i: (points[i][0] - points[i][1], points[i][0]))
    orderings.append(order_anti)
    
    # Radial from center
    cx, cy = N/2.0, N/2.0
    order_radial = sorted(range(num_points), key=lambda i: (points[i][0]-cx)**2 + (points[i][1]-cy)**2)
    orderings.append(order_radial)
    
    # Row-major
    order_rows = list(range(num_points))
    orderings.append(order_rows)
    
    # Column-major (transposed)
    order_cols = [y*N + x for y in range(N) for x in range(N)]
    orderings.append(order_cols)
    
    # Stride patterns
    for stride in [2, 3, 5, 7]:
        order_stride = list(range(0, num_points, stride))
        orderings.append(order_stride)
    
    # Run greedy on each ordering
    for order in orderings:
        S, used = greedy_construct(order, time_limit=deadline - 30)
        if len(S) > len(best_S):
            best_S = S[:]
            best_used = {k: v.copy() for k, v in used.items()}
    
    # Phase 2: Local search on best solution
    S, used = local_search(best_S, best_used, time_limit=deadline - 10)
    
    # Phase 3: Try a few more restarts if time remains
    if time.time() < deadline - 60:
        for _ in range(3):
            order = list(range(num_points))
            rng.shuffle(order)
            S_new, used_new = greedy_construct(order, time_limit=deadline - 30)
            if len(S_new) > len(S):
                S, used = S_new, used_new
            else:
                # Try local search on this too if it's close
                if len(S_new) >= len(S) - 2 and time.time() < deadline - 20:
                    S_try, used_try = local_search(S_new, used_new, time_limit=deadline - 10)
                    if len(S_try) > len(S):
                        S, used = S_try, used_try
    
    return [points[i] for i in S]
# EVOLVE-BLOCK-END
