import numpy as np
import time
from scipy.optimize import linear_sum_assignment

def solve(a, b):
    n = len(a)
    deadline = time.time() + 115.0
    
    # Precompute
    a_sym = a + a.T
    b_sym = b + b.T # Not strictly needed if b is symmetric, but safe
    
    def compute_cost(p):
        return int(np.sum(a * b[np.ix_(p, p)]))
    
    # Initial Solution
    flow_cent = a.sum(axis=1).astype(np.float64)
    dist_cent = b.sum(axis=1).astype(np.float64)
    C = np.outer(flow_cent, dist_cent)
    row_ind, col_ind = linear_sum_assignment(C)
    p = np.empty(n, dtype=np.int64)
    p[row_ind] = col_ind
    
    best_p = p.copy()
    best_cost = compute_cost(best_p)
    
    # 2-Swap Local Search (Vectorized)
    def local_search_2swap(p, max_iters=100):
        for _ in range(max_iters):
            if time.time() > deadline - 5.0:
                break
            # Compute deltas
            Bp = b[np.ix_(p, p)]
            M1 = a.T @ Bp
            M2 = a @ Bp
            
            # T1, T2, T3 logic from current program
            # ... (reuse logic from current program)
            
            # Find best
            # ...
            
            if no_improvement: break
        return p

    # 3-Cycle Local Search (Vectorized O(n^3))
    def local_search_3cycle(p, max_iters=10):
        for _ in range(max_iters):
            if time.time() > deadline - 5.0:
                break
            
            Bp = b[np.ix_(p, p)]
            M = a_sym @ Bp # (n,n)
            
            # Build Delta Tensor
            # Indices i, j, k
            # We want D[i,j,k]
            
            # External Part
            # M[i,j] - M[i,i]
            # ...
            
            # Find best
            # ...
            
            if no_improvement: break
        return p

    # Main Loop
    while time.time() < deadline - 5.0:
        # Polish with 2-swap
        p = local_search_2swap(p)
        
        # Try 3-cycle
        p_new = local_search_3cycle(p)
        cost_new = compute_cost(p_new)
        
        if cost_new < best_cost:
            best_cost = cost_new
            best_p = p_new.copy()
            p = p_new
        else:
            # Kick
            # Random swaps
            for _ in range(3):
                i, j = np.random.choice(n, 2, replace=False)
                p[i], p[j] = p[j], p[i]
            # Refine
            p = local_search_2swap(p)
            cost = compute_cost(p)
            if cost < best_cost:
                best_cost = cost
                best_p = p.copy()

    return best_p.tolist()
