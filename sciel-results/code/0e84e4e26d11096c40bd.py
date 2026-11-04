import numpy as np
import time

def solve(a, b):
    n = len(a)
    
    # 1. Initialization: Rank Matching
    heavy = np.argsort(-a.sum(axis=1), kind="stable")
    central = np.argsort(b.sum(axis=1), kind="stable")
    p = np.empty(n, dtype=np.int64)
    p[heavy] = central
    
    def calc_cost(p):
        B_perm = b[np.ix_(p, p)]
        return np.sum(a * B_perm)

    current_cost = calc_cost(p)
    best_cost = current_cost
    best_p = p.copy()
    
    start_time = time.time()
    time_limit = 115.0
    
    rng = np.random.default_rng(42)
    
    # Local Search: Single Pass Best Improvement
    def local_search_best_improvement(p, cost):
        best_delta = 0
        best_i = -1
        best_j = -1
        
        # Precompute B_p for faster access within this pass
        B_p = b[np.ix_(p, p)]
        
        for i in range(n):
            for j in range(i + 1, n):
                u = p[i]
                v = p[j]
                diff_a = a[i] - a[j]
                diff_b = B_p[v, :] - B_p[u, :]
                S = np.dot(diff_a, diff_b)
                term_i = diff_a[i] * diff_b[i]
                term_j = diff_a[j] * diff_b[j]
                delta = S - term_i - term_j
                
                if delta < best_delta:
                    best_delta = delta
                    best_i = i
                    best_j = j
                    
        if best_i != -1:
            p[best_i], p[best_j] = p[best_j], p[best_i]
            cost += best_delta
            return p, cost, True
        else:
            return p, cost, False

    # Iterated Local Search
    p_curr = best_p.copy()
    cost_curr = best_cost
    
    while True:
        elapsed = time.time() - start_time
        if elapsed > time_limit:
            break
        
        # Run one pass of Best Improvement
        p_curr, cost_curr, improved = local_search_best_improvement(p_curr, cost_curr)
        
        if improved:
            if cost_curr < best_cost:
                best_cost = cost_curr
                best_p = p_curr.copy()
        else:
            # Local optimum reached, perturb
            # Perform 15 random swaps
            for _ in range(15):
                i = rng.integers(n)
                j = rng.integers(n)
                if i != j:
                    p_curr[i], p_curr[j] = p_curr[j], p_curr[i]
                    
            # Recalculate cost
            cost_curr = calc_cost(p_curr)
            
    return best_p.tolist()
