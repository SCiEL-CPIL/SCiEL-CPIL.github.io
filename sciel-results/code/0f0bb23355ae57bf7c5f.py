"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def solve(a, b):
    n = len(a)
    a = np.asarray(a, dtype=np.int64)
    b = np.asarray(b, dtype=np.int64)
    
    start_time = time.time()
    timeout = 115.0  # safety margin before 120s kill
    
    def calc_cost(p):
        b_perm = b[np.ix_(p, p)]
        return int(np.sum(a * b_perm))
    
    def get_delta(p, i, j):
        u = p[i]
        v = p[j]
        
        b_v_p = b[v, p]
        b_u_p = b[u, p]
        diff_b = b_v_p - b_u_p
        
        diff_a = a[i] - a[j]
        
        sum_all = np.dot(diff_a, diff_b)
        
        term_i = (a[i, i] - a[j, i]) * (b[v, u] - b[u, u])
        term_j = (a[i, j] - a[j, j]) * (b[v, v] - b[u, v])
        
        delta = 2 * (sum_all - term_i - term_j)
        return delta
    
    def local_search(p, current_cost, max_time=None):
        """Best-improvement 2-opt local search."""
        if max_time is None:
            max_time = timeout
        
        while time.time() < max_time:
            improved = False
            for i in range(n):
                if time.time() >= max_time:
                    return p, current_cost
                
                u = p[i]
                b_u_p = b[u, p]
                a_i = a[i]
                
                best_delta = 0
                best_j = -1
                
                for j in range(i + 1, n):
                    v = p[j]
                    diff_b = b[v, p] - b_u_p
                    diff_a = a_i - a[j]
                    sum_all = np.dot(diff_a, diff_b)
                    
                    term_i = (a[i, i] - a[j, i]) * (b[v, u] - b[u, u])
                    term_j = (a[i, j] - a[j, j]) * (b[v, v] - b[u, v])
                    
                    delta = 2 * (sum_all - term_i - term_j)
                    
                    if delta < best_delta:
                        best_delta = delta
                        best_j = j
                
                if best_j != -1:
                    j = best_j
                    p[i], p[j] = p[j], p[i]
                    current_cost += best_delta
                    improved = True
                    break
            
            if not improved:
                break
        
        return p, current_cost
    
    def spectral_init():
        """Generate initial solution using spectral decomposition."""
        # Compute Laplacian-like matrices
        # For QAP, we can use the approach of finding eigenvectors
        
        # Simple rank-matching as fallback
        flow_sums = a.sum(axis=1)
        dist_sums = b.sum(axis=1)
        
        heavy_facilities = np.argsort(-flow_sums, kind="stable")
        central_locations = np.argsort(dist_sums, kind="stable")
        
        p = np.empty(n, dtype=np.int64)
        p[heavy_facilities] = central_locations
        
        return p
    
    def multi_start_ils(p_init, current_cost, num_restarts=None):
        """Iterated Local Search with multiple restarts."""
        if num_restarts is None:
            num_restarts = 50  # will be limited by time
        
        best_p = p_init.copy()
        best_cost = current_cost
        no_improve = 0
        
        for restart in range(num_restarts):
            if time.time() >= timeout - 10.0:
                break
            
            # Perturbation strength based on stagnation
            if no_improve < 3:
                num_swaps = 3
            elif no_improve < 8:
                num_swaps = 5
            elif no_improve < 15:
                num_swaps = 8
            else:
                num_swaps = 12
            
            # Apply random swaps
            p = best_p.copy()
            cost = best_cost
            for _ in range(num_swaps):
                i = np.random.randint(n)
                j = np.random.randint(n)
                if i == j:
                    continue
                delta = get_delta(p, i, j)
                p[i], p[j] = p[j], p[i]
                cost += delta
            
            # Local search
            p, cost = local_search(p, cost, max_time=timeout - 5.0)
            
            if cost < best_cost:
                best_cost = cost
                best_p = p.copy()
                no_improve = 0
            else:
                no_improve += 1
            
            # If stuck for too long, try a bigger perturbation
            if no_improve >= 20:
                # Random restart from scratch
                p = spectral_init()
                cost = calc_cost(p)
                p, cost = local_search(p, cost, max_time=timeout - 5.0)
                if cost < best_cost:
                    best_cost = cost
                    best_p = p.copy()
                no_improve = 0
        
        return best_p, best_cost
    
    # Generate multiple diverse initial solutions
    np.random.seed(42)
    
    best_overall_p = None
    best_overall_cost = float('inf')
    
    # Strategy 1: Rank matching
    p1 = spectral_init()
    cost1 = calc_cost(p1)
    p1, cost1 = local_search(p1, cost1, max_time=timeout - 60.0)
    
    if cost1 < best_overall_cost:
        best_overall_cost = cost1
        best_overall_p = p1.copy()
    
    # Strategy 2: Try a few different random initializations
    for attempt in range(3):
        if time.time() >= timeout - 40.0:
            break
        
        # Random initialization
        p_rand = np.random.permutation(n).astype(np.int64)
        cost_rand = calc_cost(p_rand)
        
        # Quick local search
        p_rand, cost_rand = local_search(p_rand, cost_rand, max_time=timeout - 30.0)
        
        if cost_rand < best_overall_cost:
            best_overall_cost = cost_rand
            best_overall_p = p_rand.copy()
    
    # Strategy 3: ILS from the best solution found so far
    if time.time() < timeout - 10.0:
        p_ils, cost_ils = multi_start_ils(best_overall_p, best_overall_cost, num_restarts=30)
        
        if cost_ils < best_overall_cost:
            best_overall_cost = cost_ils
            best_overall_p = p_ils.copy()
    
    # Final check: if we have time, do one more polish
    if time.time() < timeout - 3.0 and best_overall_p is not None:
        p_final = best_overall_p.copy()
        cost_final = best_overall_cost
        p_final, cost_final = local_search(p_final, cost_final, max_time=timeout - 1.0)
        
        if cost_final < best_overall_cost:
            best_overall_cost = cost_final
            best_overall_p = p_final.copy()
    
    return best_overall_p.tolist()
# EVOLVE-BLOCK-END
