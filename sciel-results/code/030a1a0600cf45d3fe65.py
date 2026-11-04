"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def solve(a, b):
    n = len(a)
    start_time = time.time()
    time_limit = 115.0
    rng = np.random.default_rng(12345)
    
    def compute_cost(p):
        B = b[np.ix_(p, p)]
        return int(np.sum(a * B))
    
    def delta_cost_fast(i, j, p, contrib):
        """
        Fast delta cost using precomputed contribution matrix.
        contrib[k] = sum of a[k,m]*b[p[k],p[m]] + a[m,k]*b[p[m],p[k]] for m != k
        Actually, let's use a different approach: maintain c[i] = sum_j a[i,j]*b[p[i],p[j]] + sum_j a[j,i]*b[p[j],p[i]]
        """
        u = p[i]
        v = p[j]
        
        # The change in cost when swapping facilities i and j
        # Old contribution involving i: sum_{k!=i} a[i,k]*b[u,p[k]] + sum_{k!=i} a[k,i]*b[p[k],u]
        # New contribution involving i (now at v): sum_{k!=i} a[i,k]*b[v,p[k]] + sum_{k!=i} a[k,i]*b[p[k],v]
        # Similarly for j.
        
        # Use vectorized computation
        p_arr = p
        b_u = b[u, :]
        b_v = b[v, :]
        b_p_u = b[:, u]
        b_p_v = b[:, v]
        
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        mask[j] = False
        
        # Delta for facility i moving from u to v
        delta_i = np.dot(a[i, mask], b_v[mask] - b_u[mask]) + np.dot(a[mask, i], b_p_v[mask] - b_p_u[mask])
        
        # Delta for facility j moving from v to u
        delta_j = np.dot(a[j, mask], b_u[mask] - b_v[mask]) + np.dot(a[mask, j], b_p_u[mask] - b_p_v[mask])
        
        # Cross term between i and j
        old_cross = a[i, j] * b[u, v] + a[j, i] * b[v, u]
        new_cross = a[i, j] * b[v, u] + a[j, i] * b[u, v]
        cross_delta = new_cross - old_cross
        
        return int(delta_i + delta_j + cross_delta)
    
    def simulated_annealing(p_init, t_start, t_end, cooling_rate, max_iter):
        p = p_init.copy()
        current_cost = compute_cost(p)
        best_cost = current_cost
        best_p = p.copy()
        
        current_temp = t_start
        iteration = 0
        
        while current_temp > t_end and iteration < max_iter and (time.time() - start_time) < time_limit:
            # Random swap
            i = rng.integers(n)
            j = rng.integers(n)
            while i == j:
                j = rng.integers(n)
            
            delta = delta_cost_fast(i, j, p, None)
            
            # Accept if improving or with probability exp(-delta/temp)
            if delta < 0 or (current_temp > 0 and rng.random() < np.exp(-delta / current_temp)):
                p[i], p[j] = p[j], p[i]
                current_cost += delta
                
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_p = p.copy()
            
            # Cool down
            current_temp *= cooling_rate
            iteration += 1
        
        return best_p, best_cost
    
    # Strategy: Multiple restarts with different initial solutions
    # 1. Greedy initialization
    # 2. Random initialization
    # 3. Greedy with random perturbation
    
    best_overall_cost = float('inf')
    best_overall_p = None
    
    # Precompute row sums for greedy
    a_row_sums = a.sum(axis=1)
    b_row_sums = b.sum(axis=1)
    
    # Run multiple SA restarts
    num_restarts = 5
    restarts_done = 0
    
    for restart in range(num_restarts):
        if (time.time() - start_time) >= time_limit:
            break
        
        # Different initializations
        if restart == 0:
            # Greedy: sort facilities by a_row_sums desc, locations by b_row_sums asc
            facility_order = np.argsort(-a_row_sums, kind='stable')
            location_order = np.argsort(b_row_sums, kind='stable')
            p_init = np.empty(n, dtype=np.int64)
            p_init[facility_order] = location_order
        elif restart == 1:
            # Random
            p_init = rng.permutation(n)
        elif restart == 2:
            # Greedy with random swap
            facility_order = np.argsort(-a_row_sums, kind='stable')
            location_order = np.argsort(b_row_sums, kind='stable')
            p_init = np.empty(n, dtype=np.int64)
            p_init[facility_order] = location_order
            # Apply a few random swaps
            for _ in range(10):
                i = rng.integers(n)
                j = rng.integers(n)
                while i == j:
                    j = rng.integers(n)
                p_init[i], p_init[j] = p_init[j], p_init[i]
        elif restart == 3:
            # Sort by different criteria: a_row_sums * b_row_sums
            scores = a_row_sums * b_row_sums
            facility_order = np.argsort(-scores, kind='stable')
            location_order = np.arange(n)
            p_init = np.empty(n, dtype=np.int64)
            p_init[facility_order] = location_order
        else:
            # Another random
            p_init = rng.permutation(n)
        
        # SA parameters
        t_start = 1000.0
        t_end = 1.0
        cooling_rate = 0.995
        max_iter = 50000
        
        p_best, cost_best = simulated_annealing(p_init, t_start, t_end, cooling_rate, max_iter)
        
        if cost_best < best_overall_cost:
            best_overall_cost = cost_best
            best_overall_p = p_best.copy()
        
        restarts_done += 1
    
    if best_overall_p is None:
        # Fallback
        best_overall_p = np.arange(n)
    
    return best_overall_p.tolist()
# EVOLVE-BLOCK-END
