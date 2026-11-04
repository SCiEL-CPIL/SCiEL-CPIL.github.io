"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def solve(a, b):
    n = len(a)
    
    # Helper to compute full cost
    def compute_cost(p):
        B = b[np.ix_(p, p)]
        return int(np.sum(a * B))
        
    # Helper to compute delta cost for swapping i and j
    def delta_cost(i, j, p):
        u = p[i]
        v = p[j]
        
        # Vectorized delta calculation
        b_v_p = b[v, p]
        b_u_p = b[u, p]
        b_p_v = b[p, v]
        b_p_u = b[p, u]
        
        diff_v_u_out = b_v_p - b_u_p
        diff_v_u_in = b_p_v - b_p_u
        
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        mask[j] = False
        
        s1 = np.dot(a[i, mask], diff_v_u_out[mask])
        s2 = np.dot(a[mask, i], diff_v_u_in[mask])
        s3 = np.dot(a[j, mask], -diff_v_u_out[mask])
        s4 = np.dot(a[mask, j], -diff_v_u_in[mask])
        
        sum_part = s1 + s2 + s3 + s4
        
        cross_delta = a[i, j] * b[v, u] + a[j, i] * b[u, v] - (a[i, j] * b[u, v] + a[j, i] * b[v, u])
        
        return sum_part + cross_delta

    # Initial Solution: Greedy
    a_row_sums = a.sum(axis=1)
    b_row_sums = b.sum(axis=1)
    
    facility_order = np.argsort(-a_row_sums, kind='stable')
    location_order = np.argsort(b_row_sums, kind='stable')
    
    p = np.empty(n, dtype=np.int64)
    p[facility_order] = location_order
    
    current_cost = compute_cost(p)
    best_cost = current_cost
    best_p = p.copy()
    
    # Quick local search warm-up (Steepest Descent) to get out of bad greedy start
    for _ in range(50):
        improved = False
        for i in range(n - 1):
            for j in range(i + 1, n):
                d = delta_cost(i, j, p)
                if d < 0:
                    p[i], p[j] = p[j], p[i]
                    current_cost += d
                    if current_cost < best_cost:
                        best_cost = current_cost
                        best_p = p.copy()
                    improved = True
                    break
            if improved:
                break
        if not improved:
            break

    # Simulated Annealing Parameters
    start_time = time.time()
    time_limit = 115.0
    
    rng = np.random.default_rng(42)
    
    # Cooling parameters
    T = 1000.0
    T_min = 1.0
    alpha = 0.999 # Cooling rate
    
    # We will iterate until time limit
    while time.time() - start_time < time_limit:
        # Choose a random move
        i = rng.integers(n)
        j = rng.integers(n)
        while i == j:
            j = rng.integers(n)
        
        d = delta_cost(i, j, p)
        
        # Acceptance criterion
        if d < 0:
            # Always accept improving moves
            accept = True
        else:
            # Accept worsening moves with probability exp(-d/T)
            if T > T_min:
                accept = (rng.random() < np.exp(-d / T))
            else:
                accept = False
        
        if accept:
            p[i], p[j] = p[j], p[i]
            current_cost += d
            
            if current_cost < best_cost:
                best_cost = current_cost
                best_p = p.copy()
                
            # Cool down
            T *= alpha
            
            # If T is too low, restart with a higher temperature to escape local minima
            if T < T_min:
                # Simple restart: randomize a few positions or reset T
                # Here we just reset T to a fraction of the initial T to allow exploration
                T = 500.0 
        else:
            # Even if rejected, we might cool down slightly to progress
            T *= alpha
            if T < T_min:
                T = 500.0

    return best_p.tolist()
# EVOLVE-BLOCK-END
