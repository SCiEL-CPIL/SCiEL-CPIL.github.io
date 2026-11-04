"""Rewritten QAP solver using Tabu Search with vectorized delta evaluation."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def solve(a, b):
    n = len(a)
    
    # Precompute row sums for greedy initialization
    a_row_sums = a.sum(axis=1)
    b_row_sums = b.sum(axis=1)
    
    # Greedy initialization: sort facilities by descending flow, locations by ascending distance sum
    facility_order = np.argsort(-a_row_sums, kind='stable')
    location_order = np.argsort(b_row_sums, kind='stable')
    
    p = np.empty(n, dtype=np.int64)
    p[facility_order] = location_order
    
    # Compute initial cost
    B = b[np.ix_(p, p)]
    current_cost = int(np.sum(a * B))
    best_cost = current_cost
    best_p = p.copy()
    
    # Tabu Search parameters
    tabu_tenure = 20
    max_iter_no_improve = 500
    start_time = time.time()
    time_limit = 115.0
    
    # Tabu list: for each pair (i, j) with i < j, store the iteration when it becomes non-tabu
    # Use a 2D array of size n x n, only upper triangle used
    tabu_until = np.zeros((n, n), dtype=np.int64)
    
    rng = np.random.default_rng(42)
    
    # Helper to compute delta cost for swapping i and j
    def delta_cost(i, j, p):
        u = p[i]
        v = p[j]
        
        b_v_p = b[v, p]
        b_u_p = b[u, p]
        b_p_v = b[p, v]
        b_p_u = b[p, u]
        
        diff_out = b_v_p - b_u_p
        diff_in = b_p_v - b_p_u
        
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        mask[j] = False
        
        s1 = np.dot(a[i, mask], diff_out[mask])
        s2 = np.dot(a[mask, i], diff_in[mask])
        s3 = np.dot(a[j, mask], -diff_out[mask])
        s4 = np.dot(a[mask, j], -diff_in[mask])
        
        sum_part = s1 + s2 + s3 + s4
        
        cross_delta = a[i, j] * b[v, u] + a[j, i] * b[u, v] - (a[i, j] * b[u, v] + a[j, i] * b[v, u])
        
        return sum_part + cross_delta
    
    # Quick local search warm-up (Steepest Descent)
    for _ in range(30):
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
    
    iter_count = 0
    no_improve_count = 0
    
    while time.time() - start_time < time_limit:
        iter_count += 1
        
        # Find best non-tabu move (or aspiration)
        best_delta = float('inf')
        best_i = -1
        best_j = -1
        is_aspiration = False
        
        for i in range(n):
            # Vectorized computation of deltas for swapping i with all j != i
            u = p[i]
            
            b_u_p = b[u, p]
            b_p_u = b[p, u]
            
            # For each j, delta involves:
            # Terms with k (k != i, j): 
            #   a[i,k] * (b[p[j],p[k]] - b[u,p[k]]) + a[k,i] * (b[p[k],p[j]] - b[p[k],u])
            #   a[j,k] * (b[u,p[k]] - b[p[j],p[k]]) + a[k,j] * (b[p[k],u] - b[p[k],p[j]])
            # Cross term:
            #   a[i,j]*b[p[j],u] + a[j,i]*b[u,p[j]] - a[i,j]*b[u,p[j]] - a[j,i]*b[p[j],u]
            
            # Let's compute for all j at once
            # b_p_u is (n,), b_u_p is (n,)
            
            # For term s1-like: a[i, :] dot (b[p[j], :] - b[u, :]) for each j
            # This is: a[i, :] @ (B_p - b[u, :]) where B_p[j, k] = b[p[j], p[k]]
            # But we need to exclude j=i and handle the cross term separately.
            
            # Compute B_p = b[np.ix_(p, p)] is expensive each iteration.
            # Instead, let's loop j but vectorize over k.
            # Actually, let's just loop j since n=150 is small enough for inner loop.
            
            for j in range(n):
                if i == j:
                    continue
                
                # Check tabu
                ii, jj = (i, j) if i < j else (j, i)
                if iter_count < tabu_until[ii, jj]:
                    # Check aspiration
                    d = delta_cost(i, j, p)
                    if current_cost + d < best_cost:
                        if d < best_delta:
                            best_delta = d
                            best_i = i
                            best_j = j
                            is_aspiration = True
                    continue
                
                d = delta_cost(i, j, p)
                if d < best_delta:
                    best_delta = d
                    best_i = i
                    best_j = j
                    is_aspiration = False
        
        if best_i == -1:
            # No valid move found, diversify
            for _ in range(10):
                i = rng.integers(n)
                j = rng.integers(n)
                while i == j:
                    j = rng.integers(n)
                p[i], p[j] = p[j], p[i]
                current_cost += delta_cost(i, j, p)
            no_improve_count += 10
            if no_improve_count > max_iter_no_improve:
                break
            continue
        
        # Apply best move
        i, j = best_i, best_j
        p[i], p[j] = p[j], p[i]
        current_cost += best_delta
        
        # Update tabu
        ii, jj = (i, j) if i < j else (j, i)
        tabu_until[ii, jj] = iter_count + tabu_tenure
        
        if current_cost < best_cost:
            best_cost = current_cost
            best_p = p.copy()
            no_improve_count = 0
        else:
            no_improve_count += 1
        
        if no_improve_count > max_iter_no_improve:
            # Diversify: random swaps
            for _ in range(50):
                i = rng.integers(n)
                j = rng.integers(n)
                while i == j:
                    j = rng.integers(n)
                p[i], p[j] = p[j], p[i]
                current_cost += delta_cost(i, j, p)
            no_improve_count = 0
            # Clear tabu list
            tabu_until.fill(0)
    
    return best_p.tolist()
# EVOLVE-BLOCK-END
