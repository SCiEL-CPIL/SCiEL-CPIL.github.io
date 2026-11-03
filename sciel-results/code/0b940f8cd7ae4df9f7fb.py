"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def solve(a, b):
    n = len(a)
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    
    # Initial solution: Rank matching
    # Sort facilities by sum of flows, locations by sum of distances
    heavy = np.argsort(-a.sum(axis=1), kind="stable")
    central = np.argsort(b.sum(axis=1), kind="stable")
    p = np.empty(n, dtype=np.int64)
    p[heavy] = central
    
    # Compute initial cost
    def compute_cost(p):
        # Sum a[i, j] * b[p[i], p[j]]
        # Vectorized:
        # b_perm = b[np.ix_(p, p)]
        # return np.sum(a * b_perm)
        b_perm = b[p][:, p]
        return np.sum(a * b_perm)
        
    current_cost = compute_cost(p)
    best_cost = current_cost
    best_p = p.copy()
    
    # Precomputation for fast delta
    # B_p[k, l] = b[k, p[l]]
    # X[i, u] = sum_k a[i, k] * b[u, p[k]] = (a @ B_p)[i, u]
    B_p = b[:, p]
    X = a @ B_p
    
    # Tabu Search parameters
    tabu_tenure = 20
    tabu_list = [] # (i, j, time)
    iteration = 0
    deadline = time.time() + 110.0
    
    # Helper to get delta
    # Delta_ij = X[i, u] - X[i, v] - X[j, u] + X[j, v] - corrections
    # u = p[i], v = p[j]
    # Corrections:
    # - a[i, i] * (b[u, u] - b[v, u])
    # - a[i, j] * (b[u, v] - b[v, v])
    # + a[j, i] * (b[u, u] - b[v, u])
    # + a[j, j] * (b[u, v] - b[v, v])
    
    while time.time() < deadline:
        iteration += 1
        best_delta = np.inf
        best_i = -1
        best_j = -1
        best_u = -1
        best_v = -1
        
        # Find best move
        for i in range(n):
            u = p[i]
            for j in range(i + 1, n):
                v = p[j]
                
                # Check tabu
                # Simple check: if (i, j) or (j, i) in tabu_list
                # Optimization: Use a set for tabu pairs?
                # For n=100, linear scan of tabu list (size ~20) is fast.
                is_tabu = False
                for ti, tj, t_time in tabu_list:
                    if (ti == i and tj == j) or (ti == j and tj == i):
                        is_tabu = True
                        break
                
                # Calculate delta
                # Main term
                d = X[i, u] - X[i, v] - X[j, u] + X[j, v]
                
                # Corrections
                # b[u, u], b[u, v], b[v, u], b[v, v]
                # Note b is symmetric, so b[u, v] == b[v, u]
                buu = b[u, u]
                buv = b[u, v]
                bvv = b[v, v]
                
                corr = 0
                if a[i, i] != 0:
                    corr -= a[i, i] * (buu - buv) # b[v, u] is buv
                if a[i, j] != 0:
                    corr -= a[i, j] * (buv - bvv)
                if a[j, i] != 0:
                    corr += a[j, i] * (buu - buv)
                if a[j, j] != 0:
                    corr += a[j, j] * (buv - bvv)
                
                d += corr
                
                if d < best_delta:
                    # Aspiration criterion: if d improves best_cost, accept even if tabu
                    if not is_tabu or d < best_cost - current_cost: # Wait, aspiration is usually d < best_cost_found_so_far
                         # Actually, standard aspiration: accept if d + current_cost < best_cost
                         # Here d is delta. new_cost = current_cost + d.
                         # If new_cost < best_cost, it's aspiration.
                         pass 
                    
                    # We need to decide if we accept tabu.
                    # Let's store the best non-tabu and best overall separately?
                    # Or just pick best non-tabu. If no non-tabu, pick best overall.
                    
                    # To simplify, let's just track best non-tabu.
                    # If is_tabu, we skip unless it's the only option?
                    # Better: Track best_non_tabu_delta and best_overall_delta.
                    
                    # Let's refine the loop to track both.
                    pass
        
        # The above loop is O(n^2 * tabu_size).
        # Let's rewrite the inner loop properly.
        
        best_non_tabu_delta = np.inf
        best_non_tabu_i = -1
        best_non_tabu_j = -1
        best_overall_delta = np.inf
        best_overall_i = -1
        best_overall_j = -1
        
        for i in range(n):
            u = p[i]
            for j in range(i + 1, n):
                v = p[j]
                
                d = X[i, u] - X[i, v] - X[j, u] + X[j, v]
                
                buu = b[u, u]
                buv = b[u, v]
                bvv = b[v, v]
                
                corr = 0
                if a[i, i] != 0:
                    corr -= a[i, i] * (buu - buv)
                if a[i, j] != 0:
                    corr -= a[i, j] * (buv - bvv)
                if a[j, i] != 0:
                    corr += a[j, i] * (buu - buv)
                if a[j, j] != 0:
                    corr += a[j, j] * (buv - bvv)
                
                d += corr
                
                if d < best_overall_delta:
                    best_overall_delta = d
                    best_overall_i = i
                    best_overall_j = j
                    
                # Check tabu
                is_tabu = False
                for ti, tj, t_time in tabu_list:
                    if (ti == i and tj == j) or (ti == j and tj == i):
                        is_tabu = True
                        break
                        
                if not is_tabu:
                    if d < best_non_tabu_delta:
                        best_non_tabu_delta = d
                        best_non_tabu_i = i
                        best_non_tabu_j = j
                        
        # Decide move
        if best_non_tabu_delta < np.inf:
            i, j = best_non_tabu_i, best_non_tabu_j
            delta = best_non_tabu_delta
        else:
            i, j = best_overall_i, best_overall_j
            delta = best_overall_delta
            
        if i == -1:
            break
            
        # Apply move
        u = p[i]
        v = p[j]
        p[i] = v
        p[j] = u
        
        # Update cost
        current_cost += delta
        
        if current_cost < best_cost:
            best_cost = current_cost
            best_p = p.copy()
            
        # Update Tabu List
        tabu_list.append((i, j, iteration))
        if len(tabu_list) > tabu_tenure:
            tabu_list.pop(0)
            
        # Update X
        # B_p columns i and j swapped
        B_p[:, i], B_p[:, j] = B_p[:, j], B_p[:, i]
        
        # X[:, i] = a @ B_p[:, i]
        # X[:, j] = a @ B_p[:, j]
        X[:, i] = a @ B_p[:, i]
        X[:, j] = a @ B_p[:, j]
        
    return best_p.tolist()
# EVOLVE-BLOCK-END
