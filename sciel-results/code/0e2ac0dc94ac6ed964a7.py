"""
QAP solver: Linearized Greedy Construction + Fast Incremental 2-Opt Local Search.
Combines the structural idea of the current program (greedy construction for diverse
initial solutions) with the refinement machinery from the inspiration (O(n) incremental
updates for local search using precomputed dot products).
"""
# EVOLVE-BLOCK-START
import numpy as np
import time


def compute_cost(a, b, p):
    """Compute QAP cost: sum_{i,j} a[i,j] * b[p[i], p[j]]"""
    return np.sum(a * b[np.ix_(p, p)])


def greedy_construct(a, b, rng, noise_scale=0.0):
    """
    Constructs a permutation using linearized greedy approach.
    Maintains a cost matrix D where D[i,j] is the marginal cost of assigning
    facility i to location j given the current partial assignment.
    """
    n = a.shape[0]
    p = np.full(n, -1, dtype=np.int64)
    assigned_fac = np.zeros(n, dtype=bool)
    assigned_loc = np.zeros(n, dtype=bool)
    
    # D[i, j] = sum_{k in assigned} (a[i, k] * b[j, p[k]] + a[k, i] * b[p[k], j])
    D = np.zeros((n, n), dtype=np.float64)
    
    # If noise_scale > 0, add noise to D initially or during selection to diversify
    if noise_scale > 0:
        D += rng.random((n, n)) * noise_scale

    for step in range(n):
        # Mask out assigned facilities and locations
        mask = np.ones((n, n), dtype=bool)
        mask[assigned_fac, :] = False
        mask[:, assigned_loc] = False
        
        # Find the minimum
        D_masked = np.where(mask, D, np.inf)
        
        flat_idx = np.argmin(D_masked)
        i, j = np.unravel_index(flat_idx, (n, n))
        
        # Assign
        p[i] = j
        assigned_fac[i] = True
        assigned_loc[j] = True
        
        # Update D for all unassigned i', j'
        # D[i', j'] += a[i', i] * b[j', j] + a[i, i'] * b[j, j']
        D += np.outer(a[:, i], b[:, j]) + np.outer(a[i, :], b[j, :])
        
    return p


def local_search_2opt(a, b, p, time_limit):
    """
    Fast 2-opt local search with O(n) incremental updates.
    Uses precomputed dot_row and dot_col for O(1) delta computation per swap.
    """
    n = len(p)
    p = p.copy()
    current_cost = compute_cost(a, b, p)
    best_p = p.copy()
    best_cost = current_cost
    
    start_time = time.time()
    deadline = start_time + time_limit
    
    # Ensure float64 for consistent arithmetic
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    
    # Precompute dot_row and dot_col
    # dot_row[i, l] = sum_k a[i, k] * b[l, p[k]]
    # dot_col[i, l] = sum_k a[k, i] * b[p[k], l]
    
    def recompute_precomp(p):
        b_col_p = b[:, p]  # (n, n): [l, k] = b[l, p[k]]
        b_row_p = b[p, :]  # (n, n): [k, l] = b[p[k], l]
        dot_row = a @ b_col_p.T
        dot_col = a.T @ b_row_p
        return dot_row, dot_col
    
    dot_row, dot_col = recompute_precomp(p)
    
    # Precompute b diagonal
    b_diag = np.diag(b)
    
    max_passes = 1000
    pass_count = 0
    
    while pass_count < max_passes:
        if time.time() > deadline:
            break
            
        improved = False
        pass_count += 1
        
        for i in range(n):
            if time.time() > deadline:
                break
            for j in range(i + 1, n):
                if time.time() > deadline:
                    break
                    
                u = p[i]
                v = p[j]
                
                buu = b_diag[u]
                bvv = b_diag[v]
                buv = b[u, v]
                bvu = b[v, u]
                
                # Compute delta using precomputed dot products
                delta = 0.0
                # Terms 1-4: direct i-j interactions
                delta += a[i, i] * (bvv - buu)
                delta += a[i, j] * (bvu - buv)
                delta += a[j, i] * (buv - bvu)
                delta += a[j, j] * (buu - bvv)
                
                # Term 5: sum_{k not in {i,j}} a[i,k] * (b[v, p[k]] - b[u, p[k]])
                delta += (dot_row[i, v] - dot_row[i, u]) - a[i,i]*(bvv - buu) - a[i,j]*(bvu - buv)
                
                # Term 6: sum_{k not in {i,j}} a[j,k] * (b[u, p[k]] - b[v, p[k]])
                delta += (dot_row[j, u] - dot_row[j, v]) - a[j,i]*(buv - bvu) - a[j,j]*(buu - bvv)
                
                # Term 7: sum_{k not in {i,j}} a[k,i] * (b[p[k], v] - b[p[k], u])
                delta += (dot_col[i, v] - dot_col[i, u]) - a[i,i]*(bvv - buu) - a[j,i]*(bvu - buv)
                
                # Term 8: sum_{k not in {i,j}} a[k,j] * (b[p[k], u] - b[p[k], v])
                delta += (dot_col[j, u] - dot_col[j, v]) - a[i,j]*(buv - bvu) - a[j,j]*(buu - bvv)
                
                if delta < -1e-9:
                    # Accept the swap
                    p[i], p[j] = p[j], p[i]
                    current_cost += delta
                    improved = True
                    
                    if current_cost < best_cost:
                        best_cost = current_cost
                        best_p = p.copy()
                    
                    # Incremental update of dot_row and dot_col in O(n)
                    # When p[i] and p[j] swap (u <-> v):
                    #
                    # dot_row[k, l] = sum_m a[k,m] * b[l, p[m]]
                    # For k != i,j: dot_row[k, l] changes by:
                    #   a[k,i]*(b[l,v] - b[l,u]) + a[k,j]*(b[l,u] - b[l,v])
                    #   = (a[k,i] - a[k,j]) * (b[l,v] - b[l,u])
                    # For k = i: dot_row[i, l] changes by:
                    #   a[i,i]*(b[l,v] - b[l,u]) - a[i,j]*(b[l,v] - b[l,u])
                    #   = (a[i,i] - a[i,j]) * (b[l,v] - b[l,u])
                    # For k = j: dot_row[j, l] changes by:
                    #   a[j,i]*(b[l,u] - b[l,v]) - a[j,j]*(b[l,u] - b[l,v])
                    #   = (a[j,i] - a[j,j]) * (b[l,u] - b[l,v])
                    #
                    # Similarly for dot_col.
                    
                    # Update dot_row
                    # For all l:
                    # dot_row[:, l] += (a[:, i] - a[:, j]) * (b[l, v] - b[l, u])
                    # But we need to handle k=i and k=j specially.
                    
                    # Let's do it more carefully:
                    # dot_row[k, l] = sum_m a[k,m] * b[l, p[m]]
                    # After swap, p[i]=v, p[j]=u
                    # 
                    # For k not in {i,j}:
                    #   new_dot_row[k, l] = old_dot_row[k, l] + a[k,i]*(b[l,v] - b[l,u]) + a[k,j]*(b[l,u] - b[l,v])
                    #   = old_dot_row[k, l] + (a[k,i] - a[k,j]) * (b[l,v] - b[l,u])
                    #
                    # For k = i:
                    #   new_dot_row[i, l] = sum_m a[i,m] * b[l, p[m]]
                    #   = a[i,i]*b[l,v] + a[i,j]*b[l,u] + sum_{m not in {i,j}} a[i,m]*b[l,p[m]]
                    #   old_dot_row[i, l] = a[i,i]*b[l,u] + a[i,j]*b[l,v] + sum_{m not in {i,j}} a[i,m]*b[l,p[m]]
                    #   diff = a[i,i]*(b[l,v] - b[l,u]) + a[i,j]*(b[l,u] - b[l,v])
                    #        = (a[i,i] - a[i,j]) * (b[l,v] - b[l,u])
                    #
                    # For k = j:
                    #   new_dot_row[j, l] = a[j,i]*b[l,u] + a[j,j]*b[l,v] + sum_{m not in {i,j}} a[j,m]*b[l,p[m]]
                    #   old_dot_row[j, l] = a[j,i]*b[l,v] + a[j,j]*b[l,u] + sum_{m not in {i,j}} a[j,m]*b[l,p[m]]
                    #   diff = a[j,i]*(b[l,u] - b[l,v]) + a[j,j]*(b[l,v] - b[l,u])
                    #        = (a[j,j] - a[j,i]) * (b[l,v] - b[l,u])
                    
                    # So for all k:
                    #   dot_row[k, l] += (a[k,i] - a[k,j]) * (b[l,v] - b[l,u])
                    #   except k=i and k=j have different coefficients.
                    
                    # Actually, let's just recompute the affected rows.
                    # dot_row[i, :] and dot_row[j, :] need special handling.
                    # For k not in {i,j}, dot_row[k, :] += (a[k,i] - a[k,j]) * (b[:, v] - b[:, u])
                    
                    # Update dot_row for k not in {i,j}
                    mask_k = np.ones(n, dtype=bool)
                    mask_k[i] = False
                    mask_k[j] = False
                    dot_row[mask_k, :] += np.outer(a[mask_k, i] - a[mask_k, j], b[:, v] - b[:, u])
                    
                    # Update dot_row for k=i
                    dot_row[i, :] = a[i, i] * b[:, v] + a[i, j] * b[:, u] + dot_row[i, :].copy()
                    # Wait, that's not right. Let me recompute from scratch for i and j.
                    # Actually, it's easier to just recompute dot_row[i, :] and dot_row[j, :] directly.
                    dot_row[i, :] = np.sum(a[i, :] * b[:, p].T, axis=1)
                    dot_row[j, :] = np.sum(a[j, :] * b[:, p].T, axis=1)
                    
                    # Update dot_col similarly
                    # dot_col[k, l] = sum_m a[m, k] * b[p[m], l]
                    # After swap, p[i]=v, p[j]=u
                    #
                    # For l not in {i,j}:
                    #   new_dot_col[k, l] = old_dot_col[k, l] + a[i,k]*(b[v,l] - b[u,l]) + a[j,k]*(b[u,l] - b[v,l])
                    #   = old_dot_col[k, l] + (a[i,k] - a[j,k]) * (b[v,l] - b[u,l])
                    #
                    # For l = i:
                    #   new_dot_col[k, i] = sum_m a[m,k] * b[p[m], i]
                    #   = a[i,k]*b[v,i] + a[j,k]*b[u,i] + sum_{m not in {i,j}} a[m,k]*b[p[m],i]
                    #   old_dot_col[k, i] = a[i,k]*b[u,i] + a[j,k]*b[v,i] + sum_{m not in {i,j}} a[m,k]*b[p[m],i]
                    #   diff = a[i,k]*(b[v,i] - b[u,i]) + a[j,k]*(b[u,i] - b[v,i])
                    #        = (a[i,k] - a[j,k]) * (b[v,i] - b[u,i])
                    #
                    # For l = j:
                    #   new_dot_col[k, j] = a[i,k]*b[v,j] + a[j,k]*b[u,j] + sum_{m not in {i,j}} a[m,k]*b[p[m],j]
                    #   old_dot_col[k, j] = a[i,k]*b[u,j] + a[j,k]*b[v,j] + sum_{m not in {i,j}} a[m,k]*b[p[m],j]
                    #   diff = a[i,k]*(b[v,j] - b[u,j]) + a[j,k]*(b[u,j] - b[v,j])
                    #        = (a[i,k] - a[j,k]) * (b[v,j] - b[u,j])
                    
                    # So for all l:
                    #   dot_col[:, l] += (a[i, l] - a[j, l]) * (b[v, l] - b[u, l])
                    #   except l=i and l=j have different coefficients.
                    
                    # Update dot_col for l not in {i,j}
                    mask_l = np.ones(n, dtype=bool)
                    mask_l[i] = False
                    mask_l[j] = False
                    dot_col[:, mask_l] += np.outer(a[i, mask_l] - a[j, mask_l], b[v, mask_l] - b[u, mask_l])
                    
                    # Update dot_col for l=i and l=j
                    dot_col[:, i] = np.sum(a[:, i] * b[p, i], axis=0)
                    dot_col[:, j] = np.sum(a[:, j] * b[p, j], axis=0)
        
        if not improved:
            break
            
    return best_p, best_cost


def solve(a, b):
    n = len(a)
    start_time = time.time()
    # Safety margin: 5 seconds
    deadline = start_time + 145.0
    
    # Ensure float64 for consistent arithmetic
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    
    best_p = None
    best_cost = np.inf
    
    # Strategy:
    # 1. Generate several diverse initial solutions using Greedy Construction with noise.
    # 2. Run local search on each.
    # 3. Keep the best.
    
    rng = np.random.default_rng(42)
    
    # Initial simple heuristic: rank matching
    heavy = np.argsort(-a.sum(axis=1), kind="stable")
    central = np.argsort(b.sum(axis=1), kind="stable")
    p_init = np.empty(n, dtype=np.int64)
    p_init[heavy] = central
    
    # Run local search on rank matching first to get a baseline
    remaining = deadline - time.time()
    if remaining > 5.0:
        p1, c1 = local_search_2opt(a, b, p_init, remaining * 0.2)
        if c1 < best_cost:
            best_cost = c1
            best_p = p1
    
    # Now try greedy constructions
    noise_levels = [0.0, 1.0, 5.0, 10.0, 50.0]
    
    for noise in noise_levels:
        remaining = deadline - time.time()
        if remaining < 10.0:
            break
            
        # Generate solution
        p_g = greedy_construct(a, b, rng, noise_scale=noise)
        
        # Run local search with a time slice
        time_slice = min(15.0, remaining * 0.3)
        
        if time_slice < 1.0:
            continue
            
        p2, c2 = local_search_2opt(a, b, p_g, time_slice)
        
        if c2 < best_cost:
            best_cost = c2
            best_p = p2
            
    # Final polish with all remaining time
    remaining = deadline - time.time()
    if remaining > 1.0 and best_p is not None:
        best_p, best_cost = local_search_2opt(a, b, best_p, remaining)
        
    return best_p.tolist()
# EVOLVE-BLOCK-END
