"""
Rewritten QAP solver: Linearized Greedy Construction + Fast Vectorized 2-Opt.
Uses a dynamic cost matrix update for O(n^3) greedy construction to generate
high-quality diverse initial solutions, followed by an optimized local search.
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
        # We want to find min D[i, j] where not assigned_fac[i] and not assigned_loc[j]
        
        # Create a mask
        mask = np.ones((n, n), dtype=bool)
        mask[assigned_fac, :] = False
        mask[:, assigned_loc] = False
        
        # Find the minimum
        # np.argmin on masked array is tricky, so we set non-masked to inf
        D_masked = np.where(mask, D, np.inf)
        
        flat_idx = np.argmin(D_masked)
        i, j = np.unravel_index(flat_idx, (n, n))
        
        # Assign
        p[i] = j
        assigned_fac[i] = True
        assigned_loc[j] = True
        
        # Update D for all unassigned i', j'
        # D[i', j'] += a[i', i] * b[j', j] + a[i, i'] * b[j, j']
        
        # Vectorized update
        # Term 1: a[:, i] * b[:, j] -> shape (n, 1) * (1, n) -> (n, n)
        # But we only care about unassigned. The update formula is valid for all, 
        # but we will mask out assigned in the next step's selection.
        
        D += np.outer(a[:, i], b[:, j]) + np.outer(a[i, :], b[j, :])
        
    return p


def local_search_2opt(a, b, p, time_limit):
    """
    Fast 2-opt local search with vectorized delta computation.
    """
    n = len(p)
    p = p.copy()
    current_cost = compute_cost(a, b, p)
    best_p = p.copy()
    best_cost = current_cost
    
    start_time = time.time()
    deadline = start_time + time_limit
    
    # Precompute nothing specific, but ensure a, b are float64 for speed
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    
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
                    
                x = p[i]
                y = p[j]
                
                # Compute delta for swapping p[i]=x and p[j]=y
                # Delta = sum_{k != i, j} [ a[i,k](b[y,p[k]] - b[x,p[k]]) + a[k,i](b[p[k],y] - b[p,k],x) ]
                #       + sum_{k != i, j} [ a[j,k](b[x,p[k]] - b[y,p[k]]) + a[k,j](b[p[k],x] - b[p[k],y]) ]
                #       + a[i,j](b[y,x] - b[x,y]) + a[j,i](b[x,y] - b[y,x])
                
                # Simplified:
                # Let v1 = a[i, :] - a[j, :]
                # Let v2 = a[:, i] - a[:, j]
                # Let u1 = b[y, :] - b[x, :]
                # Let u2 = b[:, y] - b[:, x]
                
                # The sum over k is:
                # sum_k v1[k] * u1[p[k]] + sum_k v2[k] * u2[p[k]]
                # Note: u1[p[k]] is b[y, p[k]] - b[x, p[k]]
                #       u2[p[k]] is b[p[k], y] - b[p[k], x]
                
                # We compute the full sum including k=i and k=j, then subtract their contributions.
                
                # Vectorized sum
                # term1 = dot(v1, u1[p])
                # term2 = dot(v2, u2[p])
                
                v1 = a[i, :] - a[j, :]
                v2 = a[:, i] - a[:, j]
                u1 = b[y, :] - b[x, :]
                u2 = b[:, y] - b[:, x]
                
                # u1_at_p[k] = u1[p[k]]
                # u2_at_p[k] = u2[p[k]]
                
                # This creates temporary arrays of size n.
                # To avoid memory allocation overhead in tight loop, we can try to be clever,
                # but for n=150, allocation is small.
                
                delta = np.dot(v1, u1[p]) + np.dot(v2, u2[p])
                
                # Subtract k=i
                # k=i term in sum:
                # v1[i] * u1[p[i]] + v2[i] * u2[p[i]]
                # p[i] = x
                # u1[x] = b[y,x] - b[x,x]
                # u2[x] = b[x,y] - b[x,x]
                delta -= v1[i] * (b[y, x] - b[x, x]) + v2[i] * (b[x, y] - b[x, x])
                
                # Subtract k=j
                # k=j term in sum:
                # v1[j] * u1[p[j]] + v2[j] * u2[p[j]]
                # p[j] = y
                # u1[y] = b[y,y] - b[x,y]
                # u2[y] = b[y,y] - b[y,x]
                delta -= v1[j] * (b[y, y] - b[x, y]) + v2[j] * (b[y, y] - b[y, x])
                
                # Add direct i-j interaction change
                # Old: a[i,j]b[x,y] + a[j,i]b[y,x]
                # New: a[i,j]b[y,x] + a[j,i]b[x,y]
                # Delta_direct = a[i,j](b[y,x] - b[x,y]) + a[j,i](b[x,y] - b[y,x])
                #             = (a[i,j] - a[j,i]) * (b[y,x] - b[x,y])
                delta += (a[i, j] - a[j, i]) * (b[y, x] - b[x, y])
                
                if delta < 0:
                    p[i], p[j] = p[j], p[i]
                    current_cost += delta
                    improved = True
                    if current_cost < best_cost:
                        best_cost = current_cost
                        best_p = p.copy()
                    
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
    
    # Number of restarts depends on time.
    # Greedy is ~0.01s. Local search is ~1-10s depending on convergence.
    # Let's try 5-10 restarts.
    
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
    # We will allocate a portion of time for each greedy+LS combo
    # Let's try 5 different noise levels
    
    noise_levels = [0.0, 1.0, 5.0, 10.0, 50.0]
    
    for noise in noise_levels:
        remaining = deadline - time.time()
        if remaining < 10.0:
            break
            
        # Generate solution
        p_g = greedy_construct(a, b, rng, noise_scale=noise)
        
        # Run local search with a time slice
        # We want to spend most of the remaining time on the best candidates,
        # but we don't know which is best until we try.
        # Strategy: give each candidate a fixed slice, say 20% of remaining, 
        # but ensure we have time for a final polish.
        
        # Let's say we have T remaining.
        # We want to test K candidates.
        # Time per candidate = (T - polish_time) / K
        # But K is variable.
        
        # Simpler: just give each greedy candidate a slice of 15 seconds, 
        # capped by remaining time.
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
