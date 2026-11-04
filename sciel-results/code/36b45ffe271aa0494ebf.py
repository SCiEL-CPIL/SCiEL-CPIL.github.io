"""
3-Opt Cyclic Exchange Local Search for QAPLIB wil100.
Strategy:
1. Use a 3-opt cyclic exchange neighborhood (rotating three facilities among three locations)
   instead of 2-swap moves to escape 2-opt local optima.
2. Compute the cost delta for all O(n^3) triplets via vectorized NumPy operations.
3. Combine with a multi-start framework: Hungarian initialization + random restarts.
4. Use adaptive restarts when stuck in local optima.
"""
import numpy as np
import time
import random
from scipy.optimize import linear_sum_assignment

def _qap_cost(a, b, p):
    """Compute the exact integer cost sum_{i,j} a[i,j] * b[p[i], p[j]]."""
    return int(np.sum(a * b[np.ix_(p, p)]))

def _hungarian_initial(a, b, mode='row'):
    """
    Generate an initial permutation using the Hungarian algorithm on a surrogate cost matrix.
    """
    n = len(a)
    if mode == 'row':
        a_sums = a.sum(axis=1)
        b_sums = b.sum(axis=1)
        C = -np.outer(a_sums, b_sums)
    elif mode == 'col':
        a_sums = a.sum(axis=0)
        b_sums = b.sum(axis=0)
        C = -np.outer(a_sums, b_sums)
    else:
        C = -np.outer(a.sum(axis=1), b.sum(axis=1))
        
    row_ind, col_ind = linear_sum_assignment(C)
    p = np.empty(n, dtype=np.int64)
    p[row_ind] = col_ind
    return p

def _compute_3opt_deltas(a, b, p):
    """
    Compute the cost delta for all 3-opt cyclic exchanges.
    A 3-opt cyclic exchange rotates three facilities among three locations:
    (i, j, k) -> p[i] goes to j, p[j] goes to k, p[k] goes to i.
    
    Returns a 3D array D where D[i,j,k] is the cost change for the cyclic exchange.
    Due to memory constraints for n=100, we compute this in a memory-efficient way.
    """
    n = len(p)
    C = b[np.ix_(p, p)]
    
    # Precompute some useful matrices
    # S1[i,j] = sum_l a[i,l] * C[j,l]
    S1 = a @ C
    # S2[i,j] = sum_l a[l,i] * C[l,j]  (this is a.T @ C)
    S2 = a.T @ C
    
    # For a cyclic exchange (i,j,k):
    # Old cost contribution involving i,j,k:
    #   sum over all l: a[i,l]*C[j,l] + a[j,l]*C[k,l] + a[k,l]*C[i,l]
    #   + sum over all l: a[l,i]*C[l,j] + a[l,j]*C[l,k] + a[l,k]*C[l,i]
    #   - terms where l is in {i,j,k} need special handling
    # 
    # New cost contribution:
    #   sum over all l: a[i,l]*C[k,l] + a[j,l]*C[i,l] + a[k,l]*C[j,l]
    #   + sum over all l: a[l,i]*C[l,k] + a[l,j]*C[l,i] + a[l,k]*C[l,j]
    #   - terms where l is in {i,j,k} need special handling
    #
    # The delta is new - old.
    #
    # Let's define:
    # delta_from_i = sum_l a[i,l]*(C[k,l] - C[j,l])  (i's row changes from j to k)
    # delta_from_j = sum_l a[j,l]*(C[i,l] - C[k,l])  (j's row changes from k to i)
    # delta_from_k = sum_l a[k,l]*(C[j,l] - C[i,l])  (k's row changes from i to j)
    # delta_to_i = sum_l a[l,i]*(C[l,k] - C[l,j])    (i's column changes from j to k)
    # delta_to_j = sum_l a[l,j]*(C[l,i] - C[l,k])    (j's column changes from k to i)
    # delta_to_k = sum_l a[l,k]*(C[l,j] - C[l,i])    (k's column changes from i to j)
    #
    # But we need to be careful about the diagonal terms where l is in {i,j,k}.
    # When l=i: in old, a[i,i]*C[j,i] and a[i,i]*C[i,j] contribute. In new, a[i,i]*C[k,i] and a[i,i]*C[i,k].
    # This gets complex. Let's use a different approach.
    
    # Alternative: compute the full cost before and after for small batches.
    # For n=100, n^3 = 1,000,000 triplets. We can process them in batches.
    
    # Actually, let's use the formula more carefully.
    # The total cost is sum_{i,j} a[i,j] * C[p[i], p[j]] where C = b[p,p].
    # Wait, C is already b[p,p], so cost = sum_{i,j} a[i,j] * C[i,j].
    
    # When we do a cyclic exchange (i,j,k):
    # p_new[i] = p[j], p_new[j] = p[k], p_new[k] = p[i]
    # C_new[i,j] = b[p_new[i], p_new[j]] = b[p[j], p[k]] = C[j,k]
    # C_new[j,k] = b[p_new[j], p_new[k]] = b[p[k], p[i]] = C[k,i]
    # C_new[k,i] = b[p_new[k], p_new[i]] = b[p[i], p[j]] = C[i,j]
    # C_new[i,k] = b[p_new[i], p_new[k]] = b[p[j], p[i]] = C[j,i]
    # C_new[j,i] = b[p_new[j], p_new[i]] = b[p[k], p[j]] = C[k,j]
    # C_new[k,j] = b[p_new[k], p_new[j]] = b[p[i], p[k]] = C[i,k]
    # For all other (u,v) not in {i,j,k} x {i,j,k}: C_new[u,v] = C[u,v]
    
    # So the delta is:
    # sum over (u,v) in {i,j,k}x{i,j,k} of a[u,v]*(C_new[u,v] - C[u,v])
    # + sum over u in {i,j,k}, v not in {i,j,k} of a[u,v]*(C_new[u,v] - C[u,v])
    # + sum over u not in {i,j,k}, v in {i,j,k} of a[u,v]*(C_new[u,v] - C[u,v])
    
    # For u in {i,j,k}, v not in {i,j,k}: C_new[u,v] = C[u,v] (since p_new[u] changes but v doesn't, 
    # wait no: C_new[u,v] = b[p_new[u], p[v]]. If u is in {i,j,k}, p_new[u] != p[u], so C_new[u,v] != C[u,v] generally.
    
    # This is getting complicated. Let's just compute it directly for batches.
    
    # For efficiency, let's precompute:
    # Row contributions: for each facility i and each location l, a[i,l]
    # Col contributions: for each facility i and each location l, a[l,i]
    
    # Let's use a simpler approach: for each triplet (i,j,k), compute the delta by
    # considering only the affected terms.
    
    # The affected terms are those where at least one of the indices is in {i,j,k}.
    # There are 3*3 + 3*(n-3) + (n-3)*3 = 9 + 6n - 18 = 6n - 9 terms.
    # For n=100, that's ~591 terms per triplet. With 1M triplets, that's ~600M operations.
    # This might be too slow in pure Python, but vectorized it should be okay.
    
    # Let's try a different approach: use the fact that we can compute the delta as:
    # delta = sum_{u,v in {i,j,k}^2} a[u,v]*(C_new[u,v] - C[u,v])
    #       + sum_{u in {i,j,k}, v notin {i,j,k}} a[u,v]*(b[p_new[u],p[v]] - b[p[u],p[v]])
    #       + sum_{u notin {i,j,k}, v in {i,j,k}} a[u,v]*(b[p[u],p_new[v]] - b[p[u],p[v]])
    
    # For the second term: p_new[i]=p[j], p_new[j]=p[k], p_new[k]=p[i]
    # So for u=i: b[p[j],p[v]] - b[p[i],p[v]]
    # For u=j: b[p[k],p[v]] - b[p[j],p[v]]
    # For u=k: b[p[i],p[v]] - b[p[k],p[v]]
    
    # Let's precompute for each pair of locations (x,y): b[x,y]
    # That's just b itself.
    
    # I think the most practical approach for n=100 is to:
    # 1. Generate all triplets (i,j,k) with i<j<k
    # 2. For each triplet, compute the delta using vectorized operations on the affected subset
    
    # But 1M triplets * ~600 ops each = 600M ops. In NumPy, this should be feasible.
    
    # Actually, let's try a smarter vectorized approach.
    
    # Precompute:
    # For each i, let R_i = a[i,:] (row i of a)
    # For each i, let C_i = a[:,i] (column i of a)
    
    # The delta for triplet (i,j,k) is:
    # Part 1: internal terms (u,v both in {i,j,k})
    # Part 2: row terms (u in {i,j,k}, v not in {i,j,k})
    # Part 3: col terms (u not in {i,j,k}, v in {i,j,k})
    
    # Let's just implement a batched version that processes triplets in chunks.
    
    n = len(p)
    # Number of triplets: n*(n-1)*(n-2)/6 ~ 161,700 for n=100
    # That's much more manageable!
    
    # Generate all triplets
    triplets = []
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                triplets.append((i, j, k))
    
    triplets = np.array(triplets)
    num_triplets = len(triplets)
    
    # Process in batches
    batch_size = 10000
    deltas = np.empty(num_triplets, dtype=np.float64)
    
    for start in range(0, num_triplets, batch_size):
        end = min(start + batch_size, num_triplets)
        batch = triplets[start:end]
        i_idx = batch[:, 0]
        j_idx = batch[:, 1]
        k_idx = batch[:, 2]
        
        # Get the relevant p values
        pi = p[i_idx]
        pj = p[j_idx]
        pk = p[k_idx]
        
        # For each triplet, compute the delta
        # Internal terms: 9 terms per triplet
        # C[i,j] = b[p[i],p[j]], etc.
        cij = b[pi, pj]
        cjk = b[pj, pk]
        cki = b[pk, pi]
        cji = b[pj, pi]
        cik = b[pi, pk]
        ckj = b[pk, pj]
        
        # New values
        # C_new[i,j] = C[j,k] = cjk
        # C_new[j,k] = C[k,i] = cki
        # C_new[k,i] = C[i,j] = cij
        # C_new[i,k] = C[j,i] = cji
        # C_new[j,i] = C[k,j] = ckj
        # C_new[k,j] = C[i,k] = cik
        
        # a[i,i]*b[p_new[i],p_new[i]] - a[i,i]*b[p[i],p[i]]
        # p_new[i] = p[j], so b[p[j],p[j]] - b[p[i],p[i]]
        # This is a[i,i]*(b[pj,pj] - b[pi,pi])
        
        # This is getting very complex. Let me just compute the full cost for the triplet's affected region.
        
        # Actually, let's just compute the delta by directly computing the old and new contributions
        # for the affected facilities.
        
        # For facility i:
        # Old: sum_v a[i,v]*b[p[i],p[v]]
        # New: sum_v a[i,v]*b[p[j],p[v]]
        # Delta_i_row = sum_v a[i,v]*(b[p[j],p[v]] - b[p[i],p[v]])
        
        # For facility i as column:
        # Old: sum_u a[u,i]*b[p[u],p[i]]
        # New: sum_u a[u,i]*b[p[u],p[j]]
        # Delta_i_col = sum_u a[u,i]*(b[p[u],p[j]] - b[p[u],p[i]])
        
        # But we need to be careful about double counting when both u and v are in {i,j,k}.
        
        # Let's use the inclusion-exclusion approach:
        # Total delta = (Delta_i_row + Delta_j_row + Delta_k_row)
        #             + (Delta_i_col + Delta_j_col + Delta_k_col)
        #             - (correction for double-counted terms where both indices are in {i,j,k})
        
        # The double-counted terms are:
        # For (i,i): a[i,i]*(b[p[j],p[j]] - b[p[i],p[i]]) counted in both row and col
        # etc.
        
        # This is still complex. Let me just compute it directly.
        
        # For each triplet, the affected terms are:
        # 1. Terms where row index is in {i,j,k}: 3*n terms
        # 2. Terms where col index is in {i,j,k}: 3*n terms
        # 3. Subtract terms where both are in {i,j,k}: 9 terms (counted twice)
        
        # So total affected terms: 6n - 9
        
        # Let's compute this in a vectorized way for the batch.
        
        # For row terms:
        # For facility i: sum_v a[i,v]*(b[p[j],p[v]] - b[p[i],p[v]])
        # = sum_v a[i,v]*b[p[j],p[v]] - sum_v a[i,v]*b[p[i],p[v]]
        # = a[i,:] @ b[p[j],:] - a[i,:] @ b[p[i],:]
        
        # Similarly for j and k.
        
        # For col terms:
        # For facility i: sum_u a[u,i]*(b[p[u],p[j]] - b[p[u],p[i]])
        # = a[:,i] @ b[:,p[j]] - a[:,i] @ b[:,p[i]]
        
        # Let's precompute some matrix-vector products.
        
        # Actually, let's just do it directly for the batch.
        
        batch_deltas = np.zeros(end - start, dtype=np.float64)
        
        for t in range(end - start):
            i, j, k = i_idx[t], j_idx[t], k_idx[t]
            pi_t, pj_t, pk_t = pi[t], pj[t], pk[t]
            
            # Compute old and new costs for affected terms
            # Old cost involving i,j,k:
            # sum_{u,v} a[u,v]*b[p[u],p[v]] where u or v is in {i,j,k}
            
            # New cost:
            # sum_{u,v} a[u,v]*b[p_new[u],p_new[v]] where u or v is in {i,j,k}
            # p_new[i] = p[j] = pj_t
            # p_new[j] = p[k] = pk_t
            # p_new[k] = p[i] = pi_t
            
            # Let's compute the difference directly.
            # Delta = sum_{u,v: u in {i,j,k} or v in {i,j,k}} a[u,v]*(b[p_new[u],p_new[v]] - b[p[u],p[v]])
            
            # We can compute this as:
            # sum_{u in {i,j,k}} sum_{v} a[u,v]*(b[p_new[u],p[v]] - b[p[u],p[v]])
            # + sum_{u not in {i,j,k}} sum_{v in {i,j,k}} a[u,v]*(b[p[u],p_new[v]] - b[p[u],p[v]])
            
            # First part:
            part1 = 0.0
            for u in [i, j, k]:
                p_new_u = [pj_t, pk_t, pi_t][[i, j, k].index(u)]
                part1 += np.dot(a[u, :], b[p_new_u, :] - b[p[u], :])
            
            # Second part:
            part2 = 0.0
            for v in [i, j, k]:
                p_new_v = [pj_t, pk_t, pi_t][[i, j, k].index(v)]
                part2 += np.dot(a[:, v], b[:, p_new_v] - b[:, p[v]])
            
            # But this double-counts terms where both u and v are in {i,j,k}.
            # In part1, we have a[u,v]*(b[p_new[u],p[v]] - b[p[u],p[v]]) for u in {i,j,k}, all v.
            # In part2, we have a[u,v]*(b[p[u],p_new[v]] - b[p[u],p[v]]) for v in {i,j,k}, all u.
            # The overlap is when u in {i,j,k} AND v in {i,j,k}.
            # In part1: a[u,v]*(b[p_new[u],p[v]] - b[p[u],p[v]])
            # In part2: a[u,v]*(b[p[u],p_new[v]] - b[p[u],p[v]])
            # Total: a[u,v]*(b[p_new[u],p[v]] + b[p[u],p_new[v]] - 2*b[p[u],p[v]])
            # But the correct delta is: a[u,v]*(b[p_new[u],p_new[v]] - b[p[u],p[v]])
            # So we need to subtract: a[u,v]*(b[p_new[u],p[v]] + b[p[u],p_new[v]] - b[p_new[u],p_new[v]] - b[p[u],p[v]])
            # = a[u,v]*(b[p_new[u],p[v]] - b[p_new[u],p_new[v]] + b[p[u],p_new[v]] - b[p[u],p[v]])
            
            # Let's just compute the correction.
            correction = 0.0
            for u in [i, j, k]:
                for v in [i, j, k]:
                    p_new_u = [pj_t, pk_t, pi_t][[i, j, k].index(u)]
                    p_new_v = [pj_t, pk_t, pi_t][[i, j, k].index(v)]
                    # We added a[u,v]*(b[p_new[u],p[v]] - b[p[u],p[v]]) in part1
                    # and a[u,v]*(b[p[u],p_new[v]] - b[p[u],p[v]]) in part2
                    # Total added: a[u,v]*(b[p_new[u],p[v]] + b[p[u],p_new[v]] - 2*b[p[u],p[v]])
                    # Correct: a[u,v]*(b[p_new[u],p_new[v]] - b[p[u],p[v]])
                    # Difference: a[u,v]*(b[p_new[u],p[v]] + b[p[u],p_new[v]] - b[p_new[u],p_new[v]] - b[p[u],p[v]])
                    correction += a[u, v] * (b[p_new_u, p[v]] + b[p[u], p_new_v] - b[p_new_u, p_new_v] - b[p[u], p[v]])
            
            batch_deltas[t] = part1 + part2 - correction
        
        deltas[start:end] = batch_deltas
    
    return deltas, triplets

def _local_search_3opt(a, b, p, max_iters=50, time_limit=None, start_time=None):
    """
    Steepest descent 3-opt cyclic exchange local search.
    """
    if time_limit is None:
        time_limit = float('inf')
    if start_time is None:
        start_time = time.time()
        
    current_cost = _qap_cost(a, b, p)
    n = len(p)
    best_p = p.copy()
    best_cost = current_cost
    
    for it in range(max_iters):
        if time.time() - start_time > time_limit:
            break
        
        deltas, triplets = _compute_3opt_deltas(a, b, p)
        
        min_idx = np.argmin(deltas)
        min_delta = deltas[min_idx]
        
        if min_delta >= -1e-10:
            break  # Local minimum reached
        
        i, j, k = triplets[min_idx]
        
        # Apply cyclic exchange: p[i]->j, p[j]->k, p[k]->i
        pi, pj, pk = p[i], p[j], p[k]
        p[i] = pj
        p[j] = pk
        p[k] = pi
        
        current_cost += int(min_delta)
        
        if current_cost < best_cost:
            best_cost = current_cost
            best_p = p.copy()
    
    return best_p, best_cost

def _local_search_2opt(a, b, p, max_iters=50, time_limit=None, start_time=None):
    """
    Steepest descent 2-opt local search.
    """
    if time_limit is None:
        time_limit = float('inf')
    if start_time is None:
        start_time = time.time()
        
    current_cost = _qap_cost(a, b, p)
    n = len(p)
    best_p = p.copy()
    best_cost = current_cost
    
    for it in range(max_iters):
        if time.time() - start_time > time_limit:
            break
        
        C = b[np.ix_(p, p)]
        S1 = a.T @ C
        T1 = a @ C
        
        D_S1 = np.diag(S1)
        D_T1 = np.diag(T1)
        
        Term1 = S1 - D_S1[:, None] - D_S1[None, :] + S1.T
        Term2 = T1 - D_T1[:, None] - D_T1[None, :] + T1.T
        
        Raw = Term1 + Term2
        Error = -2.0 * (a + a.T) * C
        Delta = Raw - Error
        np.fill_diagonal(Delta, 0.0)
        
        min_delta = np.min(Delta)
        
        if min_delta >= -1e-10:
            break
        
        i, j = np.unravel_index(np.argmin(Delta), Delta.shape)
        p[i], p[j] = p[j], p[i]
        current_cost += int(min_delta)
        
        if current_cost < best_cost:
            best_cost = current_cost
            best_p = p.copy()
    
    return best_p, best_cost

def _perturb(p, num_swaps=5):
    """
    Apply random swaps to create a perturbed version of the permutation.
    """
    n = len(p)
    p = p.copy()
    for _ in range(num_swaps):
        i = random.randint(0, n - 1)
        j = random.randint(0, n - 1)
        if i != j:
            p[i], p[j] = p[j], p[i]
    return p

def solve(a, b):
    n = len(a)
    total_deadline = 140.0  # 10s buffer
    start_time = time.time()
    
    # Multi-start framework
    best_p = None
    best_cost = float('inf')
    
    # Initial solutions
    initial_solutions = []
    
    # Hungarian inits
    for mode in ['row', 'col']:
        p = _hungarian_initial(a, b, mode)
        initial_solutions.append(p)
    
    # Random inits
    for _ in range(8):
        initial_solutions.append(np.random.permutation(n))
    
    # Process each initial solution
    for init_p in initial_solutions:
        if time.time() - start_time > total_deadline - 10.0:
            break
        
        # 2-opt polish first (faster)
        p, c = _local_search_2opt(a, b, init_p.copy(), max_iters=30, time_limit=total_deadline, start_time=start_time)
        
        # Then 3-opt polish
        p, c = _local_search_3opt(a, b, p, max_iters=20, time_limit=total_deadline, start_time=start_time)
        
        if c < best_cost:
            best_cost = c
            best_p = p.copy()
    
    # Adaptive restarts
    no_improve_count = 0
    max_no_improve = 5
    
    while time.time() - start_time < total_deadline - 5.0:
        # Perturb the best solution
        num_swaps = random.randint(3, 8)
        p = _perturb(best_p, num_swaps)
        
        # 2-opt polish
        p, c = _local_search_2opt(a, b, p, max_iters=20, time_limit=total_deadline, start_time=start_time)
        
        # 3-opt polish
        p, c = _local_search_3opt(a, b, p, max_iters=15, time_limit=total_deadline, start_time=start_time)
        
        if c < best_cost:
            best_cost = c
            best_p = p.copy()
            no_improve_count = 0
        else:
            no_improve_count += 1
        
        if no_improve_count >= max_no_improve:
            # Try a fresh random start
            p = np.random.permutation(n)
            p, c = _local_search_2opt(a, b, p, max_iters=20, time_limit=total_deadline, start_time=start_time)
            p, c = _local_search_3opt(a, b, p, max_iters=15, time_limit=total_deadline, start_time=start_time)
            if c < best_cost:
                best_cost = c
                best_p = p.copy()
            no_improve_count = 0
    
    # Final polish
    time_left = total_deadline - (time.time() - start_time)
    if time_left > 3.0:
        best_p, best_cost = _local_search_2opt(a, b, best_p, max_iters=20, time_limit=time_left, start_time=time.time())
        best_p, best_cost = _local_search_3opt(a, b, best_p, max_iters=10, time_limit=time_left, start_time=time.time())
    
    return best_p.tolist()
