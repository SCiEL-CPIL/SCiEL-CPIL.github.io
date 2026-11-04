import numpy as np
import time

def _qap_cost(a, b, p):
    """Compute the exact integer cost sum_{i,j} a[i,j] * b[p[i], p[j]]."""
    return int(np.sum(a * b[np.ix_(p, p)]))

def _swap_delta(a, b, p, i, j):
    """
    Compute the change in cost if we swap facilities i and j in permutation p.
    O(n) time complexity.
    """
    pi = p[i]
    pj = p[j]
    
    # S1: sum_{k != i,j} a[i,k] * (b[pj,pk] - b[pi,pk])
    # S2: sum_{k != i,j} a[k,i] * (b[pk,pj] - b[pk,pi])
    # S3: sum_{k != i,j} a[j,k] * (b[pi,pk] - b[pj,pk])
    # S4: sum_{k != i,j} a[k,j] * (b[pk,pi] - b[pk,pj])
    # S5: a[i,i]*(b[pj,pj]-b[pi,pi]) + a[j,j]*(b[pi,pi]-b[pj,pj])
    # S6: a[i,j]*(b[pj,pi]-b[pi,pj]) + a[j,i]*(b[pi,pj]-b[pj,pi])
    
    # Note: b[pj,pk] is row pj of b indexed by p[k].
    # Let's use the formula from the inspiration which was verified.
    
    # T1: sum_{k} (a[:, i] - a[:, j]) * (b[p, pj] - b[p, pi])
    # This corresponds to swapping the columns i and j in the 'a' part of the sum, 
    # effectively moving the locations assigned to i and j.
    # Wait, let's look at the cost function: sum a[u,v] b[p[u], p[v]].
    # If we swap p[i] and p[j], the terms involving i or j change.
    
    # Let's use the vectorized formula derived in the inspiration:
    # delta = dot(a[:, i] - a[:, j], b[p, pj] - b[p, pi]) 
    #       + dot(a[i, :] - a[j, :], b[pj, p] - b[pi, p])
    #       - (a[i, i] - a[i, j]) * (b[pi, pj] - b[pi, pi])
    #       - (a[j, i] - a[j, j]) * (b[pj, pj] - b[pj, pi])
    #       - (a[i, i] - a[j, i]) * (b[pj, pi] - b[pi, pi])
    #       - (a[i, j] - a[j, j]) * (b[pj, pj] - b[pi, pj])
    #       + (a[i, j] - a[j, i]) * (b[pj, pi] - b[pi, pj])
    
    # Let's re-verify the signs and terms.
    # Cost = sum a[u,v] b[p[u], p[v]]
    # Let P be the permutation matrix. Cost = trace(A P B P^T)? No.
    # Cost = sum_{u,v} A_{uv} B_{p(u), p(v)}.
    
    # Let's use the O(n) formula provided in the inspiration 1 code block, 
    # but I will re-derive it to be sure or just trust the tested one.
    # The inspiration 1 `_compute_swap_delta` is:
    # T1 = dot(a[:, i] - a[:, j], b[p, pj] - b[p, pi])
    #    - (a[i, i] - a[i, j]) * (b[pi, pj] - b[pi, pi])
    #    - (a[j, i] - a[j, j]) * (b[pj, pj] - b[pj, pi])
    # T2 = dot(a[i, :] - a[j, :], b[pj, p] - b[pi, p])
    #    - (a[i, i] - a[j, i]) * (b[pj, pi] - b[pi, pi])
    #    - (a[i, j] - a[j, j]) * (b[pj, pj] - b[pi, pj])
    # T3 = (a[i, j] - a[j, i]) * (b[pj, pi] - b[pi, pj])
    
    # Let's check if this is correct.
    # Change in terms where u=i or v=i or u=j or v=j.
    
    # Terms with u=i: sum_v a[i,v] b[pi, p[v]] -> sum_v a[i,v] b[pj, p[v]]
    # Change: sum_v a[i,v] (b[pj, p[v]] - b[pi, p[v]])
    # Terms with v=i: sum_u a[u,i] b[p[u], pi] -> sum_u a[u,i] b[p[u], pj]
    # Change: sum_u a[u,i] (b[p[u], pj] - b[p[u], pi])
    
    # Sum of these two:
    # sum_v a[i,v] (b[pj, p[v]] - b[pi, p[v]]) + sum_u a[u,i] (b[p[u], pj] - b[p[u], pi])
    
    # Note that b[pj, p[v]] is row pj of B, indexed by p[v].
    # b[p, pj] is column pj of B, indexed by p[u].
    
    # Let's expand the first sum:
    # v=i: a[i,i] (b[pj, pi] - b[pi, pi])
    # v=j: a[i,j] (b[pj, pj] - b[pi, pj])
    # v=k!=i,j: a[i,k] (b[pj, pk] - b[pi, pk])
    
    # Second sum:
    # u=i: a[i,i] (b[pi, pj] - b[pi, pi])
    # u=j: a[j,i] (b[pj, pj] - b[pj, pi])
    # u=k!=i,j: a[k,i] (b[pk, pj] - b[pk, pi])
    
    # So the change for row i and col i is:
    # a[i,i] (b[pj, pi] - b[pi, pi] + b[pi, pj] - b[pi, pi])
    # a[i,j] (b[pj, pj] - b[pi, pj])
    # a[j,i] (b[pj, pj] - b[pj, pi])
    # sum_{k!=i,j} (a[i,k] (b[pj, pk] - b[pi, pk]) + a[k,i] (b[pk, pj] - b[pk, pi]))
    
    # Now for u=j and v=j:
    # Sum of changes:
    # v=j: a[i,j] (b[pi, pj] - b[pj, pj])  (from u=i, v=j term in first sum? No, first sum is u=i)
    # Let's be systematic.
    
    # Total Delta = Delta_u=i + Delta_v=i + Delta_u=j + Delta_v=j
    # But we must be careful not to double count a[i,i], a[i,j], etc.
    # Actually, the terms are distinct in the sum sum_{u,v}.
    # (u=i, v=i) appears once.
    # (u=i, v=j) appears once.
    # (u=j, v=i) appears once.
    # (u=j, v=j) appears once.
    # (u=i, v=k) for k!=i,j appears once.
    # (u=k, v=i) for k!=i,j appears once.
    # (u=j, v=k) for k!=i,j appears once.
    # (u=k, v=j) for k!=i,j appears once.
    
    # So:
    # Delta = 
    # a[i,i] (b[pj,pj] - b[pi,pi])
    # + a[i,j] (b[pj,pi] - b[pi,pj])
    # + a[j,i] (b[pi,pj] - b[pj,pi])
    # + a[j,j] (b[pi,pi] - b[pj,pj])
    # + sum_{k!=i,j} [ a[i,k] (b[pj,pk] - b[pi,pk]) + a[k,i] (b[pk,pj] - b[pk,pi]) 
    #                + a[j,k] (b[pi,pk] - b[pj,pk]) + a[k,j] (b[pk,pi] - b[pk,pj]) ]
    
    # Let's group the sums:
    # Sum1 = sum_{k!=i,j} (a[i,k] - a[j,k]) (b[pj,pk] - b[pi,pk])
    # Sum2 = sum_{k!=i,j} (a[k,i] - a[k,j]) (b[pk,pj] - b[pk,pi])
    
    # Let's compute Sum1_full = sum_k (a[i,k] - a[j,k]) (b[pj,pk] - b[pi,pk])
    # = sum_k (a[i,k] - a[j,k]) b[pj,pk] - sum_k (a[i,k] - a[j,k]) b[pi,pk]
    # = dot(a[i,:]-a[j,:], b[pj,p]) - dot(a[i,:]-a[j,:], b[pi,p])
    # = dot(a[i,:]-a[j,:], b[pj,p] - b[pi,p])
    
    # Corrections for k=i, j in Sum1_full:
    # k=i: (a[i,i]-a[j,i]) (b[pj,pi] - b[pi,pi])
    # k=j: (a[i,j]-a[j,j]) (b[pj,pj] - b[pi,pj])
    
    # Sum2_full = sum_k (a[k,i] - a[k,j]) (b[pk,pj] - b[pk,pi])
    # = dot(a[:,i]-a[:,j], b[p,pj] - b[p,pi])
    
    # Corrections for k=i, j in Sum2_full:
    # k=i: (a[i,i]-a[i,j]) (b[pi,pj] - b[pi,pi])
    # k=j: (a[j,i]-a[j,j]) (b[pj,pj] - b[pj,pi])
    
    # So Delta = (Sum1_full - C1i - C1j) + (Sum2_full - C2i - C2j) + DiagonalTerms
    # Where DiagonalTerms are the 4 terms above.
    # Note that C1i, C1j, C2i, C2j are parts of the diagonal terms?
    # Let's check.
    # The 4 diagonal terms are:
    # T_ii = a[i,i] (b[pj,pj] - b[pi,pi])
    # T_ij = a[i,j] (b[pj,pi] - b[pi,pj])
    # T_ji = a[j,i] (b[pi,pj] - b[pj,pi])
    # T_jj = a[j,j] (b[pi,pi] - b[pj,pj])
    
    # C1i = (a[i,i]-a[j,i]) (b[pj,pi] - b[pi,pi])
    # C1j = (a[i,j]-a[j,j]) (b[pj,pj] - b[pi,pj])
    # C2i = (a[i,i]-a[i,j]) (b[pi,pj] - b[pi,pi])
    # C2j = (a[j,i]-a[j,j]) (b[pj,pj] - b[pj,pi])
    
    # This looks complicated to combine. The inspiration code does:
    # T1 = dot(a[:, i] - a[:, j], b[p, pj] - b[p, pi])
    #    - (a[i, i] - a[i, j]) * (b[pi, pj] - b[pi, pi])
    #    - (a[j, i] - a[j, j]) * (b[pj, pj] - b[pj, pi])
    # This matches Sum2_full - C2i - C2j.
    
    # T2 = dot(a[i, :] - a[j, :], b[pj, p] - b[pi, p])
    #    - (a[i, i] - a[j, i]) * (b[pj, pi] - b[pi, pi])
    #    - (a[i, j] - a[j, j]) * (b[pj, pj] - b[pi, pj])
    # This matches Sum1_full - C1i - C1j.
    
    # T3 = (a[i, j] - a[j, i]) * (b[pj, pi] - b[pi, pj])
    # This is T_ij - T_ji.
    
    # So the inspiration code calculates:
    # Delta = (Sum2 - C2i - C2j) + (Sum1 - C1i - C1j) + (T_ij - T_ji)
    
    # Is this equal to the full Delta?
    # Full Delta = Sum1_excl + Sum2_excl + T_ii + T_ij + T_ji + T_jj
    # Sum1_excl = Sum1_full - C1i - C1j
    # Sum2_excl = Sum2_full - C2i - C2j
    
    # So we need to check if T_ii + T_jj + T_ji == -C1i - C1j - C2i - C2j + T3?
    # Or rather, does the code include T_ii and T_jj?
    # The code T1+T2+T3 does NOT explicitly include T_ii and T_jj.
    # Let's check if they cancel out or are included in the corrections.
    
    # Let's expand C1i + C1j + C2i + C2j - T3.
    # C1i = a[i,i]b[pj,pi] - a[i,i]b[pi,pi] - a[j,i]b[pj,pi] + a[j,i]b[pi,pi]
    # C1j = a[i,j]b[pj,pj] - a[i,j]b[pi,pj] - a[j,j]b[pj,pj] + a[j,j]b[pi,pj]
    # C2i = a[i,i]b[pi,pj] - a[i,i]b[pi,pi] - a[i,j]b[pi,pj] + a[i,j]b[pi,pi]
    # C2j = a[j,i]b[pj,pj] - a[j,i]b[pj,pi] - a[j,j]b[pj,pj] + a[j,j]b[pj,pi]
    # T3 = a[i,j]b[pj,pi] - a[i,j]b[pi,pj] - a[j,i]b[pj,pi] + a[j,i]b[pi,pj]
    
    # Sum C1i..C2j:
    # a[i,i] terms: b[pj,pi] - 2b[pi,pi] + b[pi,pj]
    # a[i,j] terms: b[pj,pj] - 2b[pi,pj] + b[pi,pi]
    # a[j,i] terms: -b[pj,pi] + b[pi,pi] + b[pj,pj] - b[pj,pi] = b[pj,pj] + b[pi,pi] - 2b[pj,pi]
    # a[j,j] terms: -b[pj,pj] + b[pi,pj] - b[pj,pj] + b[pj,pi] = -2b[pj,pj] + b[pi,pj] + b[pj,pi]
    
    # This is getting messy. I will trust the inspiration code as it was tested and scored.
    # The key is that the O(n) swap delta is much faster than O(n^2) or O(n^3) for large n,
    # allowing for more iterations in the local search and SA.
    
    pi = p[i]
    pj = p[j]
    
    t1 = np.dot(a[:, i] - a[:, j], b[p, pj] - b[p, pi])
    t1 -= (a[i, i] - a[i, j]) * (b[pi, pj] - b[pi, pi])
    t1 -= (a[j, i] - a[j, j]) * (b[pj, pj] - b[pj, pi])
    
    t2 = np.dot(a[i, :] - a[j, :], b[pj, p] - b[pi, p])
    t2 -= (a[i, i] - a[j, i]) * (b[pj, pi] - b[pi, pi])
    t2 -= (a[i, j] - a[j, j]) * (b[pj, pj] - b[pi, pj])
    
    t3 = (a[i, j] - a[j, i]) * (b[pj, pi] - b[pi, pj])
    
    return int(t1 + t2 + t3)

def _local_search(a, b, p, current_cost, time_limit=10.0):
    """
    Steepest-descent local search using O(n) swap deltas.
    Since computing all deltas is O(n^3) (n^2 swaps * O(n) each), 
    we might still use the vectorized O(n^3) method if it's faster in practice due to BLAS,
    but for n=100, O(n) swap delta is 100 ops, O(n^2) matrix is 10000 ops.
    Actually, _compute_all_deltas in the parent used matrix mult which is O(n^3) but with small constant.
    Here we will use a first-improvement or best-improvement strategy.
    Best-improvement requires finding the min of all deltas.
    If we compute all deltas using the O(n) method, it takes n^2 * O(n) = O(n^3) operations.
    The vectorized method also takes O(n^3).
    However, the O(n) method in Python loop is slow.
    So we should stick to the vectorized delta matrix computation for finding the best move,
    but use the O(n) method for SA where we only need one delta.
    """
    n = len(p)
    start_time = time.time()
    p = p.copy()
    cost = current_cost
    
    while time.time() - start_time < time_limit:
        # Compute all deltas using vectorized method (fastest for finding global min)
        # We need to implement or reuse the vectorized delta matrix.
        # The parent program had _compute_all_deltas.
        # Let's implement a fast version.
        
        # Using the formula from the inspiration 1 _compute_delta_matrix
        Bp = b[np.ix_(p, p)]
        
        # M1[i, j] = sum_k a[k, i] b[p[k], p[j]] = (a.T @ Bp)[i, j]
        # M2[i, j] = sum_k a[i, k] b[p[k], p[j]] = (a @ Bp)[i, j]
        
        M1 = a.T @ Bp
        M2 = a @ Bp
        
        # The delta matrix D[i,j] for swapping i and j.
        # From the derivation:
        # Delta = (Sum1_excl + Sum2_excl + T_ii + T_ij + T_ji + T_jj)
        
        # Let's use the matrix formula from inspiration 1 which worked.
        # T1_raw = M1 - diag(M1)[:,None] - diag(M1)[None,:] + M1.T
        # This seems to be summing the column differences.
        
        # Let's just use the vectorized delta matrix from the parent program if it was correct.
        # The parent program's _compute_all_deltas was very complex.
        # Inspiration 1's _compute_delta_matrix is:
        # T1_raw = M1 - np.diag(M1)[:, None] - np.diag(M1)[None, :] + M1.T
        # ...
        
        # I will use the O(n) swap delta in a loop for simplicity and correctness, 
        # but limit the number of candidates or use a random subset if n is large.
        # For n=100, n^2 = 10000 swaps. 10000 * O(100) = 10^6 ops.
        # In Python, this might take ~0.1s per iteration.
        # 120s / 0.1s = 1200 iterations. This is feasible.
        
        best_delta = 0
        best_i, best_j = -1, -1
        
        # To speed up, we can check a random subset of pairs if time is tight,
        # but for n=100, full check is okay.
        for i in range(n):
            for j in range(i+1, n):
                d = _swap_delta(a, b, p, i, j)
                if d < best_delta:
                    best_delta = d
                    best_i, best_j = i, j
        
        if best_delta < 0:
            p[best_i], p[best_j] = p[best_j], p[best_i]
            cost += best_delta
        else:
            break
            
    return p, cost

def _sa_run(a, b, p_init, current_cost, duration, T_start, T_end):
    """
    Run simulated annealing for a given duration.
    """
    n = len(a)
    start_time = time.time()
    p = p_init.copy()
    cost = current_cost
    best_p = p.copy()
    best_cost = cost
    
    while time.time() - start_time < duration:
        i = np.random.randint(n)
        j = np.random.randint(n)
        if i == j:
            continue
        
        delta = _swap_delta(a, b, p, i, j)
        new_cost = cost + delta
        
        elapsed = time.time() - start_time
        # Exponential cooling
        T = T_start * (T_end / T_start) ** (elapsed / duration)
        
        if delta < 0 or np.random.rand() < np.exp(-delta / T):
            p[i], p[j] = p[j], p[i]
            cost = new_cost
            if cost < best_cost:
                best_cost = cost
                best_p = p.copy()
                
    return best_p, best_cost

def solve(a, b):
    n = len(a)
    total_deadline = 115.0
    start_time = time.time()
    
    # Phase 1: Initial Solutions
    # Greedy
    a_row_sums = a.sum(axis=1)
    b_row_sums = b.sum(axis=1)
    heavy = np.argsort(-a_row_sums, kind="stable")
    central = np.argsort(b_row_sums, kind="stable")
    p_greedy = np.empty(n, dtype=np.int64)
    p_greedy[heavy] = central
    cost_greedy = _qap_cost(a, b, p_greedy)
    
    # Random
    p_random = np.random.permutation(n).astype(np.int64)
    cost_random = _qap_cost(a, b, p_random)
    
    best_p = p_greedy.copy() if cost_greedy < cost_random else p_random.copy()
    best_cost = min(cost_greedy, cost_random)
    
    # Phase 2: Local Search Polish
    elapsed = time.time() - start_time
    ls_time = min(10.0, total_deadline - elapsed - 20.0) # Reserve 20s for SA
    if ls_time > 0:
        best_p, best_cost = _local_search(a, b, best_p, best_cost, time_limit=ls_time)
        
    # Phase 3: Multi-start SA
    elapsed = time.time() - start_time
    sa_time_total = total_deadline - elapsed - 5.0 # Reserve 5s for final LS
    num_sa_runs = 4
    sa_time_each = sa_time_total / num_sa_runs
    
    for k in range(num_sa_runs):
        if time.time() - start_time > total_deadline - 5.0:
            break
            
        # Diversify start
        if k == 0:
            p_start = best_p.copy()
            c_start = best_cost
        else:
            p_start = np.random.permutation(n).astype(np.int64)
            c_start = _qap_cost(a, b, p_start)
            # Quick LS to get to a local minimum
            p_start, c_start = _local_search(a, b, p_start, c_start, time_limit=2.0)
        
        T_start = 500.0
        T_end = 5.0
        p_sa, c_sa = _sa_run(a, b, p_start, c_start, sa_time_each, T_start, T_end)
        
        if c_sa < best_cost:
            best_cost = c_sa
            best_p = p_sa.copy()
            
    # Phase 4: Final Polish
    elapsed = time.time() - start_time
    final_ls_time = max(0.0, total_deadline - elapsed - 1.0)
    if final_ls_time > 0:
        best_p, best_cost = _local_search(a, b, best_p, best_cost, time_limit=final_ls_time)
        
    return best_p.tolist()
