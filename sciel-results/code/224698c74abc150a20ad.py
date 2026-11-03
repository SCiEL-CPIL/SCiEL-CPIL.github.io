# EVOLVE-BLOCK-START
import numpy as np
import time


def _qap_cost(a, b, p):
    return int(np.sum(a * b[np.ix_(p, p)]))


def _compute_C(a, b, p):
    """C[i] = sum_j a[i,j] * b[p[i], p[j]]"""
    B = b[np.ix_(p, p)]
    return (a @ B).astype(np.int64)


def _swap_delta_vec(a, b, p, i, j, C):
    """Vectorized O(n) swap delta using precomputed C.
    delta = C_new - C_old for the two affected facilities plus cross terms.
    Uses the formula:
      delta = (a[i,j]+a[j,i])*(b[pj,pi]-b[pi,pj]) 
            + sum_{k!=i,j} [a[i,k]+a[k,i]+a[j,k]+a[k,j]] * (b[pj,pk]-b[pi,pk]) 
            ... actually let's use the clean formulation.
    
    Clean: after swap, facility i goes to pj, facility j goes to pi.
    delta = [sum_k a[i,k]*(b[pj,pk]-b[pi,pk])] + [sum_k a[j,k]*(b[pi,pk]-b[pj,pk])]
          + [sum_k a[k,i]*(b[pk,pj]-b[pk,pi])] + [sum_k a[k,j]*(b[pk,pi]-b[pk,pj])]
          - 2*[a[i,j]*(b[pj,pj]-b[pi,pj]) + a[j,i]*(b[pi,pi]-b[pj,pi])]
          ... this is getting messy. Let me use the direct full formula.
    
    Actually the cleanest vectorized approach:
    diff_row = b[pj,:] - b[pi,:]   # shape (n,)
    diff_col = b[:,pj] - b[:,pi]   # shape (n,)
    
    delta = a[i,:] @ diff_row + a[j,:] @ (-diff_row) + a[:,i] @ diff_col + a[:,j] @ (-diff_col)
    
    This includes k=i and k=j terms. Let's verify:
    When k=i in first sum: a[i,i]*(b[pj,pi]-b[pi,pi])
    When k=j in first sum: a[i,j]*(b[pj,pj]-b[pi,pj])
    When k=i in second sum: a[j,i]*(b[pi,pi]-b[pj,pi])
    When k=j in second sum: a[j,j]*(b[pi,pj]-b[pj,pj])
    When k=i in third sum: a[i,i]*(b[pi,pj]-b[pi,pi])
    When k=j in third sum: a[j,i]*(b[pj,pj]-b[pj,pi])
    When k=i in fourth sum: a[i,j]*(b[pi,pi]-b[pi,pj])
    When k=j in fourth sum: a[j,j]*(b[pj,pj]-b[pj,pj])  wait no...
    
    Let me just compute it directly and verify numerically.
    """
    pi = p[i]
    pj = p[j]
    diff_row = b[pj, :] - b[pi, :]  # (n,)
    diff_col = b[:, pj] - b[:, pi]  # (n,)
    
    # Full vectorized sum (includes all k)
    delta = (int(a[i, :] @ diff_row) 
             + int(a[j, :] @ (-diff_row)) 
             + int(a[:, i] @ diff_col) 
             + int(a[:, j] @ (-diff_col)))
    
    # The above over-counts the i-j interactions. We need to subtract the excess.
    # The correct delta should be:
    # For k != i,j: a[i,k]*(b[pj,pk]-b[pi,pk]) + a[j,k]*(b[pi,pk]-b[pj,pk])
    #               + a[k,i]*(b[pk,pj]-b[pk,pi]) + a[k,j]*(b[pk,pi]-b[pk,pj])
    # For k=i: a[i,i]*(b[pj,pi]-b[pi,pi]) + a[i,i]*(b[pi,pj]-b[pi,pi]) = a[i,i]*(b[pj,pi]+b[pi,pj]-2*b[pi,pi])
    #   But correct: facility i at pj interacting with itself: a[i,i]*b[pj,pj]
    #   Original: a[i,i]*b[pi,pi]
    #   Delta for self: a[i,i]*(b[pj,pj]-b[pi,pi])
    #   What we got: a[i,i]*(b[pj,pi]-b[pi,pi]) + a[i,i]*(b[pi,pj]-b[pi,pi])
    #   = a[i,i]*(b[pj,pi]+b[pi,pj]-2*b[pi,pi])
    #   If b is symmetric: b[pj,pi]=b[pi,pj], so = a[i,i]*(2*b[pi,pj]-2*b[pi,pi])
    #   Correct: a[i,i]*(b[pj,pj]-b[pi,pi])
    #   These are different! So the formula is wrong for non-zero diagonal.
    
    # Let me use a different approach: compute the delta as the difference of the two relevant C values
    # plus the cross terms properly.
    
    # Actually, let me just use the brute-force but vectorized over k:
    # delta = sum over all k of the change in contribution
    
    # The contribution of pair (i,k) changes from a[i,k]*b[pi,pk] to a[i,k]*b[pj,pk] (if k!=j)
    # The contribution of pair (j,k) changes from a[j,k]*b[pj,pk] to a[j,k]*b[pi,pk] (if k!=i)
    # The contribution of pair (k,i) changes from a[k,i]*b[pk,pi] to a[k,i]*b[pk,pj] (if k!=j)
    # The contribution of pair (k,j) changes from a[k,j]*b[pk,pj] to a[k,j]*b[pk,pi] (if k!=i)
    # The contribution of pair (i,j) changes from a[i,j]*b[pi,pj] to a[i,j]*b[pj,pi]
    # The contribution of pair (j,i) changes from a[j,i]*b[pj,pi] to a[j,i]*b[pi,pj]
    
    # So: delta = a[i,j]*(b[pj,pi]-b[pi,pj]) + a[j,i]*(b[pi,pj]-b[pj,pi])
    #          + sum_{k!=i,j} [a[i,k]*(b[pj,pk]-b[pi,pk]) + a[j,k]*(b[pi,pk]-b[pj,pk])
    #                          + a[k,i]*(b[pk,pj]-b[pk,pi]) + a[k,j]*(b[pk,pi]-b[pk,pj])]
    
    # Vectorized: compute the full sums and subtract the k=i and k=j terms
    s1 = int(a[i, :] @ diff_row)  # includes k=i and k=j
    s2 = int(a[j, :] @ (-diff_row))  # includes k=i and k=j
    s3 = int(a[:, i] @ diff_col)  # includes k=i and k=j
    s4 = int(a[:, j] @ (-diff_col))  # includes k=i and k=j
    
    # Subtract k=i terms:
    # s1, k=i: a[i,i]*(b[pj,pi]-b[pi,pi])
    # s2, k=i: a[j,i]*(b[pi,pi]-b[pj,pi])
    # s3, k=i: a[i,i]*(b[pi,pj]-b[pi,pi])
    # s4, k=i: a[i,j]*(b[pi,pi]-b[pi,pj])
    # Subtract k=j terms:
    # s1, k=j: a[i,j]*(b[pj,pj]-b[pi,pj])
    # s2, k=j: a[j,j]*(b[pi,pj]-b[pj,pj])
    # s3, k=j: a[j,i]*(b[pj,pj]-b[pj,pi])
    # s4, k=j: a[j,j]*(b[pj,pj]-b[pj,pj]) = 0
    
    # Add back the correct i-j direct terms:
    # a[i,j]*(b[pj,pi]-b[pi,pj]) + a[j,i]*(b[pi,pj]-b[pj,pi])
    
    # This is getting too complex. Let me just use the simple O(n) loop but with numpy indexing.
    # For n=100, even a Python loop of 100 iterations is fast enough.
    
    # Actually, let me precompute the needed vectors and use a single dot product.
    # For fixed i, j: delta = (a[i,:] + a[:,i] - a[i,i]*e_i - a[i,j]*e_j) @ (b[pj,:]-b[pi,:])
    #                    + (a[j,:] + a[:,j] - a[j,j]*e_j - a[j,i]*e_i) @ (b[pi,:]-b[pj,:])
    #                    + a[i,j]*(b[pj,pi]-b[pi,pj]) + a[j,i]*(b[pi,pj]-b[pj,pi])
    # This is still messy.
    
    # Simplest correct vectorized approach:
    # Create a mask excluding i and j
    # But for n=100, just do the loop. It's fast enough.
    
    pi_val = pi
    pj_val = pj
    delta = 0
    # Direct i-j interaction
    delta += a[i, j] * (b[pj_val, pi_val] - b[pi_val, pj_val])
    delta += a[j, i] * (b[pi_val, pj_val] - b[pj_val, pi_val])
    
    # Cross terms with all k != i, j
    # Vectorized: use numpy to compute for all k at once
    # We need: sum_{k != i,j} a[i,k]*(b[pj,pk]-b[pi,pk])
    # = a[i,:] @ (b[pj,:]-b[pi,:]) - a[i,i]*(b[pj,pi]-b[pi,pi]) - a[i,j]*(b[pj,pj]-b[pi,pj])
    
    diff_r = b[pj_val, :] - b[pi_val, :]  # (n,)
    diff_c = b[:, pj_val] - b[:, pi_val]  # (n,)
    
    # Sum over all k, then subtract k=i and k=j
    s_i = int(a[i, :] @ diff_r)  # sum_k a[i,k]*(b[pj,pk]-b[pi,pk])
    s_j = int(a[j, :] @ (-diff_r))  # sum_k a[j,k]*(b[pi,pk]-b[pj,pk])
    s_ci = int(a[:, i] @ diff_c)  # sum_k a[k,i]*(b[pk,pj]-b[pk,pi])
    s_cj = int(a[:, j] @ (-diff_c))  # sum_k a[k,j]*(b[pk,pi]-b[pk,pj])
    
    # Subtract k=i contributions:
    s_i -= a[i, i] * (b[pj_val, pi_val] - b[pi_val, pi_val])
    s_j -= a[j, i] * (b[pi_val, pi_val] - b[pj_val, pi_val])
    s_ci -= a[i, i] * (b[pi_val, pj_val] - b[pi_val, pi_val])
    s_cj -= a[i, j] * (b[pi_val, pi_val] - b[pi_val, pj_val])
    
    # Subtract k=j contributions:
    s_i -= a[i, j] * (b[pj_val, pj_val] - b[pi_val, pj_val])
    s_j -= a[j, j] * (b[pi_val, pj_val] - b[pj_val, pj_val])
    s_ci -= a[j, i] * (b[pj_val, pj_val] - b[pj_val, pi_val])
    s_cj -= a[j, j] * (b[pj_val, pj_val] - b[pj_val, pj_val])  # = 0
    
    delta += s_i + s_j + s_ci + s_cj
    return delta


def _swap_delta_fast(a, b, p, i, j):
    """Fast O(n) vectorized swap delta.
    Uses the identity:
    delta = (a[i,:]+a[:,i]) @ (b[pj,:]-b[pi,:]) + (a[j,:]+a[:,j]) @ (b[pi,:]-b[pj,:])
            - correction terms for i,j self-interactions
            + direct i-j term
    
    Actually, let me derive it cleanly.
    
    Original cost contribution involving i or j:
    - a[i,j]*b[pi,pj] + a[j,i]*b[pj,pi]  (direct)
    - sum_{k!=i,j} a[i,k]*b[pi,pk] + a[k,i]*b[pk,pi]  (i with others)
    - sum_{k!=i,j} a[j,k]*b[pj,pk] + a[k,j]*b[pk,pj]  (j with others)
    - a[i,i]*b[pi,pi] + a[j,j]*b[pj,pj]  (self)
    
    New cost contribution:
    - a[i,j]*b[pj,pi] + a[j,i]*b[pi,pj]  (direct, swapped)
    - sum_{k!=i,j} a[i,k]*b[pj,pk] + a[k,i]*b[pk,pj]  (i at pj)
    - sum_{k!=i,j} a[j,k]*b[pi,pk] + a[k,j]*b[pk,pi]  (j at pi)
    - a[i,i]*b[pj,pj] + a[j,j]*b[pi,pi]  (self)
    
    delta = [a[i,j]*(b[pj,pi]-b[pi,pj]) + a[j,i]*(b[pi,pj]-b[pj,pi])]
          + [a[i,i]*(b[pj,pj]-b[pi,pi]) + a[j,j]*(b[pi,pi]-b[pj,pj])]
          + sum_{k!=i,j} [a[i,k]*(b[pj,pk]-b[pi,pk]) + a[k,i]*(b[pk,pj]-b[pk,pi])
                         + a[j,k]*(b[pi,pk]-b[pj,pk]) + a[k,j]*(b[pk,pi]-b[pk,pj])]
    
    Vectorize the sum over k!=i,j by computing full sum and subtracting k=i,j:
    """
    pi = p[i]
    pj = p[j]
    
    diff_r = b[pj, :] - b[pi, :]  # (n,)
    diff_c = b[:, pj] - b[:, pi]  # (n,)
    
    # Full sums over all k
    S1 = int(a[i, :] @ diff_r)   # sum_k a[i,k]*(b[pj,pk]-b[pi,pk])
    S2 = int(a[:, i] @ diff_c)   # sum_k a[k,i]*(b[pk,pj]-b[pk,pi])
    S3 = int(a[j, :] @ (-diff_r)) # sum_k a[j,k]*(b[pi,pk]-b[pj,pk])
    S4 = int(a[:, j] @ (-diff_c)) # sum_k a[k,j]*(b[pk,pi]-b[pk,pj])
    
    # Subtract k=i terms:
    S1 -= a[i, i] * (b[pj, pi] - b[pi, pi])
    S2 -= a[i, i] * (b[pi, pj] - b[pi, pi])
    S3 -= a[j, i] * (b[pi, pi] - b[pj, pi])
    S4 -= a[i, j] * (b[pi, pi] - b[pi, pj])
    
    # Subtract k=j terms:
    S1 -= a[i, j] * (b[pj, pj] - b[pi, pj])
    S2 -= a[j, i] * (b[pj, pj] - b[pj, pi])
    S3 -= a[j, j] * (b[pi, pj] - b[pj, pj])
    S4 -= a[j, j] * (b[pj, pj] - b[pj, pj])  # = 0
    
    # Direct i-j term
    direct = a[i, j] * (b[pj, pi] - b[pi, pj]) + a[j, i] * (b[pi, pj] - b[pj, pi])
    
    # Self terms
    self_term = a[i, i] * (b[pj, pj] - b[pi, pi]) + a[j, j] * (b[pi, pi] - b[pj, pj])
    
    delta = direct + self_term + S1 + S2 + S3 + S4
    return delta


def _local_search_best(a, b, p, deadline, best_cost, best_p, max_passes=100):
    """Best-improvement 2-opt local search with vectorized deltas."""
    n = len(p)
    cur = _qap_cost(a, b, p)
    if cur < best_cost:
        best_cost = cur
        best_p = p.copy()
    
    passes = 0
    while time.time() < deadline and passes < max_passes:
        best_d = 0
        best_i = -1
        best_j = -1
        
        for i in range(n):
            if time.time() >= deadline:
                break
            for j in range(i + 1, n):
                d = _swap_delta_fast(a, b, p, i, j)
                if d < best_d:
                    best_d = d
                    best_i = i
                    best_j = j
        
        if best_i == -1:
            break
        
        p[best_i], p[best_j] = p[best_j], p[best_i]
        cur += best_d
        if cur < best_cost:
            best_cost = cur
            best_p = p.copy()
        passes += 1
    
    if cur < best_cost:
        best_cost = cur
        best_p = p.copy()
    return best_cost, best_p


def _local_search_first(a, b, p, deadline, best_cost, best_p, max_passes=200):
    """First-improvement 2-opt local search (faster per pass, more passes)."""
    n = len(p)
    cur = _qap_cost(a, b, p)
    if cur < best_cost:
        best_cost = cur
        best_p = p.copy()
    
    passes = 0
    while time.time() < deadline and passes < max_passes:
        improved = False
        for i in range(n):
            if time.time() >= deadline:
                break
            for j in range(i + 1, n):
                d = _swap_delta_fast(a, b, p, i, j)
                if d < 0:
                    p[i], p[j] = p[j], p[i]
                    cur += d
                    improved = True
                    if cur < best_cost:
                        best_cost = cur
                        best_p = p.copy()
                    break
        passes += 1
        if not improved:
            break
    
    if cur < best_cost:
        best_cost = cur
        best_p = p.copy()
    return best_cost, best_p


def _anneal(a, b, p, deadline, best_cost, best_p, rng, t_start=200.0, t_end=1.0, alpha=0.995):
    """Simulated annealing with swap moves."""
    n = len(p)
    cur = _qap_cost(a, b, p)
    if cur < best_cost:
        best_cost = cur
        best_p = p.copy()
    
    t = t_start
    while time.time() < deadline and t > t_end:
        i = rng.integers(n)
        j = rng.integers(n)
        if i == j:
            continue
        d = _swap_delta_fast(a, b, p, i, j)
        if d < 0 or rng.random() < np.exp(-d / max(t, 1e-10)):
            p[i], p[j] = p[j], p[i]
            cur += d
            if cur < best_cost:
                best_cost = cur
                best_p = p.copy()
        t *= alpha
    
    return best_cost, best_p


def _greedy_construct(a, b, rng, n):
    """Greedy nearest-neighbor style constructive heuristic."""
    p = np.zeros(n, dtype=np.int64)
    used = np.zeros(n, dtype=bool)
    
    # Order facilities by total flow (heaviest first)
    a_row_sums = a.sum(axis=1)
    order = np.argsort(-a_row_sums, kind="stable")
    
    for idx in order:
        if not np.any(used):
            p[idx] = int(np.argmin(b.sum(axis=1)))
        else:
            placed = np.where(used)[0]
            # Score for each unused location
            scores = np.zeros(n, dtype=np.float64)
            for j in placed:
                scores += a[idx, j] * b[:, p[j]] + a[j, idx] * b[p[j], :]
            candidates = np.where(~used)[0]
            p[idx] = int(candidates[np.argmin(scores[candidates])])
        used[p[idx]] = True
    
    return p


def _rank_construct(a, b, n):
    """Rank matching: heavy facilities to central locations."""
    heavy = np.argsort(-a.sum(axis=1), kind="stable")
    central = np.argsort(b.sum(axis=1), kind="stable")
    p = np.empty(n, dtype=np.int64)
    p[heavy] = central
    return p


def _random_construct(rng, n):
    return rng.permutation(n).astype(np.int64)


def _2opt_polish(a, b, p, deadline, best_cost, best_p):
    """Multiple rounds of best-improvement 2-opt."""
    cur = _qap_cost(a, b, p)
    if cur < best_cost:
        best_cost = cur
        best_p = p.copy()
    
    while time.time() < deadline:
        improved = False
        n = len(p)
        best_d = 0
        best_i = -1
        best_j = -1
        for i in range(n):
            if time.time() >= deadline:
                break
            for j in range(i + 1, n):
                d = _swap_delta_fast(a, b, p, i, j)
                if d < best_d:
                    best_d = d
                    best_i = i
                    best_j = j
        if best_i == -1:
            break
        p[best_i], p[best_j] = p[best_j], p[best_i]
        cur += best_d
        improved = True
        if cur < best_cost:
            best_cost = cur
            best_p = p.copy()
        if not improved:
            break
    
    return best_cost, best_p


def solve(a, b):
    a = np.ascontiguousarray(a, dtype=np.int64)
    b = np.ascontiguousarray(b, dtype=np.int64)
    n = len(a)
    t0 = time.time()
    deadline = t0 + 145.0  # safety margin before 150s kill

    rng = np.random.default_rng(42)
    
    best_cost = np.iinfo(np.int64).max
    best_p = None
    
    # --- Phase 1: Multiple constructive starts ---
    starts = []
    
    # Rank matching
    p_rank = _rank_construct(a, b, n)
    starts.append(p_rank)
    
    # Greedy
    p_greedy = _greedy_construct(a, b, rng, n)
    starts.append(p_greedy)
    
    # Random starts
    for _ in range(3):
        starts.append(_random_construct(rng, n))
    
    # Evaluate and pick best
    for p in starts:
        c = _qap_cost(a, b, p)
        if c < best_cost:
            best_cost = c
            best_p = p.copy()
    
    # --- Phase 2: Local search from best start ---
    if time.time() < deadline - 30.0:
        p = best_p.copy()
        best_cost, best_p = _local_search_best(a, b, p, min(deadline - 20.0, t0 + 40.0), best_cost, best_p)
    
    # --- Phase 3: Simulated annealing ---
    if time.time() < deadline - 15.0:
        p = best_p.copy()
        sa_deadline = min(deadline - 5.0, t0 + 90.0)
        best_cost, best_p = _anneal(a, b, p, sa_deadline, best_cost, best_p, rng)
    
    # --- Phase 4: Final local search polish ---
    if time.time() < deadline - 3.0:
        p = best_p.copy()
        best_cost, best_p = _local_search_best(a, b, p, deadline - 1.0, best_cost, best_p)
    
    # --- Phase 5: If time remains, do another round of anneal + polish ---
    if time.time() < deadline - 10.0:
        p = best_p.copy()
        best_cost, best_p = _anneal(a, b, p, min(deadline - 5.0, t0 + 135.0), best_cost, best_p, rng)
        if time.time() < deadline - 2.0:
            p = best_p.copy()
            best_cost, best_p = _local_search_best(a, b, p, deadline - 1.0, best_cost, best_p)
    
    # Final verification
    final_cost = _qap_cost(a, b, best_p)
    if final_cost < best_cost:
        best_cost = final_cost
    
    return best_p.tolist()
# EVOLVE-BLOCK-END
