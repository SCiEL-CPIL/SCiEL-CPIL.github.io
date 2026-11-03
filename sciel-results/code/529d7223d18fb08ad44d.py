"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _cost(a, b, p):
    """Compute the exact QAP cost for permutation p."""
    Bp = b[np.ix_(p, p)]
    return int(np.sum(a * Bp))


def _delta_swap(a, b, p, i, j):
    """
    Compute the change in cost if we swap facilities i and j in the assignment p.
    p is a list/array; p[i] and p[j] are the locations.
    """
    pi = p[i]
    pj = p[j]
    
    # Vectorized computation of the four sums over all k
    # S1 = sum_k a[i,k] * (b[pj,pk] - b[pi,pk])
    # S2 = sum_k a[j,k] * (b[pi,pk] - b[pj,pk])
    # S3 = sum_k a[k,i] * (b[pk,pj] - b[pk,pi])
    # S4 = sum_k a[k,j] * (b[pk,pi] - b[pk,pj])
    
    diff_row = b[pj, p] - b[pi, p]
    diff_col = b[p, pj] - b[p, pi]
    
    S1 = int(np.dot(a[i, :], diff_row))
    S2 = int(np.dot(a[j, :], -diff_row))
    S3 = int(np.dot(a[:, i], diff_col))
    S4 = int(np.dot(a[:, j], -diff_col))
    
    total = S1 + S2 + S3 + S4
    
    # Subtract the double-counted i-j interaction terms.
    b_pi_pi = b[pi, pi]
    b_pj_pj = b[pj, pj]
    b_pi_pj = b[pi, pj]
    b_pj_pi = b[pj, pi]
    
    Rem_ii = a[i, i] * (b_pj_pi - b_pi_pi + b_pi_pj - b_pi_pi)
    Rem_ij = a[i, j] * (b_pj_pj - b_pi_pj + b_pi_pi - b_pi_pj)
    Rem_ji = a[j, i] * (b_pi_pi - b_pj_pi + b_pj_pj - b_pj_pi)
    Rem_jj = a[j, j] * (b_pi_pj - b_pj_pj + b_pj_pi - b_pj_pj)
    
    Rem = Rem_ii + Rem_ij + Rem_ji + Rem_jj
    
    Direct = a[i, j] * (b_pj_pi - b_pi_pj) + a[j, i] * (b_pi_pj - b_pj_pi)
    
    return total - Rem + Direct


def _local_search(a, b, p, deadline, best_cost, best_p, first_improvement=True):
    """
    Local search with 2-opt swaps.
    If first_improvement is True, use first-improvement strategy (faster per pass).
    If False, use best-improvement strategy (finds the best swap in each pass).
    """
    n = len(p)
    cur = _cost(a, b, p)
    if cur < best_cost:
        best_cost = cur
        best_p = p.copy()
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        best_delta = 0
        best_i = -1
        best_j = -1
        
        for i in range(n):
            if time.time() >= deadline:
                break
            pi = p[i]
            for j in range(i + 1, n):
                d = _delta_swap(a, b, p, i, j)
                if first_improvement:
                    if d < 0:
                        p[i], p[j] = p[j], p[i]
                        cur += d
                        improved = True
                        if cur < best_cost:
                            best_cost = cur
                            best_p = p.copy()
                        break  # Restart inner loop for this i
                else:
                    if d < best_delta:
                        best_delta = d
                        best_i = i
                        best_j = j
        
        if not first_improvement and best_i >= 0:
            p[best_i], p[best_j] = p[best_j], p[best_i]
            cur += best_delta
            improved = True
            if cur < best_cost:
                best_cost = cur
                best_p = p.copy()
    
    if cur < best_cost:
        best_cost = cur
        best_p = p.copy()
        
    return best_cost, best_p


def _anneal(a, b, p, deadline, best_cost, best_p, rng, t_start=50.0, t_end=0.1, alpha=0.995):
    """Simulated annealing polish."""
    n = len(p)
    cur = _cost(a, b, p)
    if cur < best_cost:
        best_cost = cur
        best_p = p.copy()
    
    t = t_start
    while time.time() < deadline and t > t_end:
        i = int(rng.integers(n))
        j = int(rng.integers(n))
        if i == j:
            continue
        d = _delta_swap(a, b, p, i, j)
        if d < 0 or (d > 0 and rng.random() < np.exp(-d / max(t, 1e-10))):
            p[i], p[j] = p[j], p[i]
            cur += d
            if cur < best_cost:
                best_cost = cur
                best_p = p.copy()
        t *= alpha
    
    # Final local search
    cur = _cost(a, b, p)
    if cur < best_cost:
        best_cost = cur
        best_p = p.copy()
    best_cost, best_p = _local_search(a, b, p, deadline, best_cost, best_p, first_improvement=True)
    return best_cost, best_p


def solve(a, b):
    a = np.ascontiguousarray(a, dtype=np.int64)
    b = np.ascontiguousarray(b, dtype=np.int64)
    n = len(a)
    t0 = time.time()
    deadline = t0 + 115.0  # Safety margin before 120s kill

    rng = np.random.default_rng(12345)
    
    # --- Constructive starts ---
    best_cost = np.iinfo(np.int64).max
    best_p = None
    
    # Strategy 1: Rank matching
    heavy = np.argsort(-a.sum(axis=1), kind="stable")
    central = np.argsort(b.sum(axis=1), kind="stable")
    p1 = np.empty(n, dtype=np.int64)
    p1[heavy] = central
    c1 = _cost(a, b, p1)
    if c1 < best_cost:
        best_cost = c1
        best_p = p1.copy()
        
    # Strategy 2: Greedy constructive heuristic
    p2 = np.zeros(n, dtype=np.int64)
    used = np.zeros(n, dtype=bool)
    order = np.argsort(-a.sum(axis=1), kind="stable")
    for idx in order:
        if not np.any(used):
            p2[idx] = int(np.argmin(b.sum(axis=1)))
        else:
            placed = np.where(~used)[0]
            # Vectorized score calculation for greedy step
            # score[l] = sum_{j in placed} (a[idx, j] * b[l, p2[j]] + a[j, idx] * b[p2[j], l])
            scores = np.zeros(n, dtype=np.float64)
            for j in placed:
                scores += a[idx, j] * b[:, p2[j]] + a[j, idx] * b[p2[j], :]
            scores[used] = np.inf
            p2[idx] = int(np.argmin(scores))
        used[p2[idx]] = True
        
    c2 = _cost(a, b, p2)
    if c2 < best_cost:
        best_cost = c2
        best_p = p2.copy()
        
    # Strategy 3: Random starts
    for _ in range(5):
        p_r = rng.permutation(n)
        c_r = _cost(a, b, p_r)
        if c_r < best_cost:
            best_cost = c_r
            best_p = p_r.copy()
    
    # --- Local search on the best start ---
    if best_p is not None:
        best_cost, best_p = _local_search(a, b, best_p.copy(), deadline, best_cost, best_p, first_improvement=True)
    
    # --- Simulated annealing with restarts ---
    if time.time() < deadline - 30:
        p_sa = best_p.copy()
        best_cost, best_p = _anneal(a, b, p_sa, deadline - 10, best_cost, best_p, rng)
    
    # --- Additional random restarts with local search ---
    while time.time() < deadline - 15:
        p_r = rng.permutation(n)
        c_r = _cost(a, b, p_r)
        # Only do local search if the random start is not too far from best
        if c_r < best_cost * 1.5:
            best_cost, best_p = _local_search(a, b, p_r, deadline - 5, best_cost, best_p, first_improvement=True)
    
    return best_p.tolist()
# EVOLVE-BLOCK-END
