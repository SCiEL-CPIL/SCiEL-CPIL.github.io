"""Vectorized Simulated Annealing for QAP wil100."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _qap_cost(a, b, p):
    """Exact cost sum_{i,j} a[i,j] * b[p[i], p[j]]."""
    return int(np.sum(a * b[np.ix_(p, p)]))


def solve(a, b):
    n = len(a)
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)

    deadline = time.time() + 110.0  # safety margin under 120s

    # Rank-matching start
    heavy = np.argsort(-a.sum(axis=1), kind="stable")
    central = np.argsort(b.sum(axis=1), kind="stable")
    p = np.empty(n, dtype=np.int64)
    p[heavy] = central

    best_p = p.copy()
    best_cost = _qap_cost(a, b, p)

    # Simulated Annealing
    T = float(best_cost) * 0.05 + 1.0
    T_min = 1e-6
    cooling = 0.9999
    rng = np.random.default_rng(12345)

    while time.time() < deadline:
        # Batch of random swaps
        batch = 500
        idx = rng.integers(0, n, size=(batch, 2))
        i1, i2 = idx[:, 0], idx[:, 1]
        mask = i1 != i2
        i1, i2 = i1[mask], i2[mask]
        
        if len(i1) == 0:
            T *= cooling
            continue

        # Vectorized Delta Calculation
        # i1, i2 are positions. pi, pj are facilities.
        pi = p[i1]
        pj = p[i2]
        
        # Term 1: (a[pi, pj] - a[pj, pi]) * (b[i2, i1] - b[i1, i2])
        a_pi_pj = a[pi, pj]
        a_pj_pi = a[pj, pi]
        b_i2_i1 = b[i2, i1]
        b_i1_i2 = b[i1, i2]
        term1 = (a_pi_pj - a_pj_pi) * (b_i2_i1 - b_i1_i2)
        
        # Term 2: a[pi, pi] * (b[i2, i2] - b[i1, i1])
        a_pi_pi = a[pi, pi]
        b_i2_i2 = b[i2, i2]
        b_i1_i1 = b[i1, i1]
        term2 = a_pi_pi * (b_i2_i2 - b_i1_i1)
        
        # Term 3: a[pj, pj] * (b[i1, i1] - b[i2, i2])
        a_pj_pj = a[pj, pj]
        term3 = a_pj_pj * (b_i1_i1 - b_i2_i2)
        
        # Term 4: Sum over m != pi, pj of (a[m, pi] - a[m, pj]) * (b[p[m], i2] - b[p[m], i1])
        # Let vecA = a[:, pi] - a[:, pj]  (Shape N)
        # Let vecB = b[:, p[i2]] - b[:, p[i1]] (Shape N) -- Wait, b is indexed by location.
        # b[p[m], i2] means row p[m], col i2.
        # So we need columns i2 and i1 of B.
        # col_i2 = b[:, i2]
        # col_i1 = b[:, i1]
        # diffB = col_i2 - col_i1
        # We need sum_m vecA[m] * diffB[m]
        # But we must exclude m=pi and m=pj.
        
        # Vectorize for batch:
        # M_A shape (Batch, N). Row k is a[:, pi_k] - a[:, pj_k]
        # M_B shape (Batch, N). Row k is b[:, i2_k] - b[:, i1_k]
        
        # Construct M_A
        # a[:, pi] is hard to vectorize directly as a matrix of columns.
        # a[pi, :] are rows. We need columns.
        # a.T[pi, :] are rows of a.T which are columns of a.
        # So M_A = a.T[pi, :] - a.T[pj, :]
        
        M_A = a.T[pi, :] - a.T[pj, :]
        
        # M_B: b[:, i2] - b[:, i1]
        # b[i2, :] are rows. We need columns.
        # b.T[i2, :] are columns.
        M_B = b.T[i2, :] - b.T[i1, :]
        
        # S_full = sum(M_A * M_B, axis=1)
        S_full = np.sum(M_A * M_B, axis=1)
        
        # Subtract excluded terms m=pi and m=pj
        # m=pi term in S_full: (a[pi, pi] - a[pi, pj]) * (b[p[pi], i2] - b[p[pi], i1])
        # p[pi] is the location of facility pi. That is i1.
        # So b[p[pi], i2] = b[i1, i2]
        # b[p[pi], i1] = b[i1, i1]
        # Term_pi = (a[pi, pi] - a[pi, pj]) * (b[i1, i2] - b[i1, i1])
        
        # m=pj term in S_full: (a[pj, pi] - a[pj, pj]) * (b[p[pj], i2] - b[p[pj], i1])
        # p[pj] is the location of facility pj. That is i2.
        # So b[p[pj], i2] = b[i2, i2]
        # b[p[pj], i1] = b[i2, i1]
        # Term_pj = (a[pj, pi] - a[pj, pj]) * (b[i2, i2] - b[i2, i1])
        
        term_pi = (a[pi, pi] - a[pi, pj]) * (b[i1, i2] - b[i1, i1])
        term_pj = (a[pj, pi] - a[pj, pj]) * (b[i2, i2] - b[i2, i1])
        
        S_excluded = term_pi + term_pj
        term4 = S_full - S_excluded
        
        delta = term1 + term2 + term3 + term4
        
        # Accept improving or probabilistically
        accept = delta < 0
        pos = np.where(~accept)[0]
        if len(pos) > 0:
            d = delta[pos]
            prob = np.exp(-d / (T + 1e-12))
            accept_pos = rng.random(len(pos)) < prob
            accept[pos] = accept_pos

        if not np.any(accept):
            T *= cooling
            if T < T_min:
                break
            continue

        # Apply accepted swaps
        ai1 = i1[accept]
        ai2 = i2[accept]
        
        # We must update p and recalculate cost carefully.
        # Since swaps are independent in terms of "state" only if we update p, 
        # but delta calculation assumed original p for all.
        # For small N, applying sequentially and recalculating cost is safer for correctness 
        # but expensive. 
        # However, N=100. Cost calc is O(N^2)=10k ops. 
        # If we accept ~100 swaps, that's 1M ops. Very fast.
        # Let's just apply all accepted swaps to p, then calculate new cost.
        
        for u, v in zip(ai1, ai2):
            p[u], p[v] = p[v], p[u]
            
        # Calculate new cost
        cur_cost = _qap_cost(a, b, p)
        
        if cur_cost < best_cost:
            best_cost = cur_cost
            best_p = p.copy()
            
        T *= cooling
        if T < T_min:
            break

    return best_p.tolist()
# EVOLVE-BLOCK-END
