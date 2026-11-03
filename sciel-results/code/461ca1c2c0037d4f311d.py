"""Vectorized Simulated Annealing for QAP wil100."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _qap_cost(a, b, p):
    return int(np.sum(a * b[np.ix_(p, p)]))


def _swap_delta(a, b, p, inv, i, j):
    """Exact cost change when swapping positions i and j."""
    pi = inv[i]
    pj = inv[j]
    if pi == pj:
        return 0
    bpi_pi = b[pi, pi]
    bpj_pj = b[pj, pj]
    bpi_pj = b[pi, pj]
    bpj_pi = b[pj, pi]
    aii = a[pi, pi]
    aij = a[pi, pj]
    ajj = a[pj, pj]
    apj_i = a[pj, pi]
    # Swapping changes: pair (pi,pi),(pi,pj),(pj,pi),(pj,pj) contributions
    before = aii * bpi_pi + aij * bpi_pj + apj_i * bpj_pi + ajj * bpj_pj
    after = aii * bpi_pj + aij * bpj_pi + apj_i * bpi_pj + ajj * bpi_pi
    return int(after - before)


def solve(a, b):
    n = len(a)
    a = np.asarray(a, dtype=np.int64)
    b = np.asarray(b, dtype=np.int64)

    deadline = time.time() + 110.0

    # Rank-matching start
    heavy = np.argsort(-a.sum(axis=1), kind="stable")
    central = np.argsort(b.sum(axis=1), kind="stable")
    p = np.empty(n, dtype=np.int64)
    p[heavy] = central
    inv = np.empty(n, dtype=np.int64)
    inv[p] = np.arange(n)

    cur_cost = _qap_cost(a, b, p)
    best_p = p.copy()
    best_cost = cur_cost

    rng = np.random.default_rng(2024)
    T = max(float(cur_cost) * 0.02, 10.0)
    T_min = 1e-3
    cooling = 0.9995

    while time.time() < deadline:
        # Vectorized candidate deltas
        batch = 400
        idx = rng.integers(0, n, size=(batch, 2))
        i1, i2 = idx[:, 0], idx[:, 1]
        mask = i1 != i2
        i1, i2 = i1[mask], i2[mask]
        if len(i1) == 0:
            T *= cooling
            if T < T_min:
                break
            continue

        pi = inv[i1]
        pj = inv[i2]
        bpi_pi = b[np.ix_(pi, pi)]
        bpj_pj = b[np.ix_(pj, pj)]
        bpi_pj = b[np.ix_(pi, pj)]
        bpj_pi = b[np.ix_(pj, pi)]
        aii = np.array([a[pi[k], pi[k]] for k in range(len(pi))])
        aij = np.array([a[pi[k], pj[k]] for k in range(len(pi))])
        ajj = np.array([a[pj[k], pj[k]] for k in range(len(pi))])
        apj_i = np.array([a[pj[k], pi[k]] for k in range(len(pi))])

        before = aii * bpi_pi + aij * bpi_pj + apj_i * bpj_pi + ajj * bpj_pj
        after = aii * bpi_pj + aij * bpj_pi + apj_i * bpi_pj + ajj * bpi_pi
        delta = after - before

        accept = delta < 0
        pos = np.where(~accept)[0]
        if len(pos) > 0:
            d = delta[pos]
            prob = np.exp(-d / (T + 1e-12))
            accept[pos] = rng.random(len(pos)) < prob

        if not np.any(accept):
            T *= cooling
            if T < T_min:
                break
            continue

        ai1 = i1[accept]
        ai2 = i2[accept]
        for i, j in zip(ai1, ai2):
            d = _swap_delta(a, b, p, inv, i, j)
            pi = inv[i]
            pj = inv[j]
            p[i], p[j] = p[j], p[i]
            inv[i], inv[j] = inv[j], inv[i]
            cur_cost += d

        if cur_cost < best_cost:
            best_cost = cur_cost
            best_p = p.copy()

        T *= cooling
        if T < T_min:
            break

    return best_p.tolist()
# EVOLVE-BLOCK-END
