"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _qap_cost(a, b, p):
    return int(np.sum(a * b[np.ix_(p, p)]))


def _swap_delta(a, b, p, i, j):
    pi = p[i]
    pj = p[j]
    bpj = b[:, pj][p]
    bpi = b[:, pi][p]
    brj = b[pj, :][p]
    bri = b[pi, :][p]
    delta = (a[:, i] * (bpj - bpi) +
             a[:, j] * (bpi - bpj) +
             a[i, :] * (brj - bri) +
             a[j, :] * (bri - brj))
    delta[i] = 0
    delta[j] = 0
    direct = a[i, j] * (b[pj, pi] - b[pi, pj]) + a[j, i] * (b[pi, pj] - b[pj, pi])
    return int(np.sum(delta) + direct)


def solve(a, b):
    n = len(a)
    deadline = time.time() + 110
    heavy = np.argsort(-a.sum(axis=1), kind="stable")
    central = np.argsort(b.sum(axis=1), kind="stable")
    p = np.empty(n, dtype=np.int64)
    p[heavy] = central
    current_cost = _qap_cost(a, b, p)
    best_p = p.copy()
    best_cost = current_cost
    T = 5000.0
    T_min = 1.0
    cooling = 0.99999
    rng = np.random.default_rng(42)
    while time.time() < deadline:
        if T < T_min:
            T = 5000.0
            p = best_p.copy()
            current_cost = best_cost
            continue
        i, j = rng.integers(0, n, size=2)
        if i == j:
            continue
        d = _swap_delta(a, b, p, i, j)
        if d < 0 or rng.random() < np.exp(-d / T):
            p[i], p[j] = p[j], p[i]
            current_cost += d
            if current_cost < best_cost:
                best_cost = current_cost
                best_p = p.copy()
        T *= cooling
    return best_p.tolist()
# EVOLVE-BLOCK-END
