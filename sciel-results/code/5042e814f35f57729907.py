# EVOLVE-BLOCK-START
import time
import numpy as np


def _qap_cost(a, b, p):
    return int(np.sum(a * b[np.ix_(p, p)]))


def _local_search(c, d, p, deadline):
    n = len(p)
    p = p.copy()
    cost = int(np.sum(c * d[np.ix_(p, p)]))
    # Precompute the pairwise flow between facilities.
    cmat = c
    # Candidate pool: highest-flow facility pairs (most impactful swaps).
    ii, jj = np.triu_indices(n, k=1)
    order = np.argsort(-cmat[ii, jj], kind="stable")
    pool = np.stack([ii[order], jj[order]], axis=1)
    # Limit the pool to a manageable size for speed.
    pool = pool[: min(len(pool), n * 8)]
    improved = True
    while improved:
        improved = False
        pi = p
        for idx in range(pool.shape[0]):
            if time.time() >= deadline:
                break
            i, j = int(pool[idx, 0]), int(pool[idx, 1])
            if i == j:
                continue
            pi_i = pi[i]
            pi_j = pi[j]
            if pi_i == pi_j:
                continue
            # delta = sum_k c[i,k]*(d[pi_i,pi_k]-d[pi_k,pi_j])
            #       + sum_k c[j,k]*(d[pi_j,pi_k]-d[pi_k,pi_i])
            #       + c[i,j]*(d[pi_i,pi_j]-d[pi_j,pi_i])
            delta = (
                np.dot(cmat[i], d[pi_i] - d[pi, pi_j])
                + np.dot(cmat[j], d[pi_j] - d[pi, pi_i])
                + cmat[i, j] * (d[pi_i, pi_j] - d[pi_j, pi_i])
            )
            if delta < 0:
                pi[i], pi[j] = pi_j, pi_i
                cost += int(delta)
                improved = True
        # If we improved, re-evaluate pool from scratch next pass (cheap enough).
    return p, cost


def solve(a, b):
    start = time.time()
    deadline = start + 110.0  # safety margin under 120s
    n = len(a)

    # Symmetric matrices (cost is invariant under symmetrization of a and b).
    c = a + a.T
    d = b + b.T

    best_p = None
    best_cost = None

    def consider(p):
        nonlocal best_p, best_cost
        p, cost = _local_search(c, d, p, deadline)
        if best_cost is None or cost < best_cost:
            best_cost = cost
            best_p = p
        return p, cost

    # Restart 0: rank-matching (heavy flows -> short distances).
    heavy = np.argsort(-a.sum(axis=1), kind="stable")
    central = np.argsort(b.sum(axis=1), kind="stable")
    p0 = np.empty(n, dtype=np.int64)
    p0[heavy] = central
    consider(p0)

    # Additional random restarts while time permits.
    rng = np.random.default_rng(12345)
    while time.time() < deadline - 15.0:
        pr = rng.permutation(n)
        consider(pr)

    if best_p is None:
        best_p = p0
    return best_p.tolist()
# EVOLVE-BLOCK-END
