# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import combinations


def ring_routing() -> np.ndarray:
    """
    Route 15 demands on a 10-node ring to minimize the maximum edge load.
    Returns an array of shape (15,) where entry >= 0.5 means CCW, else CW.
    """
    n = 10
    D = 15
    demands = [
        (0, 5, 3), (1, 7, 2), (2, 9, 4), (3, 6, 1), (0, 4, 5),
        (2, 8, 2), (5, 9, 3), (1, 4, 2), (6, 0, 4), (3, 8, 1),
        (7, 2, 3), (4, 9, 2), (8, 1, 5), (0, 6, 1), (5, 2, 2),
    ]

    # Precompute the set of edges used by each demand in each direction.
    # CW from s to t uses edges s, s+1, ..., t-1 (mod n). Edge e joins e and (e+1)%n.
    cw_edges = []
    ccw_edges = []
    for (s, t, w) in demands:
        d_cw = (t - s) % n
        if d_cw == 0:
            cw_edges.append(frozenset())
            ccw_edges.append(frozenset())
        else:
            cw_set = frozenset(((s + k) % n) for k in range(d_cw))
            ccw_set = frozenset(((s + k) % n) for k in range(n - d_cw))
            cw_edges.append(cw_set)
            ccw_edges.append(ccw_set)

    weights = np.array([w for (_, _, w) in demands], dtype=np.float64)
    # For each demand and each direction, a vector of length n: 1 if edge on path else 0.
    cw_mat = np.zeros((D, n), dtype=np.float64)
    ccw_mat = np.zeros((D, n), dtype=np.float64)
    for i in range(D):
        for e in cw_edges[i]:
            cw_mat[i, e] = 1.0
        for e in ccw_edges[i]:
            ccw_mat[i, e] = 1.0

    # A routing is a binary vector r of length D: 1 = CCW, 0 = CW.
    # Edge load = sum_i w_i * M_i[e] where M_i = ccw_mat[i] if r_i else cw_mat[i].
    # Build base = sum w_i * cw_mat[i] (all CW). Then flipping i to CCW adds
    # w_i * (ccw_mat[i] - cw_mat[i]).

    base_load = weights @ cw_mat  # shape (n,)
    delta = (ccw_mat - cw_mat) * weights[:, None]  # shape (D, n)

    def max_load(r):
        # r: 0/1 array length D
        if np.any(r):
            load = base_load + delta[r == 1].sum(axis=0)
        else:
            load = base_load
        return float(np.max(load)), load

    def compute_load(r):
        if np.any(r):
            load = base_load + delta[r == 1].sum(axis=0)
        else:
            load = base_load
        return load

    start = time.time()
    deadline = start + 590.0  # safety margin well under 630s

    # Initial solution: greedy. Start all CW, then try flipping each demand if it
    # reduces the max load; repeat passes until no improvement.
    r = np.zeros(D, dtype=np.int8)
    best_r = r.copy()
    best_val, _ = max_load(best_r)

    # Greedy local improvement (single flips)
    improved = True
    while improved:
        improved = False
        for i in range(D):
            r[i] = 1 - r[i]
            v, _ = max_load(r)
            if v < best_val - 1e-12:
                best_val = v
                best_r = r.copy()
                improved = True
            else:
                r[i] = 1 - r[i]  # revert

    # Also try starting from all-CCW
    r_ccw = np.ones(D, dtype=np.int8)
    v_ccw, _ = max_load(r_ccw)
    if v_ccw < best_val - 1e-12:
        best_val = v_ccw
        best_r = r_ccw.copy()
        # Do greedy from this start too
        r = r_ccw.copy()
        improved = True
        while improved:
            improved = False
            for i in range(D):
                r[i] = 1 - r[i]
                v, _ = max_load(r)
                if v < best_val - 1e-12:
                    best_val = v
                    best_r = r.copy()
                    improved = True
                else:
                    r[i] = 1 - r[i]

    # Pairwise swap neighborhood: flip two bits at once
    def pairwise_improve(r, best_val, best_r):
        improved = True
        while improved:
            improved = False
            for i in range(D):
                for j in range(i + 1, D):
                    r[i] = 1 - r[i]
                    r[j] = 1 - r[j]
                    v, _ = max_load(r)
                    if v < best_val - 1e-12:
                        best_val = v
                        best_r = r.copy()
                        improved = True
                    else:
                        r[i] = 1 - r[i]
                        r[j] = 1 - r[j]
        return best_val, best_r

    # Apply pairwise improvement
    r = best_r.copy()
    best_val, best_r = pairwise_improve(r, best_val, best_r)

    # Simulated annealing / random restarts to escape local optima, with time guard.
    rng = np.random.default_rng(12345)
    cur_r = best_r.copy()
    cur_val, _ = max_load(cur_r)

    # Hill climbing with random perturbation + acceptance (SA-like).
    t_max = 0.05
    t_min = 1e-4
    iters_budget = 400000
    it = 0
    n_improve = 0
    while it < iters_budget:
        if time.time() > deadline:
            break
        i = int(rng.integers(D))
        j = int(rng.integers(D))
        if i == j:
            # single flip
            cur_r[i] = 1 - cur_r[i]
            v, _ = max_load(cur_r)
            cur_r[i] = 1 - cur_r[i]  # revert tentatively
            # accept?
            if v < cur_val or (time.time() > deadline - 60 and v < best_val - 1e-12):
                # do it for real
                cur_r[i] = 1 - cur_r[i]
                cur_val = v
                if v < best_val - 1e-12:
                    best_val = v
                    best_r = cur_r.copy()
                    n_improve += 1
        else:
            # double flip
            cur_r[i] = 1 - cur_r[i]
            cur_r[j] = 1 - cur_r[j]
            v, _ = max_load(cur_r)
            if v < cur_val or (time.time() > deadline - 60 and v < best_val - 1e-12):
                cur_val = v
                if v < best_val - 1e-12:
                    best_val = v
                    best_r = cur_r.copy()
                    n_improve += 1
            else:
                cur_r[i] = 1 - cur_r[i]
                cur_r[j] = 1 - cur_r[j]
        it += 1
        # Occasional random restart from a random config, keep if better.
        if it % 20000 == 0:
            cand = rng.integers(0, 2, size=D).astype(np.int8)
            cv, _ = max_load(cand)
            if cv < best_val - 1e-12:
                best_val = cv
                best_r = cand.copy()
                cur_r = cand.copy()
                cur_val = cv

    # Final: ensure best_r is a 0/1 int array; map to output (>=0.5 CCW else CW).
    out = best_r.astype(np.float64)
    # Per contract: entry >= 0.5 routes CCW. Our 1 = CCW, 0 = CW, so out in {0,1} works.
    return out


# EVOLVE-BLOCK-END
