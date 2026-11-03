# EVOLVE-BLOCK-START
import time
import numpy as np


def _compute_cost(a, b, p):
    """Exact integer QAP cost for permutation p (facility i at location p[i])."""
    return int(np.sum(a * b[np.ix_(p, p)]))


def _init_greedy(a, b, n, rng):
    """Pairwise greedy initialization.

    Different modelling family than rank-matching: instead of matching row sums,
    we build the assignment facility-by-facility, placing each facility at the
    location that minimizes the incremental interaction cost with already-placed
    facilities. This is a min-assignment style construction that directly
    accounts for the flow-distance cross terms.
    """
    placed = np.zeros(n, dtype=bool)
    p = np.empty(n, dtype=np.int64)
    # row_j[j] = sum_i a[i,j] * b[p[i], j]  (cost contribution of putting facility j at location j... tracked per location)
    # We maintain for each candidate location L the marginal cost of placing the next facility there.
    # marginal(L) for facility f = sum over placed facilities i of a[f,i]*b[L,p[i]] + a[i,f]*b[p[i],L]
    # We process facilities in order of total flow (most connected first) for stability.
    order = np.argsort(-a.sum(axis=1), kind="stable")
    # acc[L] = sum over placed i of (a[?,i]*b[L,p[i]] + a[i,?]*b[p[i],L]) placeholder per facility
    # Efficient: keep two vectors updated per placement.
    # For a newly placed facility i at location L_i:
    #   it adds to marginal(L) for a future facility f: a[f,i]*b[L,L_i] + a[i,f]*b[L_i,L]
    # We precompute nothing; instead keep arrays:
    #   contrib1[L] = sum_i a[?,i]*b[L,p[i]]  -> depends on f via a[f,i]
    # So marginal_f(L) = sum_i a[f,i]*b[L,p[i]] + sum_i a[i,f]*b[p[i],L]
    #                  = (A_col @ b_col[L]) where A_col[i]=a[f,i], b_col[L]=sum_i b[L,p[i]]? No.
    # Let's define:
    #   S1[L] = sum over placed i of b[L, p[i]]   (independent of f) -- not enough.
    # We need per-f. Use:
    #   termA_f(L) = sum_i a[f,i]*b[L,p[i]]
    #   termB_f(L) = sum_i a[i,f]*b[p[i],L]
    # Keep placed facility set; update incrementally is O(n) per placement, O(n^2) total -- fine.
    placed_list = []
    for f in order:
        bestL = -1
        bestC = None
        for L in range(n):
            if placed[L]:
                continue
            c = 0
            for i in placed_list:
                c += a[f, i] * b[L, p[i]] + a[i, f] * b[p[i], L]
            if bestC is None or c < bestC:
                bestC = c
                bestL = L
        p[f] = bestL
        placed[bestL] = True
        placed_list.append(f)
    return p


def _init_rank(a, b, n):
    """Rank-matching initialization (baseline)."""
    heavy = np.argsort(-a.sum(axis=1), kind="stable")
    central = np.argsort(b.sum(axis=1), kind="stable")
    p = np.empty(n, dtype=np.int64)
    p[heavy] = central
    return p


def _two_opt(a, b, p, max_iters, deadline, rng):
    """Vectorized 2-opt local search with O(n) incremental delta per swap."""
    n = len(p)
    a = np.ascontiguousarray(a)
    b = np.ascontiguousarray(b)
    p = p.copy()

    # Precompute per-facility row/col contributions for fast delta.
    # cost = sum_i sum_j a[i,j]*b[p[i],p[j]]
    # Let rowA[i] = a[i,:], colA[i] = a[:,i]
    # When swapping locations of facilities i and j (p[i]<->p[j]):
    # delta = (a[i,j]+a[j,i])*(b[p[j],p[i]]+b[p[i],p[j]])
    #        - (a[i,j]+a[j,i])*(b[p[i],p[j]]+b[p[j],p[i]])  [same, cancels]
    # Actually standard: delta = (a[i,j]+a[j,i])*(b[p[j],p[i]]+b[p[i],p[j]] - b[p[i],p[j]] - b[p[j],p[i]])
    # Let me derive properly.
    # Terms involving i and j in cost:
    #   a[i,j]*b[p[i],p[j]] + a[j,i]*b[p[j],p[i]]  (cross)
    #   a[i,k]*b[p[i],p[k]] for k!=i,j  and a[k,i]*b[p[k],p[i]]
    #   a[j,k]*b[p[j],p[k]] for k!=i,j  and a[k,j]*b[p[k],p[j]]
    # After swap (p[i],p[j] exchanged):
    #   cross: a[i,j]*b[p[j],p[i]] + a[j,i]*b[p[i],p[j]]
    #   a[i,k]*b[p[j],p[k]] + a[k,i]*b[p[k],p[j]]
    #   a[j,k]*b[p[i],p[k]] + a[k,j]*b[p[k],p[i]]
    # delta = [a[i,j]*b[p[j],p[i]] + a[j,i]*b[p[i],p[j]]
    #          - a[i,j]*b[p[i],p[j]] - a[j,i]*b[p[j],p[i]]]
    #       + sum_{k!=i,j} [a[i,k]*b[p[j],p[k]] + a[k,i]*b[p[k],p[j]]
    #          - a[i,k]*b[p[i],p[k]] - a[k,i]*b[p[k],p[i]]
    #          + a[j,k]*b[p[i],p[k]] + a[k,j]*b[p[k],p[i]]
    #          - a[j,k]*b[p[j],p[k]] - a[k,j]*b[p[k],p[j]]]
    # Group cross: (a[i,j]-a[j,i])*(b[p[j],p[i]]-b[p[i],p[j]])
    # Group k-sum: sum_k [(a[i,k]-a[j,k])*(b[p[j],p[k]]-b[p[i],p[k]])
    #                 + (a[k,i]-a[k,j])*(b[p[k],p[j]]-b[p[k],p[i]])]
    # This is O(n) per candidate pair but we want vectorized over k for a fixed (i,j).

    def delta_ij(i, j):
        li, lj = p[i], p[j]
        # cross term
        cross = (a[i, j] - a[j, i]) * (b[lj, li] - b[li, lj])
        # k-sum via vectorization
        ai = a[i, :] - a[j, :]
        ak = a[:, i] - a[:, j]
        bli = b[li, :] - b[lj, :]
        bki = b[:, li] - b[:, lj]
        # exclude i and j from k
        s1 = np.dot(ai, bli)
        s2 = np.dot(ak, bki)
        # remove k=i and k=j contributions
        s1 -= ai[i] * bli[i] + ai[j] * bli[j]
        s2 -= ak[i] * bki[i] + ak[j] * bki[j]
        return cross + s1 + s2

    improved = True
    iters = 0
    while improved and iters < max_iters:
        improved = False
        iters += 1
        if time.time() > deadline:
            break
        # First-improvement scan: for each i, find best j
        for i in range(n):
            if time.time() > deadline:
                break
            best_j = -1
            best_d = 0
            for j in range(i + 1, n):
                d = delta_ij(i, j)
                if d < best_d:
                    best_d = d
                    best_j = j
            if best_j >= 0:
                p[i], p[best_j] = p[best_j], p[i]
                improved = True
    return p


def _perturb(p, k, rng):
    """Random k-swap perturbation to escape local minima."""
    n = len(p)
    p = p.copy()
    idx = rng.choice(n, size=min(2 * k, n), replace=False)
    for t in range(len(idx) // 2):
        i, j = idx[2 * t], idx[2 * t + 1]
        p[i], p[j] = p[j], p[i]
    return p


def solve(a, b):
    t0 = time.time()
    deadline = t0 + 145.0  # safety margin below 150s kill
    n = len(a)
    a = np.ascontiguousarray(a, dtype=np.int64)
    b = np.ascontiguousarray(b, dtype=np.int64)
    rng = np.random.default_rng(12345)

    best_p = None
    best_cost = None

    # Try multiple diverse initializations and optimize each.
    inits = [
        ("rank", _init_rank(a, b, n)),
        ("greedy", _init_greedy(a, b, n, rng)),
    ]
    # Add a few random diversification seeds
    for r in range(3):
        heavy = np.argsort(-a.sum(axis=1), kind="stable")
        central = np.argsort(b.sum(axis=1), kind="stable")
        p = np.empty(n, dtype=np.int64)
        p[heavy] = central
        # random perturbation
        p = _perturb(p, 5, rng)
        inits.append((f"rand{r}", p))

    for name, p0 in inits:
        if time.time() > deadline - 5.0:
            break
        p0 = p0.copy()
        # Local search
        p = _two_opt(a, b, p0, max_iters=200, deadline=deadline, rng=rng)
        c = _compute_cost(a, b, p)
        if best_cost is None or c < best_cost:
            best_cost = c
            best_p = p.copy()

        # Stagnation escape: if we have time, perturb and re-optimize
        while time.time() < deadline - 5.0:
            p2 = _perturb(p, 4, rng)
            p2 = _two_opt(a, b, p2, max_iters=50, deadline=deadline, rng=rng)
            c2 = _compute_cost(a, b, p2)
            if c2 < c:
                p = p2
                c = c2
                if best_cost is None or c < best_cost:
                    best_cost = c
                    best_p = p.copy()
            else:
                # no improvement, break out of perturbation loop for this init
                break

    if best_p is None:
        best_p = _init_rank(a, b, n)
    return best_p.tolist()
# EVOLVE-BLOCK-END
