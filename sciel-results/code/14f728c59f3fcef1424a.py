"""Fresh design: smooth log-sum-exp objective + simulated annealing + polish."""
import time
import numpy as np

N = 14
D = 3

_START = time.time()
_DEADLINE = 115.0  # safety margin below the 120s wall-clock limit


def _pairwise_sq(P):
    """Squared pairwise distance matrix (n,n)."""
    g = P[:, None, :] - P[None, :, :]
    return np.einsum("ijk,ijk->ij", g, g)


def _objective(P, beta):
    """Smooth proxy: (max d)^2 / (min d)^2 approximated via log-sum-exp."""
    D2 = _pairwise_sq(P)
    iu = np.triu_indices(N, 1)
    v = D2[iu]
    vmax = v.max()
    vmin = v.min()
    if vmin <= 1e-12:
        return 1e12
    Lmax = beta * np.log(np.sum(np.exp(beta * (v - vmax)))) + beta * vmax
    Lmin = beta * np.log(np.sum(np.exp(-beta * (v - vmin)))) - beta * vmin
    # (max/min)^2 = exp(2*(Lmax - Lmin))
    return float(np.exp(2.0 * (Lmax - Lmin)))


def _anneal(P0, beta, iters, rng):
    """Simulated annealing with random-walk steps."""
    cur = P0.copy()
    cur_obj = _objective(cur, beta)
    best = cur.copy()
    best_obj = cur_obj
    for it in range(iters):
        if time.time() - _START > _DEADLINE:
            break
        step = 0.02 + 0.05 * (it / max(1, iters))
        cand = cur + step * rng.standard_normal((N, D))
        cand_obj = _objective(cand, beta)
        if cand_obj < cur_obj or rng.random() < np.exp(-(cand_obj - cur_obj) / (0.5 + 0.5 * it / iters)):
            cur = cand
            cur_obj = cand_obj
            if cur_obj < best_obj:
                best_obj = cur_obj
                best = cur.copy()
    return best, best_obj


def construct():
    global _START, _DEADLINE
    _START = time.time()
    _DEADLINE = _START + 115.0

    rng = np.random.default_rng(12345)
    beta = 4.0
    best_P = None
    best_obj = np.inf

    # Multiple diverse random restarts
    for trial in range(200):
        if time.time() - _START > _DEADLINE - 15.0:
            break
        # random start on unit ball
        P0 = rng.standard_normal((N, D))
        P0 /= np.linalg.norm(P0, axis=1, keepdims=True)
        # small jitter to break symmetry
        P0 += 0.01 * rng.standard_normal((N, D))
        P, obj = _anneal(P0, beta, 400, rng)
        if obj < best_obj:
            best_obj = obj
            best_P = P

    # Deterministic polish: a few gradient-free Nelder-Mead-like local steps
    if best_P is not None and time.time() - _START < _DEADLINE - 10.0:
        # simple coordinate descent polish
        P = best_P.copy()
        obj = _objective(P, beta)
        for it in range(3000):
            if time.time() - _START > _DEADLINE:
                break
            improved = False
            for i in range(N):
                for d in range(D):
                    for sign in (1, -1):
                        for step in (0.01, 0.001):
                            P2 = P.copy()
                            P2[i, d] += sign * step
                            o2 = _objective(P2, beta)
                            if o2 < obj:
                                P = P2
                                obj = o2
                                improved = True
            if not improved:
                break
        if obj < best_obj:
            best_obj = obj
            best_P = P

    if best_P is None:
        # fallback grid
        side = int(np.ceil(N ** (1 / D)))
        import itertools
        grid = list(itertools.product(range(side), repeat=D))[:N]
        return np.array(grid, dtype=float)
    return best_P
