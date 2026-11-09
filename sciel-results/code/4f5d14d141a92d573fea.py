# EVOLVE-BLOCK-START
import time
import numpy as np

_DEADLINE = time.time() + 600.0
_MARGIN = 12.0  # seconds of safety margin before the 630s kill


def _vertex_scores(P):
    """Per-vertex score in [0,1]. Vectorized over vertices."""
    n = P.shape[0]
    D = np.linalg.norm(P[:, None, :] - P[None, :, :], axis=2)
    np.fill_diagonal(D, np.inf)
    d = np.sort(D, axis=1)  # (n, n-1), d[:,0] <= ... <= d[:,n-2]
    m = n - 1
    gaps = d[:, 1:] - d[:, :-1]  # (n, m-1)
    k = np.argmin(gaps, axis=1)   # first index minimizing the gap
    Dmid = 0.5 * (d[np.arange(n), k] + d[np.arange(n), k + 1])

    # indices of the four distances around the gap
    idx = np.empty((n, 4), dtype=int)
    idx[:, 0] = np.where(k == 0, -1, k - 1)
    idx[:, 1] = k
    idx[:, 2] = k + 1
    idx[:, 3] = k + 2
    idx = np.clip(idx, 0, m - 1)
    sel = d[np.arange(n)[:, None], idx]  # (n,4)

    ratio = sel / Dmid[:, None]
    raw = np.mean(np.maximum(ratio, 1.0 / ratio), axis=1)
    s = np.minimum(1.0, 1.0 / raw)
    s[Dmid < 0.1] = 0.0
    s[d[:, 0] < 0.05] = 0.0
    return s


def _build_polygon(coeffs):
    """Coefficients: c1, c2, c3 (for k=1,2,3). Returns (20,2) polygon."""
    n = 20
    thetas = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    r = 1.0
    for k, c in zip((1, 2, 3), coeffs):
        r = r + c * np.cos(k * thetas)
    r = np.maximum(r, 0.3)
    return np.stack([r * np.cos(thetas), r * np.sin(thetas)], axis=1)


def _objective(coeffs):
    P = _build_polygon(coeffs)
    return float(np.min(_vertex_scores(P)))


class _DeadlineHit(Exception):
    pass


def equidistant_points_polygon20() -> np.ndarray:
    """
    Convex 20-gon maximizing the minimum per-vertex equidistance score.
    Radius parametrized as a low-order cosine series; optimized with
    Differential Evolution. Returns (20,2) array in cyclic (CCW) order.
    """
    from scipy.optimize import differential_evolution, minimize

    bounds = [(-0.12, 0.12), (-0.12, 0.12), (-0.08, 0.08)]

    # deterministic baseline: regular polygon (score ~0.818)
    best_P = _build_polygon(np.zeros(3))
    best_val = _objective(np.zeros(3))

    # local polish from the regular starting point
    try:
        res = minimize(lambda c: -_objective(c), np.zeros(3), method="Nelder-Mead",
                       options={"maxiter": 2000, "xatol": 1e-7, "fatol": 1e-9})
        if res.x is not None and np.all(np.isfinite(res.x)):
            v = _objective(res.x)
            if v > best_val:
                best_val = v
                best_P = _build_polygon(res.x)
    except Exception:
        pass

    # global search with Differential Evolution, guarded by deadline
    try:
        def _wrapped(c):
            if time.time() > _DEADLINE - _MARGIN:
                raise _DeadlineHit()
            return -_objective(c)

        de_res = differential_evolution(
            _wrapped, bounds,
            seed=1234,
            maxiter=150,
            popsize=12,
            tol=1e-8,
            mutation=(0.3, 0.9),
            recombination=0.8,
            polish=True,
            init="latinhypercube",
            updating="immediate",
        )
        if de_res.x is not None and np.all(np.isfinite(de_res.x)):
            v = _objective(de_res.x)
            if v > best_val:
                best_val = v
                best_P = _build_polygon(de_res.x)
    except _DeadlineHit:
        pass
    except Exception:
        pass

    # ensure finite
    if not np.all(np.isfinite(best_P)):
        best_P = _build_polygon(np.zeros(3))
    return best_P


# EVOLVE-BLOCK-END
