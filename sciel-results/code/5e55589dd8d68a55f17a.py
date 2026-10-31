"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 11

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution, minimize
from itertools import combinations
import time

_TRI_INDICES = np.array(list(combinations(range(N), 3)))
_SQRT3 = np.sqrt(3.0)
_TRI_AREA = _SQRT3 / 4.0


def _bary_to_cartesian(params):
    a = params[:, 0]
    b = params[:, 1]
    c = 1.0 - a - b
    neg = c < 0
    if np.any(neg):
        denom = a[neg] + b[neg]
        mask = denom > 0
        a_sub = a[neg]
        b_sub = b[neg]
        a_sub[mask] *= -c[neg][mask] / denom[mask]
        b_sub[mask] *= -c[neg][mask] / denom[mask]
        a[neg] = a_sub
        b[neg] = b_sub
        c = 1.0 - a - b
    pts = np.column_stack([b + 0.5 * c, c * _SQRT3 / 2.0])
    return pts


def _min_area_norm(params):
    pts = _bary_to_cartesian(params)
    p1 = pts[_TRI_INDICES[:, 0]]
    p2 = pts[_TRI_INDICES[:, 1]]
    p3 = pts[_TRI_INDICES[:, 2]]
    areas = 0.5 * np.abs((p2[:, 0] - p1[:, 0]) * (p3[:, 1] - p1[:, 1]) -
                          (p3[:, 0] - p1[:, 0]) * (p2[:, 1] - p1[:, 1]))
    return np.min(areas) / _TRI_AREA


def _obj_de(params_flat):
    params = params_flat.reshape(N, 2)
    return -_min_area_norm(params)


def _obj_soft(params_flat, alpha):
    params = params_flat.reshape(N, 2)
    pts = _bary_to_cartesian(params)
    p1 = pts[_TRI_INDICES[:, 0]]
    p2 = pts[_TRI_INDICES[:, 1]]
    p3 = pts[_TRI_INDICES[:, 2]]
    areas = 0.5 * np.abs((p2[:, 0] - p1[:, 0]) * (p3[:, 1] - p1[:, 1]) -
                          (p3[:, 0] - p1[:, 0]) * (p2[:, 1] - p1[:, 1]))
    an = areas / _TRI_AREA
    mn = np.min(an)
    return -(mn - np.log(np.sum(np.exp(-alpha * (an - mn)))) / alpha)


def _make_constraints():
    cons = []
    for i in range(N):
        def f(x, i=i):
            return 1.0 - x[2 * i] - x[2 * i + 1]
        cons.append({'type': 'ineq', 'fun': f})
    return cons


def construct():
    deadline = time.time() + 140.0
    bounds = [(0.0, 1.0)] * (2 * N)
    best_points = None
    best_score = 0.0

    de_res = differential_evolution(
        _obj_de,
        bounds=bounds,
        seed=42,
        maxiter=150,
        popsize=15,
        tol=1e-8,
        polish=False,
        workers=1,
        init='sobol'
    )

    if hasattr(de_res, 'x') and de_res.x is not None:
        best_params = de_res.x.reshape(N, 2)
        best_points = _bary_to_cartesian(best_params)
        best_score = -de_res.fun

    if best_points is None:
        rng = np.random.RandomState(123)
        best_params = rng.uniform(0.05, 0.45, (N, 2))
        best_points = _bary_to_cartesian(best_params)
        best_score = _min_area_norm(best_params)

    if time.time() < deadline - 10.0:
        alpha = 50.0
        cons = _make_constraints()
        res2 = minimize(
            lambda x: _obj_soft(x, alpha),
            best_params.ravel(),
            method='SLSQP',
            bounds=bounds,
            constraints=cons,
            options={'maxiter': 200, 'ftol': 1e-10}
        )
        if res2.success and -res2.fun > best_score:
            best_params = res2.x.reshape(N, 2)
            best_points = _bary_to_cartesian(best_params)
            best_score = -res2.fun

    if time.time() < deadline - 3.0:
        cons = _make_constraints()
        res3 = minimize(
            _obj_de,
            best_params.ravel(),
            method='SLSQP',
            bounds=bounds,
            constraints=cons,
            options={'maxiter': 300, 'ftol': 1e-12}
        )
        if res3.success and -res3.fun > best_score:
            best_params = res3.x.reshape(N, 2)
            best_points = _bary_to_cartesian(best_params)
            best_score = -res3.fun

    if best_points is None:
        rng = np.random.RandomState(1)
        best_params = rng.uniform(0.05, 0.45, (N, 2))
        best_points = _bary_to_cartesian(best_params)

    x = best_points[:, 0]
    y = best_points[:, 1]
    x = np.clip(x, 0.001, 0.999)
    y = np.clip(y, 0.001, _SQRT3 / 2.0 - 0.001)
    best_points = np.column_stack([x, y])

    return best_points

# EVOLVE-BLOCK-END
