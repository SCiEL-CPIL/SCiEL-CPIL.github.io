"""Covering 50 points on the unit sphere: minimize the covering radius.

Fresh design:
  - Icosahedral symmetric seed (12 vertices + 38 face-centers).
  - Optimize the covering radius (max over sphere of distance to nearest
    point) with a smooth 'softmax' surrogate via L-BFGS-B.
  - Fast nearest-point evaluation on a fixed fine spherical grid.
  - Hard wall-clock deadline guard.
"""
import time

import numpy as np
from scipy.optimize import minimize

N = 50
T_LIMIT = 120.0
T_START = time.time()


def _time_left():
    return T_LIMIT - (time.time() - T_START)


def _norm_rows(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n = np.where(n > 1e-15, n, 1.0)
    return X / n


def _nearest_index(P, G):
    # squared distance from each grid point to each sphere point
    d2 = (G @ G.T) + (P @ P.T).T - 2.0 * (G @ P.T)
    return np.argmin(d2, axis=1)


def _covering_radius(P, G):
    idx = _nearest_index(P, G)
    v = np.take_along_axis(P, idx[:, None], axis=0)
    cosang = np.einsum('ij,ij->i', G, v)
    cosang = np.clip(cosang, -1.0, 1.0)
    return np.arccos(cosang).max()


def _icosahedral_seed():
    t = (1.0 + np.sqrt(5.0)) / 2.0
    base = np.array([
        [-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0],
        [0, -1, t], [0, 1, t], [0, -1, -t], [0, 1, -t],
        [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1],
    ], dtype=float)
    base = _norm_rows(base)
    pts = [base]
    # 20 faces of the icosahedron
    faces = [
        (0, 11, 5), (0, 5, 1), (0, 1, 7), (0, 7, 10), (0, 10, 11),
        (1, 5, 9), (5, 11, 4), (11, 10, 2), (10, 7, 6), (7, 1, 8),
        (3, 9, 4), (3, 4, 2), (3, 2, 6), (3, 6, 8), (3, 8, 9),
        (4, 9, 5), (2, 4, 11), (6, 2, 10), (8, 6, 7), (9, 8, 1),
    ]
    for (a, b, c) in faces:
        pts.append(base[a])
        pts.append(base[b])
        pts.append(base[c])
    P = np.array(pts, dtype=float)
    assert P.shape == (N, 3)
    return _norm_rows(P)


def _objective_and_grad(x, G, beta):
    P = _norm_rows(x.reshape(N, 3))
    idx = _nearest_index(P, G)
    v = np.take_along_axis(P, idx[:, None], axis=0)
    cosang = np.einsum('ij,ij->i', G, v)
    cosang = np.clip(cosang, -1.0, 1.0)
    u = beta * cosang
    m = u.max()
    e = np.exp(u - m)
    s = e.sum()
    w = e / s
    # objective
    val = -(m + np.log(s)) / beta
    # gradient (chain rule through softmax, arccos, and the normalization)
    grad = np.zeros_like(x)
    for i in range(N):
        wi = w[i]
        if wi <= 0.0:
            continue
        g = -wi / (beta * np.sqrt(max(1.0 - cosang[i] ** 2, 1e-14)))
        p = P[i]
        grad[3 * i:3 * i + 3] += g * (G[i] - np.dot(G[i], p) * p)
    return val, grad


def construct():
    # deterministic fine grid (sufficiently dense to resolve ~18 deg radius)
    nlat = 120
    nlon = 240
    theta = np.linspace(0.0, np.pi, nlat)
    phi = np.linspace(0.0, 2.0 * np.pi, nlon, endpoint=False)
    TH, PH = np.meshgrid(theta, phi, indexing='ij')
    G = np.stack([
        np.sin(TH) * np.cos(PH),
        np.sin(TH) * np.sin(PH),
        np.cos(TH),
    ], axis=-1).reshape(-1, 3)
    G = _norm_rows(G)

    P0 = _icosahedral_seed()
    best = P0.copy()
    best_cr = _covering_radius(best, G)

    if _time_left() < 15.0:
        return best

    # smoothness of the soft-max surrogate (in radians)
    beta = 30.0
    x0 = best.reshape(-1)
    try:
        res = minimize(
            _objective_and_grad,
            x0,
            args=(G, beta),
            method='L-BFGS-B',
            jac=True,
            options={'maxiter': 200, 'maxfun': 2000, 'gtol': 1e-12,
                     'disp': False, 'maxls': 40},
        )
        if _time_left() > 5.0 and res.x is not None:
            P1 = _norm_rows(res.x.reshape(N, 3))
            if np.all(np.isfinite(P1)) and np.linalg.norm(P1, axis=1).min() > 1e-9:
                cr1 = _covering_radius(P1, G)
                if cr1 < best_cr:
                    best, best_cr = P1, cr1
    except Exception:
        pass

    if _time_left() < 15.0:
        return best

    # refine the smoothness and polish
    for beta in (60.0, 120.0, 250.0):
        if _time_left() < 15.0:
            break
        try:
            res = minimize(
                _objective_and_grad,
                best.reshape(-1),
                args=(G, beta),
                method='L-BFGS-B',
                jac=True,
                options={'maxiter': 120, 'maxfun': 1200, 'gtol': 1e-12,
                         'disp': False, 'maxls': 30},
            )
            if _time_left() > 5.0 and res.x is not None:
                P1 = _norm_rows(res.x.reshape(N, 3))
                if np.all(np.isfinite(P1)) and np.linalg.norm(P1, axis=1).min() > 1e-9:
                    cr1 = _covering_radius(P1, G)
                    if cr1 < best_cr:
                        best, best_cr = P1, cr1
        except Exception:
            break

    return best
