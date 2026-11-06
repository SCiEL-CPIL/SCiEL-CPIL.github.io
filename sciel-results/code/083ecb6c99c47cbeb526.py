"""Optimize 50 equal circles in the unit disk via smooth soft-min gradient ascent."""
import time
import math
import numpy as np
from scipy.optimize import minimize

N = 50
_DEADLINE = time.time() + 130.0  # safety margin before the 150 s kill
_RNG = np.random.default_rng(12345)


def _min_pair(p):
    d = p[:, None, :] - p[None, :, :]
    dd = d * d
    np.fill_diagonal(dd, 1e18)
    return np.sqrt(dd.min(axis=2))


def _wall(p):
    return 1.0 - np.hypot(p[:, 0], p[:, 1])


def _score(p):
    return min(_min_pair(p).min(), _wall(p).min())


def _softmin_grad(p, beta):
    """Gradient of the soft-min of all constraints (pair distances + walls)."""
    n = p.shape[0]
    d = p[:, None, :] - p[None, :, :]
    dist = np.sqrt(np.einsum('ijk,ijk->ij', d, d) + 1e-15)
    np.fill_diagonal(dist, 1e18)
    inv = 1.0 / dist
    w = np.exp(-beta * dist)
    w *= inv
    np.fill_diagonal(w, 0.0)
    g = (2.0 * (w @ d)).sum(axis=0)
    r = np.hypot(p[:, 0], p[:, 1])
    r = np.where(r < 1e-9, 1e-9, r)
    wv = np.exp(-beta * (1.0 - r))
    g += wv * (p / r)[:, None]
    s = w.sum() + wv.sum()
    return g / s


def _softmin_val(p, beta):
    d = p[:, None, :] - p[None, :, :]
    dist = np.sqrt(np.einsum('ijk,ijk->ij', d, d) + 1e-15)
    np.fill_diagonal(dist, 1e18)
    r = np.hypot(p[:, 0], p[:, 1])
    wall = 1.0 - r
    m = np.min(np.concatenate([dist, wall]))
    a = np.exp(-beta * (dist - m))
    b = np.exp(-beta * (wall - m))
    return m + np.log(a.sum() + b.sum()) / beta


def _lbfgs(x, beta, maxiter, ftol, gtol, time_budget):
    t0 = time.time()
    def obj(x):
        p = x.reshape(N, 2)
        return -_softmin_val(p, beta)
    def grad(x):
        p = x.reshape(N, 2)
        return _softmin_grad(p, beta).reshape(-1)
    def cb(xk):
        return (time.time() - t0) > time_budget
    res = minimize(obj, x, jac=grad, method='L-BFGS-B',
                   options={'maxiter': maxiter, 'ftol': ftol,
                            'gtol': gtol, 'maxfun': 10000, 'callback': cb})
    return res.x.reshape(N, 2)


def _try_init(p, beta, maxiter, ftol, gtol, time_budget):
    p = _lbfgs(p.reshape(-1), beta, maxiter, ftol, gtol, time_budget)
    return p


def _polish(p):
    """High-precision polish: anneal beta to infinity (exact min) with tight tol."""
    t0 = time.time()
    best = p
    best_s = _score(p)
    for beta in [50.0, 120.0, 300.0, 700.0]:
        if time.time() - t0 > 12.0:
            break
        p = _lbfgs(p.reshape(-1), beta, 3000, 1e-15, 1e-12, 12.0)
        s = _score(p)
        if s > best_s + 1e-12:
            best = p
            best_s = s
    return best


def _ring_init(rng, n, radius, jitter=0.0):
    p = np.zeros((n, 2))
    ang = (np.arange(n) + 0.5) * 2.0 * math.pi / n
    p[:, 0] = radius * np.cos(ang)
    p[:, 1] = radius * np.sin(ang)
    if jitter > 0.0:
        p += rng.normal(0.0, jitter, p.shape)
    return p


def _hex_init(rng):
    pts = []
    for r in [0.0, 0.2, 0.4, 0.6, 0.8]:
        cnt = int(2 * math.pi * r / 0.22) if r > 0 else 1
        if r == 0:
            pts.append([0.0, 0.0])
            continue
        for i in range(cnt):
            a = (i + 0.5) * 2.0 * math.pi / cnt
            pts.append([r * math.cos(a), r * math.sin(a)])
    pts = np.array(pts[:N])
    return pts + rng.normal(0.0, 0.02, pts.shape)


def construct():
    t0 = time.time()
    best = None
    best_s = -1.0
    # diverse initializations
    inits = []
    for r in [0.35, 0.5, 0.62, 0.72, 0.8]:
        inits.append(_ring_init(_RNG, N, r, jitter=0.03))
    inits.append(_hex_init(_RNG))
    # add a couple random
    for _ in range(2):
        a = _RNG.uniform(0, 2 * math.pi, N)
        rr = _RNG.uniform(0, 0.8, N)
        inits.append(np.stack([rr * np.cos(a), rr * np.sin(a)], axis=1))

    for it, p0 in enumerate(inits):
        if time.time() > _DEADLINE - 25.0:
            break
        # coarse to fine beta
        p = p0
        for beta, mi, ft, gt in [(20.0, 1500, 1e-8, 1e-6),
                                  (60.0, 2000, 1e-10, 1e-8),
                                  (150.0, 2500, 1e-12, 1e-10)]:
            p = _try_init(p, beta, mi, ft, gt, 15.0)
            if time.time() > _DEADLINE - 25.0:
                break
        s = _score(p)
        if s > best_s:
            best_s = s
            best = p

    if best is None:
        best = inits[0]
    # final polish on the best
    if time.time() < _DEADLINE - 8.0:
        best = _polish(best)
        best_s = _score(best)

    return best
