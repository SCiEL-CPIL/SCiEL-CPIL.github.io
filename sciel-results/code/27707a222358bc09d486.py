# EVOLVE-BLOCK-START
import numpy as np
from scipy.special import lpmv
from scipy.optimize import minimize

_K = 10


def _build_quadrature():
    """Gauss-Legendre (cos theta) x equally spaced phi, matching the evaluator."""
    n_theta = 70
    n_phi = 130
    x, w = np.polynomial.legendre.leggauss(n_theta)          # x in (-1,1), sum w = 2
    phi = np.linspace(0.0, 2.0 * np.pi, n_phi, endpoint=False)
    dphi = 2.0 * np.pi / n_phi
    return x, w, phi, dphi


def _norms(m):
    m = np.asarray(m)
    return np.sqrt(21.0 / (4.0 * np.pi) * np.exp(
        np.lgamma(_K - m + 1.0) - np.lgamma(_K + m + 1.0)))


def _build_harmonics(x, w, phi, dphi):
    """Return (F, W) with F[i, m+K] = Y_m at grid point i, W[i] = quadrature weight."""
    n = len(x) * len(phi)
    m_idx = np.arange(-_K, _K + 1)
    F = np.empty((n, 2 * _K + 1), dtype=np.float64)
    m = np.abs(m_idx)
    N = _norms(m)
    P = lpmv(m, _K, x)[:, None] * N[None, :]
    mgrid = m_idx[None, :]
    cosm = np.cos(mgrid * phi[:, None])
    sinm = np.sin(np.abs(mgrid) * phi[:, None])
    Y = np.where(mgrid > 0, np.sqrt(2.0) * P * cosm,
                 np.where(mgrid < 0, np.sqrt(2.0) * P * sinm, P))
    W = np.repeat(w * dphi, len(phi))
    F[:] = Y
    return F, W


def _ratio2(c, F, W):
    """Return (R^2, f) where R = ||f||_4/||f||_2, computed via quadrature."""
    f = F @ c
    w2 = W * f * f
    n2 = float(np.dot(w2, f * f))
    if n2 <= 0.0:
        return 1e30, f
    n4 = float(np.dot(w2, f * f))
    return n4 / n2, f


def _tr_sphere(c, F, W, iters=140, delta0=1.0):
    """Trust-region method on the unit sphere (retraction + quadratic model)."""
    n = len(c)
    c = np.asarray(c, dtype=np.float64)
    c = c / np.linalg.norm(c)
    delta = delta0
    r2, f = _ratio2(c, F, W)
    best = (r2, c.copy())
    for _ in range(iters):
        g = F.T @ (W * (f * f * f))
        g = g - (g @ c) * c                      # tangential gradient of R^2
        gn = np.linalg.norm(g)
        if gn < 1e-12:
            break
        d = -g / gn
        # quadratic model on the tangent plane:  q(s) = <g,d> s + 0.5 s^2
        s = max(-1.0, min(1.0, -float(g @ d)))
        trial = c + s * d
        trial = trial / np.linalg.norm(trial)     # retraction to sphere
        tr2, tf = _ratio2(trial, F, W)
        if tr2 < best[0]:
            best = (tr2, trial.copy())
        rho = (r2 - tr2) / (abs(float(g @ d)) * abs(s) + 1e-30)
        if rho > 0.75:
            delta = min(1.0, 2.0 * delta)
        elif rho < 0.12:
            delta = 0.5 * delta
        else:
            delta = 0.75 * delta
        r2, f = tr2, tf
        c = trial
        if delta < 1e-5 and gn < 1e-9:
            break
    return best[1]


def _refine(c, F, W):
    """Polish a unit vector with L-BFGS-B then a trust-region pass."""
    c = np.asarray(c, dtype=np.float64)
    c = c / np.linalg.norm(c)
    res = minimize(lambda x: _ratio2(x, F, W)[0], c, method='L-BFGS-B',
                   options={'maxiter': 400, 'ftol': 1e-16, 'gtol': 1e-10})
    c = res.x
    c = c / np.linalg.norm(c)
    c = _tr_sphere(c, F, W)
    return c


def rudin_problem10() -> np.ndarray:
    x, w, phi, dphi = _build_quadrature()
    F, W = _build_harmonics(x, w, phi, dphi)

    rng = np.random.default_rng(20240517)
    candidates = []
    # deterministic structured starts
    for a in np.linspace(-1.0, 1.0, 9):
        c = np.cos(np.arange(-_K, _K + 1) * a)
        candidates.append(c)
    for a in np.linspace(-1.0, 1.0, 9):
        c = np.sin(np.arange(-_K, _K + 1) * a)
        candidates.append(c)
    for a in np.linspace(0.3, 1.3, 6):
        c = np.exp(-a * np.arange(-_K, _K + 1) ** 2 / (2.0 * _K))
        candidates.append(c)
    # random starts
    for _ in range(40):
        candidates.append(rng.standard_normal(2 * _K + 1))

    best = None
    for c0 in candidates:
        c0 = np.asarray(c0, dtype=np.float64)
        if np.linalg.norm(c0) < 1e-12:
            continue
        c0 = c0 / np.linalg.norm(c0)
        c = _refine(c0, F, W)
        r2, _ = _ratio2(c, F, W)
        if best is None or r2 < best[0]:
            best = (r2, c)

    c = best[1]
    c = c / np.linalg.norm(c)
    # scale into the valid range, well away from the 1e-3 / 1e5 bounds
    c = c * 1e-1
    c = np.nan_to_num(c, nan=0.0, posinf=0.0, neginf=0.0)
    return c


# EVOLVE-BLOCK-END
