"""Minimize the C4 uncertainty bound via gradient-based search + high-precision polish."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize, fsolve

_DEADLINE = time.time() + 170.0  # safety margin under the 210 s wall-clock kill

# Physicists' Hermite polynomials H_0, H_4, H_8, H_12
_H0 = np.array([0.0, 0.0, 0.0, 0.0, 1.0])
_H4 = np.array([12.0, 0.0, -48.0, 0.0, 16.0])
_H8 = np.array([168.0, 0.0, -1344.0, 0.0, 2688.0, 0.0, -1280.0, 0.0, 256.0])
_H12 = np.array([3024.0, 0.0, -30240.0, 0.0, 120960.0, 0.0, -184320.0, 0.0, 129024.0, 0.0, -38400.0, 0.0, 4096.0])
_POW2 = np.array([2.0 ** (k / 2.0) for k in range(13)])


def _moments(a):
    """Return the moment array m_k = <g, H_k g> for the Hermite combination."""
    c = np.zeros(13)
    c[0] = a[0]
    c[4] = a[1]
    c[8] = a[2]
    c[12] = -(a[0] + 12.0 * a[1] + 168.0 * a[2]) / 3024.0  # enforce g(0) = 0
    return c * _POW2 * c


def _bound(a):
    """Upper bound on A(g)*A(ĝ) = C4 * (sqrt(1+mu6) - 1) / (mu4 - mu2^2)."""
    m = _moments(a)
    mu2 = m[2]
    mu4 = m[4]
    mu6 = m[6]
    den = mu4 - mu2 * mu2
    if den <= 1e-30:
        return 1e300
    num = np.sqrt(1.0 + mu6) - 1.0
    if num < 0.0:
        return 1e300
    return 4.0 * num / den


def _logsign_to_lin(z):
    """z -> (sign, log|.|) -> linear value."""
    s = np.tanh(z[0:3])
    logm = z[3:6]
    return s * np.exp(logm)


def _neg_bound_lin(a):
    return -_bound(a)


def _grad_neg_bound(z):
    a = _logsign_to_lin(z)
    m = _moments(a)
    mu2 = m[2]
    mu4 = m[4]
    mu6 = m[6]
    den = mu4 - mu2 * mu2
    if den <= 1e-30:
        return np.zeros(6)
    num = np.sqrt(1.0 + mu6) - 1.0
    if num < 0.0:
        return np.zeros(6)
    b = 4.0 * num / den
    # db/da = 4 * [ (0.5/(sqrt(1+mu6))) * dmu6/dai * den - num * dden/dai ] / den^2
    dmu = np.zeros(6)
    dmu[0] = 2.0 * (m[0] * a[0] + m[2] * a[1] + m[6] * a[2] + m[10] * (-(a[0] + 12.0 * a[1] + 168.0 * a[2]) / 3024.0))
    dmu[2] = 2.0 * (m[2] * a[0] + m[4] * a[1] + m[8] * a[2] + m[12] * (-(a[0] + 12.0 * a[1] + 168.0 * a[2]) / 3024.0))
    dmu[4] = 2.0 * (m[4] * a[0] + m[6] * a[1] + m[10] * a[2] + m[14] * 0.0)  # m[14] unused
    dmu[6] = 2.0 * (m[6] * a[0] + m[8] * a[1] + m[12] * a[2] + m[16] * 0.0)
    dden = dmu[4] - 2.0 * mu2 * dmu[2]
    dsq = 0.5 / np.sqrt(1.0 + mu6)
    db = 4.0 * (dsq * dmu[6] * den - num * dden) / (den * den)
    # chain rule through log/sign parameterization
    s = np.tanh(z[0:3])
    logm = z[3:6]
    lin = s * np.exp(logm)
    grad_z = np.zeros(6)
    for i in range(3):
        # d lin_i / d z_i (sign part) = (1 - s^2) * exp(logm_i)
        # d lin_i / d z_{3+i} (log part) = s * exp(logm_i)
        grad_z[i] = db[i] * (1.0 - s[i] * s[i]) * np.exp(logm[i])
        grad_z[3 + i] = db[i] * s[i] * np.exp(logm[i])
    return -grad_z


def _refine_lin(a0):
    """High-precision polish: solve gradient = 0 in linear space."""
    def gfun(a):
        m = _moments(a)
        mu2 = m[2]
        mu4 = m[4]
        mu6 = m[6]
        den = mu4 - mu2 * mu2
        if den <= 1e-30:
            return [1e6, 1e6, 1e6]
        num = np.sqrt(1.0 + mu6) - 1.0
        if num < 0.0:
            return [1e6, 1e6, 1e6]
        dsq = 0.5 / np.sqrt(1.0 + mu6)
        db = np.zeros(3)
        for i in range(3):
            dmu6 = 2.0 * (m[6] * a[0] + m[8] * a[1] + m[12] * a[2] + m[16] * 0.0) if i == 0 else \
                  2.0 * (m[6] * a[0] + m[8] * a[1] + m[12] * a[2] + m[16] * 0.0) if i == 1 else \
                  2.0 * (m[6] * a[0] + m[8] * a[1] + m[12] * a[2] + m[16] * 0.0)
        # Recompute per-component derivatives properly
        def dmu_k(k):
            # d m_k / d a_j
            out = np.zeros(3)
            out[0] = 2.0 * (m[k - 0] * a[0] + m[k - 4] * a[1] + m[k - 8] * a[2] + m[k - 12] * (-(a[0] + 12.0 * a[1] + 168.0 * a[2]) / 3024.0)) if k >= 12 else \
                    2.0 * (m[k] * a[0] + (m[k + 4] * a[1] if k + 4 <= 12 else 0.0) + (m[k + 8] * a[2] if k + 8 <= 12 else 0.0) + (m[k + 12] * (-(a[0] + 12.0 * a[1] + 168.0 * a[2]) / 3024.0) if k + 12 <= 12 else 0.0))
            return out
        dmu2 = dmu_k(2)
        dmu4 = dmu_k(4)
        dmu6 = dmu_k(6)
        dden = dmu4 - 2.0 * mu2 * dmu2
        db = 4.0 * (dsq * dmu6 * den - num * dden) / (den * den)
        return db
    sol, info, ier, msg = fsolve(gfun, a0, full_output=True, xtol=1e-13, maxfev=400)
    if ier == 1 and _bound(sol) < _bound(a0):
        return sol
    return a0


def construct():
    global _DEADLINE
    best_a = np.array([-1.13, 0.04, 1.0 / 3240.0])
    best_val = _bound(best_a)

    # Multi-start in log/sign space
    starts = []
    base = np.array([-1.13, 0.04, 1.0 / 3240.0])
    # seed point
    starts.append(base.copy())
    # perturbations
    rng = np.random.RandomState(12345)
    for _ in range(40):
        p = base * np.exp(rng.normal(0, 1.0, 3))
        p[1] = base[1] * np.exp(rng.normal(0, 1.5, 1)[0])
        starts.append(p)
    # grid of scales
    for s0 in [-3.0, -2.0, -1.5, -1.13, -0.8, -0.5, -0.3]:
        for s4 in [0.01, 0.04, 0.1, 0.2, 0.5]:
            for s8 in [0.0, 0.0003, 0.001, 0.01, 0.05]:
                starts.append(np.array([s0, s4, s8]))

    for a0 in starts:
        if time.time() > _DEADLINE - 10.0:
            break
        # Convert to log/sign space
        signs = np.sign(a0)
        signs[signs == 0] = 1.0
        logm = np.log(np.abs(a0) + 1e-12)
        z0 = np.concatenate([np.arctanh(np.clip(signs, -0.999, 0.999)), logm])
        try:
            res = minimize(_neg_bound_lin, z0, jac=_grad_neg_bound, method='L-BFGS-B',
                           options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-12})
            a_lin = _logsign_to_lin(res.x)
            val = _bound(a_lin)
            if val < best_val:
                best_val = val
                best_a = a_lin
        except Exception:
            pass

    # High-precision polish on the best
    if time.time() < _DEADLINE - 5.0:
        try:
            a_pol = _refine_lin(best_a)
            val_pol = _bound(a_pol)
            if val_pol < best_val:
                best_val = val_pol
                best_a = a_pol
        except Exception:
            pass

    return np.array([best_a[0], best_a[1], best_a[2]])
# EVOLVE-BLOCK-END
