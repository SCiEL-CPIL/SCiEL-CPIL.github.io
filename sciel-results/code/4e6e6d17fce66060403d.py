import numpy as np
from scipy.optimize import minimize
import time
import math

_START_TIME = time.time()
_DEADLINE = _START_TIME + 205

# Physicists' Hermite polynomials: H_n(x) = (-1)^n e^{x^2} d^n/dx^n e^{-x^2}
# H_0 = 1
# H_4 = 16x^4 - 48x^2 + 12
# H_8 = 256x^8 - 3072x^6 + 13440x^4 - 13440x^2 + 1680
# H_12 = 4096x^12 - 98304x^10 + 1290240x^8 - 6451200x^6 + 12902400x^4 - 6451200x^2 + 665280
# Coefficients for x^(2k) in each Hermite polynomial, k=0..6
_H_COEFFS = np.array([
    [1.0, 0, 0, 0, 0, 0, 0],
    [12.0, -48.0, 16.0, 0, 0, 0, 0],
    [1680.0, -13440.0, 13440.0, -3072.0, 256.0, 0, 0],
    [665280.0, -6451200.0, 12902400.0, -6451200.0, 1290240.0, -98304.0, 4096.0]
])

# H_n(0) values for constraint g(0) = 0
_H_AT_ZERO = np.array([1.0, 12.0, 1680.0, 665280.0])

# Precompute I_n = ∫ x^(2n) e^{-x^2} dx = (2n)!/(4^n n!) * sqrt(pi) for n=0..14
def _compute_I_table():
    I = np.zeros(15)
    for n in range(15):
        if n == 0:
            I[n] = np.sqrt(np.pi)
        else:
            log_num = sum(math.log(k) for k in range(1, 2*n+1))
            log_den = n * math.log(4) + sum(math.log(k) for k in range(1, n+1))
            I[n] = math.exp(log_num - log_den) * np.sqrt(np.pi)
    return I

_I_TABLE = _compute_I_table()

# Matrix M where M[k, i, j] = I_{k+i+j} for k=0,1; i,j=0..6
_M_SPATIAL = np.zeros((2, 7, 7))
_M_FREQ = np.zeros((2, 7, 7))
for k in range(2):
    for i in range(7):
        for j in range(7):
            idx = k + i + j
            if idx < 15:
                _M_SPATIAL[k, i, j] = _I_TABLE[idx]
                _M_FREQ[k, i, j] = 2.0 * np.pi * _I_TABLE[idx]

_SIGNS = np.array([(-1.0)**k for k in range(7)])


def _compute_bound_and_grad(params):
    """
    Compute the uncertainty bound sqrt(<x^2><xi^2>) and its analytic gradient.
    g(x) = (a0*H0 + a4*H4 + a8*H8 + a12*H12) * exp(-x^2/2)
    a12 is determined by g(0)=0: a12 = -(a0 + 12*a4 + 1680*a8) / 665280
    """
    a0, a4, a8 = params[0], params[1], params[2]
    a12 = -(a0 + 12.0*a4 + 1680.0*a8) / 665280.0
    coeffs = np.array([a0, a4, a8, a12])
    p = _H_COEFFS @ coeffs
    m = np.array([p @ _M_SPATIAL[k] @ p for k in range(2)])
    m0, m2 = m[0], m[1]
    q = p * _SIGNS
    n = np.array([q @ _M_FREQ[k] @ q for k in range(2)])
    n0, n2 = n[0], n[1]

    if m0 <= 0 or n0 <= 0 or not np.isfinite(m0) or not np.isfinite(n0):
        return 1e18, np.array([1e18, 1e18, 1e18])
    vf = m2 / m0
    vF = n2 / n0
    if vf <= 0 or vF <= 0:
        return 1e18, np.array([1e18, 1e18, 1e18])
    bound = np.sqrt(vf * vF)

    da12_vec = -_H_AT_ZERO[:3] / _H_AT_ZERO[3]
    grad = np.zeros(3)
    for c_idx in range(3):
        dp = _H_COEFFS[c_idx] + da12_vec[c_idx] * _H_COEFFS[3]
        dm = 2.0 * np.array([dp @ _M_SPATIAL[k] @ p for k in range(2)])
        dq = dp * _SIGNS
        dn = 2.0 * np.array([dq @ _M_FREQ[k] @ q for k in range(2)])
        dvm2 = (dm[1]*m0 - m2*dm[0]) / (m0*m0)
        dvn2 = (dn[1]*n0 - n2*dn[0]) / (n0*n0)
        grad[c_idx] = (vF * dvm2 + vf * dvn2) / (2.0 * bound)
    return bound, grad


def _compute_log_obj_and_grad(params):
    """Compute log(vf) + log(vF) and its gradient."""
    a0, a4, a8 = params[0], params[1], params[2]
    a12 = -(a0 + 12.0*a4 + 1680.0*a8) / 665280.0
    coeffs = np.array([a0, a4, a8, a12])
    p = _H_COEFFS @ coeffs
    m = np.array([p @ _M_SPATIAL[k] @ p for k in range(2)])
    m0, m2 = m[0], m[1]
    q = p * _SIGNS
    n = np.array([q @ _M_FREQ[k] @ q for k in range(2)])
    n0, n2 = n[0], n[1]

    if m0 <= 0 or n0 <= 0 or not np.isfinite(m0) or not np.isfinite(n0):
        return 1e18, np.array([1e18, 1e18, 1e18])
    if m2 <= 0 or n2 <= 0:
        return 1e18, np.array([1e18, 1e18, 1e18])
    vf = m2 / m0
    vF = n2 / n0
    if vf <= 0 or vF <= 0:
        return 1e18, np.array([1e18, 1e18, 1e18])

    log_obj = math.log(vf) + math.log(vF)
    da12_vec = -_H_AT_ZERO[:3] / _H_AT_ZERO[3]
    grad = np.zeros(3)
    for c_idx in range(3):
        dp = _H_COEFFS[c_idx] + da12_vec[c_idx] * _H_COEFFS[3]
        dm = 2.0 * np.array([dp @ _M_SPATIAL[k] @ p for k in range(2)])
        dq = dp * _SIGNS
        dn = 2.0 * np.array([dq @ _M_FREQ[k] @ q for k in range(2)])
        dlog_vf = dm[1]/m2 - dm[0]/m0
        dlog_vF = dn[1]/n2 - dn[0]/n0
        grad[c_idx] = dlog_vf + dlog_vF
    return log_obj, grad


def _newton_polish(params, max_iter=200, tol=1e-16):
    """High-precision Newton's method using the Hessian of the log-objective."""
    x = params.copy().astype(float)
    for it in range(max_iter):
        if time.time() > _DEADLINE - 5:
            break
        f, g = _compute_log_obj_and_grad(x)
        if not np.isfinite(f):
            break
        if np.max(np.abs(g)) < tol:
            break

        h = 1e-8
        H = np.zeros((3, 3))
        for i in range(3):
            for j in range(3):
                ei = np.zeros(3); ei[i] = h
                ej = np.zeros(3); ej[j] = h
                _, gp = _compute_log_obj_and_grad(x + ei + ej)
                _, gm1 = _compute_log_obj_and_grad(x - ei + ej)
                _, gm2 = _compute_log_obj_and_grad(x + ei - ej)
                _, gm3 = _compute_log_obj_and_grad(x - ei - ej)
                H[i, j] = (gp[i] - gm1[i] - gm2[i] + gm3[i]) / (4.0 * h * h)

        try:
            dx = np.linalg.solve(H, -g)
        except np.linalg.LinAlgError:
            break

        alpha = 1.0
        f_new = f
        for _ in range(60):
            x_trial = x + alpha * dx
            f_trial, _ = _compute_log_obj_and_grad(x_trial)
            if np.isfinite(f_trial) and f_trial < f_new:
                x = x_trial
                f_new = f_trial
                break
            alpha *= 0.5
        else:
            break

        if np.max(np.abs(dx)) < tol:
            break
    return x


def construct():
    """
    Multi-start L-BFGS-B on log-objective + Newton polish.
    Returns three coefficients (a0, a4, a8) for H0, H4, H8.
    """
    best_val = np.inf
    best_params = None

    def _try_optimize(start, use_log=True, maxiter=5000):
        nonlocal best_val, best_params
        try:
            if use_log:
                obj_func = lambda p: _compute_log_obj_and_grad(p)
            else:
                obj_func = lambda p: _compute_bound_and_grad(p)
            res = minimize(obj_func, start, method='L-BFGS-B',
                          jac=True, options={'maxiter': maxiter, 'ftol': 1e-16, 'gtol': 1e-14})
            actual_val, _ = _compute_bound_and_grad(res.x)
            if np.isfinite(actual_val) and actual_val < best_val:
                best_val = actual_val
                best_params = res.x.copy()
        except Exception:
            pass

    # Well-chosen starting points
    starts = [
        np.array([-1.13, 0.04, 0.000308642]),   # seed
        np.array([-1.0, 0.05, 0.0004]),
        np.array([-1.5, 0.03, 0.0002]),
        np.array([-0.5, 0.02, 0.0001]),
        np.array([-2.0, 0.06, 0.0005]),
        np.array([0.0, 0.0, 0.0]),
        np.array([-1.2, 0.035, 0.00025]),
        np.array([-0.8, 0.045, 0.00035]),
        np.array([-1.4, 0.045, 0.00045]),
        np.array([-0.6, 0.03, 0.0002]),
    ]
    rng = np.random.default_rng(42)
    for scale in [0.3, 0.8, 1.5]:
        for _ in range(5):
            starts.append(rng.normal(0, scale, 3))

    # Phase 1: L-BFGS-B from each start
    for start in starts:
        if time.time() > _DEADLINE - 60:
            break
        _try_optimize(start, use_log=True, maxiter=3000)

    # Phase 2: Newton polish on best
    if best_params is not None and time.time() < _DEADLINE - 20:
        polished = _newton_polish(best_params, max_iter=100, tol=1e-15)
        actual_val, _ = _compute_bound_and_grad(polished)
        if np.isfinite(actual_val) and actual_val < best_val:
            best_val = actual_val
            best_params = polished.copy()

    # Phase 3: Fine L-BFGS-B around best with log objective
    if best_params is not None and time.time() < _DEADLINE - 15:
        _try_optimize(best_params, use_log=True, maxiter=10000)

    # Phase 4: Final Newton polish
    if best_params is not None and time.time() < _DEADLINE - 5:
        polished = _newton_polish(best_params, max_iter=200, tol=1e-16)
        actual_val, _ = _compute_bound_and_grad(polished)
        if np.isfinite(actual_val) and actual_val < best_val:
            best_val = actual_val
            best_params = polished.copy()

    if best_params is None:
        return np.array([-1.13, 0.04, 0.000308642])
    return best_params
