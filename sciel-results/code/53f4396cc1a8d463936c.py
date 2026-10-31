"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

_START_TIME = time.time()
_DEADLINE = _START_TIME + 200.0


def _I_table():
    """Precompute I[n] = integral x^{2n} e^{-x^2} dx for n=0..13."""
    I = np.zeros(14)
    for n in range(14):
        # (2n)! / (4^n n!) * sqrt(pi)
        if n == 0:
            I[n] = np.sqrt(np.pi)
        else:
            # Use recurrence: I[n] = I[n-1] * (2n-1) / (4n)
            I[n] = I[n-1] * (2*n - 1) / (4*n)
    return I


_I = _I_table()


def _bound_and_grad(params):
    """
    Compute uncertainty product sqrt(vf*vF) and its analytic gradient.
    g(x) = (a0*H0 + a4*H4 + a8*H8 + a12*H12) * exp(-x^2/2)
    a12 = -(a0 + 3*a4 + 105*a8)/945  (enforces g(0)=0)
    """
    a0, a4, a8 = params[0], params[1], params[2]
    a12 = -(a0 + 3.0*a4 + 105.0*a8) / 945.0

    # Polynomial P(x) = sum_{k=0}^{6} p_k x^{2k}
    p = np.zeros(7)
    p[0] = a0 + 3.0*a4 + 105.0*a8 + 945.0*a12
    p[1] = -6.0*a4 - 420.0*a8 - 7560.0*a12
    p[2] = a4 + 210.0*a8 + 13860.0*a12
    p[3] = -28.0*a8 - 6930.0*a12
    p[4] = a8 + 1155.0*a12
    p[5] = -66.0*a12
    p[6] = a12

    # Moments m_k = integral x^{2k} |P|^2 e^{-x^2} dx for k=0,1,2
    # m_k = sum_{i,j=0}^{6} p_i p_j I[k+i+j]
    m = np.zeros(3)
    for i in range(7):
        pi = p[i]
        if pi == 0.0:
            continue
        for j in range(7):
            pij = pi * p[j]
            if pij == 0.0:
                continue
            for k in range(3):
                idx = k + i + j
                if idx < 14:
                    m[k] += pij * _I[idx]

    # Fourier side: Q(x) = P(-ix) => q_k = p_k * (-1)^k
    q = p * np.array([1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0])

    # n_k = 2*pi * sum_{i,j} q_i q_j I[k+i+j]
    n = np.zeros(3)
    for i in range(7):
        qi = q[i]
        if qi == 0.0:
            continue
        for j in range(7):
            qij = qi * q[j]
            if qij == 0.0:
                continue
            for k in range(3):
                idx = k + i + j
                if idx < 14:
                    n[k] += 2.0 * np.pi * qij * _I[idx]

    m0, m2, m4 = m[0], m[1], m[2]
    n0, n2, n4 = n[0], n[1], n[2]

    if m0 <= 1e-30 or n0 <= 1e-30 or not np.isfinite(m0) or not np.isfinite(n0):
        return 1e18, np.array([1e18, 1e18, 1e18])

    vf = m2 / m0
    vF = n2 / n0

    if vf <= 0 or vF <= 0 or not np.isfinite(vf) or not np.isfinite(vF):
        return 1e18, np.array([1e18, 1e18, 1e18])

    bound = np.sqrt(vf * vF)

    # Analytic gradient
    da12 = np.array([-1.0/945.0, -3.0/945.0, -105.0/945.0])

    grad = np.zeros(3)
    for c in range(3):
        dp = np.zeros(7)
        dp[0] = np.array([1.0, 3.0, 105.0])[c] + 945.0*da12[c]
        dp[1] = np.array([0.0, -6.0, -420.0])[c] - 7560.0*da12[c]
        dp[2] = np.array([0.0, 1.0, 210.0])[c] + 13860.0*da12[c]
        dp[3] = np.array([0.0, 0.0, -28.0])[c] - 6930.0*da12[c]
        dp[4] = np.array([0.0, 0.0, 1.0])[c] + 1155.0*da12[c]
        dp[5] = -66.0*da12[c]
        dp[6] = da12[c]

        # dm_k/da_c = 2 * sum_i dp_i * sum_j p_j I[k+i+j]
        dm = np.zeros(3)
        for i in range(7):
            dpi = dp[i]
            if dpi == 0.0:
                continue
            for k in range(3):
                s = 0.0
                for j in range(7):
                    idx = k + i + j
                    if idx < 14:
                        s += p[j] * _I[idx]
                dm[k] += 2.0 * dpi * s

        dq = dp * np.array([1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0])
        dn = np.zeros(3)
        for i in range(7):
            dqi = dq[i]
            if dqi == 0.0:
                continue
            for k in range(3):
                s = 0.0
                for j in range(7):
                    idx = k + i + j
                    if idx < 14:
                        s += q[j] * _I[idx]
                dn[k] += 4.0 * np.pi * dqi * s

        dvm2 = (dm[1]*m0 - m2*dm[0]) / (m0*m0)
        dvn2 = (dn[1]*n0 - n2*dn[0]) / (n0*n0)

        grad[c] = (vF * dvm2 + vf * dvn2) / (2.0 * bound)

    return bound, grad


def _objective(params):
    val, _ = _bound_and_grad(params)
    return val


def _objective_grad(params):
    val, grad = _bound_and_grad(params)
    return val, grad


def construct():
    # Multiple diverse starting points
    starts = [
        np.array([-113.0/100.0, 1.0/25.0, 1.0/3240.0]),  # seed
        np.array([0.0, 0.0, 0.0]),
        np.array([-1.0, 0.05, 0.0005]),
        np.array([-2.0, 0.03, 0.0003]),
        np.array([-0.5, 0.02, 0.0002]),
        np.array([-1.5, 0.08, 0.0008]),
        np.array([1.0, -0.04, 0.0004]),
        np.array([-3.0, 0.15, 0.002]),
        np.array([-0.8, 0.06, 0.0006]),
        np.array([-1.2, 0.035, 0.00035]),
    ]

    rng = np.random.default_rng(42)
    for i in range(30):
        starts.append(rng.normal(0, 0.6, 3))

    best_val = np.inf
    best_params = None

    for start in starts:
        if time.time() > _DEADLINE - 15:
            break
        try:
            res = minimize(_objective_grad, start, method='L-BFGS-B',
                          jac=True, options={'maxiter': 8000, 'ftol': 1e-16, 'gtol': 1e-15, 'maxls': 50})
            if res.fun < best_val and np.isfinite(res.fun):
                best_val = res.fun
                best_params = res.x.copy()
        except Exception:
            pass

    # Polish with Nelder-Mead from best
    if best_params is not None and time.time() < _DEADLINE - 5:
        try:
            res2 = minimize(_objective, best_params, method='Nelder-Mead',
                           options={'maxiter': 50000, 'xatol': 1e-14, 'fatol': 1e-16, 'adaptive': True})
            if res2.fun < best_val and np.isfinite(res2.fun):
                best_val = res2.fun
                best_params = res2.x.copy()
        except Exception:
            pass

    # Final L-BFGS-B polish from best
    if best_params is not None and time.time() < _DEADLINE - 2:
        try:
            res3 = minimize(_objective_grad, best_params, method='L-BFGS-B',
                           jac=True, options={'maxiter': 10000, 'ftol': 1e-18, 'gtol': 1e-16})
            if res3.fun < best_val and np.isfinite(res3.fun):
                best_val = res3.fun
                best_params = res3.x.copy()
        except Exception:
            pass

    if best_params is None:
        return np.array([-113.0/100.0, 1.0/25.0, 1.0/3240.0])

    return np.array(best_params)
# EVOLVE-BLOCK-END
