"""
Analytic uncertainty bound with exact gradient + L-BFGS-B optimization.
Correctly handles Fourier transform phases: Q(k) = P(ik), q_k = (-1)^k p_k.
Position variance vf = m2/m0, frequency variance vF = n2/n0, bound = sqrt(vf*vF).
"""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

_START_TIME = time.time()
_DEADLINE = _START_TIME + 190  # Safety margin before 210s kill


def _precompute():
    """
    Precompute constants:
    - I[n] = integral x^{2n} e^{-x^2} dx = (2n-1)!! sqrt(pi) / 2^n
    - Reduced coefficient matrix mapping [c0, c4, c8] -> [p0..p6]
      where c12 = -(c0 + 3*c4 + 105*c8)/945
    - sign_k = (-1)^k for k=0..6 (for Q polynomial)
    """
    sp = np.sqrt(np.pi)
    # I[n] for n = 0..13 (need up to x^{24}, so n up to 12, plus margin)
    I = np.zeros(14)
    I[0] = sp
    for n in range(1, 14):
        I[n] = (2*n - 1) / (2*n) * I[n - 1]

    # Coefficient matrix: rows p_k (k=0..6), cols [c0, c4, c8, c12]
    coeff_matrix = np.array([
        [1.0,     3.0,    105.0,    945.0],
        [0.0,    -6.0,   -420.0,  -7560.0],
        [0.0,     1.0,    210.0,  13860.0],
        [0.0,     0.0,    -28.0,  -6930.0],
        [0.0,     0.0,     1.0,   1155.0],
        [0.0,     0.0,     0.0,    -66.0],
        [0.0,     0.0,     0.0,     1.0]
    ])

    # c12 = -(c0 + 3*c4 + 105*c8)/945
    # So [c0,c4,c8,c12] = [c0,c4,c8] @ (I_3 + outer([c12 coeffs]))
    A_c12 = np.array([-1.0/945.0, -3.0/945.0, -105.0/945.0])
    reduced_matrix = coeff_matrix[:, :3] + np.outer(coeff_matrix[:, 3], A_c12)

    # sign_k = (-1)^k for k=0..6
    sign_k = np.array([(-1.0)**k for k in range(7)])

    return I, reduced_matrix, sign_k


_I, _REDUCED_MATRIX, _SIGN_K = _precompute()


def _compute_moments(params):
    """
    Compute:
    m0 = int P(x)^2 e^{-x^2} dx
    m2 = int x^2 P(x)^2 e^{-x^2} dx
    n0 = 2*pi * int Q(k)^2 e^{-k^2} dk
    n2 = 2*pi * int k^2 Q(k)^2 e^{-k^2} dk
    where Q(k) = P(ik), q_k = (-1)^k p_k
    """
    c0, c4, c8 = params
    p = _REDUCED_MATRIX @ np.array([c0, c4, c8])
    q = p * _SIGN_K  # q_k = (-1)^k p_k

    m0 = 0.0
    m2 = 0.0
    n0 = 0.0
    n2 = 0.0

    for i in range(7):
        pi_val = p[i]
        qi_val = q[i]
        if pi_val == 0.0 and qi_val == 0.0:
            continue
        for j in range(7):
            pp = pi_val * p[j]
            qq = qi_val * q[j]
            idx0 = i + j
            idx1 = i + j + 1
            if idx0 < 14:
                m0 += pp * _I[idx0]
                n0 += qq * _I[idx0]
            if idx1 < 14:
                m2 += pp * _I[idx1]
                n2 += qq * _I[idx1]

    n0 *= 2.0 * np.pi
    n2 *= 2.0 * np.pi

    return m0, m2, n0, n2


def _bound_and_grad(params):
    """
    bound = sqrt(vf * vF) where vf = m2/m0, vF = n2/n0
    Gradient via chain rule.
    """
    m0, m2, n0, n2 = _compute_moments(params)

    if m0 <= 0 or n0 <= 0 or not np.isfinite(m0) or not np.isfinite(n0) \
       or not np.isfinite(m2) or not np.isfinite(n2):
        return 1e18, np.array([1e18, 1e18, 1e18])

    vf = m2 / m0
    vF = n2 / n0

    if vf <= 0 or vF <= 0 or not np.isfinite(vf) or not np.isfinite(vF):
        return 1e18, np.array([1e18, 1e18, 1e18])

    bound = np.sqrt(vf * vF)

    # Gradient: d(bound)/dc = (1/(2*bound)) * (vF * d(vf)/dc + vf * d(vF)/dc)
    # d(vf)/dc = (dm2/dc * m0 - m2 * dm0/dc) / m0^2
    # d(vF)/dc = (dn2/dc * n0 - n2 * dn0/dc) / n0^2

    p = _REDUCED_MATRIX @ np.array(params)
    q = p * _SIGN_K

    # dp/dc_k for each k=0,1,2 (columns of _REDUCED_MATRIX)
    dp = _REDUCED_MATRIX  # shape (7, 3)
    dq = dp * _SIGN_K[:, np.newaxis]  # shape (7, 3)

    grad = np.zeros(3)
    for k in range(3):
        # dm0/dc_k = 2 * sum_i dp_i * sum_j p_j I[i+j]
        # dm2/dc_k = 2 * sum_i dp_i * sum_j p_j I[i+j+1]
        dm0 = 0.0
        dm2 = 0.0
        dn0 = 0.0
        dn2 = 0.0

        for i in range(7):
            dpi = dp[i, k]
            dqi = dq[i, k]
            if dpi == 0.0 and dqi == 0.0:
                continue
            for j in range(7):
                idx0 = i + j
                idx1 = i + j + 1
                if idx0 < 14:
                    dm0 += 2.0 * dpi * p[j] * _I[idx0]
                    dn0 += 2.0 * dqi * q[j] * _I[idx0]
                if idx1 < 14:
                    dm2 += 2.0 * dpi * p[j] * _I[idx1]
                    dn2 += 2.0 * dqi * q[j] * _I[idx1]

        dn0 *= 2.0 * np.pi
        dn2 *= 2.0 * np.pi

        dvm2 = (dm2 * m0 - m2 * dm0) / (m0 * m0)
        dvn2 = (dn2 * n0 - n2 * dn0) / (n0 * n0)

        grad[k] = (vF * dvm2 + vf * dvn2) / (2.0 * bound)

    return bound, grad


def _objective(params):
    val, _ = _bound_and_grad(params)
    return val


def _objective_grad(params):
    val, grad = _bound_and_grad(params)
    return val, grad


def construct():
    """
    Return three coefficients (c0, c4, c8) that minimize the uncertainty bound.
    c12 is determined by the constraint g(0) = 0.
    """
    # Seed initial guess
    seed = np.array([-113.0/100.0, 1.0/25.0, 1.0/3240.0])

    # Multiple diverse starting points
    starts = [
        seed,
        np.array([0.0, 0.0, 0.0]),
        np.array([-1.0, 0.05, 0.0005]),
        np.array([-2.0, 0.1, 0.001]),
        np.array([-0.5, 0.02, 0.0002]),
        np.array([-1.5, 0.08, 0.0008]),
        np.array([1.0, -0.05, 0.0005]),
        np.array([-3.0, 0.15, 0.002]),
        np.array([-0.8, 0.03, 0.0003]),
        np.array([-1.2, 0.06, 0.0006]),
        np.array([-1.13, 0.04, 0.0003]),
        np.array([-1.13, 0.04, 0.00031]),
        np.array([-1.13, 0.041, 0.00029]),
        np.array([-1.12, 0.039, 0.00032]),
        np.array([-1.14, 0.041, 0.00028]),
    ]

    # Add random starts
    rng = np.random.default_rng(42)
    for _ in range(25):
        starts.append(rng.normal(0, 0.8, 3))

    best_val = np.inf
    best_params = seed.copy()

    # Phase 1: L-BFGS-B from multiple starts
    for start in starts:
        if time.time() > _DEADLINE - 30:
            break
        try:
            res = minimize(_objective_grad, start, method='L-BFGS-B',
                          jac=True, options={'maxiter': 5000, 'ftol': 1e-16, 'gtol': 1e-15})
            if res.fun < best_val and np.isfinite(res.fun):
                best_val = res.fun
                best_params = res.x.copy()
        except Exception:
            pass

    # Phase 2: Polish with Nelder-Mead from best
    if best_params is not None and time.time() < _DEADLINE - 15:
        try:
            res = minimize(_objective, best_params, method='Nelder-Mead',
                          options={'maxiter': 80000, 'xatol': 1e-15, 'fatol': 1e-18, 'adaptive': True})
            if res.fun < best_val and np.isfinite(res.fun):
                best_val = res.fun
                best_params = res.x.copy()
        except Exception:
            pass

    # Phase 3: Additional random restarts if time permits
    if time.time() < _DEADLINE - 10:
        for _ in range(15):
            if time.time() > _DEADLINE - 10:
                break
            x_rand = rng.normal(0, 0.5, 3)
            try:
                res = minimize(_objective_grad, x_rand, method='L-BFGS-B',
                              jac=True, options={'maxiter': 3000, 'ftol': 1e-16, 'gtol': 1e-15})
                if res.fun < best_val and np.isfinite(res.fun):
                    best_val = res.fun
                    best_params = res.x.copy()
            except Exception:
                pass

    return best_params

# EVOLVE-BLOCK-END
