"""Optimized Hermite-Gaussian uncertainty bound minimizer."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

_DEADLINE = time.time() + 200  # 10s safety margin before 210s kill


def _compute_bound(a0, a4, a8):
    """Compute the uncertainty product bound for given Hermite coefficients."""
    # a12 from constraint P(0) = 0: a0 + 3*a4 + 105*a8 + 945*a12 = 0
    a12 = -(a0 + 3.0 * a4 + 105.0 * a8) / 945.0

    # P(x) = a0*H0 + a4*H4 + a8*H8 + a12*H12
    # H0 = 1
    # H4 = x^4 - 6x^2 + 3
    # H8 = x^8 - 28x^6 + 210x^4 - 420x^2 + 105
    # H12 = x^12 - 66x^10 + 1155x^8 - 6930x^6 + 13860x^4 - 7560x^2 + 945

    # P(x) in powers of x^2:
    # p0 (const): a0 + 3*a4 + 105*a8 + 945*a12 = 0 (by constraint)
    # p2 (x^2): -6*a4 - 420*a8 - 7560*a12
    # p4 (x^4): a4 + 210*a8 + 13860*a12
    # p6 (x^6): -28*a8 - 6930*a12
    # p8 (x^8): a8 + 1155*a12
    # p10 (x^10): -66*a12
    # p12 (x^12): a12

    p0 = 0.0  # enforced by constraint
    p2 = -6.0 * a4 - 420.0 * a8 - 7560.0 * a12
    p4 = a4 + 210.0 * a8 + 13860.0 * a12
    p6 = -28.0 * a8 - 6930.0 * a12
    p8 = a8 + 1155.0 * a12
    p10 = -66.0 * a12
    p12 = a12

    # Gaussian moments: int x^{2k} e^{-x^2} dx = sqrt(pi) * (2k)! / (4^k k!)
    # m_k = int x^{2k} e^{-x^2} dx / sqrt(pi) = (2k-1)!! = 1*3*5*...*(2k-1)
    # m_0 = 1, m_1 = 1, m_2 = 3, m_3 = 15, m_4 = 105, m_5 = 945,
    # m_6 = 10395, m_7 = 135135, m_8 = 2027025, m_9 = 34459425,
    # m_10 = 654729075, m_11 = 13749310575, m_12 = 316234143225

    m = np.zeros(13)
    m[0] = 1.0
    for k in range(1, 13):
        m[k] = m[k-1] * (2.0 * k - 1.0)

    # |P|^2 = sum_i sum_j p_i p_j x^{2(i+j)}
    # ||g||^2 / sqrt(pi) = sum_i sum_j p_i p_j m_{i+j}
    # where p_i is the coefficient of x^{2i}

    coeffs = np.array([p0, p2, p4, p6, p8, p10, p12])  # index i -> x^{2i}
    n = len(coeffs)

    # norm2 = sum_{i,j} c_i c_j m_{i+j}
    norm2 = 0.0
    for i in range(n):
        if coeffs[i] == 0.0:
            continue
        for j in range(n):
            norm2 += coeffs[i] * coeffs[j] * m[i + j]

    if norm2 <= 0.0:
        return np.inf

    # x^2 |P|^2: coefficient of x^{2k} in x^2|P|^2 is sum_{i+j=k-1} c_i c_j
    # ||xg||^2 / sqrt(pi) = sum_k (sum_{i+j=k-1} c_i c_j) m_k
    x2_norm2 = 0.0
    for k in range(1, 2 * n):  # k = i+j+1, ranges from 1 to 2n-1
        s = 0.0
        for i in range(n):
            j = k - 1 - i
            if 0 <= j < n:
                s += coeffs[i] * coeffs[j]
        x2_norm2 += s * m[k]

    if x2_norm2 < 0.0:
        x2_norm2 = 0.0

    # P'(x): d/dx of sum c_i x^{2i} = sum 2i c_i x^{2i-1}
    # P'(x) = sum_{i=1}^{6} 2i c_i x^{2i-1}
    # P'(x) in odd powers: q_k = 2k * c_k for x^{2k-1}, k=1..6
    # P'(x) = q_1 x + q_3 x^3 + q_5 x^5 + ... 
    # Let's write P'(x) = sum_{k=1}^{6} 2k c_k x^{2k-1}

    # (P' + xP)^2: 
    # P' + xP = sum_{k=1}^{6} 2k c_k x^{2k-1} + sum_{k=0}^{6} c_k x^{2k+1}
    # = c_0 x + sum_{k=1}^{6} (2k c_k + c_{k-1}) x^{2k-1}
    # Let r_k = coefficient of x^{2k-1} for k=1..7:
    # r_1 = c_0 (from xP) ... wait let me redo.
    # xP = c_0 x + c_1 x^3 + c_2 x^5 + ... + c_6 x^{13}
    # P' = 2c_1 x + 4c_2 x^3 + 6c_3 x^5 + ... + 12c_6 x^{11}
    # P' + xP = (c_0 + 2c_1)x + (c_1 + 4c_2)x^3 + (c_2 + 6c_3)x^5 + ... + (c_5 + 12c_6)x^{11} + c_6 x^{13}
    # = sum_{k=0}^{6} s_k x^{2k+1} where:
    # s_k = c_k + 2(k+1) c_{k+1} for k=0..5
    # s_6 = c_6

    s = np.zeros(7)
    for k in range(6):
        s[k] = coeffs[k] + 2.0 * (k + 1) * coeffs[k + 1]
    s[6] = coeffs[6]

    # ||g'||^2 / sqrt(pi) = int (P'+xP)^2 e^{-x^2} dx / sqrt(pi)
    # = sum_{i,j=0}^{6} s_i s_j int x^{2i+1} x^{2j+1} e^{-x^2} dx / sqrt(pi)
    # = sum_{i,j} s_i s_j m_{i+j+1}

    deriv2 = 0.0
    for i in range(7):
        if s[i] == 0.0:
            continue
        for j in range(7):
            deriv2 += s[i] * s[j] * m[i + j + 1]

    if deriv2 < 0.0:
        deriv2 = 0.0

    # C4 = sqrt( (x2_norm2/norm2) * (deriv2/norm2) )
    val = x2_norm2 * deriv2 / (norm2 * norm2)
    if val < 0.0:
        return np.inf
    return np.sqrt(val)


def _objective(v):
    """Objective function for optimization."""
    a0, a4, a8 = v
    return _compute_bound(a0, a4, a8)


def construct():
    """Find optimal Hermite coefficients minimizing the uncertainty bound."""
    best_val = np.inf
    best_params = np.array([-113.0 / 100.0, 1.0 / 25.0, 1.0 / 3240.0])

    # Seed value
    seed_val = _compute_bound(best_params[0], best_params[1], best_params[2])
    best_val = seed_val

    # Multi-start Nelder-Mead + L-BFGS-B
    starts = [
        [-1.13, 0.04, 0.000308642],  # seed
        [-1.0, 0.05, 0.0003],
        [-1.2, 0.03, 0.0002],
        [-1.1, 0.04, 0.0004],
        [-1.15, 0.045, 0.00035],
        [-1.05, 0.035, 0.00025],
        [-1.25, 0.05, 0.0002],
        [-0.9, 0.06, 0.0004],
        [-1.3, 0.02, 0.0001],
        [-1.13, 0.04, 0.0],
        [-1.13, 0.0, 0.000308642],
        [-1.0, 0.04, 0.0003],
        [-1.1, 0.05, 0.0002],
        [-1.2, 0.04, 0.0004],
        [-1.05, 0.03, 0.0003],
    ]

    for x0 in starts:
        if time.time() > _DEADLINE:
            break

        # Nelder-Mead
        try:
            res = minimize(_objective, x0, method='Nelder-Mead',
                           options={'maxiter': 5000, 'xatol': 1e-12, 'fatol': 1e-14,
                                    'adaptive': True})
            if res.fun < best_val:
                best_val = res.fun
                best_params = res.x
        except Exception:
            pass

        # L-BFGS-B from the Nelder-Mead result or original
        if time.time() > _DEADLINE:
            break
        try:
            start2 = res.x if 'res' in dir() and hasattr(res, 'x') else x0
            res2 = minimize(_objective, start2, method='L-BFGS-B',
                            options={'maxiter': 5000, 'gtol': 1e-12})
            if res2.fun < best_val:
                best_val = res2.fun
                best_params = res2.x
        except Exception:
            pass

    # Final polish with tighter tolerances
    if time.time() < _DEADLINE - 5:
        try:
            res_final = minimize(_objective, best_params, method='Nelder-Mead',
                                 options={'maxiter': 10000, 'xatol': 1e-14, 'fatol': 1e-16,
                                          'adaptive': True})
            if res_final.fun < best_val:
                best_val = res_final.fun
                best_params = res_final.x
        except Exception:
            pass

    return np.array(best_params)
# EVOLVE-BLOCK-END
