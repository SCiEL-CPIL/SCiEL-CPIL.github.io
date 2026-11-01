"""Combined uncertainty bound optimization with analytic gradient."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

_DEADLINE = time.time() + 195

# Precompute I[n] = integral x^{2n} e^{-x^2} dx = (2n)! / (4^n n!) * sqrt(pi)
# Use log-space for stability, then convert
def _compute_I_array(max_n=14):
    I = np.zeros(max_n + 1)
    for n in range(max_n + 1):
        if n == 0:
            I[n] = np.sqrt(np.pi)
        else:
            # I[n] = I[n-1] * (2n-1) / (2n)
            I[n] = I[n-1] * (2*n - 1) / (2*n)
    return I

_I = _compute_I_array(14)
_SP = np.sqrt(np.pi)
_TWO_PI = 2.0 * np.pi

# Hermite polynomial coefficients (physicist's convention: H_0=1, H_4=x^4-6x^2+3, etc.)
# P(x) = a0*H0 + a4*H4 + a8*H8 + a12*H12
# H0 = 1
# H4 = x^4 - 6x^2 + 3
# H8 = x^8 - 28x^6 + 210x^4 - 420x^2 + 105
# H12 = x^12 - 66x^10 + 1155x^8 - 6930x^6 + 13860x^4 - 7560x^2 + 945

# Coefficients of each Hermite polynomial in basis {1, x^2, x^4, x^6, x^8, x^10, x^12}
# Index: 0->1, 1->x^2, 2->x^4, 3->x^6, 4->x^8, 5->x^10, 6->x^12
_H_COEFFS = np.array([
    [1, 0, 0, 0, 0, 0, 0],       # H0
    [3, -6, 1, 0, 0, 0, 0],      # H4
    [105, -420, 210, -28, 1, 0, 0],  # H8
    [945, -7560, 13860, -6930, 1155, -66, 1]  # H12
], dtype=float)

# For the Fourier transform: Q_k = p_k * (-1)^k
_NEG_ONE_POW = np.array([1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0])


def _compute_bound_and_grad(params):
    """
    Compute uncertainty bound sqrt(vf * vF) and its gradient.
    g(x) = P(x) * exp(-x^2/2) where P(x) = sum c_i H_i(x)
    a12 = -(a0 + 3*a4 + 105*a8) / 945 to enforce g(0) = 0
    """
    a0, a4, a8 = params[0], params[1], params[2]
    a12 = -(a0 + 3.0*a4 + 105.0*a8) / 945.0
    
    coeffs = np.array([a0, a4, a8, a12])
    
    # P(x) coefficients in basis {1, x^2, ..., x^12}
    p = _H_COEFFS @ coeffs  # shape (7,)
    
    # Verify p[0] should be ~0 due to constraint
    # p[0] = a0 + 3*a4 + 105*a8 + 945*a12 = 0 by construction
    
    # Q coefficients for Fourier transform
    q = p * _NEG_ONE_POW
    
    # Compute moments m_k = integral x^{2k} |P|^2 e^{-x^2} dx for k=0,1,2
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
                if idx <= 14:
                    m[k] += pij * _I[idx]
    
    # n_k = 2*pi * integral xi^{2k} |Q|^2 e^{-xi^2} dxi
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
                if idx <= 14:
                    n[k] += _TWO_PI * qij * _I[idx]
    
    m0, m2 = m[0], m[1]
    n0, n2 = n[0], n[1]
    
    if m0 <= 1e-30 or n0 <= 1e-30 or not np.isfinite(m0) or not np.isfinite(n0):
        return 1e18, np.array([1e18, 1e18, 1e18])
    
    vf = m2 / m0
    vF = n2 / n0
    
    if vf <= 0 or vF <= 0 or not np.isfinite(vf) or not np.isfinite(vF):
        return 1e18, np.array([1e18, 1e18, 1e18])
    
    bound = np.sqrt(vf * vF)
    
    if not np.isfinite(bound):
        return 1e18, np.array([1e18, 1e18, 1e18])
    
    # Compute gradient
    # da12/da_c for c in {a0, a4, a8}: [-1/945, -3/945, -105/945]
    da12_coeffs = np.array([-1.0, -3.0, -105.0]) / 945.0
    
    # dp_k/da_c = sum over hermite indices: _H_COEFFS[k, c] + da12_coeffs[c] * _H_COEFFS[k, 3]
    # _H_COEFFS has 4 rows (H0,H4,H8,H12) and 7 cols
    # For parameter c (0,1,2), dp = _H_COEFFS[:, c] + da12_coeffs[c] * _H_COEFFS[:, 3]
    
    grad = np.zeros(3)
    
    for c in range(3):
        dp = _H_COEFFS[:, c] + da12_coeffs[c] * _H_COEFFS[:, 3]
        
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
                    if idx <= 14:
                        s += p[j] * _I[idx]
                dm[k] += 2.0 * dpi * s
        
        dq = dp * _NEG_ONE_POW
        
        dn = np.zeros(3)
        for i in range(7):
            dqi = dq[i]
            if dqi == 0.0:
                continue
            for k in range(3):
                s = 0.0
                for j in range(7):
                    idx = k + i + j
                    if idx <= 14:
                        s += q[j] * _I[idx]
                dn[k] += 2.0 * _TWO_PI * dqi * s
        
        dvm2 = (dm[1]*m0 - m2*dm[0]) / (m0*m0)
        dvn2 = (dn[1]*n0 - n2*dn[0]) / (n0*n0)
        
        grad[c] = (vF * dvm2 + vf * dvn2) / (2.0 * bound)
    
    return bound, grad


def _objective(params):
    val, _ = _compute_bound_and_grad(params)
    return val


def _objective_grad(params):
    return _compute_bound_and_grad(params)


def construct():
    start_time = time.time()
    
    # Seed and diverse starting points
    starts = [
        np.array([-1.13, 0.04, 0.000308642]),  # original seed
        np.array([0.0, 0.0, 0.0]),
        np.array([-1.0, 0.05, 0.0005]),
        np.array([-1.5, 0.06, 0.0006]),
        np.array([-2.0, 0.08, 0.0008]),
        np.array([-0.5, 0.02, 0.0002]),
        np.array([-2.5, 0.1, 0.001]),
        np.array([-3.0, 0.12, 0.0012]),
        np.array([0.5, -0.03, 0.0003]),
        np.array([-1.2, 0.045, 0.00035]),
    ]
    
    rng = np.random.default_rng(42)
    for _ in range(15):
        starts.append(rng.normal(0, 0.8, 3))
    
    best_val = np.inf
    best_params = None
    
    # Phase 1: L-BFGS-B with analytic gradient from multiple starts
    for start in starts:
        if time.time() - start_time > _DEADLINE - 30:
            break
        try:
            res = minimize(_objective_grad, start, method='L-BFGS-B',
                          jac=True, options={'maxiter': 3000, 'ftol': 1e-16, 'gtol': 1e-15})
            if res.fun < best_val and np.isfinite(res.fun):
                best_val = res.fun
                best_params = res.x.copy()
        except Exception:
            pass
    
    # Phase 2: Nelder-Mead polish from best
    if best_params is not None and time.time() - start_time < _DEADLINE - 15:
        try:
            res = minimize(_objective, best_params, method='Nelder-Mead',
                          options={'maxiter': 50000, 'xatol': 1e-15, 'fatol': 1e-17})
            if res.fun < best_val and np.isfinite(res.fun):
                best_val = res.fun
                best_params = res.x.copy()
        except Exception:
            pass
    
    # Phase 3: Final high-precision L-BFGS-B polish
    if best_params is not None and time.time() - start_time < _DEADLINE - 5:
        try:
            res = minimize(_objective_grad, best_params, method='L-BFGS-B',
                          jac=True, options={'maxiter': 10000, 'ftol': 1e-18, 'gtol': 1e-16})
            if res.fun < best_val and np.isfinite(res.fun):
                best_val = res.fun
                best_params = res.x.copy()
        except Exception:
            pass
    
    if best_params is None:
        best_params = np.array([-1.13, 0.04, 0.000308642])
    
    return best_params
# EVOLVE-BLOCK-END
