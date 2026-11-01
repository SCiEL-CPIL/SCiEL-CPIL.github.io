"""Combined analytic uncertainty bound with exact gradient + L-BFGS-B optimization."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

_DEADLINE = time.time() + 200.0
SAFETY_MARGIN = 10.0

def _check_deadline():
    return (time.time() - _DEADLINE) < -SAFETY_MARGIN

# Precompute I_n = integral x^{2n} e^{-x^2} dx = (2n)! / (4^n n!) * sqrt(pi)
# We need up to n=12 (since max degree in |P|^2 is 2*6=12, plus k up to 2, so max index 14)
_I_CACHE = np.zeros(15)
def _compute_I(n):
    """Compute (2n)! / (4^n n!) * sqrt(pi) using log-sum for stability."""
    if n == 0:
        return np.sqrt(np.pi)
    log_val = 0.0
    for m in range(1, 2*n+1):
        log_val += np.log(m)
    log_val -= n * np.log(4.0)
    for m in range(1, n+1):
        log_val -= np.log(m)
    log_val += 0.5 * np.log(np.pi)
    return np.exp(log_val)

for _n in range(15):
    _I_CACHE[_n] = _compute_I(_n)

def _bound_and_grad(params):
    """
    Compute uncertainty product and its analytic gradient.
    g(x) = (a0*H0 + a4*H4 + a8*H8 + a12*H12) * exp(-x^2/2)
    a12 = -(a0 + 3*a4 + 105*a8)/945  (enforces g(0)=0)
    
    A(f)^2 = <x^2>/<1> for |g|^2
    A(F)^2 = <xi^2>/<1> for |G|^2
    """
    a0, a4, a8 = params[0], params[1], params[2]
    a12 = -(a0 + 3.0*a4 + 105.0*a8) / 945.0
    
    # P(x) = a0*H0 + a4*H4 + a8*H8 + a12*H12
    # Coefficients of P(x) = sum_{k=0}^{6} p_k x^{2k}
    # H0 = 1
    # H4 = x^4 - 6x^2 + 3
    # H8 = x^8 - 28x^6 + 210x^4 - 420x^2 + 105
    # H12 = x^12 - 66x^10 + 1155x^8 - 6930x^6 + 13860x^4 - 7560x^2 + 945
    
    p0 = a0 + 3.0*a4 + 105.0*a8 + 945.0*a12
    p1 = -6.0*a4 - 420.0*a8 - 7560.0*a12
    p2 = a4 + 210.0*a8 + 13860.0*a12
    p3 = -28.0*a8 - 6930.0*a12
    p4 = a8 + 1155.0*a12
    p5 = -66.0*a12
    p6 = a12
    
    p = np.array([p0, p1, p2, p3, p4, p5, p6])
    
    # Moments m_k = integral x^{2k} |P|^2 e^{-x^2} dx for k=0,1,2
    # m_k = sum_{i,j=0}^{6} p_i p_j I[k+i+j]
    m = np.zeros(3)
    for i in range(7):
        for j in range(7):
            pp = p[i] * p[j]
            for k in range(3):
                idx = k + i + j
                if idx < 15:
                    m[k] += pp * _I_CACHE[idx]
    
    # Fourier transform: G(xi) = sqrt(2*pi) * P(-i*xi) * e^{-xi^2/2}
    # P(-i*xi) = sum p_k (-i)^{2k} xi^{2k} = sum p_k (-1)^k xi^{2k}
    q = p * np.array([1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0])
    
    # |G|^2 = 2*pi * |Q|^2 * e^{-xi^2}
    # n_k = 2*pi * sum_{i,j} q_i q_j I[k+i+j]
    n = np.zeros(3)
    for i in range(7):
        for j in range(7):
            qq = q[i] * q[j]
            for k in range(3):
                idx = k + i + j
                if idx < 15:
                    n[k] += 2.0*np.pi * qq * _I_CACHE[idx]
    
    m0, m2 = m[0], m[1]
    n0, n2 = n[0], n[1]
    
    if m0 <= 0 or n0 <= 0 or not np.isfinite(m0) or not np.isfinite(n0):
        return 1e18, np.array([1e18, 1e18, 1e18])
    
    vf = m2 / m0
    vF = n2 / n0
    
    if vf <= 0 or vF <= 0:
        return 1e18, np.array([1e18, 1e18, 1e18])
    
    bound = np.sqrt(vf * vF)
    
    # Compute gradient analytically
    da12 = np.array([-1.0/945.0, -3.0/945.0, -105.0/945.0])
    
    grad = np.zeros(3)
    
    # Precompute S_m[k][i] = sum_j p_j I[k+i+j] for k=0,1,2 and i=0..6
    S_m = np.zeros((3, 7))
    for k in range(3):
        for i in range(7):
            s = 0.0
            for j in range(7):
                idx = k + i + j
                if idx < 15:
                    s += p[j] * _I_CACHE[idx]
            S_m[k, i] = s
    
    # Precompute S_n[k][i] = sum_j q_j I[k+i+j] for k=0,1,2 and i=0..6
    S_n = np.zeros((3, 7))
    for k in range(3):
        for i in range(7):
            s = 0.0
            for j in range(7):
                idx = k + i + j
                if idx < 15:
                    s += q[j] * _I_CACHE[idx]
            S_n[k, i] = s
    
    for c in range(3):
        # dp_k/da_c
        dp = np.zeros(7)
        dp[0] = np.array([1.0, 3.0, 105.0])[c] + 945.0*da12[c]
        dp[1] = np.array([0.0, -6.0, -420.0])[c] - 7560.0*da12[c]
        dp[2] = np.array([0.0, 1.0, 210.0])[c] + 13860.0*da12[c]
        dp[3] = np.array([0.0, 0.0, -28.0])[c] - 6930.0*da12[c]
        dp[4] = np.array([0.0, 0.0, 1.0])[c] + 1155.0*da12[c]
        dp[5] = np.array([0.0, 0.0, 0.0])[c] - 66.0*da12[c]
        dp[6] = np.array([0.0, 0.0, 0.0])[c] + 1.0*da12[c]
        
        # dm_k/da_c = 2 * sum_i dp_i * S_m[k, i]
        dm = np.zeros(3)
        for k in range(3):
            dm[k] = 2.0 * np.dot(dp, S_m[k, :])
        
        dq = dp * np.array([1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0])
        
        # dn_k/da_c = 2 * 2*pi * sum_i dq_i * S_n[k, i]
        dn = np.zeros(3)
        for k in range(3):
            dn[k] = 4.0 * np.pi * np.dot(dq, S_n[k, :])
        
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
    seed = np.array([-113.0 / 100.0, 1.0 / 25.0, 1.0 / 3240.0])
    
    # Multiple starting points
    starts = [
        seed,
        np.array([0.0, 0.0, 0.0]),
        np.array([-1.0, 0.1, 0.001]),
        np.array([-2.0, 0.05, 0.0005]),
        np.array([-0.5, 0.02, 0.0002]),
        np.array([-1.5, 0.08, 0.0008]),
        np.array([1.0, -0.05, 0.0005]),
        np.array([-3.0, 0.15, 0.002]),
        np.array([-1.13, 0.04, 0.0003]),
        np.array([-1.2, 0.05, 0.0004]),
        np.array([-1.05, 0.035, 0.00035]),
    ]
    
    rng = np.random.default_rng(42)
    for i in range(15):
        starts.append(rng.normal(0, 0.5, 3))
    
    best_val = np.inf
    best_params = seed.copy()
    
    for start in starts:
        if not _check_deadline():
            break
        try:
            res = minimize(_objective_grad, start, method='L-BFGS-B',
                          jac=True, options={'maxiter': 3000, 'ftol': 1e-15, 'gtol': 1e-14})
            if res.fun < best_val and np.isfinite(res.fun):
                best_val = res.fun
                best_params = res.x.copy()
        except Exception:
            pass
    
    # Polish with Nelder-Mead from best
    if best_params is not None and _check_deadline():
        try:
            res = minimize(_objective, best_params, method='Nelder-Mead',
                          options={'maxiter': 50000, 'xatol': 1e-16, 'fatol': 1e-18})
            if res.fun < best_val and np.isfinite(res.fun):
                best_val = res.fun
                best_params = res.x.copy()
        except Exception:
            pass
    
    # Final high-precision polish with L-BFGS-B
    if best_params is not None and _check_deadline():
        try:
            res = minimize(_objective_grad, best_params, method='L-BFGS-B',
                          jac=True, options={'maxiter': 10000, 'ftol': 1e-16, 'gtol': 1e-15})
            if res.fun < best_val and np.isfinite(res.fun):
                best_val = res.fun
                best_params = res.x.copy()
        except Exception:
            pass
    
    if best_params is not None and np.isfinite(best_val):
        return best_params
    return seed

# EVOLVE-BLOCK-END
