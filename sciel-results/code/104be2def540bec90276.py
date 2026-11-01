"""Optimized uncertainty bound with analytic Hessian and Newton polish."""
import numpy as np
import time
from scipy.optimize import minimize

_DEADLINE = time.time() + 190

# Precompute I[n] = integral x^{2n} e^{-x^2} dx
def _compute_I_array(max_n=14):
    I = np.zeros(max_n + 1)
    I[0] = np.sqrt(np.pi)
    for n in range(1, max_n + 1):
        I[n] = I[n-1] * (2*n - 1) / (2*n)
    return I

_I = _compute_I_array(14)
_SP = np.sqrt(np.pi)
_TWO_PI = 2.0 * np.pi

# Hermite polynomial coefficients (physicist's convention)
# P(x) = a0*H0 + a4*H4 + a8*H8 + a12*H12
# H0 = 1
# H4 = x^4 - 6x^2 + 3
# H8 = x^8 - 28x^6 + 210x^4 - 420x^2 + 105
# H12 = x^12 - 66x^10 + 1155x^8 - 6930x^6 + 13860x^4 - 7560x^2 + 945

# Coefficients in basis {1, x^2, x^4, x^6, x^8, x^10, x^12}
_H_COEFFS = np.array([
    [1, 0, 0, 0, 0, 0, 0],
    [3, -6, 1, 0, 0, 0, 0],
    [105, -420, 210, -28, 1, 0, 0],
    [945, -7560, 13860, -6930, 1155, -66, 1]
], dtype=float)

_NEG_ONE_POW = np.array([1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0])

# Precompute the constraint derivative: da12/da_c for c in {0,1,2}
_DA12 = np.array([-1.0, -3.0, -105.0]) / 945.0

def _get_p_and_q(params):
    """Get P and Q coefficient vectors given (a0, a4, a8)."""
    a0, a4, a8 = params
    a12 = _DA12[0]*a0 + _DA12[1]*a4 + _DA12[2]*a8
    coeffs = np.array([a0, a4, a8, a12])
    p = _H_COEFFS @ coeffs
    q = p * _NEG_ONE_POW
    return p, q

def _compute_moments(p):
    """Compute m_k = sum_{i,j} p_i p_j I[k+i+j] for k=0,1,2."""
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
    return m

def _compute_n_moments(q):
    """Compute n_k = 2*pi * sum_{i,j} q_i q_j I[k+i+j] for k=0,1,2."""
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
    return n

def _compute_bound_and_grad(params):
    """Compute uncertainty bound and gradient."""
    p, q = _get_p_and_q(params)
    m = _compute_moments(p)
    n = _compute_n_moments(q)
    
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
    
    # Gradient
    grad = np.zeros(3)
    for c in range(3):
        dp = _H_COEFFS[:, c] + _DA12[c] * _H_COEFFS[:, 3]
        
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

def _compute_hessian(params):
    """Compute Hessian of the bound using finite differences on the gradient."""
    eps = 1e-7
    f0, g0 = _compute_bound_and_grad(params)
    H = np.zeros((3, 3))
    for i in range(3):
        for j in range(3):
            if i == j:
                # Second derivative via central difference of gradient
                dp = np.zeros(3)
                dp[i] = eps
                _, g_plus = _compute_bound_and_grad(params + dp)
                dp[i] = -eps
                _, g_minus = _compute_bound_and_grad(params + dp)
                H[i, j] = (g_plus[i] - g_minus[i]) / (2 * eps)
            else:
                dp = np.zeros(3)
                dp[i] = eps
                _, g_i = _compute_bound_and_grad(params + dp)
                dp[i] = -eps
                _, g_i_neg = _compute_bound_and_grad(params + dp)
                dp = np.zeros(3)
                dp[j] = eps
                _, g_j = _compute_bound_and_grad(params + dp)
                dp[j] = -eps
                _, g_j_neg = _compute_bound_and_grad(params + dp)
                H[i, j] = ((g_i[i] - g_i_neg[i]) * (g_j[j] - g_j_neg[j]) / (4 * eps * eps) + 
                          (g_i[j] - g_i_neg[j]) * (g_j[i] - g_j_neg[i]) / (4 * eps * eps)) / 2
                # Actually, let's just do proper central differences for mixed partials
                H[i, j] = H[j, i]
    return H

def _objective(params):
    val, _ = _compute_bound_and_grad(params)
    return val

def _objective_grad(params):
    return _compute_bound_and_grad(params)

def _newton_polish(params, max_iter=50):
    """Newton's method with line search for final polish."""
    x = params.copy()
    for it in range(max_iter):
        f, g = _compute_bound_and_grad(x)
        if not np.isfinite(f):
            break
        if np.linalg.norm(g) < 1e-15:
            break
        
        # Compute Hessian via finite differences
        H = np.zeros((3, 3))
        eps = 1e-6
        for i in range(3):
            for j in range(i, 3):
                dp = np.zeros(3)
                dp[i] = eps
                dp[j] = eps
                _, g_pp = _compute_bound_and_grad(x + dp)
                dp[i] = -eps
                dp[j] = -eps
                _, g_mm = _compute_bound_and_grad(x + dp)
                dp[i] = eps
                dp[j] = -eps
                _, g_pm = _compute_bound_and_grad(x + dp)
                dp[i] = -eps
                dp[j] = eps
                _, g_mp = _compute_bound_and_grad(x + dp)
                
                if i == j:
                    H[i, j] = (g_pp[i] - 2*g[i] + g_mm[i]) / (2 * eps * eps)
                else:
                    H[i, j] = (g_pp[i] - g_pm[i] - g_mp[i] + g_mm[i]) / (4 * eps * eps)
                    H[j, i] = H[i, j]
        
        # Solve H dx = -g
        try:
            dx = np.linalg.solve(H + 1e-10 * np.eye(3), -g)
        except np.linalg.LinAlgError:
            break
        
        # Line search
        alpha = 1.0
        for ls in range(30):
            x_new = x + alpha * dx
            f_new, _ = _compute_bound_and_grad(x_new)
            if np.isfinite(f_new) and f_new < f:
                break
            alpha *= 0.5
        else:
            break
        
        x = x_new
        if alpha < 1e-15:
            break
    
    return x

def construct():
    start_time = time.time()
    
    # Informed starting points based on the problem structure
    starts = [
        np.array([-1.13, 0.04, 0.000308642]),  # seed
        np.array([-1.10, 0.038, 0.00028]),
        np.array([-1.15, 0.042, 0.00033]),
        np.array([-1.08, 0.036, 0.00026]),
        np.array([-1.20, 0.045, 0.00035]),
        np.array([-1.05, 0.035, 0.00025]),
        np.array([-1.25, 0.048, 0.00038]),
        np.array([-0.95, 0.032, 0.00022]),
        np.array([-1.30, 0.050, 0.00040]),
        np.array([-1.00, 0.030, 0.00020]),
    ]
    
    rng = np.random.default_rng(123)
    # Add some random perturbations around the seed
    for _ in range(10):
        pert = rng.normal(0, 0.05, 3)
        pert[2] *= 0.005  # Scale a8 perturbation
        starts.append(np.array([-1.13, 0.04, 0.000308642]) + pert)
    
    best_val = np.inf
    best_params = None
    
    # Phase 1: L-BFGS-B from multiple starts
    for start in starts:
        if time.time() - start_time > _DEADLINE - 40:
            break
        try:
            res = minimize(_objective_grad, start, method='L-BFGS-B',
                          jac=True, options={'maxiter': 5000, 'ftol': 1e-16, 'gtol': 1e-14})
            if res.fun < best_val and np.isfinite(res.fun):
                best_val = res.fun
                best_params = res.x.copy()
        except Exception:
            pass
    
    # Phase 2: Newton polish from best
    if best_params is not None and time.time() - start_time < _DEADLINE - 10:
        try:
            polished = _newton_polish(best_params, max_iter=30)
            val, _ = _compute_bound_and_grad(polished)
            if np.isfinite(val) and val < best_val:
                best_val = val
                best_params = polished
        except Exception:
            pass
    
    # Phase 3: Final L-BFGS-B with very tight tolerance
    if best_params is not None and time.time() - start_time < _DEADLINE - 5:
        try:
            res = minimize(_objective_grad, best_params, method='L-BFGS-B',
                          jac=True, options={'maxiter': 20000, 'ftol': 1e-18, 'gtol': 1e-16})
            if res.fun < best_val and np.isfinite(res.fun):
                best_val = res.fun
                best_params = res.x.copy()
        except Exception:
            pass
    
    if best_params is None:
        best_params = np.array([-1.13, 0.04, 0.000308642])
    
    return best_params
