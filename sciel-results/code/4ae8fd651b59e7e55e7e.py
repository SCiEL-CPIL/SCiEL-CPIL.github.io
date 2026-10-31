"""Analytic uncertainty bound with exact gradient + L-BFGS-B optimization."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, root

_DEADLINE = time.time() + 190

# Precompute I_n = integral x^{2n} e^{-x^2} dx = (2n)! / (4^n n!) * sqrt(pi)
def _precompute_I():
    """Precompute I_n for n=0..13"""
    I = np.zeros(14)
    sp = np.sqrt(np.pi)
    # Use recurrence: I_n = (2n-1) * I_{n-1} / (4*n) for n>=1, I_0 = sqrt(pi)
    I[0] = sp
    for n in range(1, 14):
        I[n] = (2*n - 1) * I[n-1] / (4.0 * n)
    return I

_I = _precompute_I()

# Precompute the index matrices for faster evaluation
_ARANGE7 = np.arange(7)
_IDX_MATS = [k + np.add.outer(_ARANGE7, _ARANGE7) for k in range(3)]
_SIGNS = np.array([1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0])

def _bound_and_grad(params):
    """
    Compute uncertainty product and its analytic gradient.
    g(x) = (a0*H0 + a4*H4 + a8*H8 + a12*H12) * exp(-x^2/2)
    a12 = -(a0 + 3*a4 + 105*a8)/945  (enforces g(0)=0)
    """
    a0, a4, a8 = params[0], params[1], params[2]
    a12 = -(a0 + 3.0*a4 + 105.0*a8) / 945.0
    
    # Coefficients of P(x) = sum_{k=0}^{6} p_k x^{2k}
    p = np.empty(7)
    p[0] = 0.0  # exactly zero by g(0)=0 constraint
    p[1] = -6.0*a4 - 420.0*a8 - 7560.0*a12
    p[2] = a4 + 210.0*a8 + 13860.0*a12
    p[3] = -28.0*a8 - 6930.0*a12
    p[4] = a8 + 1155.0*a12
    p[5] = -66.0*a12
    p[6] = a12
    
    I = _I
    
    # Vectorized computation of m_k and n_k
    pp = np.outer(p, p)
    m = np.zeros(3)
    for k in range(3):
        m[k] = np.sum(pp * I[_IDX_MATS[k]])
    
    # For Fourier: q_k = p_k * (-1)^k
    q = p * _SIGNS
    qq = np.outer(q, q)
    n = np.zeros(3)
    for k in range(3):
        n[k] = 2.0 * np.pi * np.sum(qq * I[_IDX_MATS[k]])
    
    m0, m2 = m[0], m[1]
    n0, n2 = n[0], n[1]
    
    if m0 <= 0 or n0 <= 0 or not np.isfinite(m0) or not np.isfinite(n0):
        return 1e18, np.array([1e18, 1e18, 1e18])
    
    vf = m2 / m0
    vF = n2 / n0
    
    if vf <= 0 or vF <= 0:
        return 1e18, np.array([1e18, 1e18, 1e18])
    
    bound = np.sqrt(vf * vF)
    
    # Precompute S_p and S_q using matrix operations
    # S_p[k,i] = sum_j p_j I[k+i+j]
    S_p = np.zeros((3, 7))
    S_q = np.zeros((3, 7))
    for k in range(3):
        idx_mat = _IDX_MATS[k]
        S_p[k, :] = I[idx_mat] @ p
        S_q[k, :] = I[idx_mat] @ q
    
    # dp/da matrix: shape (3, 7), dp[da_c, i]
    # Precompute as constant
    dpda = np.array([
        [0.0, 8.0, -13860.0/945.0, 6930.0/945.0, -1155.0/945.0, 66.0/945.0, -1.0/945.0],
        [0.0, -30.0, 45.0, -22.0, 3465.0/945.0, -198.0/945.0, -3.0/945.0],
        [0.0, -1260.0, 1750.0, -798.0, 1 + 1155.0*105.0/945.0, -6930.0/945.0, -105.0/945.0]
    ])
    
    # Vectorized gradient computation
    # dm = 2 * (dpda @ S_p.T) -> shape (3, 3)
    dm = 2.0 * (dpda @ S_p.T)
    
    # dq = dpda * signs (broadcast)
    dq = dpda * _SIGNS[np.newaxis, :]
    # dn = 4*pi * (dq @ S_q.T) -> shape (3, 3)
    dn = 4.0 * np.pi * (dq @ S_q.T)
    
    # dvm2[c] = (dm[c,1]*m0 - m2*dm[c,0]) / (m0*m0)
    dvm2 = (dm[:, 1]*m0 - m2*dm[:, 0]) / (m0*m0)
    dvn2 = (dn[:, 1]*n0 - n2*dn[:, 0]) / (n0*n0)
    
    grad = (vF * dvm2 + vf * dvn2) / (2.0 * bound)
    
    return bound, grad

def _objective(params):
    val, _ = _bound_and_grad(params)
    return val

def _objective_grad(params):
    val, grad = _bound_and_grad(params)
    return val, grad

def _grad_only(params):
    """Return only the gradient for root-finding."""
    _, grad = _bound_and_grad(params)
    return grad

def _jacobian(params):
    """Compute the Jacobian of the gradient (Hessian of objective) via finite differences."""
    eps = 1e-6
    J = np.zeros((3, 3))
    g0 = _grad_only(params)
    for j in range(3):
        p_plus = params.copy()
        p_minus = params.copy()
        p_plus[j] += eps
        p_minus[j] -= eps
        J[:, j] = (_grad_only(p_plus) - _grad_only(p_minus)) / (2 * eps)
    return J

def construct():
    # Multiple diverse starting points
    starts = [
        np.array([-1.13, 0.04, 0.000308642]),  # seed
        np.array([0.0, 0.0, 0.0]),
        np.array([-1.0, 0.1, 0.001]),
        np.array([-2.0, 0.05, 0.0005]),
        np.array([-0.5, 0.02, 0.0002]),
        np.array([-1.5, 0.08, 0.0008]),
        np.array([1.0, -0.05, 0.0005]),
        np.array([-3.0, 0.15, 0.002]),
        np.array([-0.8, 0.03, 0.0002]),
        np.array([-1.3, 0.05, 0.0004]),
        np.array([-1.7, 0.06, 0.0005]),
        np.array([-2.5, 0.12, 0.0015]),
        np.array([0.5, -0.02, 0.0001]),
        np.array([-0.3, 0.01, 0.0001]),
        np.array([-1.2, 0.045, 0.00035]),
        np.array([-1.4, 0.035, 0.00025]),
        np.array([-1.6, 0.055, 0.00045]),
        np.array([-1.0, 0.05, 0.0004]),
        np.array([-2.2, 0.07, 0.0006]),
        np.array([-1.1, 0.038, 0.00032]),
        np.array([-1.25, 0.042, 0.00038]),
        np.array([-1.35, 0.048, 0.00042]),
        np.array([-1.45, 0.052, 0.00048]),
        np.array([-1.55, 0.058, 0.00052]),
    ]
    
    rng = np.random.default_rng(42)
    for i in range(30):
        starts.append(rng.normal(0, 0.8, 3))
    
    best_val = np.inf
    best_params = None
    all_candidates = []
    
    # Phase 1: L-BFGS-B from multiple starts
    for start in starts:
        if time.time() > _DEADLINE - 50:
            break
        try:
            res = minimize(_objective_grad, start, method='L-BFGS-B',
                          jac=True, options={'maxiter': 10000, 'ftol': 1e-16, 'gtol': 1e-14, 'maxls': 50})
            if res.fun < best_val and np.isfinite(res.fun):
                best_val = res.fun
                best_params = res.x.copy()
            if np.isfinite(res.fun):
                all_candidates.append(res.x.copy())
        except Exception:
            pass
    
    # Phase 2: Root-finding for stationary points (grad = 0) from top candidates
    if time.time() < _DEADLINE - 35:
        # Use the best few candidates as starting points for root-finding
        top_candidates = all_candidates[:15] if len(all_candidates) >= 15 else all_candidates
        if best_params is not None:
            top_candidates.insert(0, best_params)
        
        for start in top_candidates:
            if time.time() > _DEADLINE - 35:
                break
            try:
                # Try to find a stationary point near this candidate
                res = root(_grad_only, start, method='hybr', tol=1e-12)
                if res.success:
                    val, _ = _bound_and_grad(res.x)
                    if val < best_val and np.isfinite(val):
                        best_val = val
                        best_params = res.x.copy()
                        all_candidates.append(res.x.copy())
            except Exception:
                pass
    
    # Phase 3: Newton's method with Hessian from best candidate
    if best_params is not None and time.time() < _DEADLINE - 15:
        x = best_params.copy()
        for _ in range(50):
            if time.time() > _DEADLINE - 15:
                break
            try:
                g = _grad_only(x)
                if np.linalg.norm(g) < 1e-14:
                    break
                H = _jacobian(x)
                # Solve H dx = -g
                try:
                    dx = np.linalg.solve(H, -g)
                except np.linalg.LinAlgError:
                    dx = np.linalg.lstsq(H, -g, rcond=None)[0]
                
                # Line search
                alpha = 1.0
                f0 = _objective(x)
                for _ in range(30):
                    x_new = x + alpha * dx
                    f_new = _objective(x_new)
                    if f_new < f0 - 1e-4 * alpha * np.dot(g, dx):
                        break
                    alpha *= 0.5
                x = x_new
                val, _ = _bound_and_grad(x)
                if val < best_val and np.isfinite(val):
                    best_val = val
                    best_params = x.copy()
            except Exception:
                break
    
    # Polish with Nelder-Mead from best
    if best_params is not None and time.time() < _DEADLINE - 10:
        try:
            res = minimize(_objective, best_params, method='Nelder-Mead',
                          options={'maxiter': 100000, 'xatol': 1e-18, 'fatol': 1e-20, 'adaptive': True})
            if res.fun < best_val and np.isfinite(res.fun):
                best_val = res.fun
                best_params = res.x.copy()
        except Exception:
            pass
    
    # Final high-precision polish with L-BFGS-B
    if best_params is not None and time.time() < _DEADLINE - 5:
        try:
            res = minimize(_objective_grad, best_params, method='L-BFGS-B',
                          jac=True, options={'maxiter': 30000, 'ftol': 1e-18, 'gtol': 1e-16, 'maxls': 50})
            if res.fun < best_val and np.isfinite(res.fun):
                best_val = res.fun
                best_params = res.x.copy()
        except Exception:
            pass
    
    if best_params is None:
        return np.array([-1.13, 0.04, 0.000308642])
    
    return np.array(best_params)
# EVOLVE-BLOCK-END
