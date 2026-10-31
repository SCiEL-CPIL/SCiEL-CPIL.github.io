"""Analytic uncertainty bound with exact gradient + multi-start L-BFGS-B + trust-constr polish."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, minimize_scalar

_DEADLINE = time.time() + 195

# Precompute I_n = integral x^{2n} e^{-x^2} dx = (2n)! / (4^n n!) * sqrt(pi)
def _precompute_I():
    I = np.zeros(14)
    sp = np.sqrt(np.pi)
    I[0] = sp
    for n in range(1, 14):
        I[n] = (2*n - 1) * I[n-1] / (4.0 * n)
    return I

_I = _precompute_I()
_sp = np.sqrt(np.pi)

# Precompute the index matrices for m and n computation
_ar7 = np.arange(7)
_idx_mat = np.add.outer(_ar7, _ar7)  # 7x7 matrix of i+j
_signs = np.array([1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0])


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
    p[0] = 0.0
    p[1] = -6.0*a4 - 420.0*a8 - 7560.0*a12
    p[2] = a4 + 210.0*a8 + 13860.0*a12
    p[3] = -28.0*a8 - 6930.0*a12
    p[4] = a8 + 1155.0*a12
    p[5] = -66.0*a12
    p[6] = a12
    
    I = _I
    
    # m_k = sum_{i,j} p_i p_j I[k+i+j]
    pp = np.outer(p, p)
    m0 = np.sum(pp * I[_idx_mat])
    m2 = np.sum(pp * I[1 + _idx_mat])
    
    # For Fourier: q_k = p_k * (-1)^k
    q = p * signs
    qq = np.outer(q, q)
    n0 = 2.0 * np.pi * np.sum(qq * I[_idx_mat])
    n2 = 2.0 * np.pi * np.sum(qq * I[1 + _idx_mat])
    
    if m0 <= 0 or n0 <= 0 or not np.isfinite(m0) or not np.isfinite(n0):
        return 1e18, np.array([1e18, 1e18, 1e18])
    
    vf = m2 / m0
    vF = n2 / n0
    
    if vf <= 0 or vF <= 0:
        return 1e18, np.array([1e18, 1e18, 1e18])
    
    bound = np.sqrt(vf * vF)
    
    # Compute gradient analytically
    # S_p[k,i] = sum_j p_j I[k+i+j]
    # S_q[k,i] = sum_j q_j I[k+i+j]
    # For k=0: S_p[0,i] = sum_j p_j I[i+j]
    # For k=1: S_p[1,i] = sum_j p_j I[1+i+j]
    
    S_p0 = np.zeros(7)
    S_p1 = np.zeros(7)
    S_q0 = np.zeros(7)
    S_q1 = np.zeros(7)
    for i in range(7):
        idx0 = i + _ar7
        idx1 = 1 + i + _ar7
        S_p0[i] = np.dot(p, I[idx0])
        S_p1[i] = np.dot(p, I[idx1])
        S_q0[i] = np.dot(q, I[idx0])
        S_q1[i] = np.dot(q, I[idx1])
    
    # dp/da matrix: shape (3, 7)
    dpda = np.zeros((3, 7))
    dpda[0, 1] = 8.0
    dpda[1, 1] = -30.0
    dpda[2, 1] = -1260.0
    dpda[0, 2] = -13860.0/945.0
    dpda[1, 2] = 45.0
    dpda[2, 2] = 1750.0
    dpda[0, 3] = 6930.0/945.0
    dpda[1, 3] = -22.0
    dpda[2, 3] = -798.0
    dpda[0, 4] = -1155.0/945.0
    dpda[1, 4] = 3465.0/945.0
    dpda[2, 4] = 1 + 1155.0*105.0/945.0
    dpda[0, 5] = 66.0/945.0
    dpda[1, 5] = -198.0/945.0
    dpda[2, 5] = -6930.0/945.0
    dpda[0, 6] = -1.0/945.0
    dpda[1, 6] = -3.0/945.0
    dpda[2, 6] = -105.0/945.0
    
    grad = np.zeros(3)
    for c in range(3):
        dp = dpda[c]
        
        dm0 = 2.0 * (dp @ S_p0)
        dm2 = 2.0 * (dp @ S_p1)
        
        dq = dp * signs
        dn0 = 4.0 * np.pi * (dq @ S_q0)
        dn2 = 4.0 * np.pi * (dq @ S_q1)
        
        dvm2 = (dm2*m0 - m2*dm0) / (m0*m0)
        dvn2 = (dn2*n0 - n2*dn0) / (n0*n0)
        
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
        np.array([-1.1, 0.042, 0.00032]),
        np.array([-1.15, 0.038, 0.00029]),
        np.array([-1.25, 0.035, 0.00028]),
        np.array([-0.9, 0.045, 0.00035]),
    ]
    
    rng = np.random.default_rng(42)
    for i in range(25):
        starts.append(rng.normal(0, 0.8, 3))
    
    best_val = np.inf
    best_params = None
    
    for start in starts:
        if time.time() > _DEADLINE - 60:
            break
        try:
            res = minimize(_objective_grad, start, method='L-BFGS-B',
                          jac=True, options={'maxiter': 20000, 'ftol': 1e-18, 'gtol': 1e-16, 'maxls': 50})
            if res.fun < best_val and np.isfinite(res.fun):
                best_val = res.fun
                best_params = res.x.copy()
        except Exception:
            pass
    
    # Polish with Nelder-Mead from best
    if best_params is not None and time.time() < _DEADLINE - 40:
        try:
            res = minimize(_objective, best_params, method='Nelder-Mead',
                          options={'maxiter': 200000, 'xatol': 1e-18, 'fatol': 1e-20, 'adaptive': True})
            if res.fun < best_val and np.isfinite(res.fun):
                best_val = res.fun
                best_params = res.x.copy()
        except Exception:
            pass
    
    # Final high-precision polish with L-BFGS-B
    if best_params is not None and time.time() < _DEADLINE - 25:
        try:
            res = minimize(_objective_grad, best_params, method='L-BFGS-B',
                          jac=True, options={'maxiter': 50000, 'ftol': 1e-18, 'gtol': 1e-16, 'maxls': 50})
            if res.fun < best_val and np.isfinite(res.fun):
                best_val = res.fun
                best_params = res.x.copy()
        except Exception:
            pass
    
    # One more polish round if time permits
    if best_params is not None and time.time() < _DEADLINE - 15:
        try:
            res = minimize(_objective, best_params, method='Nelder-Mead',
                          options={'maxiter': 150000, 'xatol': 1e-20, 'fatol': 1e-22, 'adaptive': True})
            if res.fun < best_val and np.isfinite(res.fun):
                best_val = res.fun
                best_params = res.x.copy()
        except Exception:
            pass
    
    # Final ultra-precise L-BFGS-B pass
    if best_params is not None and time.time() < _DEADLINE - 8:
        try:
            res = minimize(_objective_grad, best_params, method='L-BFGS-B',
                          jac=True, options={'maxiter': 100000, 'ftol': 1e-20, 'gtol': 1e-18, 'maxls': 50})
            if res.fun < best_val and np.isfinite(res.fun):
                best_val = res.fun
                best_params = res.x.copy()
        except Exception:
            pass
    
    # Coordinate descent polish for extreme precision
    if best_params is not None and time.time() < _DEADLINE - 15:
        try:
            bp = best_params.copy()
            # Multiple passes with shrinking bounds
            for pass_num in range(3):
                for dim in range(3):
                    def line_obj(t):
                        trial = bp.copy()
                        trial[dim] = t
                        return _objective(trial)
                    # Adaptive bounds: start wide, shrink each pass
                    bound_width = 0.5 * (0.1 ** pass_num)
                    res1d = minimize_scalar(line_obj, bounds=(bp[dim]-bound_width, bp[dim]+bound_width),
                                           method='bounded', options={'xatol': 1e-15})
                    if res1d.fun < best_val and np.isfinite(res1d.fun):
                        best_val = res1d.fun
                        bp[dim] = res1d.x
            best_params = bp
        except Exception:
            pass
    
    # Final ultra-precise 1D scan along each coordinate with very tight bounds
    if best_params is not None and time.time() < _DEADLINE - 5:
        try:
            bp = best_params.copy()
            for dim in range(3):
                def final_line_obj(t):
                    trial = bp.copy()
                    trial[dim] = t
                    return _objective(trial)
                # Very tight final scan
                res_final = minimize_scalar(final_line_obj, bounds=(bp[dim]-1e-6, bp[dim]+1e-6),
                                           method='bounded', options={'xatol': 1e-18, 'maxiter': 10000})
                if res_final.fun < best_val and np.isfinite(res_final.fun):
                    best_val = res_final.fun
                    bp[dim] = res_final.x
            best_params = bp
        except Exception:
            pass
    
    # Newton-Raphson polish: solve grad = 0 with Hessian approximation
    if best_params is not None and time.time() < _DEADLINE - 2:
        try:
            bp = best_params.copy()
            for _ in range(50):
                if time.time() > _DEADLINE - 1:
                    break
                f, g = _objective_grad(bp)
                if not np.isfinite(f) or not np.isfinite(g).all():
                    break
                # Finite-difference Hessian
                h = np.zeros((3, 3))
                eps = 1e-7
                for i in range(3):
                    for j in range(3):
                        if i == j:
                            fp = bp.copy(); fp[i] += eps
                            fm = bp.copy(); fm[i] -= eps
                            h[i, j] = (_objective_grad(fp)[1][i] - _objective_grad(fm)[1][i]) / (2*eps)
                        else:
                            fp = bp.copy(); fp[i] += eps; fp[j] += eps
                            fm = bp.copy(); fm[i] -= eps; fm[j] -= eps
                            h[i, j] = (_objective_grad(fp)[1][i] - _objective_grad(fm)[1][i]) / (2*eps)
                # Damped Newton step
                try:
                    step = np.linalg.solve(h, -g)
                except np.linalg.LinAlgError:
                    step = -0.01 * g
                # Line search to ensure decrease
                alpha = 1.0
                for _ in range(20):
                    trial = bp + alpha * step
                    f_trial, _ = _objective_grad(trial)
                    if np.isfinite(f_trial) and f_trial < f:
                        break
                    alpha *= 0.5
                if alpha < 1e-12:
                    break
                bp_new = bp + alpha * step
                f_new, _ = _objective_grad(bp_new)
                if np.isfinite(f_new) and f_new < best_val:
                    best_val = f_new
                    bp = bp_new
                else:
                    break
            if np.isfinite(best_val):
                best_params = bp
        except Exception:
            pass
    
    if best_params is None:
        return np.array([-1.13, 0.04, 0.000308642])
    
    return np.array(best_params)
# EVOLVE-BLOCK-END
