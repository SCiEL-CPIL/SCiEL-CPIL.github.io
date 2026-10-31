"""Analytic Hermite-Gaussian uncertainty bound with generalized eigenvalue + L-BFGS-B polish."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.linalg import eigh

_DEADLINE = time.time() + 200  # 10s safety margin before the 210s kill


def _bound_and_grad(a0, a4, a8):
    """
    Compute the uncertainty product and its gradient with respect to (a0, a4, a8).
    a12 is determined by the constraint g(0) = 0: a12 = -(a0 + 3*a4 + 105*a8)/945.
    Key insight: vF = n2/n0 = 7/3 (constant), so bound = sqrt(7/3 * m2/m0).
    """
    a12 = -(a0 + 3.0 * a4 + 105.0 * a8) / 945.0
    g2 = 2.0 * a0 + 210.0 * a8
    sqrt_pi = np.sqrt(np.pi)

    m0_inner = a0 * a0 + 3.0 * a4 * a4 + 105.0 * a8 * a8 + 945.0 * a12 * a12
    m0 = sqrt_pi * m0_inner

    m2_inner = (a0 * (3.0 * a0 + 12.0 * a4)
                + a4 * (12.0 * a0 + 105.0 * a4 + 840.0 * a8)
                + a8 * (840.0 * a4 + 2940.0 * a8)
                + 11340.0 * a8 * a12)
    m2 = sqrt_pi * m2_inner

    n0 = 18.0 * np.pi * g2 * g2
    n2 = 42.0 * np.pi * g2 * g2

    if m0 <= 0.0 or n0 <= 0.0 or not np.isfinite(m0) or not np.isfinite(n0):
        return 1e18, np.array([0.0, 0.0, 0.0])

    vf = m2 / m0
    vF = n2 / n0

    if vf <= 0.0 or vF <= 0.0:
        return 1e18, np.array([0.0, 0.0, 0.0])

    bound = np.sqrt(vf * vF)

    # Gradient
    dm0_da0 = sqrt_pi * (2.0 * a0 + 2.0 * 945.0 * a12 * (-1.0 / 945.0))
    dm0_da4 = sqrt_pi * (6.0 * a4 + 2.0 * 945.0 * a12 * (-3.0 / 945.0))
    dm0_da8 = sqrt_pi * (210.0 * a8 + 2.0 * 945.0 * a12 * (-105.0 / 945.0))

    dm2_inner_da0 = 6.0 * a0 + 24.0 * a4
    dm2_inner_da4 = 24.0 * a0 + 210.0 * a4 + 1680.0 * a8
    dm2_inner_da8 = 1680.0 * a4 + 4620.0 * a8 + 11340.0 * a12

    dm2_da0 = sqrt_pi * dm2_inner_da0
    dm2_da4 = sqrt_pi * dm2_inner_da4
    dm2_da8 = sqrt_pi * dm2_inner_da8

    dn0_da0 = 72.0 * np.pi * g2
    dn0_da4 = 0.0
    dn0_da8 = 7560.0 * np.pi * g2
    dn2_da0 = 168.0 * np.pi * g2
    dn2_da4 = 0.0
    dn2_da8 = 17640.0 * np.pi * g2

    dvf_da0 = (dm2_da0 * m0 - m2 * dm0_da0) / (m0 * m0)
    dvf_da4 = (dm2_da4 * m0 - m2 * dm0_da4) / (m0 * m0)
    dvf_da8 = (dm2_da8 * m0 - m2 * dm0_da8) / (m0 * m0)
    dvF_da0 = (dn2_da0 * n0 - n2 * dn0_da0) / (n0 * n0)
    dvF_da4 = (dn2_da4 * n0 - n2 * dn0_da4) / (n0 * n0)
    dvF_da8 = (dn2_da8 * n0 - n2 * dn0_da8) / (n0 * n0)

    grad_da0 = 0.5 / bound * (vF * dvf_da0 + vf * dvF_da0)
    grad_da4 = 0.5 / bound * (vF * dvf_da4 + vf * dvF_da4)
    grad_da8 = 0.5 / bound * (vF * dvf_da8 + vf * dvF_da8)

    return bound, np.array([grad_da0, grad_da4, grad_da8])


def _newton_polish(x, max_iter=200):
    """
    Newton-Raphson on the analytic gradient to drive ||grad|| to machine precision.
    Uses finite-difference Hessian with central differences.
    """
    x = np.asarray(x, dtype=np.float64).copy()
    f, g = _bound_and_grad(*x)
    if not np.isfinite(f):
        return x

    for it in range(max_iter):
        gn = np.linalg.norm(g)
        if gn < 1e-14:
            break
        # Central-difference Hessian
        h = 1e-6 * max(1.0, np.linalg.norm(x))
        H = np.zeros((3, 3))
        for i in range(3):
            xp = x.copy(); xp[i] += h
            xm = x.copy(); xm[i] -= h
            _, gp = _bound_and_grad(*xp)
            _, gm = _bound_and_grad(*xm)
            H[i, :] = (gp - gm) / (2.0 * h)
        # Damped Newton step
        try:
            step = np.linalg.solve(H, g)
        except np.linalg.LinAlgError:
            step = -g / gn
        # Line search to ensure descent
        alpha = 1.0
        for _ in range(30):
            x_new = x - alpha * step
            f_new, g_new = _bound_and_grad(*x_new)
            if np.isfinite(f_new) and f_new < f:
                x, f, g = x_new, f_new, g_new
                break
            alpha *= 0.5
        else:
            break
    return x


def _logmag_objective(y):
    """Objective in log-magnitude coordinates: x_i = sign_i * exp(y_i)."""
    x = np.sign(np.array([y[0], y[1], y[2]])) * np.exp(y[3:6])
    # Ensure non-zero signs
    signs = np.sign(x)
    signs[signs == 0] = 1.0
    x = signs * np.exp(y[3:6])
    val, _ = _bound_and_grad(*x)
    return val if np.isfinite(val) else 1e18


def _objective(params):
    val, _ = _bound_and_grad(params[0], params[1], params[2])
    return val if np.isfinite(val) else 1e18


def _objective_grad(params):
    val, grad = _bound_and_grad(params[0], params[1], params[2])
    if not np.isfinite(val):
        return 1e18, np.array([0.0, 0.0, 0.0])
    return val, grad


def construct():
    """
    Key insight: vF = n2/n0 = 7/3 is constant (given g(0)=0 constraint).
    So bound = sqrt(7/3 * m2/m0). Minimizing m2/m0 is a Rayleigh quotient problem.
    Set up symmetric matrices A, B such that m0_inner = x^T A x, m2_inner = x^T B x.
    Solve B v = lambda A v; smallest eigenvalue gives the minimum ratio.
    """
    # Build symmetric matrices A and B
    # m0_inner = a0^2 + 3*a4^2 + 105*a8^2 + 945*a12^2
    # where a12 = -(a0+3*a4+105*a8)/945
    # 945*a12^2 = (a0+3*a4+105*a8)^2 / 945
    # Expanding: (a0^2 + 9*a4^2 + 11025*a8^2 + 6*a0*a4 + 210*a0*a8 + 630*a4*a8)/945
    
    A = np.array([
        [1.0 + 1.0/945.0,       3.0/945.0,       105.0/945.0],
        [3.0/945.0,              3.0 + 9.0/945.0, 315.0/945.0],
        [105.0/945.0,            315.0/945.0,     105.0 + 11025.0/945.0]
    ])
    
    # m2_inner = 3*a0^2 + 24*a0*a4 + 105*a4^2 + 1680*a4*a8 + 2940*a8^2 + 11340*a8*a12
    # 11340*a8*a12 = -12*a8*a0 - 36*a8*a4 - 1260*a8^2
    # m2_inner = 3*a0^2 + 24*a0*a4 + 105*a4^2 + 1644*a4*a8 + 1680*a8^2 - 12*a0*a8
    
    B = np.array([
        [3.0,     12.0,   -6.0],
        [12.0,    105.0,  822.0],
        [-6.0,    822.0,  1680.0]
    ])
    
    # Verify matrix construction with seed
    seed = np.array([-113.0/100, 1.0/25, 1.0/3240.0])
    a0, a4, a8 = seed
    a12 = -(a0 + 3.0*a4 + 105.0*a8) / 945.0
    m0_direct = a0*a0 + 3.0*a4*a4 + 105.0*a8*a8 + 945.0*a12*a12
    m2_direct = (a0*(3.0*a0+12.0*a4) + a4*(12.0*a0+105.0*a4+840.0*a8)
                 + a8*(840.0*a4+2940.0*a8) + 11340.0*a8*a12)
    m0_mat = seed @ A @ seed
    m2_mat = seed @ B @ seed
    
    # Solve generalized eigenvalue problem B v = lambda A v
    best = seed.copy()
    best_val, _ = _bound_and_grad(*seed)
    
    try:
        eigenvalues, eigenvectors = eigh(B, A)
        # Smallest eigenvalue gives minimum of x^T B x / x^T A x
        min_ratio = eigenvalues[0]
        v_min = eigenvectors[:, 0]
        
        # Check g2 != 0
        g2_min = 2.0 * v_min[0] + 210.0 * v_min[2]
        if abs(g2_min) > 1e-12:
            # Scale eigenvector to reasonable magnitude
            norm_v = np.linalg.norm(v_min)
            if norm_v > 1e-15:
                v_scaled = v_min / norm_v * 0.5  # moderate scaling
                val_scaled, _ = _bound_and_grad(*v_scaled)
                if np.isfinite(val_scaled) and val_scaled < best_val:
                    best_val = val_scaled
                    best = v_scaled.copy()
    except Exception:
        pass
    
    # Multi-start L-BFGS-B for robustness
    rng = np.random.default_rng(42)
    starts = [best.copy(), seed.copy()]
    for i in range(20):
        scale = 0.5 / (i + 1)
        starts.append(best + rng.normal(0.0, scale, 3))
    
    for s in starts:
        if time.time() > _DEADLINE - 15:
            break
        try:
            res = minimize(_objective_grad, s, method='L-BFGS-B',
                           options={'maxiter': 10000, 'gtol': 1e-15, 'ftol': 1e-15})
            if res.fun < best_val and np.isfinite(res.fun):
                best_val = res.fun
                best = res.x.copy()
        except Exception:
            pass
    
    # High-precision polish
    if time.time() < _DEADLINE - 5:
        try:
            res = minimize(_objective_grad, best, method='L-BFGS-B',
                           options={'maxiter': 50000, 'gtol': 1e-16, 'ftol': 1e-16})
            if res.fun < best_val and np.isfinite(res.fun):
                best_val = res.fun
                best = res.x.copy()
        except Exception:
            pass

    # Newton-Raphson polish on the analytic gradient
    if time.time() < _DEADLINE - 3:
        try:
            x_newton = _newton_polish(best, max_iter=200)
            val_newton, _ = _bound_and_grad(*x_newton)
            if np.isfinite(val_newton) and val_newton < best_val:
                best_val = val_newton
                best = x_newton.copy()
        except Exception:
            pass

    # Log-magnitude reparameterization search
    if time.time() < _DEADLINE - 10:
        try:
            # Initialize log-magnitude from current best
            x_best = np.asarray(best, dtype=np.float64)
            signs = np.sign(x_best)
            signs[signs == 0] = 1.0
            mags = np.abs(x_best)
            mags[mags < 1e-15] = 1e-15
            y_init = np.concatenate([signs, np.log(mags)])

            res_log = minimize(_logmag_objective, y_init, method='L-BFGS-B',
                               options={'maxiter': 20000, 'gtol': 1e-14, 'ftol': 1e-14})
            if np.isfinite(res_log.fun):
                x_log = np.sign(res_log.x[0:3]) * np.exp(res_log.x[3:6])
                x_log[np.sign(res_log.x[0:3]) == 0] = 1.0 * np.exp(res_log.x[3:6])
                val_log, _ = _bound_and_grad(*x_log)
                if np.isfinite(val_log) and val_log < best_val:
                    best_val = val_log
                    best = x_log.copy()
        except Exception:
            pass

    # Final Newton polish after log-magnitude
    if time.time() < _DEADLINE - 2:
        try:
            x_final = _newton_polish(best, max_iter=100)
            val_final, _ = _bound_and_grad(*x_final)
            if np.isfinite(val_final) and val_final < best_val:
                best_val = val_final
                best = x_final.copy()
        except Exception:
            pass

    return np.array(best)
# EVOLVE-BLOCK-END
