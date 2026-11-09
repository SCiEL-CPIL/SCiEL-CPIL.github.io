# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.special import lpmv
from scipy.optimize import minimize

_K = 10
_NPHI = 130
_NTH = 70
_NPTS = _NTH * _NPHI
_NC = 2 * _K + 1


def _build_quadrature():
    x, w = np.polynomial.legendre.leggauss(_NTH)
    phi = np.linspace(0.0, 2.0 * np.pi, _NPHI, endpoint=False)
    dphi = 2.0 * np.pi / _NPHI
    W = np.outer(w, np.full(_NPHI, dphi))

    Y = np.zeros((_NTH, _NPHI, _NC))
    for idx, m in enumerate(range(-_K, _K + 1)):
        am = abs(m)
        # N_m = sqrt((2K+1)/(4pi) * (K-am)!/(K+am)!)
        denom = 1.0
        for n in range(_K - am + 1, _K + am + 1):
            denom *= n
        fact_ratio = 1.0 / denom
        Nm = np.sqrt((2 * _K + 1) / (4.0 * np.pi) * fact_ratio)
        P = lpmv(am, _K, x)
        if m == 0:
            Y[:, :, idx] = Nm * P[:, None]
        elif m > 0:
            Y[:, :, idx] = np.sqrt(2.0) * Nm * P[:, None] * np.cos(m * phi)[None, :]
        else:
            Y[:, :, idx] = np.sqrt(2.0) * Nm * P[:, None] * np.sin(am * phi)[None, :]
    return Y, W


_Y, _W = _build_quadrature()
_Wflat = _W.ravel()
_Yflat = _Y.reshape(_NPTS, _NC)


def _objective_L4(c):
    """
    Minimize H(c) = sum(W * f^4) subject to ||c||^2 = 1.
    This is equivalent to minimizing R^4 since ||f||_2^2 = 1.
    """
    f = _Yflat @ c
    return float(np.dot(_Wflat, f * f * f * f))


def _gradient_L4(c):
    """
    Gradient of H(c) = sum(W * f^4).
    dH/dc = 4 * sum(W * f^3 * Y_i) = 4 * Y^T (W * f^3)
    """
    f = _Yflat @ c
    return 4.0 * (_Yflat.T @ (_Wflat * f * f * f))


def _objective_unconstrained(c):
    """
    Minimize F(c) = H(c) / ||c||^4, which is scale-invariant.
    This removes the unit sphere constraint.
    """
    norm_sq = float(np.dot(c, c))
    if norm_sq < 1e-30:
        return 1e30
    f = _Yflat @ c
    H = float(np.dot(_Wflat, f * f * f * f))
    return H / (norm_sq * norm_sq)


def _gradient_unconstrained(c):
    """
    Gradient of F(c) = H(c) / ||c||^4.
    Using quotient rule:
    dF = (dH * ||c||^4 - H * d(||c||^4)) / ||c||^8
    dH = 4 * Y^T (W * f^3)
    d(||c||^4) = 4 * ||c||^2 * c
    So dF = 4 * (Y^T(W*f^3) - H*c/||c||^2) / ||c||^4
    """
    norm_sq = float(np.dot(c, c))
    if norm_sq < 1e-30:
        return np.zeros(_NC)
    f = _Yflat @ c
    H = float(np.dot(_Wflat, f * f * f * f))
    dH = 4.0 * (_Yflat.T @ (_Wflat * f * f * f))
    grad = (dH - H * c / norm_sq) / (norm_sq * norm_sq)
    return grad


def _conjugate_gradient_sphere(c0, max_iter=3000, ftol=1e-14):
    """
    Conjugate gradient on the unit sphere with Wolfe line search.
    Uses the Riemannian CG approach: project gradient to tangent space,
    update direction with Polak-Ribiere beta, and use backtracking line search.
    """
    c = np.asarray(c0, dtype=np.float64).copy()
    c /= np.linalg.norm(c)
    
    def fun(c):
        return _objective_L4(c)
    
    def grad(c):
        return _gradient_L4(c)
    
    f_prev = fun(c)
    g_prev = grad(c)
    # Project gradient to tangent space: g_t = g - (g·c)*c
    gt_prev = g_prev - np.dot(g_prev, c) * c
    
    d = -gt_prev.copy()  # initial search direction
    best_c = c.copy()
    best_val = f_prev
    
    for k in range(max_iter):
        if time.time() > _deadline - 5.0:
            break
        
        # Check convergence
        if np.linalg.norm(gt_prev) < ftol:
            break
        
        # Wolfe backtracking line search on the sphere
        alpha = 1.0
        c_new = c + alpha * d
        c_new /= np.linalg.norm(c_new)
        f_new = fun(c_new)
        
        # Wolfe conditions
        c1 = 1e-4  # sufficient decrease
        c2 = 0.9    # curvature
        
        for _ls in range(50):
            c_new = c + alpha * d
            c_new /= np.linalg.norm(c_new)
            f_new = fun(c_new)
            g_new = grad(c_new)
            gt_new = g_new - np.dot(g_new, c_new) * c_new
            
            # Armijo condition
            if f_new <= f_prev + c1 * alpha * np.dot(gt_prev, d):
                # Curvature condition
                if np.dot(gt_new, d) >= c2 * alpha * np.dot(gt_prev, d):
                    break
            alpha *= 0.5
        
        f_new = fun(c_new)
        g_new = grad(c_new)
        gt_new = g_new - np.dot(g_new, c_new) * c_new
        
        if f_new < best_val:
            best_val = f_new
            best_c = c_new.copy()
        
        # Polak-Ribiere beta
        if k > 0:
            beta = np.dot(gt_new, gt_new - gt_prev) / (np.dot(gt_prev, gt_prev) + 1e-30)
            beta = max(0.0, beta)
        else:
            beta = 0.0
        
        d = -gt_new + beta * d
        # Re-orthogonalize direction to tangent space
        d = d - np.dot(d, c_new) * c_new
        
        c = c_new
        f_prev = f_new
        g_prev = g_new
        gt_prev = gt_new
        
        if abs(f_prev - f_new) < ftol * max(1.0, abs(f_prev)):
            break
    
    return best_c, best_val


def _polish_gradient_descent(c0, max_iter=5000, ftol=1e-15):
    """
    High-precision gradient descent on the sphere with adaptive step size.
    Used as a final polish to squeeze out the last bits of improvement.
    """
    c = np.asarray(c0, dtype=np.float64).copy()
    c /= np.linalg.norm(c)
    
    def fun(c):
        return _objective_L4(c)
    
    def grad(c):
        return _gradient_L4(c)
    
    f_prev = fun(c)
    best_c = c.copy()
    best_val = f_prev
    alpha = 0.01
    
    for k in range(max_iter):
        if time.time() > _deadline - 3.0:
            break
        
        g = grad(c)
        gt = g - np.dot(g, c) * c
        grad_norm = np.linalg.norm(gt)
        
        if grad_norm < ftol:
            break
        
        # Adaptive step size with backtracking
        for _ls in range(40):
            c_new = c - alpha * gt
            c_new /= np.linalg.norm(c_new)
            f_new = fun(c_new)
            if f_new < f_prev:
                break
            alpha *= 0.5
        
        f_new = fun(c_new)
        if f_new < best_val:
            best_val = f_new
            best_c = c_new.copy()
        
        # Increase step size if we made good progress
        if f_prev - f_new > 0.5 * alpha * grad_norm * grad_norm:
            alpha = min(alpha * 1.2, 0.1)
        
        c = c_new
        f_prev = f_new
        
        if abs(f_prev - f_new) < ftol * max(1.0, abs(f_prev)):
            break
    
    return best_c, best_val


def _optimize_sphere(c0, deadline):
    global _deadline
    _deadline = deadline
    c0 = np.asarray(c0, dtype=np.float64)
    norm = np.linalg.norm(c0)
    if norm < 1e-12:
        c0 = np.ones(_NC)
        c0 /= np.linalg.norm(c0)
    else:
        c0 = c0 / norm

    def fun(c):
        return _objective_L4(c)

    def jac(c):
        return _gradient_L4(c)

    def cons_fun(c):
        return np.sum(c * c) - 1.0

    def cons_jac(c):
        return 2.0 * c

    constraints = [{'type': 'eq', 'fun': cons_fun, 'jac': cons_jac}]

    best_c = c0.copy()
    best_val = fun(c0)

    # Phase 1: SLSQP for initial optimization
    try:
        res = minimize(fun, c0, jac=jac, method='SLSQP', constraints=constraints,
                       options={'maxiter': 500, 'ftol': 1e-12, 'disp': False})
        if res.fun < best_val:
            best_val = res.fun
            best_c = res.x.copy()
            best_c /= np.linalg.norm(best_c)
    except Exception:
        pass

    # Phase 2: Conjugate gradient on the sphere
    if time.time() < deadline - 30.0:
        try:
            cg_c, cg_val = _conjugate_gradient_sphere(best_c, max_iter=3000, ftol=1e-14)
            if cg_val < best_val:
                best_val = cg_val
                best_c = cg_c.copy()
        except Exception:
            pass

    # Phase 3: Unconstrained L-BFGS-B polish
    if time.time() < deadline - 20.0:
        try:
            res2 = minimize(_objective_unconstrained, best_c, jac=_gradient_unconstrained,
                            method='L-BFGS-B', options={'maxiter': 2000, 'ftol': 1e-14, 'disp': False})
            if res2.fun < best_val:
                nc = res2.x / np.linalg.norm(res2.x)
                val_check = _objective_L4(nc)
                if val_check < best_val:
                    best_val = val_check
                    best_c = nc
        except Exception:
            pass

    # Phase 4: Second CG pass from improved point
    if time.time() < deadline - 15.0:
        try:
            cg_c, cg_val = _conjugate_gradient_sphere(best_c, max_iter=2000, ftol=1e-15)
            if cg_val < best_val:
                best_val = cg_val
                best_c = cg_c.copy()
        except Exception:
            pass

    # Phase 5: Final high-precision gradient descent polish
    if time.time() < deadline - 5.0:
        try:
            gd_c, gd_val = _polish_gradient_descent(best_c, max_iter=5000, ftol=1e-15)
            if gd_val < best_val:
                best_val = gd_val
                best_c = gd_c.copy()
        except Exception:
            pass

    # Phase 6: Final SLSQP polish
    if time.time() < deadline - 3.0:
        try:
            res3 = minimize(fun, best_c, jac=jac, method='SLSQP', constraints=constraints,
                            options={'maxiter': 500, 'ftol': 1e-14, 'disp': False})
            if res3.fun < best_val:
                best_val = res3.fun
                best_c = res3.x.copy()
                best_c /= np.linalg.norm(best_c)
        except Exception:
            pass

    return best_c, best_val


def rudin_problem10() -> np.ndarray:
    start_time = time.time()
    deadline = start_time + 590.0  # Safety margin

    # Initial candidate: All 1s (symmetric)
    c_init = np.ones(_NC)
    c_init /= np.linalg.norm(c_init)
    
    best_c = c_init.copy()
    best_val = _objective_L4(c_init)

    # Try a few other structured starts
    starts = [
        c_init,
        np.array([1 if m % 2 == 0 else 0 for m in range(-_K, _K+1)], dtype=float),
        np.array([1 if m >= 0 else -1 for m in range(-_K, _K+1)], dtype=float),
        np.array([1.0, 0, 0, 0, 0, 0, 1.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1.0]),
        np.array([0, 1.0, 0, 0, 0, 0, 0, 0, 0, 0, 1.0, 0, 0, 0, 0, 1.0, 0]),
        np.random.default_rng(42).standard_normal(_NC),
        np.random.default_rng(123).standard_normal(_NC),
        np.random.default_rng(456).standard_normal(_NC),
        np.random.default_rng(789).standard_normal(_NC),
        np.random.default_rng(101).standard_normal(_NC),
        np.random.default_rng(202).standard_normal(_NC),
        np.random.default_rng(303).standard_normal(_NC),
        np.random.default_rng(404).standard_normal(_NC),
        np.random.default_rng(505).standard_normal(_NC),
        np.random.default_rng(606).standard_normal(_NC),
    ]
    
    # Normalize starts
    for i in range(len(starts)):
        n = np.linalg.norm(starts[i])
        if n > 1e-12:
            starts[i] /= n
        else:
            starts[i] = c_init

    # Allocate time budget per start
    n_starts = len(starts)
    time_per_start = (deadline - start_time - 30.0) / n_starts
    
    for idx, c0 in enumerate(starts):
        remaining = deadline - time.time()
        if remaining < 10.0:
            break
        # Give later starts less time if we're running out
        alloc_time = min(time_per_start, remaining - 10.0)
        if alloc_time < 5.0:
            break
        start_this = time.time()
        try:
            c_opt, val_opt = _optimize_sphere(c0, start_this + alloc_time)
            if val_opt < best_val:
                best_val = val_opt
                best_c = c_opt.copy()
        except Exception:
            continue

    # Final polish: run unconstrained L-BFGS-B from the best found so far
    if time.time() < deadline - 10.0:
        try:
            res_final = minimize(_objective_unconstrained, best_c, jac=_gradient_unconstrained,
                                 method='L-BFGS-B', options={'maxiter': 5000, 'ftol': 1e-15, 'disp': False})
            if res_final.fun < best_val:
                nc = res_final.x / np.linalg.norm(res_final.x)
                val_check = _objective_L4(nc)
                if val_check < best_val:
                    best_val = val_check
                    best_c = nc
        except Exception:
            pass

    # Final CG polish
    if time.time() < deadline - 5.0:
        try:
            cg_c, cg_val = _conjugate_gradient_sphere(best_c, max_iter=2000, ftol=1e-15)
            if cg_val < best_val:
                best_val = cg_val
                best_c = cg_c.copy()
        except Exception:
            pass

    # Final safety check
    best_c = np.asarray(best_c, dtype=np.float64)
    if not np.all(np.isfinite(best_c)):
        best_c = c_init
    mx = float(np.max(np.abs(best_c)))
    if mx < 1e-3:
        best_c = best_c * (1.0 / mx)
    if mx > 1e5:
        best_c = best_c * (1e5 / mx)
    
    return best_c

# EVOLVE-BLOCK-END
