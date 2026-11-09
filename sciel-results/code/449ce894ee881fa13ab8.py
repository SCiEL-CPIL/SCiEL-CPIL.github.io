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
    dF = 4 * (Y^T(W*f^3) - H*c/||c||^2) / ||c||^4
    """
    norm_sq = float(np.dot(c, c))
    if norm_sq < 1e-30:
        return np.zeros(_NC)
    f = _Yflat @ c
    H = float(np.dot(_Wflat, f * f * f * f))
    dH = 4.0 * (_Yflat.T @ (_Wflat * f * f * f))
    grad = (dH - H * c / norm_sq) / (norm_sq * norm_sq)
    return grad


def _polish_gradient_descent(c, max_iter=500, tol=1e-15):
    """
    High-precision polish using projected gradient descent with momentum
    on the sphere. Uses backtracking line search for stability.
    """
    c = np.asarray(c, dtype=np.float64)
    c = c / np.linalg.norm(c)
    best_val = _objective_L4(c)
    best_c = c.copy()
    
    # Momentum-based approach
    momentum = np.zeros(_NC)
    beta_mom = 0.9
    
    for it in range(max_iter):
        f = _Yflat @ c
        grad = 4.0 * (_Yflat.T @ (_Wflat * f * f * f))
        # Project gradient onto tangent space
        grad_tan = grad - np.dot(grad, c) * c
        gnorm = np.linalg.norm(grad_tan)
        if gnorm < tol:
            break
        
        # Update momentum
        momentum = beta_mom * momentum - grad_tan
        
        # Backtracking line search along momentum direction
        step = 1.0
        f_curr = best_val
        success = False
        for _ in range(40):
            c_trial = c - step * momentum
            c_trial_norm = np.linalg.norm(c_trial)
            if c_trial_norm < 1e-12:
                step *= 0.5
                continue
            c_trial = c_trial / c_trial_norm
            f_trial = _objective_L4(c_trial)
            if f_trial < f_curr - 1e-12 * step * np.dot(grad_tan, momentum):
                success = True
                break
            step *= 0.5
        
        if not success:
            # Try pure gradient step
            step = 1.0
            for _ in range(30):
                c_trial = c - step * grad_tan
                c_trial = c_trial / np.linalg.norm(c_trial)
                f_trial = _objective_L4(c_trial)
                if f_trial < f_curr - 1e-15 * step * gnorm * gnorm:
                    success = True
                    break
                step *= 0.5
            if not success:
                break
        
        c = c_trial
        if f_trial < best_val:
            best_val = f_trial
            best_c = c.copy()
        else:
            break
    
    return best_c, best_val


def _hessian_polish(c, max_iter=50, tol=1e-16):
    """
    Newton-type polish using the Hessian of the Lagrangian projected
    onto the tangent space. Quadratic convergence near the optimum.
    """
    c = np.asarray(c, dtype=np.float64)
    c = c / np.linalg.norm(c)
    best_val = _objective_L4(c)
    best_c = c.copy()
    
    for it in range(max_iter):
        f = _Yflat @ c
        f2 = f * f
        f3 = f2 * f
        f4 = f2 * f2
        
        # Gradient: 4 * Y^T (W * f^3)
        grad = 4.0 * (_Yflat.T @ (_Wflat * f3))
        grad_tan = grad - np.dot(grad, c) * c
        gnorm = np.linalg.norm(grad_tan)
        if gnorm < tol:
            break
        
        # Hessian of H(c) = sum(W * f^4): 12 * (Y^T diag(W*f^2) Y)
        # Project onto tangent space: H_tan = P * H * P where P = I - c c^T
        Wf2 = _Wflat * f2
        H = 12.0 * (_Yflat.T @ (Wf2[:, None] * _Yflat))
        
        # Project Hessian onto tangent space
        P = np.eye(_NC) - np.outer(c, c)
        H_tan = P @ H @ P
        
        # Add Lagrange multiplier for constraint: lambda * I (approximate)
        # lambda = c^T H c / 2 (from constraint d/dc (c^T c - 1) = 2c)
        # Actually, the full Lagrangian Hessian is H + 2*lambda*I
        # At optimum, lambda = c^T grad (since grad_tan = 0 implies grad = lambda*c)
        lam = np.dot(c, grad)
        H_tan += 2.0 * lam * P  # Projected version
        
        # Solve H_tan * delta = -grad_tan
        try:
            delta = np.linalg.solve(H_tan + 1e-10 * np.eye(_NC), -grad_tan)
        except np.linalg.LinAlgError:
            # Fallback to gradient step
            delta = -grad_tan / (gnorm + 1e-30)
        
        # Line search along delta
        step = 1.0
        f_curr = best_val
        success = False
        for _ in range(30):
            c_trial = c + step * delta
            c_trial_norm = np.linalg.norm(c_trial)
            if c_trial_norm < 1e-12:
                step *= 0.5
                continue
            c_trial = c_trial / c_trial_norm
            f_trial = _objective_L4(c_trial)
            if f_trial < f_curr - 1e-14 * step * np.dot(grad_tan, delta):
                success = True
                break
            step *= 0.5
        
        if not success:
            # Fallback: gradient step
            c_trial = c - grad_tan / (gnorm + 1e-30)
            c_trial = c_trial / np.linalg.norm(c_trial)
            f_trial = _objective_L4(c_trial)
            if f_trial >= f_curr:
                break
        
        c = c_trial
        if f_trial < best_val:
            best_val = f_trial
            best_c = c.copy()
        else:
            break
    
    return best_c, best_val


def _conjugate_gradient_sphere(c, max_iter=1000, tol=1e-16):
    """
    Conjugate gradient method on the unit sphere using the Fletcher-Reeves variant
    with Wolfe line search conditions for better convergence.
    """
    c = np.asarray(c, dtype=np.float64)
    norm = np.linalg.norm(c)
    if norm < 1e-12:
        c = np.ones(_NC)
        c /= np.linalg.norm(c)
    else:
        c = c / norm
    
    best_val = _objective_L4(c)
    best_c = c.copy()
    
    # Compute initial gradient on sphere
    f = _Yflat @ c
    grad = 4.0 * (_Yflat.T @ (_Wflat * f * f * f))
    grad_tan = grad - np.dot(grad, c) * c
    
    d = -grad_tan.copy()
    prev_grad_tan = grad_tan.copy()
    prev_grad_norm_sq = np.dot(prev_grad_tan, prev_grad_tan)
    
    for it in range(max_iter):
        # Wolfe line search along direction d
        step = 1.0
        f_curr = best_val
        success = False
        
        for _ in range(60):
            c_trial = c + step * d
            c_trial_norm = np.linalg.norm(c_trial)
            if c_trial_norm < 1e-12:
                step *= 0.5
                continue
            c_trial = c_trial / c_trial_norm
            f_trial = _objective_L4(c_trial)
            
            # Check sufficient decrease (Armijo condition)
            grad_dot_d = np.dot(grad_tan, d)
            if f_trial < f_curr + 1e-4 * step * grad_dot_d:
                # Check curvature condition
                f_new = _Yflat @ c_trial
                grad_new = 4.0 * (_Yflat.T @ (_Wflat * f_new * f_new * f_new))
                grad_new_tan = grad_new - np.dot(grad_new, c_trial) * c_trial
                if np.dot(grad_new_tan, d) > -0.9 * step * grad_dot_d:
                    success = True
                    break
                else:
                    # Strong Wolfe not satisfied, try smaller step
                    step *= 0.5
                    continue
            else:
                step *= 0.5
        
        if not success:
            # Fallback: simple backtracking
            step = 1.0
            for _ in range(40):
                c_trial = c + step * d
                c_trial = c_trial / np.linalg.norm(c_trial)
                f_trial = _objective_L4(c_trial)
                if f_trial < f_curr - 1e-14 * step * np.dot(grad_tan, d):
                    success = True
                    break
                step *= 0.5
            if not success:
                break
        
        c = c_trial
        if f_trial < best_val:
            best_val = f_trial
            best_c = c.copy()
        
        # Compute new gradient
        f = _Yflat @ c
        grad = 4.0 * (_Yflat.T @ (_Wflat * f * f * f))
        grad_tan = grad - np.dot(grad, c) * c
        
        gnorm = np.linalg.norm(grad_tan)
        if gnorm < tol:
            break
        
        # Fletcher-Reeves update with restart if needed
        curr_grad_norm_sq = np.dot(grad_tan, grad_tan)
        if curr_grad_norm_sq <= 1e-10 * prev_grad_norm_sq:
            beta = 0.0
        else:
            beta = curr_grad_norm_sq / max(prev_grad_norm_sq, 1e-30)
            if beta > 1.0:
                beta = 1.0
        
        d = -grad_tan + beta * d
        prev_grad_tan = grad_tan.copy()
        prev_grad_norm_sq = curr_grad_norm_sq
    
    return best_c, best_val


def _optimize_sphere(c0, deadline):
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

    # SLSQP with tighter tolerances
    try:
        res = minimize(fun, c0, jac=jac, method='SLSQP', constraints=constraints,
                       options={'maxiter': 1000, 'ftol': 1e-14, 'disp': False})
        if res.fun < best_val:
            best_val = res.fun
            best_c = res.x.copy()
            best_c /= np.linalg.norm(best_c)
    except Exception:
        pass

    # Polish with unconstrained L-BFGS-B
    if time.time() < deadline - 10.0:
        try:
            c_uncon = best_c.copy()
            res2 = minimize(_objective_unconstrained, c_uncon, jac=_gradient_unconstrained,
                            method='L-BFGS-B', options={'maxiter': 5000, 'ftol': 1e-18, 'gtol': 1e-14, 'disp': False})
            if res2.fun < best_val:
                best_val = res2.fun
                best_c = res2.x.copy()
                best_c /= np.linalg.norm(best_c)
        except Exception:
            pass

    # Conjugate gradient on sphere
    if time.time() < deadline - 10.0:
        try:
            c_cg, val_cg = _conjugate_gradient_sphere(best_c, max_iter=3000, tol=1e-18)
            if val_cg < best_val:
                best_val = val_cg
                best_c = c_cg.copy()
        except Exception:
            pass

    # Hessian-based Newton polish for quadratic convergence
    if time.time() < deadline - 5.0:
        try:
            c_newton, val_newton = _hessian_polish(best_c, max_iter=100, tol=1e-18)
            if val_newton < best_val:
                best_val = val_newton
                best_c = c_newton.copy()
        except Exception:
            pass

    # Final high-precision gradient descent polish
    if time.time() < deadline - 3.0:
        try:
            c_polished, val_polished = _polish_gradient_descent(best_c, max_iter=10000, tol=1e-18)
            if val_polished < best_val:
                best_val = val_polished
                best_c = c_polished.copy()
        except Exception:
            pass

    # Second Hessian polish after gradient polish
    if time.time() < deadline - 2.0:
        try:
            c_newton2, val_newton2 = _hessian_polish(best_c, max_iter=50, tol=1e-18)
            if val_newton2 < best_val:
                best_val = val_newton2
                best_c = c_newton2.copy()
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

    # Build diverse set of starting points
    rng = np.random.default_rng(42)
    starts = [
        c_init,
        np.array([1 if m % 2 == 0 else 0 for m in range(-_K, _K+1)], dtype=float),
        np.array([1 if m >= 0 else -1 for m in range(-_K, _K+1)], dtype=float),
        np.array([1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0,
                  -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0], dtype=float),
        np.array([1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0,
                  0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0], dtype=float),
        np.array([0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0,
                  1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0], dtype=float),
    ]
    # Add random starts with different seeds
    for seed in [123, 456, 789, 101, 202, 303, 404, 505, 606, 707, 808, 909]:
        starts.append(rng.standard_normal(_NC))
    
    # Add structured random starts
    for seed in [111, 222, 333, 444, 555]:
        c_rand = rng.standard_normal(_NC)
        # Make it more structured by zeroing some components
        mask = rng.random(_NC) > 0.5
        c_rand[mask] = 0.0
        starts.append(c_rand)
    
    # Add single-harmonic starts and their combinations
    for m_idx in [0, 1, 2, 5, 10]:  # m = -10, -9, -8, -5, 0
        c_single = np.zeros(_NC)
        c_single[m_idx] = 1.0
        starts.append(c_single)
    
    # Add pairs of harmonics
    for m1, m2 in [(0, 10), (0, 5), (1, 10), (5, 10), (0, 15)]:
        if m1 < _NC and m2 < _NC:
            c_pair = np.zeros(_NC)
            c_pair[m1] = 1.0
            c_pair[m2] = 1.0
            starts.append(c_pair)
    
    # Normalize starts
    for i in range(len(starts)):
        n = np.linalg.norm(starts[i])
        if n > 1e-12:
            starts[i] = starts[i] / n
        else:
            starts[i] = c_init.copy()

    for c0 in starts:
        if time.time() > deadline - 15.0:
            break
        try:
            c_opt, val_opt = _optimize_sphere(c0, deadline)
            if val_opt < best_val:
                best_val = val_opt
                best_c = c_opt.copy()
        except Exception:
            continue

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
