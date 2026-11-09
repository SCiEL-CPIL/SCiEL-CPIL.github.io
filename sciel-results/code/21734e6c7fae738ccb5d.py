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
    f = _Yflat @ c
    return float(np.dot(_Wflat, f * f * f * f))


def _gradient_L4(c):
    f = _Yflat @ c
    return 4.0 * (_Yflat.T @ (_Wflat * f * f * f))


def _hessian_vec(c, v):
    """Hessian-vector product for the L4 objective on the sphere."""
    f = _Yflat @ c
    Yv = _Yflat @ v
    # H[v] = 4 * Y^T (W * 3*f^2 * Yv)
    Hv = 12.0 * (_Yflat.T @ (_Wflat * f * f * Yv))
    # Project to tangent space: H_tan[v] = Hv - (Hv·c + grad·v)*c
    grad = 4.0 * (_Yflat.T @ (_Wflat * f * f * f))
    Hv_tan = Hv - np.dot(Hv, c) * c - np.dot(grad, v) * c
    return Hv_tan


def _sphere_cg(c, max_iter=10000, tol=1e-16, deadline=None):
    """
    Conjugate gradient on the unit sphere with Wolfe line search.
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

    f = _Yflat @ c
    grad = 4.0 * (_Yflat.T @ (_Wflat * f * f * f))
    grad_tan = grad - np.dot(grad, c) * c

    d = -grad_tan.copy()
    prev_grad_tan = grad_tan.copy()
    prev_grad_norm_sq = float(np.dot(prev_grad_tan, prev_grad_tan))

    for it in range(max_iter):
        if deadline is not None and time.time() > deadline - 3.0:
            break

        # Wolfe line search
        step = 1.0
        f_curr = best_val
        grad_dot_d = float(np.dot(grad_tan, d))
        success = False
        for _ in range(60):
            c_trial = c + step * d
            c_norm = np.linalg.norm(c_trial)
            if c_norm < 1e-12:
                step *= 0.5
                continue
            c_trial = c_trial / c_norm
            f_trial = _objective_L4(c_trial)
            
            # Check Armijo (sufficient decrease)
            if f_trial < f_curr + 1e-4 * step * grad_dot_d:
                # Check curvature condition (Wolfe)
                f_new = _Yflat @ c_trial
                grad_new = 4.0 * (_Yflat.T @ (_Wflat * f_new * f_new * f_new))
                grad_new_tan = grad_new - np.dot(grad_new, c_trial) * c_trial
                if float(np.dot(grad_new_tan, d)) > -0.9 * grad_dot_d:
                    success = True
                    break
                # If curvature fails, try smaller step
            step *= 0.5

        if not success:
            # Fallback: just accept if there was any improvement
            c_trial = c + 0.01 * d
            c_trial /= np.linalg.norm(c_trial)
            f_trial = _objective_L4(c_trial)
            if f_trial < f_curr:
                success = True
            else:
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

        # Fletcher-Reeves update with periodic restart
        if it % 15 == 14:
            d = -grad_tan.copy()
        else:
            curr_grad_norm_sq = float(np.dot(grad_tan, grad_tan))
            beta = curr_grad_norm_sq / max(prev_grad_norm_sq, 1e-30)
            if beta < 0 or beta > 10.0:
                d = -grad_tan.copy()
            else:
                d = -grad_tan + beta * d

        prev_grad_tan = grad_tan.copy()
        prev_grad_norm_sq = curr_grad_norm_sq

    return best_c, best_val


def _newton_polish(c, max_iter=30, tol=1e-15, deadline=None):
    """
    Newton-type polish using Hessian-vector products and conjugate gradient
    to solve the KKT system on the tangent space.
    """
    c = np.asarray(c, dtype=np.float64)
    c = c / np.linalg.norm(c)
    best_val = _objective_L4(c)
    best_c = c.copy()

    for it in range(max_iter):
        if deadline is not None and time.time() > deadline - 2.0:
            break

        f = _Yflat @ c
        grad = 4.0 * (_Yflat.T @ (_Wflat * f * f * f))
        grad_tan = grad - np.dot(grad, c) * c
        gnorm = np.linalg.norm(grad_tan)
        if gnorm < tol:
            break

        # Build tangent basis (simple: use Gram-Schmidt on standard basis)
        # For efficiency, use CG to solve H_tan * d = -grad_tan
        # H_tan[v] = H[v] - (H[v]·c + grad·v)*c
        
        # Use CG on the Hessian in tangent space
        # Start with d0 = -grad_tan
        r = -grad_tan.copy()
        p = r.copy()
        rs = float(np.dot(r, r))
        
        for cg_iter in range(50):
            Hp = _hessian_vec(c, p)
            pnorm = float(np.dot(p, p))
            if pnorm < 1e-30:
                break
            alpha = rs / (pnorm + 1e-10)  # Regularize
            d_trial = c + alpha * p
            d_trial /= np.linalg.norm(d_trial)
            
            # Check if this Newton step improves
            f_trial = _objective_L4(d_trial)
            if f_trial < best_val:
                c = d_trial
                best_val = f_trial
                best_c = c.copy()
                break
            
            # Update CG
            r_new = _hessian_vec(c, p) + alpha * grad_tan  # Approximate
            r_new_tan = r_new - np.dot(r_new, c) * c
            rs_new = float(np.dot(r_new_tan, r_new_tan))
            if rs_new < 1e-30:
                break
            beta = rs_new / max(rs, 1e-30)
            p = r_new_tan + beta * p
            r = r_new_tan
            rs = rs_new
        
        # If CG didn't find improvement, try damped gradient step
        if _objective_L4(c) >= best_val:
            c = best_c
            step = 0.01
            for _ in range(20):
                c_trial = c - step * grad_tan
                c_trial /= np.linalg.norm(c_trial)
                f_trial = _objective_L4(c_trial)
                if f_trial < best_val:
                    c = c_trial
                    best_val = f_trial
                    best_c = c.copy()
                    break
                step *= 0.5

    return best_c, best_val


def _optimize_from_start(c0, deadline):
    """Full optimization pipeline from a single starting point."""
    c0 = np.asarray(c0, dtype=np.float64)
    norm = np.linalg.norm(c0)
    if norm < 1e-12:
        c0 = np.ones(_NC)
        c0 /= np.linalg.norm(c0)
    else:
        c0 = c0 / norm

    best_c = c0.copy()
    best_val = _objective_L4(c0)

    # Stage 1: SLSQP for good initial convergence
    if time.time() < deadline - 30.0:
        try:
            def fun(c):
                return _objective_L4(c)
            def jac(c):
                return _gradient_L4(c)
            def cons_fun(c):
                return np.sum(c * c) - 1.0
            def cons_jac(c):
                return 2.0 * c
            constraints = [{'type': 'eq', 'fun': cons_fun, 'jac': cons_jac}]
            res = minimize(fun, c0, jac=jac, method='SLSQP', constraints=constraints,
                           options={'maxiter': 300, 'ftol': 1e-14, 'disp': False})
            if res.fun < best_val:
                best_val = res.fun
                best_c = res.x / np.linalg.norm(res.x)
        except Exception:
            pass

    # Stage 2: Sphere CG (main optimizer)
    if time.time() < deadline - 20.0:
        try:
            c_cg, val_cg = _sphere_cg(best_c, max_iter=8000, tol=1e-16, deadline=deadline)
            if val_cg < best_val:
                best_val = val_cg
                best_c = c_cg
        except Exception:
            pass

    # Stage 3: Newton polish
    if time.time() < deadline - 10.0:
        try:
            c_newton, val_newton = _newton_polish(best_c, max_iter=20, tol=1e-15, deadline=deadline)
            if val_newton < best_val:
                best_val = val_newton
                best_c = c_newton
        except Exception:
            pass

    # Stage 4: Final CG pass
    if time.time() < deadline - 5.0:
        try:
            c_cg2, val_cg2 = _sphere_cg(best_c, max_iter=5000, tol=1e-17, deadline=deadline)
            if val_cg2 < best_val:
                best_val = val_cg2
                best_c = c_cg2
        except Exception:
            pass

    # Stage 5: Final Newton polish
    if time.time() < deadline - 3.0:
        try:
            c_np2, val_np2 = _newton_polish(best_c, max_iter=15, tol=1e-16, deadline=deadline)
            if val_np2 < best_val:
                best_val = val_np2
                best_c = c_np2
        except Exception:
            pass

    return best_c, best_val


def rudin_problem10() -> np.ndarray:
    start_time = time.time()
    deadline = start_time + 580.0  # Safety margin

    rng = np.random.default_rng(42)

    starts = []

    # Symmetric: all ones
    c1 = np.ones(_NC)
    c1 /= np.linalg.norm(c1)
    starts.append(c1)

    # Even m only
    c2 = np.array([1.0 if (i - _K) % 2 == 0 else 0.0 for i in range(_NC)])
    n2 = np.linalg.norm(c2)
    if n2 > 1e-12:
        c2 /= n2
    starts.append(c2)

    # Odd m only
    c3 = np.array([1.0 if (i - _K) % 2 != 0 else 0.0 for i in range(_NC)])
    n3 = np.linalg.norm(c3)
    if n3 > 1e-12:
        c3 /= n3
    starts.append(c3)

    # m >= 0 positive
    c4 = np.array([1.0 if m >= 0 else -1.0 for m in range(-_K, _K + 1)])
    c4 /= np.linalg.norm(c4)
    starts.append(c4)

    # Structured: cosine weighted
    c5 = np.array([np.cos(np.pi * (m + _K) / (2 * _K)) for m in range(-_K, _K + 1)])
    n5 = np.linalg.norm(c5)
    if n5 > 1e-12:
        c5 /= n5
    starts.append(c5)

    # Structured: m^2 weighted
    c6 = np.array([float(m * m) for m in range(-_K, _K + 1)])
    n6 = np.linalg.norm(c6)
    if n6 > 1e-12:
        c6 /= n6
    starts.append(c6)

    # Extremal: m=0, ±10
    c7 = np.zeros(_NC)
    c7[_K] = 1.0
    c7[0] = 1.0
    c7[_NC - 1] = 1.0
    c7 /= np.linalg.norm(c7)
    starts.append(c7)

    # m=0, ±5, ±10
    c8 = np.zeros(_NC)
    c8[_K] = 1.0
    c8[_K - 5] = 1.0
    c8[_K + 5] = 1.0
    c8[0] = 1.0
    c8[_NC - 1] = 1.0
    c8 /= np.linalg.norm(c8)
    starts.append(c8)

    # Random starts
    for seed in [1, 7, 13, 29, 47, 61, 79, 97]:
        cr = rng.standard_normal(_NC)
        cr /= np.linalg.norm(cr)
        starts.append(cr)

    best_c = starts[0].copy()
    best_val = _objective_L4(best_c)

    # Optimize from each starting point
    for i, c0 in enumerate(starts):
        if time.time() > deadline - 30.0:
            break
        try:
            c_opt, val_opt = _optimize_from_start(c0, deadline)
            if val_opt < best_val:
                best_val = val_opt
                best_c = c_opt.copy()
        except Exception:
            continue

    # Final polish from best
    if time.time() < deadline - 5.0:
        try:
            c_final, val_final = _sphere_cg(best_c, max_iter=10000, tol=1e-17, deadline=deadline)
            if val_final < best_val:
                best_val = val_final
                best_c = c_final
        except Exception:
            pass

    if time.time() < deadline - 3.0:
        try:
            c_np, val_np = _newton_polish(best_c, max_iter=20, tol=1e-16, deadline=deadline)
            if val_np < best_val:
                best_val = val_np
                best_c = c_np
        except Exception:
            pass

    # Safety check
    best_c = np.asarray(best_c, dtype=np.float64)
    if not np.all(np.isfinite(best_c)):
        best_c = starts[0]
    mx = float(np.max(np.abs(best_c)))
    if mx < 1e-3:
        best_c = best_c * (1.0 / mx)
    if mx > 1e5:
        best_c = best_c * (1e5 / mx)

    return best_c

# EVOLVE-BLOCK-END
