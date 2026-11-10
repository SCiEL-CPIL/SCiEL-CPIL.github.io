import time
import numpy as np
from scipy.special import lpmv, factorial
from scipy.optimize import minimize, differential_evolution

_DEADLINE = time.time() + 610.0
_K = 10
_NTH = 70
_NPHI = 130


def _build_harmonics():
    """Return Y (nphi*nth, 2k+1), weights (nphi*nth,) for quadrature."""
    k = _K
    mu, w = np.polynomial.legendre.leggauss(_NTH)
    phi = np.linspace(0.0, 2.0 * np.pi, _NPHI, endpoint=False)
    dphi = 2.0 * np.pi / _NPHI

    Y = np.zeros((_NTH, _NPHI, 2 * k + 1), dtype=np.float64)
    for idx, m in enumerate(range(-k, k + 1)):
        am = abs(m)
        N = np.sqrt((2 * k + 1) / (4.0 * np.pi)
                    * float(factorial(k - am, exact=True))
                    / float(factorial(k + am, exact=True)))
        P = lpmv(am, k, mu)
        if m == 0:
            Y[:, :, idx] = N * P[:, None]
        elif m > 0:
            Y[:, :, idx] = np.sqrt(2.0) * N * P[:, None] * np.cos(m * phi)[None, :]
        else:
            Y[:, :, idx] = np.sqrt(2.0) * N * P[:, None] * np.sin(am * phi)[None, :]

    Y_flat = Y.reshape(_NTH * _NPHI, 2 * k + 1)
    w_flat = np.outer(w, np.full(_NPHI, dphi)).ravel()

    return Y_flat, w_flat


def _ratio_val_and_grad(c, Y, w):
    """Compute R^4 = S4/S2^2 and its gradient."""
    f = Y @ c
    f2 = f * f
    f3 = f2 * f
    
    S2 = np.dot(w, f2)
    S4 = np.dot(w, f2 * f2)
    
    if S2 < 1e-15:
        return 1e12, np.zeros_like(c)
    
    val = S4 / (S2 * S2)
    
    Yt_wf3 = Y.T @ (w * f3)
    Yt_wf = Y.T @ (w * f)
    
    dS4 = 4.0 * Yt_wf3
    dS2 = 2.0 * Yt_wf
    
    grad = dS4 / (S2 * S2) - 2.0 * S4 * dS2 / (S2 * S2 * S2)
    
    return val, grad


def _trust_region_sphere(x0, Y, w, max_iter=500, tol=1e-14, time_limit=None):
    """
    Trust-region method on the unit sphere for minimizing R^4.
    Uses a second-order model with projected Newton steps.
    """
    n = len(x0)
    x = x0 / np.linalg.norm(x0)
    f_val, g = _ratio_val_and_grad(x, Y, w)
    
    # Compute Hessian approximation via finite differences of gradient
    # For the ratio S4/S2^2, we need the Hessian of the objective
    # We'll use a quasi-Newton approach with BFGS on the sphere
    
    # BFGS on sphere: maintain B (inverse Hessian approx)
    B = np.eye(n)
    xk = x.copy()
    fk = f_val
    gk = g.copy()
    
    # Project gradient to tangent space
    def project_tangent(v, x):
        return v - np.dot(v, x) * x
    
    g_tan = project_tangent(gk, xk)
    
    # Trust region radius
    delta = 0.5
    
    for it in range(max_iter):
        if time_limit is not None and time.time() > time_limit:
            break
        
        # Solve for step direction: min 0.5*d^T B d + g_tan^T d s.t. ||d|| <= delta
        # and d is in tangent space (approximately, since we project after)
        B_g = B @ g_tan
        # Newton step in tangent space
        d_newton = -B_g
        d_newton = project_tangent(d_newton, xk)
        
        dn = np.linalg.norm(d_newton)
        if dn < 1e-15:
            break
        
        # Scale to trust region
        if dn > delta:
            d = d_newton * (delta / dn)
        else:
            d = d_newton
        
        # Take step and project back to sphere
        x_trial = xk + d
        nrm = np.linalg.norm(x_trial)
        if nrm < 1e-15:
            x_trial = xk
        else:
            x_trial = x_trial / nrm
        
        f_trial, g_trial = _ratio_val_and_grad(x_trial, Y, w)
        
        # Predicted reduction
        pred = np.dot(g_tan, d) + 0.5 * np.dot(d, B @ d)
        actual = fk - f_trial
        
        # Check for sufficient decrease
        if actual > 0.1 * pred and pred < 0:
            # Successful step
            rho = actual / pred if abs(pred) > 1e-15 else 1.0
            # Update BFGS
            s = x_trial - xk
            s = project_tangent(s, (xk + x_trial) / 2)
            gs = g_trial - gk
            gs = project_tangent(gs, (xk + x_trial) / 2)
            
            sgs = np.dot(s, gs)
            if sgs > 1e-15:
                # BFGS update for B (inverse Hessian)
                B = B + (s @ s.T) / sgs - (B @ s @ s.T @ B.T) / (np.dot(s, B @ s))
                # Symmetrize
                B = (B + B.T) / 2.0
            
            xk = x_trial
            fk = f_trial
            gk = g_trial
            g_tan = project_tangent(gk, xk)
            
            # Expand trust region
            delta = min(2.0 * delta, 1.0)
            
            # Convergence check
            if np.linalg.norm(g_tan) < tol:
                break
        else:
            # Unsuccessful step, shrink trust region
            delta *= 0.5
            if delta < 1e-10:
                break
        
        # Safety: ensure we're not diverging
        if not np.isfinite(fk):
            break
    
    return xk, fk


def rudin_problem10() -> np.ndarray:
    Y, w = _build_harmonics()
    k = _K
    n = 2 * k + 1
    
    best_c = np.zeros(n)
    best_val = np.inf

    # 1. Diverse Initial Guesses
    inits = []
    v0 = np.zeros(n); v0[k] = 1.0; inits.append(v0)
    v10 = np.zeros(n); v10[2*k] = 1.0; inits.append(v10)
    v10s = np.zeros(n); v10s[k-10] = 1.0; inits.append(v10s)
    v_u = np.ones(n); inits.append(v_u / np.linalg.norm(v_u))
    
    rng = np.random.default_rng(123)
    for _ in range(15):
        v = rng.standard_normal(n)
        inits.append(v / np.linalg.norm(v))
    
    # 2. Trust-Region Optimization from Each Init
    time_limit_tr = _DEADLINE - 200.0
    for x0 in inits:
        if time.time() > time_limit_tr:
            break
        try:
            c_opt, val_opt = _trust_region_sphere(x0, Y, w, max_iter=500, tol=1e-14, time_limit=time_limit_tr)
            if val_opt < best_val and np.isfinite(val_opt):
                best_val = val_opt
                best_c = c_opt.copy()
        except Exception:
            pass
    
    # 3. Global Optimization: Differential Evolution (if time remains)
    def de_obj(x):
        norm = np.linalg.norm(x)
        if norm < 1e-12:
            return 1e10
        x_norm = x / norm
        val, _ = _ratio_val_and_grad(x_norm, Y, w)
        return val

    bounds = [(-1.0, 1.0)] * n
    
    try:
        if time.time() < _DEADLINE - 100.0:
            res_de = differential_evolution(
                de_obj, bounds, seed=42, maxiter=2000, popsize=25, 
                tol=1e-10, mutation=(0.5, 1.0), recombination=0.9,
                polish=True
            )
            c_de = res_de.x
            norm = np.linalg.norm(c_de)
            if norm > 1e-12:
                c_de_norm = c_de / norm
                val_de, _ = _ratio_val_and_grad(c_de_norm, Y, w)
                if val_de < best_val and np.isfinite(val_de):
                    best_val = val_de
                    best_c = c_de_norm.copy()
    except Exception:
        pass
    
    # 4. Final High-Precision Polish
    if time.time() < _DEADLINE - 10.0:
        def fun_final(c):
            val, grad = _ratio_val_and_grad(c, Y, w)
            return val, grad
        try:
            res_final = minimize(fun_final, best_c, method="L-BFGS-B", jac=True,
                                 options={"maxiter": 5000, "maxls": 60, "ftol": 1e-18, "gtol": 1e-16})
            c_final = res_final.x
            val_final, _ = _ratio_val_and_grad(c_final, Y, w)
            if val_final < best_val and np.isfinite(val_final):
                best_val = val_final
                best_c = c_final.copy()
        except Exception:
            pass
        
        # Additional trust-region polish
        try:
            c_polish, val_polish = _trust_region_sphere(best_c, Y, w, max_iter=1000, tol=1e-16, time_limit=_DEADLINE - 5.0)
            if val_polish < best_val and np.isfinite(val_polish):
                best_val = val_polish
                best_c = c_polish.copy()
        except Exception:
            pass

    # Normalize and validate
    out = best_c.astype(np.float64)
    nrm = np.linalg.norm(out)
    if nrm < 1e-12:
        out = np.ones(n, dtype=np.float64)
    else:
        out = out / nrm

    if out.shape != (21,):
        out = np.ones(21, dtype=np.float64)
    if np.any(np.isnan(out)) or np.any(np.isinf(out)):
        out = np.ones(21, dtype=np.float64)
    
    max_abs = np.max(np.abs(out))
    if max_abs < 1e-3:
        out = out / max_abs if max_abs > 0 else np.ones(21)
    if max_abs > 1e5:
        out = out / max_abs

    return out
