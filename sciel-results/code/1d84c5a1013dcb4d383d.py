import time
import math
import numpy as np
from scipy.optimize import differential_evolution, minimize

_DEADLINE = 85.0  # Safety margin below the 90s wall-clock kill

def _now():
    return time.time()

def _Om(a, Om0, w0, wa):
    a = np.asarray(a, dtype=float)
    wa_a = 1.0 - a
    w = w0 + wa * wa_a
    x = np.maximum(a, 1e-300)
    num = 1.0 + w
    den = 1.0 + 3.0 * w
    E2 = Om0 * np.power(x, 3.0 * den) + (1.0 - Om0) * np.power(x, -3.0 * num)
    return Om0 * np.power(x, 3.0) / E2

def _f_gr(a, Om0, w0, wa):
    Om = _Om(a, Om0, w0, wa)
    a = np.maximum(np.asarray(a, dtype=float), 1e-300)
    dOm = (3.0 * Om - 1.5 * Om * (w0 + wa * (1.0 - a))) / a
    f = 0.5 * Om + 0.6 * np.sqrt(Om * (1.0 - Om)) + 0.2 * dOm / (3.0 * a)
    return f

def _D_gr(a, Om0, w0, wa, n=2000):
    a = np.asarray(a, dtype=float)
    xs = np.linspace(np.log(1e-3), 0.0, n)
    f = _f_gr(np.exp(xs), Om0, w0, wa)
    I = np.concatenate(([0.0], np.cumsum(0.5 * (f[1:] + f[:-1]) * np.diff(xs))))
    out = np.zeros_like(a)
    for i, av in enumerate(a):
        if av <= 1e-3:
            out[i] = 1.0
            continue
        xv = min(max(math.log(av), xs[0]), 0.0)
        out[i] = math.exp(-np.interp(xv, xs, I))
    return out

def _gamma(c, a, Om0, w0, wa):
    q = 1.0 - _Om(a, Om0, w0, wa)
    u = 1.0 + w0 + wa * (1.0 - a)
    return c[0] + c[1] * q + c[2] * q * q + c[3] * u + c[4] * wa * a + c[5] * u * q

def _f_app(c, a, Om0, w0, wa):
    Om = _Om(a, Om0, w0, wa)
    g = _gamma(c, a, Om0, w0, wa)
    return np.power(np.maximum(Om, 1e-300), g)

def _D_app(c, a, Om0, w0, wa, n=2000):
    a = np.asarray(a, dtype=float)
    xs = np.linspace(np.log(1e-3), 0.0, n)
    fa = _f_app(c, np.exp(xs), Om0, w0, wa)
    I = np.concatenate(([0.0], np.cumsum(0.5 * (fa[1:] + f[:-1]) * np.diff(xs))))
    out = np.zeros_like(a)
    for i, av in enumerate(a):
        if av <= 1e-3:
            out[i] = 1.0
            continue
        xv = min(max(math.log(av), xs[0]), 0.0)
        out[i] = math.exp(-np.interp(xv, xs, I))
    return out

def construct():
    t0 = _now()
    
    # Validation models
    models = [
        (0.30, -1.0, 0.0),
        (0.30, -0.5, 1.0),
        (0.30, -1.0, 0.5),
        (0.27, -1.0, 0.3),
        (0.30, -0.9, 0.0),
    ]
    
    # Evaluation grid: log-spaced from 0.01 to 1.0
    a_eval = np.logspace(-2.0, 0.0, 50)
    a_fit = np.logspace(-2.0, 0.0, 20)

    # Pre-compute reference values
    ref_f = []
    ref_D = []
    for (Om0, w0, wa) in models:
        ref_f.append(_f_gr(a_eval, Om0, w0, wa))
        ref_D.append(_D_gr(a_eval, Om0, w0, wa))

    # Pre-compute gamma constraint grid values for speed (only a-dependent parts)
    # We need to check gamma in [0.1, 1.2] for all models on a_fit
    # gamma = c0 + c1*q + c2*q^2 + c3*u + c4*wa*a + c5*u*q
    # Let's store q, q^2, u, wa*a, u*q for each model on a_fit
    gamma_basis = []
    for (Om0, w0, wa) in models:
        q = 1.0 - _Om(a_fit, Om0, w0, wa)
        u = 1.0 + w0 + wa * (1.0 - a_fit)
        basis = np.column_stack([
            np.ones_like(a_fit), # c0
            q,                   # c1
            q*q,                 # c2
            u,                   # c3
            wa * a_fit,          # c4
            u * q                # c5
        ])
        gamma_basis.append(basis)

    def objective(c):
        if _now() > _DEADLINE:
            return 1e9
            
        # Check bounds
        if np.any(np.abs(c) > 10.0):
            return 1e9
            
        max_err = 0.0
        
        for i, (Om0, w0, wa) in enumerate(models):
            # Check gamma constraint
            g = gamma_basis[i] @ c
            if g.min() < 0.1 or g.max() > 1.2:
                # Penalty for constraint violation
                pen = np.maximum(0.0, 0.1 - g).max() + np.maximum(0.0, g - 1.2).max()
                return 1e6 + 1000.0 * pen

            # Compute f error
            fa = _f_app(c, a_eval, Om0, w0, wa)
            err_f = np.max(np.abs((fa - ref_f[i]) / ref_f[i]))
            max_err = max(max_err, err_f)
            
            # Compute D error
            Da = _D_app(c, a_eval, Om0, w0, wa)
            err_D = np.max(np.abs((Da - ref_D[i]) / ref_D[i]))
            max_err = max(max_err, err_D)
            
        return 100.0 * max_err

    # Initial best
    best_c = np.array([0.55, 0.0, 0.0, 0.0, 0.0, 0.0])
    best_v = objective(best_c)
    
    # Stage 1: Differential Evolution
    bounds = [(-10.0, 10.0)] * 6
    rng = np.random.default_rng(42)
    
    try:
        if _now() < _DEADLINE - 10:
            res = differential_evolution(
                objective, 
                bounds, 
                maxiter=20, 
                popsize=15, 
                tol=1e-5, 
                seed=42,
                polish=False
            )
            if res.fun < best_v:
                best_v = res.fun
                best_c = res.x
    except Exception:
        pass

    # Stage 2: Local Polish
    if _now() < _DEADLINE - 5:
        try:
            res = minimize(
                objective, 
                best_c, 
                method='L-BFGS-B', 
                bounds=bounds,
                options={'maxiter': 100, 'ftol': 1e-8}
            )
            if res.fun < best_v:
                best_v = res.fun
                best_c = res.x
        except Exception:
            pass
        
    # Final validation and clamp
    best_c = np.clip(best_c, -10.0, 10.0)
    # Double check constraint
    valid = True
    for i, (Om0, w0, wa) in enumerate(models):
        g = gamma_basis[i] @ best_c
        if g.min() < 0.1 or g.max() > 1.2:
            valid = False
            break
            
    if not valid:
        # Fallback to seed if constraints violated
        best_c = np.array([0.55, 0.0, 0.0, 0.0, 0.0, 0.0])
        best_v = objective(best_c)

    return [float(x) for x in best_c]
