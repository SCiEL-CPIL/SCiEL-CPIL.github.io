"""Compact dark-energy growth-index fit; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
import numpy as np

_DEADLINE = 82.0  # safety margin below the 90 s wall-clock kill


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
    f = 0.5 * Om + 0.6 * np.sqrt(np.maximum(Om * (1.0 - Om), 0.0)) + 0.2 * dOm / (3.0 * a)
    return f


def _D_gr(a, Om0, w0, wa, n=4000):
    a = np.asarray(a, dtype=float)
    xs = np.linspace(np.log(1e-4), 0.0, n)
    f = _f_gr(np.exp(xs), Om0, w0, wa)
    # Use Simpson's rule for better accuracy
    dx = xs[1] - xs[0]
    I = np.zeros(n)
    for i in range(1, n):
        I[i] = I[i-1] + dx * (f[i] + 4*f[i-1] + f[i-2])/6 if i >= 2 else dx * (f[i] + f[i-1])/2
    # Simpler: use cumulative trapezoid with higher n
    I = np.concatenate(([0.0], np.cumsum(0.5 * (f[1:] + f[:-1]) * np.diff(xs))))
    out = np.zeros_like(a)
    for i, av in enumerate(a):
        if av <= 1e-4:
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


def _D_app(c, a, Om0, w0, wa, n=4000):
    a = np.asarray(a, dtype=float)
    xs = np.linspace(np.log(1e-4), 0.0, n)
    fa = _f_app(c, np.exp(xs), Om0, w0, wa)
    I = np.concatenate(([0.0], np.cumsum(0.5 * (fa[1:] + fa[:-1]) * np.diff(xs))))
    out = np.zeros_like(a)
    for i, av in enumerate(a):
        if av <= 1e-4:
            out[i] = 1.0
            continue
        xv = min(max(math.log(av), xs[0]), 0.0)
        out[i] = math.exp(-np.interp(xv, xs, I))
    return out


def _eval(c, models, a_eval, a_fit):
    # Check gamma bounds
    for (Om0, w0, wa) in models:
        g = _gamma(c, a_fit, Om0, w0, wa)
        if np.any(g < 0.1) or np.any(g > 1.2):
            return 1e9
        if not np.all(np.isfinite(g)):
            return 1e9
    
    err_f = 0.0
    err_D = 0.0
    for (Om0, w0, wa) in models:
        fa = _f_app(c, a_eval, Om0, w0, wa)
        fg = _f_gr(a_eval, Om0, w0, wa)
        err_f = max(err_f, float(np.max(np.abs((fa - fg) / np.maximum(fg, 1e-10)))))
        
        Da = _D_app(c, a_eval, Om0, w0, wa)
        Dg = _D_gr(a_eval, Om0, w0, wa)
        err_D = max(err_D, float(np.max(np.abs((Da - Dg) / np.maximum(Dg, 1e-10)))))
    
    return 100.0 * max(err_f, err_D)


def construct():
    t0 = _now()
    # Expanded set of models for robustness as per strategy
    models = [
        (0.30, -1.0, 0.0),
        (0.30, -0.5, 1.0),
        (0.30, -1.0, 0.5),
        (0.27, -1.0, 0.3),
        (0.30, -0.9, 0.0),
        (0.315, -1.0, -0.5),
        (0.25, -1.0, 1.5),
    ]
    a_eval = np.logspace(-2.5, 0.0, 100)
    a_fit = np.logspace(-2.5, 0.0, 80)

    # Pre-compute reference data to speed up objective function
    # We store f_gr and D_gr for each model and a_eval point
    ref_data = []
    for m in models:
        Om0, w0, wa = m
        fg = _f_gr(a_eval, Om0, w0, wa)
        Dg = _D_gr(a_eval, Om0, w0, wa)
        # Store as arrays for vectorized evaluation
        ref_data.append((fg, Dg))

    def _fast_eval(c):
        """Vectorized evaluation against pre-computed references with smooth max."""
        c = np.clip(c, -10.0, 10.0)
        # Check gamma bounds quickly on a subset of a_fit to save time in inner loop
        max_err_f = 0.0
        max_err_D = 0.0
        for i, (Om0, w0, wa) in enumerate(models):
            # Calculate gamma for a_fit to check bounds
            q_fit = 1.0 - _Om(a_fit, Om0, w0, wa)
            u_fit = 1.0 + w0 + wa * (1.0 - a_fit)
            g_fit = c[0] + c[1]*q_fit + c[2]*q_fit**2 + c[3]*u_fit + c[4]*wa*a_fit + c[5]*u_fit*q_fit
            if np.any(g_fit < 0.1) or np.any(g_fit > 1.2):
                return 1e9
            
            # Calculate f_app and D_app for a_eval
            fa = _f_app(c, a_eval, Om0, w0, wa)
            Da = _D_app(c, a_eval, Om0, w0, wa)
            
            fg, Dg = ref_data[i]
            
            # Relative error for f
            err_f = np.abs(fa - fg) / np.maximum(fg, 1e-10)
            # Relative error for D
            err_D = np.abs(Da - Dg) / np.maximum(Dg, 1e-10)
            
            max_err_f = max(max_err_f, float(np.max(err_f)))
            max_err_D = max(max_err_D, float(np.max(err_D)))
        
        return 100.0 * max(max_err_f, max_err_D)

    best_c = np.array([0.55, 0.0, 0.0, 0.0, 0.0, 0.0])
    best_v = _fast_eval(best_c)

    def _try(c):
        nonlocal best_c, best_v
        if not np.all(np.isfinite(c)):
            return
        v = _fast_eval(c)
        if v < best_v:
            best_v = v
            best_c = np.clip(c, -10.0, 10.0)

    from scipy.optimize import differential_evolution, minimize

    # 1. Differential Evolution for global search
    if _now() < _DEADLINE - 40.0:
        try:
            bounds = [(-10.0, 10.0)] * 6
            de_res = differential_evolution(
                _fast_eval,
                bounds,
                maxiter=80,
                popsize=20,
                tol=1e-6,
                seed=42,
                polish=False, # We will polish separately
                init='sobol'
            )
            _try(de_res.x)
        except Exception:
            pass

    # 2. Nelder-Mead polish from best DE result and seed
    start_points = [best_c.copy(), np.array([0.55, 0.0, 0.0, 0.0, 0.0, 0.0])]
    
    # Add more random perturbations around best
    rng = np.random.default_rng(42)
    for _ in range(10):
        start_points.append(np.clip(best_c + rng.normal(0, 0.05, 6), -10, 10))

    for sp in start_points:
        if _now() > _DEADLINE - 10.0:
            break
        try:
            nm_res = minimize(
                _fast_eval,
                sp,
                method="Nelder-Mead",
                options={"maxiter": 500, "xatol": 1e-10, "fatol": 1e-12}
            )
            _try(nm_res.x)
        except Exception:
            pass
    
    # 3. Final high-precision polish with L-BFGS-B
    if _now() < _DEADLINE - 5.0:
        try:
            lbfgs_res = minimize(
                _fast_eval,
                best_c,
                method="L-BFGS-B",
                bounds=[(-10.0, 10.0)] * 6,
                options={"maxiter": 200, "ftol": 1e-15, "gtol": 1e-12}
            )
            _try(lbfgs_res.x)
        except Exception:
            pass

    # Final validation
    best_c = np.clip(best_c, -10.0, 10.0)
    # Double check gamma bounds on a_fit for all models
    for (Om0, w0, wa) in models:
        q_fit = 1.0 - _Om(a_fit, Om0, w0, wa)
        u_fit = 1.0 + w0 + wa * (1.0 - a_fit)
        g_fit = best_c[0] + best_c[1]*q_fit + best_c[2]*q_fit**2 + best_c[3]*u_fit + best_c[4]*wa*a_fit + best_c[5]*u_fit*q_fit
        if np.any(g_fit < 0.1) or np.any(g_fit > 1.2):
            # Fallback to safe seed
            best_c = np.array([0.55, 0.0, 0.0, 0.0, 0.0, 0.0])
            break

    return [float(x) for x in best_c]
# EVOLVE-BLOCK-END
