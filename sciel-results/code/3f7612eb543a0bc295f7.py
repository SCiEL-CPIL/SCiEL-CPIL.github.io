"""Compact dark-energy growth-index fit (CPL cosmologies)."""
# EVOLVE-BLOCK-START
import time
import math
import numpy as np
from scipy.optimize import minimize, linprog
from scipy.integrate import solve_ivp

_DEADLINE = 85.0  # safety margin under 90s wall-clock
_START = None

def _elapsed():
    global _START
    if _START is None:
        _START = time.time()
    return time.time() - _START

def _time_left():
    return _DEADLINE - _elapsed()

def _Om(a, Om0, w0, wa):
    a = np.asarray(a, dtype=float)
    Om_m = Om0 * a**(-3.0)
    Om_de = (1.0 - Om0) * np.exp((1.0 + w0 + wa) * np.log(a) - wa * (a - 1.0))
    return Om_m + Om_de

def _Hprime_over_H(a, Om0, w0, wa):
    a = np.asarray(a, dtype=float)
    Om_m = Om0 * a**(-3.0)
    Om_de = (1.0 - Om0) * np.exp((1.0 + w0 + wa) * np.log(a) - wa * (a - 1.0))
    Om_tot = Om_m + Om_de
    dOm_dx = -3.0 * Om_m + Om_de * ((1.0 + w0 + wa) - wa * a)
    return 0.5 * dOm_dx / Om_tot

def _f_ode(x, y, Om0, w0, wa):
    a = math.exp(x)
    Om = _Om(np.array([a]), Om0, w0, wa)[0]
    H = _Hprime_over_H(np.array([a]), Om0, w0, wa)[0]
    f = y[0]
    return [(1.5) * Om - f * (1.5 + H) - f * f]

def _reference(Om0, w0, wa, a_grid):
    """Return f_ref, D_ref, logD_ref on a_grid (ascending)."""
    x_grid = np.log(a_grid)
    x_min = x_grid[0]
    f0 = 0.05
    sol = solve_ivp(_f_ode, [x_min, 0.0], [f0], args=(Om0, w0, wa),
                    t_eval=x_grid, method='RK45', rtol=1e-9, atol=1e-12, max_step=0.01)
    f_ref = sol.y[0]
    x = x_grid
    f = f_ref
    dx_f = np.diff(x)
    trap_f = 0.5 * (f[1:] + f[:-1]) * dx_f
    cum_f = np.concatenate([[0.0], np.cumsum(trap_f)])
    logD = cum_f - cum_f[-1]
    D_ref = np.exp(logD)
    return f_ref, D_ref, logD

def _candidate(Om0, w0, wa, a_grid, c):
    a = np.asarray(a_grid, dtype=float)
    c0, c1, c2, c3, c4, c5 = c
    Om = _Om(a, Om0, w0, wa)
    q = 1.0 - Om
    u = 1.0 + w0 + wa * (1.0 - a)
    gamma = c0 + c1 * q + c2 * q**2 + c3 * u + c4 * wa * a + c5 * u * q
    f = Om**gamma
    x = np.log(a)
    dx = np.diff(x)
    trap = 0.5 * (f[1:] + f[:-1]) * dx
    cum = np.concatenate([[0.0], np.cumsum(trap)])
    logD = cum - cum[-1]
    D = np.exp(logD)
    return f, D, logD, gamma

def _model_grid():
    Om0 = np.array([0.20, 0.25, 0.30, 0.35])
    w0 = np.array([-1.5, -1.0, -0.5, 0.0])
    wa = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])
    models = []
    for o in Om0:
        for w in w0:
            for wv in wa:
                models.append((o, w, wv))
    return models

def _build_references(models, a_grid):
    refs = {}
    for (o, w, wv) in models:
        refs[(o, w, wv)] = _reference(o, w, wv, a_grid)
    return refs

def _exact_max(c, models, a_grid, refs):
    c = np.clip(c, -10.0, 10.0)
    worst = 0.0
    gv = 0.0
    for (o, w, wv) in models:
        f_ref, D_ref, logD_ref = refs[(o, w, wv)]
        f_c, D_c, logD_c, gamma = _candidate(o, w, wv, a_grid, c)
        rel_f = np.abs(f_c - f_ref) / np.maximum(np.abs(f_ref), 1e-6)
        rel_D = np.abs(D_c - D_ref) / np.maximum(np.abs(D_ref), 1e-6)
        worst = max(worst, rel_f.max(), rel_D.max())
        gv += np.sum(np.maximum(0.0, gamma - 1.2)) + np.sum(np.maximum(0.0, 0.1 - gamma))
    return 100.0 * worst + 5000.0 * gv

def _build_lp_data(models, a_grid, refs, c0_guess):
    """Build A, b, c for LP: minimize t s.t. |A_c c + b_c| <= t, c in [-10,10].
    We linearize: error in ln f ≈ (gamma - gamma_ref) * ln Om, and error in ln D ≈ integral of that.
    Actually, let's use a different approach: directly compute the feature matrix for gamma,
    then the error in f is |Om^gamma - Om^gamma_ref|, which is nonlinear.
    
    Better approach: use the fact that for small perturbations, 
    ln f = gamma * ln Om, so delta_ln_f = delta_gamma * ln Om.
    And delta_ln_D = integral of delta_ln_f dx (approximately, since D = exp(integral f dx)
    and f = Om^gamma, so ln D = integral Om^gamma dx, which is nonlinear in gamma).
    
    Let me use a simpler linearization: work in the space of ln f and ln D directly.
    For the LP, I'll approximate the error as linear in c by evaluating at c0_guess.
    """
    n_models = len(models)
    n_a = len(a_grid)
    total_points = n_models * n_a
    
    # Feature matrix for gamma: each row is [1, q, q^2, u, wa*a, u*q]
    # For each model and each a
    gamma_features = np.zeros((total_points, 6))
    lnOm = np.zeros(total_points)
    lnOm_ref = np.zeros(total_points)
    gamma_ref = np.zeros(total_points)
    
    # Precompute reference values
    f_ref_all = np.zeros(total_points)
    logD_ref_all = np.zeros(total_points)
    
    idx = 0
    for mi, (o, w, wv) in enumerate(models):
        f_ref, D_ref, logD_ref = refs[(o, w, wv)]
        a = a_grid
        Om = _Om(a, o, w, wv)
        q = 1.0 - Om
        u = 1.0 + w + wv * (1.0 - a)
        gamma_ref_vals = None  # we need gamma_ref from the ODE solution
        
        for ai in range(n_a):
            gamma_features[idx, 0] = 1.0
            gamma_features[idx, 1] = q[ai]
            gamma_features[idx, 2] = q[ai]**2
            gamma_features[idx, 3] = u[ai]
            gamma_features[idx, 4] = wv * a[ai]
            gamma_features[idx, 5] = u[ai] * q[ai]
            lnOm[idx] = np.log(Om[ai])
            f_ref_all[idx] = f_ref[ai]
            logD_ref_all[idx] = logD_ref[ai]
            idx += 1
    
    # For linearization, we need gamma_ref. Estimate it from f_ref = Om^gamma_ref
    # gamma_ref = ln(f_ref) / ln(Om)
    safe_lnOm = np.where(np.abs(lnOm) > 1e-10, lnOm, 1e-10)
    gamma_ref_est = np.where(np.abs(f_ref_all) > 1e-10, np.log(np.maximum(f_ref_all, 1e-10)) / safe_lnOm, 0.55)
    gamma_ref_est = np.clip(gamma_ref_est, 0.1, 1.2)
    
    # Linearize: delta_ln_f = (gamma - gamma_ref) * lnOm = (G c - gamma_ref) * lnOm
    # where G c is the gamma prediction. So delta_ln_f = lnOm * G c - lnOm * gamma_ref
    # The error in ln f is: |lnOm * G c - lnOm * gamma_ref|
    # For ln D: ln D = integral f dx, f = Om^gamma. Linearize: 
    # delta_ln_D ≈ integral (f_ref * lnOm * delta_gamma) dx
    # where delta_gamma = G c - gamma_ref
    
    # Build the linear system. The error in ln f at point i is:
    # e_i = lnOm[i] * (G[i,:] c - gamma_ref[i])
    # = (lnOm[i] * G[i,:]) c - lnOm[i] * gamma_ref[i]
    
    A_lnF = gamma_features * lnOm[:, None]  # (total_points, 6)
    b_lnF = -lnOm * gamma_ref_est
    
    # For ln D: we need the cumulative integral effect.
    # delta_ln_D_i ≈ sum_{j=0}^{i} 0.5 * (f_ref[j]*lnOm[j] + f_ref[j+1]*lnOm[j+1]) * dx * delta_gamma_j
    # This is complex. Let's approximate: delta_ln_D_i ≈ integral_0^{x_i} f_ref * lnOm * delta_gamma dx
    # We can build a matrix B such that delta_ln_D = B @ c + b_D
    
    # Build B for ln D error
    B_lnD = np.zeros((total_points, 6))
    b_lnD = np.zeros(total_points)
    
    idx = 0
    for mi, (o, w, wv) in enumerate(models):
        f_ref, D_ref, logD_ref = refs[(o, w, wv)]
        a = a_grid
        x = np.log(a)
        Om = _Om(a, o, w, wv)
        q = 1.0 - Om
        u = 1.0 + w + wv * (1.0 - a)
        
        # Compute gamma_ref for this model
        safe_lnOm_m = np.where(np.abs(np.log(Om)) > 1e-10, np.log(Om), 1e-10)
        gamma_ref_m = np.where(np.abs(f_ref) > 1e-10, np.log(np.maximum(f_ref, 1e-10)) / safe_lnOm_m, 0.55)
        gamma_ref_m = np.clip(gamma_ref_m, 0.1, 1.2)
        
        # delta_gamma = G c - gamma_ref_m
        # delta_ln_D_i = integral_0^{x_i} f_ref * lnOm * delta_gamma dx
        # = integral_0^{x_i} f_ref * lnOm * (G c - gamma_ref_m) dx
        # = [integral_0^{x_i} f_ref * lnOm * G dx] c - integral_0^{x_i} f_ref * lnOm * gamma_ref_m dx
        
        # Build the integrand: w(x) = f_ref(x) * lnOm(x) * G(x)
        # and the constant: w_const(x) = f_ref(x) * lnOm(x) * gamma_ref_m(x)
        
        lnOm_m = np.log(Om)
        weight = f_ref * lnOm_m  # (n_a,)
        
        # For each a_i, we need integral from x_min to x_i of weight * G c dx
        # and integral from x_min to x_i of weight * gamma_ref_m dx
        
        # G(x) = [1, q, q^2, u, wa*a, u*q]
        # So the integrand for coefficient k is weight * G_k
        # We can compute the cumulative integral of weight * G_k for each k
        
        # Build matrix: for each a_i, the row of B_lnD is the cumulative integral of weight * G up to x_i
        # Similarly for b_lnD
        
        # Compute cumulative trapezoid integrals
        dx_arr = np.diff(x)
        
        for k in range(6):
            integrand_k = weight * gamma_features[idx:idx+n_a, k]  # but gamma_features is global
            # Actually, let me recompute G for this model
            G_k = np.zeros(n_a)
            if k == 0:
                G_k[:] = 1.0
            elif k == 1:
                G_k[:] = q
            elif k == 2:
                G_k[:] = q**2
            elif k == 3:
                G_k[:] = u
            elif k == 4:
                G_k[:] = wv * a
            elif k == 5:
                G_k[:] = u * q
            
            integrand_k = weight * G_k
            # Cumulative trapezoid from x_min to x_i
            trap_k = 0.5 * (integrand_k[1:] + integrand_k[:-1]) * dx_arr
            cum_k = np.concatenate([[0.0], np.cumsum(trap_k)])
            # We want integral from x_min to x_i, which is cum_k[i]
            # But we want integral from 0 to x_i = cum_k[i] - cum_k[-1] (since x[-1]=0)
            # Wait, x is ascending from x_min to 0. So integral from x_min to x_i is cum_k[i].
            # integral from 0 to x_i = - integral from x_i to 0 = -(cum_k[-1] - cum_k[i]) = cum_k[i] - cum_k[-1]
            B_lnD[idx:idx+n_a, k] = cum_k - cum_k[-1]
        
        # b_lnD: integral of weight * gamma_ref_m
        integrand_const = weight * gamma_ref_m
        trap_const = 0.5 * (integrand_const[1:] + integrand_const[:-1]) * dx_arr
        cum_const = np.concatenate([[0.0], np.cumsum(trap_const)])
        b_lnD[idx:idx+n_a] = -(cum_const - cum_const[-1])  # negative because we want integral_0^{x_i}
        
        idx += n_a
    
    # Now we have:
    # error_ln_f = A_lnF c + b_lnF
    # error_ln_D = B_lnD c + b_lnD
    
    # The relative error in f is approximately |error_ln_f| (for small errors)
    # The relative error in D is approximately |error_ln_D|
    
    # LP: minimize t
    # s.t. error_ln_f <= t, -error_ln_f <= t
    #      error_ln_D <= t, -error_ln_D <= t
    #      -10 <= c_k <= 10
    
    # Total constraints: 2*total_points (for f) + 2*total_points (for D) + 12 (for bounds)
    # Variables: c (6) + t (1) = 7
    
    n_vars = 7
    # c vector for linprog: minimize c_vec @ x
    c_vec = np.zeros(n_vars)
    c_vec[-1] = 1.0  # minimize t
    
    # Inequality constraints: A_ub @ x <= b_ub
    # For error_ln_f: A_lnF c + b_lnF <= t  =>  A_lnF c - t <= -b_lnF
    # -error_ln_f: -A_lnF c - b_lnF <= t  =>  -A_lnF c - t <= b_lnF
    
    A_ub_parts = []
    b_ub_parts = []
    
    # error_ln_f <= t
    A_f_pos = np.hstack([A_lnF, -np.ones((total_points, 1))])
    A_ub_parts.append(A_f_pos)
    b_ub_parts.append(-b_lnF)
    
    # -error_ln_f <= t
    A_f_neg = np.hstack([-A_lnF, -np.ones((total_points, 1))])
    A_ub_parts.append(A_f_neg)
    b_ub_parts.append(b_lnF)
    
    # error_ln_D <= t
    A_D_pos = np.hstack([B_lnD, -np.ones((total_points, 1))])
    A_ub_parts.append(A_D_pos)
    b_ub_parts.append(-b_lnD)
    
    # -error_ln_D <= t
    A_D_neg = np.hstack([-B_lnD, -np.ones((total_points, 1))])
    A_ub_parts.append(A_D_neg)
    b_ub_parts.append(b_lnD)
    
    A_ub = np.vstack(A_ub_parts)
    b_ub = np.concatenate(b_ub_parts)
    
    # Bounds: -10 <= c_k <= 10, t >= 0
    bounds = [(-10, 10)] * 6 + [(0, None)]
    
    return A_ub, b_ub, c_vec, bounds

def _fit_lp(models, a_grid, refs):
    """Fit using LP-based Chebyshev approximation."""
    c_guess = np.array([0.55, 0.0, 0.0, 0.0, 0.0, 0.0])
    
    A_ub, b_ub, c_vec, bounds = _build_lp_data(models, a_grid, refs, c_guess)
    
    best_c = c_guess.copy()
    best_val = _exact_max(best_c, models, a_grid, refs)
    
    try:
        res = linprog(c_vec, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        if res.success and res.x is not None:
            c_lp = res.x[:6]
            c_lp = np.clip(c_lp, -10, 10)
            v = _exact_max(c_lp, models, a_grid, refs)
            if v < best_val:
                best_val, best_c = v, c_lp
    except Exception:
        pass
    
    # Polish with Nelder-Mead on the exact objective
    if _time_left() > 15.0:
        try:
            r = minimize(lambda cc: _exact_max(cc, models, a_grid, refs), best_c,
                        method='Nelder-Mead', options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-10})
            c = np.clip(r.x, -10, 10)
            v = _exact_max(c, models, a_grid, refs)
            if v < best_val:
                best_val, best_c = v, c
        except Exception:
            pass
    
    # Additional restarts
    if _time_left() > 10.0:
        rng = np.random.default_rng(42)
        for _ in range(5):
            if _time_left() < 5.0:
                break
            c_rand = best_c + rng.normal(0, 0.1, 6)
            c_rand = np.clip(c_rand, -10, 10)
            try:
                r = minimize(lambda cc: _exact_max(cc, models, a_grid, refs), c_rand,
                            method='Nelder-Mead', options={'maxiter': 300, 'xatol': 1e-8, 'fatol': 1e-10})
                c = np.clip(r.x, -10, 10)
                v = _exact_max(c, models, a_grid, refs)
                if v < best_val:
                    best_val, best_c = v, c
            except Exception:
                pass
    
    return best_c, best_val

def construct():
    global _START
    _START = time.time()
    models = _model_grid()
    a_grid = np.linspace(0.05, 1.0, 150)
    refs = _build_references(models, a_grid)
    c, val = _fit_lp(models, a_grid, refs)
    c = np.clip(np.asarray(c, dtype=float), -10.0, 10.0)
    # verify gamma range
    a_chk = np.linspace(0.02, 1.0, 400)
    ok = True
    for (o, w, wv) in models:
        _, _, _, gamma = _candidate(o, w, wv, a_chk, c)
        if gamma.min() < 0.1 - 1e-9 or gamma.max() > 1.2 + 1e-9:
            ok = False
            break
    if not ok:
        c = np.array([0.55, 0.0, 0.0, 0.0, 0.0, 0.0])
    return [float(x) for x in c]
# EVOLVE-BLOCK-END
