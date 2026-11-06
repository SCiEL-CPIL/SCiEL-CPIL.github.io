"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.optimize import minimize, differential_evolution
from scipy.integrate import quad

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize, differential_evolution
from scipy.integrate import quad

def _H2_of_a(a, Om0, w0, wa):
    """Hubble parameter squared normalized to H0^2 for CPL dark energy."""
    Om_m = Om0
    Om_de = 1.0 - Om0
    if Om_de < 0:
        Om_de = 1e-10
    a_safe = np.maximum(a, 1e-10)
    term_m = Om_m * a_safe ** (-3)
    exponent = -3.0 * (1.0 + w0 + wa)
    exp_factor = 3.0 * wa * (1.0 / a_safe - 1.0)
    term_de = Om_de * (a_safe ** exponent) * np.exp(exp_factor)
    return term_m + term_de

def _Om_m_of_a(a, Om0, w0, wa):
    """Matter density parameter at scale factor a."""
    H2 = _H2_of_a(a, Om0, w0, wa)
    return Om0 * a ** (-3) / H2

def _cosmology_setup():
    """
    Representative CPL models for local optimization - broad coverage.
    We want to cover the parameter space likely to be in the validation set.
    Om0 in [0.2, 0.4], w0 in [-1.5, 0.0], wa in [-1.0, 1.0].
    """
    models = [
        # Lambda CDM variants
        (0.27, -1.0, 0.0),
        (0.30, -1.0, 0.0),
        (0.32, -1.0, 0.0),
        (0.25, -1.0, 0.0),
        (0.35, -1.0, 0.0),
        # w0 variation with wa=0
        (0.27, -0.9, 0.0),
        (0.27, -1.1, 0.0),
        (0.30, -0.8, 0.0),
        (0.30, -1.2, 0.0),
        (0.28, -0.7, 0.0),
        (0.28, -1.3, 0.0),
        # wa variation with w0=-1
        (0.27, -1.0, 0.3),
        (0.27, -1.0, -0.3),
        (0.30, -1.0, 0.5),
        (0.30, -1.0, -0.5),
        (0.30, -1.0, 0.8),
        (0.30, -1.0, -0.8),
        # Combined w0 and wa (phantom crossing etc)
        (0.25, -0.8, 0.4),
        (0.25, -1.2, 0.6),
        (0.35, -0.9, 0.2),
        (0.35, -1.1, -0.4),
        (0.28, -0.85, 0.35),
        (0.28, -1.15, -0.35),
        # Extreme cases
        (0.20, -1.3, 0.7),
        (0.40, -0.7, 0.3),
        (0.22, -1.4, 0.5),
        (0.38, -1.5, -0.5),
        (0.25, -0.5, 1.0),
        (0.30, -2.0, 1.5),
    ]
    return models

def _precompute_references(models, a_values, num_steps=50000):
    """
    Pre-compute true f(a) and D(a)/D(1) for all models and a values
    by integrating the Riccati ODE for f on a fine log-a grid.
    """
    refs = []
    a_min = float(np.min(a_values))
    # Fine grid in x = ln(a) from 0 down to ln(a_min)
    x_grid = np.linspace(0.0, np.log(a_min), num_steps + 1)
    a_grid = np.exp(x_grid)
    dx = x_grid[1] - x_grid[0]
    
    x_targets = np.log(a_values)
    
    for (Om0, w0, wa) in models:
        # Precompute Om_m on the fine grid
        Om_m_grid = _Om_m_of_a(a_grid, Om0, w0, wa)
        
        # Integrate ODE: df/dx = -f(2+f) + 1.5*Om_m(a)
        f_vals = np.zeros(num_steps + 1)
        lnD_vals = np.zeros(num_steps + 1)
        
        # Boundary condition: f(1) is not exactly 0.5, but we can integrate backwards from a=1.
        # A better way is to integrate from a_min to 1 with a stable initial condition?
        # No, the ODE df/dx = -f(2+f) + 1.5 Om_m is stable for x decreasing (a decreasing)?
        # Let's check stability.
        # If f is large, df/dx is large negative.
        # If we integrate from x=0 (a=1) to x<0 (a<1), we are going "backwards" in time if x is ln a?
        # x = ln a. a goes 1 -> 0.1. x goes 0 -> -2.3.
        # So we integrate from x=0 to x=x_min.
        # The ODE is df/dx = -2f - f^2 + 1.5 Om_m.
        # This is a Riccati equation.
        # For small a, Om_m ~ 1. f ~ 1.
        # df/dx ~ -2(1) - 1 + 1.5 = -1.5.
        # So f decreases as x decreases (a decreases)?
        # If f(1) ~ 0.55, then at a=0.1, f should be closer to 1.
        # Let's check the sign.
        # If f=0.55, Om_m=0.3.
        # df/dx = -0.55(2.55) + 1.5(0.3) = -1.4025 + 0.45 = -0.95.
        # So as x decreases (dx < 0), f increases.
        # So integrating from x=0 to x<0 with dx<0:
        # f_new = f + dx * df/dx. Since dx<0 and df/dx<0, f increases. Correct.
        
        # Initial value f(1).
        # The exact value depends on the model.
        # For LCDM, f(1) ~ 0.55.
        # We can determine f(1) by integrating from a very small a where f~1, up to a=1,
        # and then use the value at a=1?
        # Or just use f(1) = 0.55 as a good approximation and refine?
        # Actually, the error in D is sensitive to the integral of f.
        # The error in f is sensitive to f itself.
        # If we get f(1) slightly wrong, it propagates.
        # Let's integrate from a_min to 1.
        # At a_min, Om_m ~ 1, f ~ 1.
        # The solution f(a) is unique for a given a_min and f(a_min).
        # As a_min -> 0, f(a_min) -> 1.
        # So we can set f(a_min) = 1.0 and integrate forward to a=1.
        # This is more stable because we start with the known asymptotic behavior.
        
        # Let's reverse the grid.
        # x_grid goes from 0 to x_min.
        # We want to integrate from x_min to 0.
        # Let's create a grid from x_min to 0.
        x_grid_fwd = np.linspace(np.log(a_min), 0.0, num_steps + 1)
        a_grid_fwd = np.exp(x_grid_fwd)
        dx_fwd = x_grid_fwd[1] - x_grid_fwd[0]
        
        Om_m_grid_fwd = _Om_m_of_a(a_grid_fwd, Om0, w0, wa)
        
        f_vals_fwd = np.zeros(num_steps + 1)
        lnD_vals_fwd = np.zeros(num_steps + 1)
        
        # Initial condition at a_min
        f_vals_fwd[0] = 1.0
        lnD_vals_fwd[0] = 0.0 # This is just a placeholder, we will normalize later.
        
        for i in range(num_steps):
            x_curr = x_grid_fwd[i]
            Om_m_curr = Om_m_grid_fwd[i]
            f_curr = f_vals_fwd[i]
            
            def df_dx(x, f):
                return -f * (2.0 + f) + 1.5 * Om_m_curr
            
            k1 = df_dx(x_curr, f_curr)
            k2 = df_dx(x_curr + 0.5 * dx_fwd, f_curr + 0.5 * dx_fwd * k1)
            k3 = df_dx(x_curr + 0.5 * dx_fwd, f_curr + 0.5 * dx_fwd * k2)
            k4 = df_dx(x_curr + dx_fwd, f_curr + dx_fwd * k3)
            
            f_vals_fwd[i + 1] = f_curr + (dx_fwd / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
            # lnD derivative is f.
            # d(ln D) / dx = f
            # So d(ln D) = f dx
            lnD_vals_fwd[i + 1] = lnD_vals_fwd[i] + 0.5 * (f_vals_fwd[i] + f_vals_fwd[i + 1]) * dx_fwd
        
        # Now we have f and lnD on x_grid_fwd (from x_min to 0).
        # We need to interpolate to x_targets (which are in [x_min, 0]).
        # x_targets is from np.log(a_values). a_values is [0.1, 1.0].
        # So x_targets is [ln(0.1), 0].
        # x_grid_fwd is [ln(0.1), 0].
        # So we can interpolate directly.
        
        f_true = np.interp(x_targets, x_grid_fwd, f_vals_fwd)
        lnD_true = np.interp(x_targets, x_grid_fwd, lnD_vals_fwd)
        
        # Normalize D such that D(1) = 1.
        # lnD_true[-1] corresponds to x=0 (a=1).
        # We want D(a)/D(1) = exp(lnD(a) - lnD(1)).
        lnD_true_norm = lnD_true - lnD_true[-1]
        D_true = np.exp(lnD_true_norm)
        
        # Precompute Om_m, q, u at target points
        Om_m_targets = _Om_m_of_a(a_values, Om0, w0, wa)
        q_targets = 1.0 - Om_m_targets
        u_targets = 1.0 + w0 + wa * (1.0 - a_values)
        
        refs.append({
            'f_true': f_true,
            'D_true': D_true,
            'Om_m': Om_m_targets,
            'q': q_targets,
            'u': u_targets,
            'w0': w0,
            'wa': wa,
            'a': a_values,
        })
    return refs

def _error_from_coeffs(coeffs, refs):
    """
    Compute 100 * max relative error of f and D over all refs.
    """
    c0, c1, c2, c3, c4, c5 = coeffs
    if not np.all(np.isfinite(coeffs)):
        return 1e10
        
    max_err = 0.0
    
    for ref in refs:
        Om_m = ref['Om_m']
        q = ref['q']
        u = ref['u']
        wa = ref['wa']
        a_vals = ref['a']
        f_true = ref['f_true']
        D_true = ref['D_true']
        
        # Compute gamma
        gamma = c0 + c1 * q + c2 * q**2 + c3 * u + c4 * wa * a_vals + c5 * u * q
        
        # Check bounds and finiteness
        if not np.all(np.isfinite(gamma)):
            return 1e10
        if np.any(gamma < 0.1) or np.any(gamma > 1.2):
            return 1e10
        
        # f approximation
        with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
            f_app = Om_m ** gamma
            
        if not np.all(np.isfinite(f_app)):
            return 1e10
        
        # Error in f
        with np.errstate(divide='ignore', invalid='ignore'):
            err_f = np.abs(f_app - f_true) / np.abs(f_true)
        err_f = np.where(np.isfinite(err_f), err_f, 1e10)
        
        # For D: compute D_app by integrating f_app
        x_vals = np.log(a_vals)
        
        # Trapezoidal cumulative integral from x_vals[0] to x_vals[i]
        # x_vals is sorted increasing (0.1 to 1.0 -> ln(0.1) to 0)
        dx_arr = np.diff(x_vals)
        
        trapz_f = np.zeros(len(x_vals))
        for i in range(1, len(x_vals)):
            trapz_f[i] = trapz_f[i-1] + 0.5 * (f_app[i-1] + f_app[i]) * dx_arr[i-1]
        
        # We want integral from 0 to x_vals[i]
        # integral from x_vals[0] to 0 = trapz_f[-1]
        # integral from x_vals[i] to 0 = trapz_f[-1] - trapz_f[i]
        # integral from 0 to x_vals[i] = -(trapz_f[-1] - trapz_f[i]) = trapz_f[i] - trapz_f[-1]
        integral_f_app = trapz_f - trapz_f[-1]
        
        with np.errstate(over='ignore'):
            D_app = np.exp(integral_f_app)
            
        if not np.all(np.isfinite(D_app)):
            return 1e10
            
        # Error in D
        with np.errstate(divide='ignore', invalid='ignore'):
            err_D = np.abs(D_app - D_true) / np.abs(D_true)
        err_D = np.where(np.isfinite(err_D), err_D, 1e10)
        
        err = np.maximum(err_f, err_D)
        if not np.all(np.isfinite(err)):
            return 1e10
            
        local_max = np.max(err)
        if not np.isfinite(local_max):
            return 1e10
        if local_max > max_err:
            max_err = local_max
    
    return 100.0 * max_err

def _initial_guess_from_refs(refs):
    """
    Derive a better initial guess by fitting gamma to true f values.
    """
    all_q = []
    all_u = []
    all_wa_a = []
    all_uq = []
    all_gamma_true = []
    
    for ref in refs:
        Om_m = ref['Om_m']
        q = ref['q']
        u = ref['u']
        wa = ref['wa']
        a_vals = ref['a']
        f_true = ref['f_true']
        
        valid = (Om_m > 0.01) & (Om_m < 0.99) & (f_true > 0.01)
        if np.sum(valid) < 5:
            continue
        
        gamma_true = np.log(f_true[valid]) / np.log(Om_m[valid])
        
        all_q.append(q[valid])
        all_u.append(u[valid])
        all_wa_a.append(wa * a_vals[valid])
        all_uq.append(u[valid] * q[valid])
        all_gamma_true.append(gamma_true)
    
    if not all_q:
        return np.array([0.55, 0.0, 0.0, 0.0, 0.0, 0.0])
    
    q_all = np.concatenate(all_q)
    u_all = np.concatenate(all_u)
    wa_a_all = np.concatenate(all_wa_a)
    uq_all = np.concatenate(all_uq)
    gamma_all = np.concatenate(all_gamma_true)
    
    A = np.column_stack([
        np.ones_like(q_all),
        q_all,
        q_all**2,
        u_all,
        wa_a_all,
        uq_all
    ])
    
    try:
        coeffs, _, _, _ = np.linalg.lstsq(A, gamma_all, rcond=None)
        coeffs = np.clip(coeffs, -5.0, 5.0)
        return coeffs
    except Exception:
        return np.array([0.55, 0.0, 0.0, 0.0, 0.0, 0.0])

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s kill
    
    # Default safe return value
    safe_coeffs = np.array([0.55, 0.0, 0.0, 0.0, 0.0, 0.0])
    
    try:
        models = _cosmology_setup()
        a_values = np.linspace(0.1, 1.0, 100)
        
        # Precompute references
        # Use a coarser grid for precomputation to save time, then refine?
        # Or just use a moderate grid. 50000 steps * 28 models = 1.4M steps.
        # In Python loop, this might be slow.
        # Let's optimize the precomputation loop.
        # Vectorize the RK4? Hard with the dependency.
        # But we can use a smaller number of steps if we are careful.
        # Or use `num_steps=10000` and check accuracy.
        # Let's try 10000 steps.
        refs = _precompute_references(models, a_values, num_steps=10000)
        
        elapsed = time.time() - start_time
        if elapsed > deadline - 20.0:
            return [float(x) for x in safe_coeffs]
        
        x0 = _initial_guess_from_refs(refs)
        if not np.all(np.isfinite(x0)):
            x0 = safe_coeffs.copy()
        x0 = np.clip(x0, -5.0, 5.0)
        
        best_score = _error_from_coeffs(x0, refs)
        best_params = x0.copy()
        
        seed = safe_coeffs.copy()
        seed_score = _error_from_coeffs(seed, refs)
        if seed_score < best_score:
            best_score = seed_score
            best_params = seed.copy()
        
        elapsed = time.time() - start_time
        if elapsed > deadline - 30.0:
            return [float(x) for x in best_params]
        
        # Use differential evolution
        try:
            bounds = [(-2.0, 2.0), (-2.0, 2.0), (-2.0, 2.0), 
                      (-2.0, 2.0), (-2.0, 2.0), (-2.0, 2.0)]
            
            res_de = differential_evolution(
                _error_from_coeffs,
                bounds,
                args=(refs,),
                maxiter=30,
                popsize=10,
                tol=1e-5,
                seed=42,
                polish=False,
                update_interval=10
            )
            
            if np.all(np.isfinite(res_de.x)) and np.isfinite(res_de.fun):
                if res_de.fun < best_score:
                    best_score = res_de.fun
                    best_params = res_de.x.copy()
        except Exception:
            pass
        
        elapsed = time.time() - start_time
        if elapsed > deadline - 20.0:
            return [float(x) for x in best_params]
        
        # Polish with Nelder-Mead
        start_points = [best_params, x0, seed]
        
        rng = np.random.default_rng(42)
        for _ in range(3):
            perturbation = rng.normal(0, 0.1, 6)
            sp = best_params + perturbation
            if np.all(np.isfinite(sp)) and np.all(np.abs(sp) < 8):
                start_points.append(sp)
        
        for sp in start_points:
            elapsed = time.time() - start_time
            if elapsed > deadline - 15.0:
                break
            
            try:
                res = minimize(
                    _error_from_coeffs,
                    sp,
                    args=(refs,),
                    method='Nelder-Mead',
                    options={'xatol': 1e-6, 'fatol': 1e-7, 'maxiter': 200, 'adaptive': True}
                )
                if np.all(np.isfinite(res.x)) and np.isfinite(res.fun):
                    if res.fun < best_score:
                        best_score = res.fun
                        best_params = res.x.copy()
            except Exception:
                pass
        
        # Final polish
        elapsed = time.time() - start_time
        if elapsed < deadline - 10.0:
            try:
                res2 = minimize(
                    _error_from_coeffs,
                    best_params,
                    args=(refs,),
                    method='Nelder-Mead',
                    options={'xatol': 1e-8, 'fatol': 1e-9, 'maxiter': 100, 'adaptive': True}
                )
                if np.all(np.isfinite(res2.x)) and np.isfinite(res2.fun):
                    if res2.fun < best_score:
                        best_score = res2.fun
                        best_params = res2.x.copy()
            except Exception:
                pass
        
        best_params = np.clip(best_params, -10.0, 10.0)
        if not np.all(np.isfinite(best_params)):
            return [float(x) for x in safe_coeffs]
            
        return [float(x) for x in best_params]
    
    except Exception:
        return [float(x) for x in safe_coeffs]

# EVOLVE-BLOCK-END
