import numpy as np
from scipy.optimize import minimize, differential_evolution
import time

N = 100
TARGET_SUM = N / 2.0
DEADLINE = 135.0  # safety margin before 150s kill

def _correlation_max(h):
    """Compute the raw value: max correlation / N * 2."""
    corr = np.correlate(h, 1.0 - h, mode='full')
    return np.max(corr) / N * 2.0

def _corr_vector(h):
    """Compute all lags of correlation."""
    return np.correlate(h, 1.0 - h, mode='full')

def _project_feasible(h):
    """Project h to satisfy 0 <= h <= 1 and sum(h) = N/2."""
    h = np.asarray(h, dtype=float).copy()
    h = np.clip(h, 0.0, 1.0)
    for _ in range(500):
        diff = TARGET_SUM - np.sum(h)
        if abs(diff) < 1e-14:
            break
        mask = (h > 1e-12) & (h < 1.0 - 1e-12)
        n_free = np.sum(mask)
        if n_free == 0:
            break
        h[mask] += diff / n_free
        h = np.clip(h, 0.0, 1.0)
    return h

def _objective(h):
    """Objective to minimize: the raw value."""
    return _correlation_max(h)

def _epigraph_objective(params):
    """
    Epigraph formulation: minimize t subject to corr_lag <= t for all lags.
    params = [h_0, ..., h_99, t]
    """
    h = params[:N]
    t = params[N]
    corr = _corr_vector(h)
    # Penalize violations: sum of max(0, corr_lag - t)^2
    violations = np.maximum(0, corr - t)
    return t + 1000.0 * np.sum(violations ** 2)

def _epigraph_grad(params):
    """Gradient of epigraph objective."""
    h = params[:N]
    t = params[N]
    corr = _corr_vector(h)
    violations = np.maximum(0, corr - t)
    
    # d/dt = 1 - 2000 * sum(violations)
    dt = 1.0 - 2000.0 * np.sum(violations)
    
    # d/dh: need gradient of corr w.r.t. h
    # corr[lag] = sum_i h[i] * (1-h[i+lag]) for lag >= 0
    # This is complex; use finite difference for now
    dh = np.zeros(N)
    eps = 1e-7
    for i in range(N):
        h_plus = h.copy()
        h_plus[i] += eps
        h_minus = h.copy()
        h_minus[i] -= eps
        corr_plus = _corr_vector(h_plus)
        corr_minus = _corr_vector(h_minus)
        dc = (corr_plus - corr_minus) / (2 * eps)
        dh[i] = np.sum(dc * 2000.0 * violations)
    
    return np.concatenate([dh, [dt]])

def _parametric_h(params):
    """
    Generate h from parametric model: sum of 3 sinusoids + constant.
    params = [c, a1, p1, a2, p2, a3, p3]
    c: constant offset
    a_i: amplitude of sinusoid i
    p_i: phase of sinusoid i
    """
    c = params[0]
    a1, p1 = params[1], params[2]
    a2, p2 = params[3], params[4]
    a3, p3 = params[5], params[6]
    
    t = np.arange(N)
    h = c + a1 * np.sin(2 * np.pi * t / N + p1) + \
             a2 * np.sin(4 * np.pi * t / N + p2) + \
             a3 * np.sin(6 * np.pi * t / N + p3)
    return _project_feasible(h)

def _parametric_objective(params):
    """Objective for parametric model."""
    h = _parametric_h(params)
    return _correlation_max(h)

def _coordinate_descent(h, max_iters=500, tol=1e-12):
    """
    Coordinate descent on individual h[i] values.
    For each i, optimize h[i] given all others fixed.
    """
    h = _project_feasible(h).copy()
    bounds_i = (0.0, 1.0)
    
    for it in range(max_iters):
        improved = False
        for i in range(N):
            # Fix all other coordinates, optimize h[i]
            # corr depends on h[i] in multiple lags
            # Use scipy minimize_scalar on 1D
            def obj_1d(x_i):
                h_temp = h.copy()
                h_temp[i] = x_i
                # Ensure sum constraint by adjusting one other element
                # Simple: just evaluate correlation
                return _correlation_max(h_temp)
            
            try:
                from scipy.optimize import minimize_scalar
                res = minimize_scalar(obj_1d, bounds=bounds_i, method='bounded',
                                     options={'xatol': tol})
                if res.success and res.fun < _correlation_max(h) - tol:
                    h[i] = res.x
                    h = _project_feasible(h)
                    improved = True
            except Exception:
                pass
        
        if not improved:
            break
    
    return h

def construct():
    start_time = time.time()
    
    best_h = np.full(N, 0.5)
    best_val = _correlation_max(best_h)
    
    # Phase 1: Parametric optimization (7-dim)
    if time.time() - start_time < DEADLINE - 60:
        param_bounds = [
            (0.3, 0.7),   # c
            (0.0, 0.5),   # a1
            (0, 2*np.pi), # p1
            (0.0, 0.5),   # a2
            (0, 2*np.pi), # p2
            (0.0, 0.5),   # a3
            (0, 2*np.pi), # p3
        ]
        
        # Multiple random starts for parametric
        rng = np.random.RandomState(42)
        param_starts = []
        for _ in range(30):
            p = np.array([
                rng.uniform(0.4, 0.6),
                rng.uniform(0.1, 0.45), rng.uniform(0, 2*np.pi),
                rng.uniform(0.05, 0.4), rng.uniform(0, 2*np.pi),
                rng.uniform(0.02, 0.3), rng.uniform(0, 2*np.pi),
            ])
            param_starts.append(p)
        
        # Also add structured starts
        for freq1, freq2, freq3 in [(1,2,3), (1,3,5), (2,3,5), (1,2,5)]:
            for amp_scale in [0.3, 0.4, 0.45]:
                p = np.array([
                    0.5,
                    amp_scale, 0,
                    amp_scale * 0.7, np.pi/2,
                    amp_scale * 0.4, np.pi,
                ])
                param_starts.append(p)
        
        for p0 in param_starts:
            if time.time() - start_time > DEADLINE - 50:
                break
            try:
                res = minimize(
                    _parametric_objective,
                    p0,
                    method='Nelder-Mead',
                    bounds=param_bounds,
                    options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-12}
                )
                h_res = _parametric_h(res.x)
                val = _correlation_max(h_res)
                if val < best_val - 1e-10:
                    best_val = val
                    best_h = h_res.copy()
            except Exception:
                pass
    
    # Phase 2: Full 100-dim SLSQP with smooth objective from best parametric solution
    if time.time() - start_time < DEADLINE - 30:
        def _smooth_objective(h, beta=0.05):
            corr = _corr_vector(h)
            x_max = np.max(corr)
            shifted = corr - x_max
            ls = x_max + beta * np.log(np.sum(np.exp(shifted / beta)))
            return ls / N * 2.0
        
        bounds = [(0.0, 1.0) for _ in range(N)]
        constraint = {'type': 'eq', 'fun': lambda x: np.sum(x) - TARGET_SUM, 'jac': lambda x: np.ones(N)}
        
        # Start from best parametric solution
        h0 = best_h.copy()
        try:
            res = minimize(
                lambda x: _smooth_objective(x, 0.05),
                h0,
                method='SLSQP',
                bounds=bounds,
                constraints=[constraint],
                options={'maxiter': 1000, 'ftol': 1e-10, 'disp': False}
            )
            h_res = _project_feasible(res.x)
            val = _correlation_max(h_res)
            if val < best_val - 1e-10:
                best_val = val
                best_h = h_res.copy()
        except Exception:
            pass
        
        # Also try from alternating patterns
        for base in [0.2, 0.3, 0.4]:
            if time.time() - start_time > DEADLINE - 20:
                break
            alt = np.where(np.arange(N) % 2 == 0, base, 1.0 - base)
            h_alt = _project_feasible(alt)
            try:
                res = minimize(
                    lambda x: _smooth_objective(x, 0.05),
                    h_alt,
                    method='SLSQP',
                    bounds=bounds,
                    constraints=[constraint],
                    options={'maxiter': 500, 'ftol': 1e-10, 'disp': False}
                )
                h_res = _project_feasible(res.x)
                val = _correlation_max(h_res)
                if val < best_val - 1e-10:
                    best_val = val
                    best_h = h_res.copy()
            except Exception:
                pass
    
    # Phase 3: Coordinate descent polish
    if time.time() - start_time < DEADLINE - 10:
        h_cd = _coordinate_descent(best_h, max_iters=200, tol=1e-12)
        val_cd = _correlation_max(h_cd)
        if val_cd < best_val - 1e-12:
            best_val = val_cd
            best_h = h_cd.copy()
    
    # Phase 4: Final SLSQP with exact objective
    if time.time() - start_time < DEADLINE - 5:
        try:
            res = minimize(
                _objective,
                best_h,
                method='SLSQP',
                bounds=bounds,
                constraints=[constraint],
                options={'maxiter': 2000, 'ftol': 1e-14, 'disp': False}
            )
            h_polish = _project_feasible(res.x)
            val_polish = _correlation_max(h_polish)
            if val_polish < best_val - 1e-12:
                best_val = val_polish
                best_h = h_polish.copy()
        except Exception:
            pass
    
    # Ensure final validity
    best_h = _project_feasible(best_h)
    
    return best_h
