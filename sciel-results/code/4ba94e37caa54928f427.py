"""Program for Erdos minimum overlap task using model-and-optimize approach."""
import numpy as np
import time
from scipy.optimize import minimize

N = 100
DEADLINE = 140.0
START = time.time()

def remaining():
    return DEADLINE - (time.time() - START)

def objective(h):
    """Compute the value to minimize."""
    c = np.correlate(h, 1.0 - h, mode="full")
    return float(np.max(c)) / N * 2.0

def objective_smooth(h, beta=200.0):
    """Smooth surrogate using log-sum-exp for the max."""
    c = np.correlate(h, 1.0 - h, mode="full")
    c_max = np.max(c)
    shifted = beta * (c - c_max)
    log_sum = np.log(np.sum(np.exp(shifted))) + beta * c_max
    return float(log_sum / N * 2.0)

def epigraph_objective(h, beta=200.0):
    """Epigraph-style objective: minimize t subject to c_m <= t, using smooth penalty."""
    c = np.correlate(h, 1.0 - h, mode="full")
    c_max = np.max(c)
    # Soft constraint violation: sum of exp(beta * (c_m - c_max)) encourages all c_m close to c_max
    # The value is still dominated by c_max, but the gradient pushes down other peaks too
    shifted = beta * (c - c_max)
    penalty = np.sum(np.exp(shifted)) / (2 * N - 1)
    return float((c_max + 0.01 * (penalty - 1.0)) / N * 2.0)

def project(h):
    """Project to [0,1]^N with sum = N/2."""
    h = np.asarray(h, dtype=np.float64).copy()
    target = N / 2.0
    
    # Bisection to find the right offset
    lo, hi = -1.0, 1.0
    for _ in range(50):
        mid = (lo + hi) / 2.0
        h_trial = np.clip(h + mid, 0.0, 1.0)
        s = h_trial.sum()
        if s < target:
            lo = mid
        else:
            hi = mid
    h = np.clip(h + (lo + hi) / 2.0, 0.0, 1.0)
    
    # Final fine-tuning
    for _ in range(10):
        s = h.sum()
        if abs(s - target) < 1e-14:
            break
        diff = target - s
        free = (h > 1e-15) & (h < 1.0 - 1e-15)
        if np.any(free):
            h[free] += diff / np.sum(free)
            h = np.clip(h, 0.0, 1.0)
        else:
            break
    return h

def lag_focused_refine(h, max_iter=50):
    """Iteratively identify dominant lag and optimize to reduce it."""
    for _ in range(max_iter):
        if remaining() < 2:
            break
        c = np.correlate(h, 1.0 - h, mode="full")
        lag = np.argmax(c)
        c_max = c[lag]
        
        def obj_weighted(x):
            h_p = project(x)
            c_p = np.correlate(h_p, 1.0 - h_p, mode="full")
            # Weight: emphasize the current dominant lag
            weights = np.ones(2*N-1)
            weights[lag] = 10.0
            # Smooth max with emphasis
            c_shifted = 50.0 * (c_p - c_max)
            log_sum = np.log(np.sum(weights * np.exp(c_shifted))) + 50.0 * c_max
            return float(log_sum / N * 2.0)
        
        try:
            result = minimize(
                obj_weighted,
                h,
                method='L-BFGS-B',
                bounds=[(0.0, 1.0)] * N,
                options={'maxiter': 50, 'ftol': 1e-14, 'gtol': 1e-10}
            )
            h_new = project(result.x)
            val_new = objective(h_new)
            val_old = objective(h)
            if val_new < val_old:
                h = h_new
            else:
                break
        except Exception:
            break
    return h

def epigraph_refine(h, max_iter=30):
    """Refine using epigraph-style objective that balances all lags."""
    for _ in range(max_iter):
        if remaining() < 2:
            break
        try:
            result = minimize(
                lambda x: epigraph_objective(project(x), beta=200.0),
                h,
                method='L-BFGS-B',
                bounds=[(0.0, 1.0)] * N,
                options={'maxiter': 100, 'ftol': 1e-14, 'gtol': 1e-10}
            )
            h_new = project(result.x)
            val_new = objective(h_new)
            val_old = objective(h)
            if val_new < val_old:
                h = h_new
            else:
                break
        except Exception:
            break
    return h

def construct():
    t0 = time.time()
    i = np.arange(N)
    
    best_h = np.full(N, 0.5)
    best_val = objective(best_h)
    
    # Strategy 1: Fourier parameterization with more modes
    for n_modes in range(1, 15):
        if remaining() < 5:
            break
        
        np.random.seed(42 + n_modes)
        params_init = np.random.randn(2 * n_modes) * 0.15
        
        def params_to_h(params):
            h = np.full(N, 0.5)
            for k in range(n_modes):
                a = params[2*k]
                b = params[2*k+1]
                h += a * np.cos(2 * np.pi * k * i / N)
                h += b * np.sin(2 * np.pi * k * i / N)
            return project(h)
        
        def obj_params(params):
            h = params_to_h(params)
            return objective_smooth(h, beta=200.0)
        
        try:
            result = minimize(
                obj_params,
                params_init,
                method='L-BFGS-B',
                options={'maxiter': 150, 'ftol': 1e-14, 'gtol': 1e-10}
            )
            
            h_candidate = params_to_h(result.x)
            val_candidate = objective(h_candidate)
            
            if val_candidate < best_val:
                best_h = h_candidate
                best_val = val_candidate
                
        except Exception:
            pass
    
    # Strategy 2: Targeted multi-start with fewer, better-chosen patterns
    for k in [1, 2, 3, 4, 5, 7, 10, 13, 17, 20]:
        if remaining() < 3:
            break
        
        for A_init in [0.3, 0.4, 0.45, 0.5]:
            if remaining() < 2:
                break
            
            for phase in [0.0, np.pi/2, np.pi/4, 3*np.pi/4]:
                if remaining() < 1:
                    break
                
                h_init = 0.5 + A_init * np.cos(2 * np.pi * k * i / N + phase)
                h_init = project(h_init)
                
                def obj_func(x):
                    return objective_smooth(project(x), beta=200.0)
                
                try:
                    result = minimize(
                        obj_func,
                        h_init,
                        method='L-BFGS-B',
                        bounds=[(0.0, 1.0)] * N,
                        options={'maxiter': 100, 'ftol': 1e-14, 'gtol': 1e-10}
                    )
                    
                    h_candidate = project(result.x)
                    val_candidate = objective(h_candidate)
                    
                    if val_candidate < best_val:
                        best_h = h_candidate
                        best_val = val_candidate
                        
                except Exception:
                    pass
    
    # Strategy 3: Lag-focused refinement on best solution
    if remaining() > 10:
        h_lag = lag_focused_refine(best_h.copy(), max_iter=30)
        val_lag = objective(h_lag)
        if val_lag < best_val:
            best_h = h_lag
            best_val = val_lag
    
    # Strategy 3b: Epigraph-style refinement
    if remaining() > 10:
        h_epi = epigraph_refine(best_h.copy(), max_iter=30)
        val_epi = objective(h_epi)
        if val_epi < best_val:
            best_h = h_epi
            best_val = val_epi
    
    # Strategy 4: Random perturbations around best solution
    if remaining() > 10:
        np.random.seed(123)
        for _ in range(15):
            if remaining() < 3:
                break
            
            scale = 0.02 + 0.04 * np.random.rand()
            h_pert = best_h + scale * np.random.randn(N)
            h_pert = project(h_pert)
            
            def obj_func2(x):
                return objective_smooth(project(x), beta=200.0)
            
            try:
                result = minimize(
                    obj_func2,
                    h_pert,
                    method='L-BFGS-B',
                    bounds=[(0.0, 1.0)] * N,
                    options={'maxiter': 100, 'ftol': 1e-14, 'gtol': 1e-10}
                )
                
                h_candidate = project(result.x)
                val_candidate = objective(h_candidate)
                
                if val_candidate < best_val:
                    best_h = h_candidate
                    best_val = val_candidate
                    
            except Exception:
                pass
    
    # Final polish with multiple refinement passes
    if remaining() > 5:
        for polish_pass in range(3):
            if remaining() < 3:
                break
            
            def obj_final(x):
                return objective_smooth(project(x), beta=200.0)
            
            try:
                result = minimize(
                    obj_final,
                    best_h,
                    method='L-BFGS-B',
                    bounds=[(0.0, 1.0)] * N,
                    options={'maxiter': 150, 'ftol': 1e-15, 'gtol': 1e-12}
                )
                
                h_candidate = project(result.x)
                val_candidate = objective(h_candidate)
                
                if val_candidate < best_val:
                    best_h = h_candidate
                    best_val = val_candidate
                    
            except Exception:
                pass
        
        # Lag-focused final refinement
        if remaining() > 5:
            h_lag2 = lag_focused_refine(best_h.copy(), max_iter=20)
            val_lag2 = objective(h_lag2)
            if val_lag2 < best_val:
                best_h = h_lag2
                best_val = val_lag2
        
        # Epigraph final refinement
        if remaining() > 5:
            h_epi2 = epigraph_refine(best_h.copy(), max_iter=20)
            val_epi2 = objective(h_epi2)
            if val_epi2 < best_val:
                best_h = h_epi2
                best_val = val_epi2
        
        # Exact objective refinement
        if remaining() > 2:
            def obj_exact(x):
                return objective(project(x))
            
            try:
                result = minimize(
                    obj_exact,
                    best_h,
                    method='L-BFGS-B',
                    bounds=[(0.0, 1.0)] * N,
                    options={'maxiter': 80, 'ftol': 1e-15, 'gtol': 1e-12}
                )
                
                h_candidate = project(result.x)
                val_candidate = objective(h_candidate)
                
                if val_candidate < best_val:
                    best_h = h_candidate
                    best_val = val_candidate
                    
            except Exception:
                pass
    
    best_h = project(best_h)
    return best_h
