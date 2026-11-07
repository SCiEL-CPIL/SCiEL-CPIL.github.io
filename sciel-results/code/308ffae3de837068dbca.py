import numpy as np
import time
from scipy.optimize import minimize

N = 50
TIME_LIMIT = 115.0  # safety margin before 120s

def compute_value(h):
    """Compute the objective value for given heights h."""
    h = np.asarray(h, dtype=float)
    g = np.convolve(h, h)
    m = len(g)
    w = 1.0 / (m + 1)  # spacing from linspace(-0.5, 0.5, m+2)
    
    # Build y = [0, g[0], ..., g[m-1], 0]
    y = np.zeros(m + 2)
    y[1:m+1] = g
    
    # l2 = sum over i in 0..m of (w/3) * (y[i]^2 + y[i]*y[i+1] + y[i+1]^2)
    l2 = (w / 3.0) * np.sum(y[:m+1]**2 + y[:m+1]*y[1:m+2] + y[1:m+2]**2)
    
    norm_1 = np.sum(np.abs(g)) / (m + 1)
    norm_inf = np.max(np.abs(g))
    
    if norm_1 * norm_inf == 0:
        return 0.0
    
    return l2 / (norm_1 * norm_inf)

def negative_value(h):
    """Negative of the objective for minimization."""
    return -compute_value(h)

def construct():
    start_time = time.time()
    
    # Seed: uniform heights
    best_h = np.ones(N)
    best_val = compute_value(best_h)
    
    # Try various initializations and local optimizations
    rng = np.random.RandomState(42)
    
    # Different initializations to try
    initializations = []
    
    # 1. Uniform (seed)
    initializations.append(np.ones(N))
    
    # 2. Triangle-like
    x = np.linspace(-1, 1, N)
    initializations.append(np.maximum(0, 1 - np.abs(x)))
    
    # 3. Gaussian-like
    x = np.linspace(-1, 1, N)
    initializations.append(np.exp(-2 * x**2))
    
    # 4. Cosine window
    x = np.linspace(0, np.pi, N)
    initializations.append(0.5 * (1 + np.cos(x)))
    
    # 5. Random non-negative
    initializations.append(np.abs(rng.randn(N)) + 0.1)
    
    # 6. Step function
    step = np.zeros(N)
    step[N//4:3*N//4] = 1.0
    initializations.append(step)
    
    # 7. Two-lobe
    x = np.linspace(-1, 1, N)
    initializations.append(np.exp(-4*(x-0.3)**2) + np.exp(-4*(x+0.3)**2))
    
    # 8. Random with smoothing
    r = np.abs(rng.randn(N))
    # Smooth it
    kernel = np.ones(5)/5
    r_smooth = np.convolve(r, kernel, mode='same')
    r_smooth = r_smooth / r_smooth.max() if r_smooth.max() > 0 else r_smooth
    initializations.append(r_smooth)
    
    # 9. Sine-based
    x = np.linspace(0, 2*np.pi, N)
    initializations.append(np.abs(np.sin(x)) + 0.1)
    
    # 10. Power-law decay
    x = np.linspace(0, 1, N)
    initializations.append((1 - x)**2 + 0.05)
    
    elapsed = time.time() - start_time
    budget_per_init = (TIME_LIMIT - elapsed - 5.0) / len(initializations)
    
    for init_h in initializations:
        elapsed = time.time() - start_time
        if elapsed > TIME_LIMIT - 10.0:
            break
        
        init_h = np.maximum(0, init_h)
        if np.max(init_h) > 0:
            init_h = init_h / np.max(init_h) * 1.0
        
        # Evaluate initial
        val = compute_value(init_h)
        if val > best_val:
            best_val = val
            best_h = init_h.copy()
        
        # Local optimization using L-BFGS-B
        try:
            result = minimize(
                negative_value,
                init_h,
                method='L-BFGS-B',
                bounds=[(0, None)] * N,
                options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8,
                         'maxls': 50, 'maxfun': 5000}
            )
            elapsed = time.time() - start_time
            if elapsed > TIME_LIMIT - 10.0:
                break
            
            opt_h = np.maximum(0, result.x)
            opt_val = compute_value(opt_h)
            if opt_val > best_val:
                best_val = opt_val
                best_h = opt_h.copy()
                
        except Exception:
            pass
    
    # Additional refinement with multiple random restarts if time permits
    elapsed = time.time() - start_time
    remaining = TIME_LIMIT - elapsed - 5.0
    if remaining > 5.0:
        n_restarts = int(min(5, remaining / 3.0))
        for _ in range(n_restarts):
            elapsed = time.time() - start_time
            if elapsed > TIME_LIMIT - 10.0:
                break
            
            # Perturb best solution
            perturbation = 0.1 * rng.randn(N)
            new_h = np.maximum(0, best_h + perturbation)
            if np.max(new_h) > 0:
                new_h = new_h / np.max(new_h)
            
            try:
                result = minimize(
                    negative_value,
                    new_h,
                    method='L-BFGS-B',
                    bounds=[(0, None)] * N,
                    options={'maxiter': 150, 'ftol': 1e-10, 'gtol': 1e-8,
                             'maxls': 50, 'maxfun': 3000}
                )
                elapsed = time.time() - start_time
                if elapsed > TIME_LIMIT - 10.0:
                    break
                
                opt_h = np.maximum(0, result.x)
                opt_val = compute_value(opt_h)
                if opt_val > best_val:
                    best_val = opt_val
                    best_h = opt_h.copy()
                    
            except Exception:
                pass
    
    return list(best_h)
