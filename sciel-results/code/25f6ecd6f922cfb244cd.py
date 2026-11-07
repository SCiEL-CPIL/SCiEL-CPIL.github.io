"""Program for Erdos minimum overlap task using model-and-optimize approach."""
import numpy as np
import time
from scipy.optimize import minimize

N = 100
DEADLINE = 145.0
START = time.time()

def remaining():
    return DEADLINE - (time.time() - START)

def objective(h):
    """Compute the value to minimize."""
    c = np.correlate(h, 1.0 - h, mode="full")
    return float(np.max(c)) / N * 2.0

def objective_smooth(h, beta=50.0):
    """Compute the smooth value to minimize using log-sum-exp."""
    c = np.correlate(h, 1.0 - h, mode="full")
    c_max = np.max(c)
    shifted = beta * (c - c_max)
    log_sum = np.log(np.sum(np.exp(shifted))) + beta * c_max
    return float(log_sum / N * 2.0)

def project(h):
    """Project to [0,1]^N with sum = N/2."""
    h = np.asarray(h, dtype=np.float64).copy()
    target = N / 2.0
    
    # Bisection to find the right offset
    lo, hi = -1.0, 1.0
    for _ in range(60):
        mid = (lo + hi) / 2.0
        h_trial = np.clip(h + mid, 0.0, 1.0)
        s = h_trial.sum()
        if s < target:
            lo = mid
        else:
            hi = mid
    h = np.clip(h + (lo + hi) / 2.0, 0.0, 1.0)
    
    # Final fine-tuning
    for _ in range(20):
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

def construct():
    t0 = time.time()
    i = np.arange(N)
    
    best_h = np.full(N, 0.5)
    best_val = objective(best_h)
    
    # Strategy 1: Model-based approach - parameterize as 0.5 + sum of sines/cosines
    # and optimize the Fourier coefficients directly
    
    # Try different numbers of Fourier modes
    for n_modes in range(1, 8):
        if remaining() < 5:
            break
        
        # Parameters: amplitude and phase for each mode
        # We'll use a combined representation: a_k * cos(2*pi*k*i/N + phi_k)
        # For simplicity, use real and imaginary parts (cos and sin coefficients)
        
        # Initialize with small random values
        np.random.seed(42 + n_modes)
        params_init = np.random.randn(2 * n_modes) * 0.05
        
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
            return objective_smooth(h)
        
        try:
            result = minimize(
                obj_params,
                params_init,
                method='L-BFGS-B',
                options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            
            h_candidate = params_to_h(result.x)
            val_candidate = objective(h_candidate)
            
            if val_candidate < best_val:
                best_h = h_candidate
                best_val = val_candidate
                
        except Exception:
            pass
    
    # Strategy 2: Multi-start with cosine patterns
    for k in range(1, 15):
        if remaining() < 3:
            break
        
        for A_init in [0.15, 0.25, 0.35, 0.45, 0.5]:
            if remaining() < 2:
                break
            
            h_init = 0.5 + A_init * np.cos(2 * np.pi * k * i / N)
            h_init = project(h_init)
            
            def obj_func(x):
                return objective_smooth(project(x))
            
            try:
                result = minimize(
                    obj_func,
                    h_init,
                    method='L-BFGS-B',
                    bounds=[(0.0, 1.0)] * N,
                    options={'maxiter': 150, 'ftol': 1e-12, 'gtol': 1e-8}
                )
                
                h_candidate = project(result.x)
                val_candidate = objective(h_candidate)
                
                if val_candidate < best_val:
                    best_h = h_candidate
                    best_val = val_candidate
                    
            except Exception:
                pass
    
    # Strategy 3: Random perturbations around best solution
    if remaining() > 10:
        np.random.seed(123)
        for _ in range(10):
            if remaining() < 3:
                break
            
            h_pert = best_h + 0.02 * np.random.randn(N)
            h_pert = project(h_pert)
            
            def obj_func2(x):
                return objective_smooth(project(x))
            
            try:
                result = minimize(
                    obj_func2,
                    h_pert,
                    method='L-BFGS-B',
                    bounds=[(0.0, 1.0)] * N,
                    options={'maxiter': 80, 'ftol': 1e-12, 'gtol': 1e-8}
                )
                
                h_candidate = project(result.x)
                val_candidate = objective(h_candidate)
                
                if val_candidate < best_val:
                    best_h = h_candidate
                    best_val = val_candidate
                    
            except Exception:
                pass
    
    # Strategy 4: Discrete Local Search
    # Try to find a better arrangement by swapping values in a quantized version
    if remaining() > 10:
        # Quantize best_h to a grid, or use binary
        # Let's try binary first, then fine quantization
        h_bin = (best_h > 0.5).astype(np.float64)
        # Ensure sum is N/2
        # If sum is not N/2, adjust
        s = h_bin.sum()
        if s != N / 2.0:
            # Sort indices by distance to 0.5 to adjust
            idx = np.argsort(np.abs(best_h - 0.5))
            if s < N / 2.0:
                for i in idx:
                    if h_bin[i] == 0.0:
                        h_bin[i] = 1.0
                        s += 1
                        if s == N / 2.0: break
            else:
                for i in idx:
                    if h_bin[i] == 1.0:
                        h_bin[i] = 0.0
                        s -= 1
                        if s == N / 2.0: break
        
        # Local search on binary vector
        improved = True
        while improved and remaining() > 5:
            improved = False
            current_val = objective(h_bin)
            for i in range(N):
                if remaining() < 5: break
                for j in range(i + 1, N):
                    if h_bin[i] == h_bin[j]: continue # Swapping identical values does nothing
                    # Swap
                    h_bin[i], h_bin[j] = h_bin[j], h_bin[i]
                    new_val = objective(h_bin)
                    if new_val < current_val:
                        current_val = new_val
                        improved = True
                    else:
                        # Swap back
                        h_bin[i], h_bin[j] = h_bin[j], h_bin[i]
            
            if current_val < best_val:
                best_h = h_bin.copy()
                best_val = current_val
        
        # Also try a fine quantized version (e.g., 0, 0.25, 0.5, 0.75, 1.0)
        # This is more expensive, so only if time permits
        if remaining() > 20:
            h_quant = np.round(best_h * 4.0) / 4.0
            # Project to sum
            h_quant = project(h_quant)
            # Simple local search on quantized values? 
            # Swapping quantized values is complex. 
            # Instead, just use the continuous optimizer on the quantized start.
            def obj_quant(x):
                return objective_smooth(project(x))
            
            try:
                result = minimize(
                    obj_quant,
                    h_quant,
                    method='L-BFGS-B',
                    bounds=[(0.0, 1.0)] * N,
                    options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-8}
                )
                h_candidate = project(result.x)
                val_candidate = objective(h_candidate)
                if val_candidate < best_val:
                    best_h = h_candidate
                    best_val = val_candidate
            except Exception:
                pass

    # Final polish
    if remaining() > 5:
        def obj_final(x):
            return objective_smooth(project(x))
        
        try:
            result = minimize(
                obj_final,
                best_h,
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
    
    best_h = project(best_h)
    return best_h
