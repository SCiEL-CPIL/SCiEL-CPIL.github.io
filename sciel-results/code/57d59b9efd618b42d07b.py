import numpy as np
import time
from scipy.optimize import minimize_scalar

N = 400

def _generate_h(r, offset=1.0):
    """
    Generate h from logistic map with parameter r.
    h[n] = offset + x_n
    """
    x = np.zeros(N)
    x[0] = 0.5
    for i in range(1, N):
        x[i] = r * x[i-1] * (1.0 - x[i-1])
    return offset + x

def _objective(r):
    """
    Objective function to minimize.
    """
    h = _generate_h(r)
    s = np.sum(h)
    if abs(s) < 1e-12:
        return 1e12
    conv = np.convolve(h, h)
    max_conv = np.max(conv)
    val = abs(2.0 * N * max_conv / (s * s))
    if not np.isfinite(val):
        return 1e12
    return val

def construct():
    """
    Construct a sequence of N step heights minimizing:
        value = abs(2 * N * np.max(np.convolve(h, h)) / np.sum(h)**2)
    """
    start_time = time.time()
    DEADLINE = 110  # Safety margin for 120s limit
    
    best_value = np.inf
    best_h = np.ones(N).astype(float)
    
    # Evaluate seed
    s_seed = np.sum(best_h)
    conv_seed = np.convolve(best_h, best_h)
    val_seed = abs(2.0 * N * np.max(conv_seed) / (s_seed * s_seed))
    best_value = val_seed
    
    # Define bounds for r (chaotic regime)
    # Logistic map is chaotic for r in approx [3.57, 4.0]
    # We can also try lower r for periodic but low amplitude variations?
    # But periodic will have high autocorrelation peaks.
    # Chaotic is better for "white" noise.
    
    def run_brent(bounds):
        nonlocal best_value, best_h
        
        if time.time() - start_time > DEADLINE:
            return
            
        try:
            # Use Brent's method to minimize the objective
            # Method 'brent' is for bounded scalar minimization
            result = minimize_scalar(
                _objective, 
                bounds=bounds, 
                method='brent',
                options={'xatol': 1e-6}
            )
            
            if time.time() - start_time > DEADLINE:
                return
                
            r_opt = result.x
            h_opt = _generate_h(r_opt)
            
            s_opt = np.sum(h_opt)
            if abs(s_opt) < 1e-12:
                return
                
            conv_opt = np.convolve(h_opt, h_opt)
            val_opt = abs(2.0 * N * np.max(conv_opt) / (s_opt * s_opt))
            
            if np.isfinite(val_opt) and val_opt < best_value:
                best_value = val_opt
                best_h = h_opt.copy()
                
        except Exception:
            pass

    # Try a few ranges for r
    # 1. Deep chaos
    run_brent((3.8, 4.0))
    
    if time.time() - start_time < DEADLINE:
        # 2. Onset of chaos
        run_brent((3.5, 3.8))
        
    if time.time() - start_time < DEADLINE:
        # 3. Periodic but maybe low peak? (Unlikely to beat chaos for this metric, but worth a shot)
        run_brent((3.0, 3.5))

    # Final check
    s_final = np.sum(best_h)
    if abs(s_final) < 1e-12 or not np.all(np.isfinite(best_h)):
        best_h = np.ones(N)
    else:
        conv_final = np.convolve(best_h, best_h)
        val_final = abs(2.0 * N * np.max(conv_final) / (s_final * s_final))
        if val_final < best_value:
            best_value = val_final

    return best_h
