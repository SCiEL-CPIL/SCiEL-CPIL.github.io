import numpy as np
import time
from scipy.optimize import differential_evolution, minimize, L_BFGS_B

N = 400
_n = np.arange(N)

def _make_h_from_cosine(coeffs, M):
    """
    Construct h[n] = (1/N) * (1 + sum_{j=1}^{M} a_j cos(2*pi*j*n/N) + b_j sin(2*pi*j*n/N))
    This ensures sum(h) = 1 (since the DC term integrates to N, divided by N gives 1,
    and all other terms integrate to 0).
    coeffs = [a_1, b_1, a_2, b_2, ..., a_M, b_M]
    """
    h = np.full(N, 1.0 / N)
    for j in range(1, M + 1):
        a = coeffs[2 * (j - 1)]
        b = coeffs[2 * (j - 1) + 1]
        h += (a * np.cos(2 * np.pi * j * _n / N) + b * np.sin(2 * np.pi * j * _n / N)) / N
    return h

def _objective(coeffs, M):
    """
    Objective function to minimize: 2 * N * max(conv(h,h)) / (sum(h))^2
    Since sum(h)=1 by construction, this is 2 * N * max(conv(h,h))
    We add abs() to handle negative max values, though for well-behaved h max should be positive.
    """
    h = _make_h_from_cosine(coeffs, M)
    s = np.sum(h)
    if abs(s) < 1e-12:
        return 1e12
    conv = np.convolve(h, h)
    max_conv = np.max(conv)
    val = abs(2.0 * N * max_conv / (s * s))
    if not np.isfinite(val):
        return 1e12
    return val

def _objective_smooth(coeffs, M, beta=10.0):
    """
    Smooth approximation of max(conv(h,h)) using soft-max.
    This makes the objective differentiable everywhere.
    """
    h = _make_h_from_cosine(coeffs, M)
    s = np.sum(h)
    if abs(s) < 1e-12:
        return 1e12
    conv = np.convolve(h, h)
    # Soft max
    max_conv_soft = (1.0 / beta) * np.log(np.sum(np.exp(beta * conv)))
    val = abs(2.0 * N * max_conv_soft / (s * s))
    if not np.isfinite(val):
        return 1e12
    return val

def construct():
    """
    Construct a sequence of N step heights minimizing:
        value = abs(2 * N * np.max(np.convolve(h, h)) / np.sum(h)**2)
    """
    start_time = time.time()
    DEADLINE = 140  # 10s safety margin
    
    best_value = np.inf
    best_h = np.ones(N).astype(float)
    
    def evaluate(h):
        s = np.sum(h)
        if abs(s) < 1e-12:
            return np.inf, False
        conv = np.convolve(h, h)
        max_conv = np.max(conv)
        val = abs(2.0 * N * max_conv / (s * s))
        valid = np.all(np.isfinite(h)) and abs(s) > 1e-12
        return val, valid
    
    val_seed, _ = evaluate(best_h)
    best_value = val_seed
    
    # Strategy:
    # 1. Global search (differential evolution) over low-dimensional cosine coefficients.
    # 2. Local refinement (L-BFGS-B) on the same coefficients.
    # 3. Try different bandwidths M.
    
    # Define bounds for coefficients. 
    # If coefficients are too large, h can become negative, which is allowed, 
    # but very large coefficients might cause numerical issues or poor conditioning.
    # Let's start with bounds [-1, 1] for each coefficient.
    
    def run_optimization(M, bounds_list, init_bounds=None, max_iter_de=100, max_iter_lbfgs=1000):
        nonlocal best_value, best_h
        
        if time.time() - start_time > DEADLINE:
            return
        
        # Differential Evolution
        # popsize=15, tol=1e-8, maxiter=max_iter_de
        # Strategy: 'best1bin'
        try:
            result_de = differential_evolution(
                _objective, 
                bounds_list, 
                args=(M,),
                maxiter=max_iter_de, 
                popsize=15, 
                tol=1e-8, 
                seed=42,
                init='sobol'
            )
            
            if time.time() - start_time > DEADLINE:
                return
                
            # Evaluate DE result
            h_de = _make_h_from_cosine(result_de.x, M)
            val_de, valid_de = evaluate(h_de)
            if valid_de and val_de < best_value:
                best_value = val_de
                best_h = h_de.copy()
            
            # Local Refinement with L-BFGS-B
            # Use the smooth objective for gradient
            def obj_smooth_wrapper(x):
                return _objective_smooth(x, M, beta=20.0)
            
            try:
                result_lbfgs = minimize(
                    obj_smooth_wrapper, 
                    result_de.x, 
                    method='L-BFGS-B', 
                    bounds=bounds_list,
                    options={'maxiter': max_iter_lbfgs, 'ftol': 1e-10, 'gtol': 1e-8}
                )
                
                if time.time() - start_time > DEADLINE:
                    return
                
                # Evaluate L-BFGS result
                h_lbfgs = _make_h_from_cosine(result_lbfgs.x, M)
                val_lbfgs, valid_lbfgs = evaluate(h_lbfgs)
                if valid_lbfgs and val_lbfgs < best_value:
                    best_value = val_lbfgs
                    best_h = h_lbfgs.copy()
                    
            except Exception:
                pass
                
        except Exception:
            pass

    # Try different bandwidths M
    # Small M (1-3) captures simple shapes.
    # Larger M (5-10) captures more complex shapes.
    
    # Bounds: 2*M dimensions, each in [-1, 1] initially
    # We can expand bounds if needed, but let's start conservative.
    
    M_list = [1, 2, 3, 5, 7, 10]
    
    for M in M_list:
        if time.time() - start_time > DEADLINE:
            break
            
        bounds = [(-1.0, 1.0)] * (2 * M)
        
        # Adaptive iterations based on time
        elapsed = time.time() - start_time
        remaining = DEADLINE - elapsed
        
        if remaining > 60:
            max_iter_de = 50
            max_iter_lbfgs = 500
        elif remaining > 30:
            max_iter_de = 30
            max_iter_lbfgs = 200
        else:
            max_iter_de = 10
            max_iter_lbfgs = 50
            
        run_optimization(M, bounds, max_iter_de=max_iter_de, max_iter_lbfgs=max_iter_lbfgs)
        
        # If we found a very good solution, we can try refining it further with higher M?
        # Or just stop if we hit a very low value.
        if best_value < 1.1:
            # Try one more refinement with a slightly larger M if time permits
            if time.time() - start_time < DEADLINE - 20 and M < 15:
                M_next = M + 3
                if M_next not in M_list:
                    bounds_next = [(-0.5, 0.5)] * (2 * M_next)
                    run_optimization(M_next, bounds_next, max_iter_de=20, max_iter_lbfgs=100)

    # Final check
    val_final, valid = evaluate(best_h)
    if not valid:
        best_h = np.ones(N)
        best_value = 2.0
    else:
        best_value = val_final
        
    return best_h
