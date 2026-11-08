"""Strategy: Differential Evolution for global search + Local Polish."""
import time
import numpy as np
from scipy.optimize import minimize, differential_evolution
from scipy.signal import fftconvolve

N = 600
DEADLINE = 145.0

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    
    # Precompute FFT size for efficient convolution
    n_fft = 2 * N
    
    def objective(x):
        x = np.maximum(x, 0)
        # Linear autoconvolution via FFT
        x_pad = np.zeros(n_fft)
        x_pad[:N] = x
        X = np.fft.fft(x_pad)
        conv = np.fft.ifft(np.abs(X)**2).real[:2*N-1]
        norm = np.sum(x * x)
        if norm < 1e-15:
            return 1e15
        return np.max(conv) / norm

    def objective_fast(x):
        # Slightly faster version for DE, avoiding some overhead if possible
        # But fftconvolve or manual FFT is similar cost. 
        # Let's stick to the standard one but ensure it's vectorized correctly.
        x = np.maximum(x, 0)
        conv = fftconvolve(x, x, mode='full')[:2*N-1]
        norm = np.sum(x * x)
        if norm < 1e-15:
            return 1e15
        return np.max(conv) / norm

    # Initial uniform value
    best_x = np.ones(N)
    best_val = objective(best_x)
    
    # Scale invariance: we can fix the norm to 1 to reduce search space dimensionality effectively
    # However, DE works on the raw values. Let's define bounds.
    # If we normalize x to have norm 1, the objective is just max(conv).
    # But DE needs bounds. Let's use [0, 1] and normalize inside?
    # No, better to let DE search in [0, 1] and the objective handles normalization.
    
    # Strategy:
    # 1. Run Differential Evolution with a tight time budget.
    #    DE is stochastic and global. It is slow but robust.
    #    With N=600, DE might be too slow for many generations.
    #    We need to be careful. 150s limit.
    #    DE with popsize=10, maxiter=50 might be feasible if objective is fast.
    #    Objective cost: FFT of size 1200. ~1200^1.5 ~ 40k ops. Very fast.
    #    DE evaluations: popsize * maxiter * 2 (approx). 10 * 50 * 2 = 1000 evals.
    #    1000 * 40k = 40M ops. < 1 sec.
    #    So we can afford more.
    
    bounds = [(0, 1)] * N
    
    # Use differential_evolution
    # tol=1e-4 is fine for this non-smooth objective
    # maxiter: we can set high, but use callback to stop if time runs out.
    
    def stop_callback(xk, convergence):
        return time.time() > DEADLINE - 20
    
    try:
        # DE is robust to local minima. 
        # Initial seed: we can provide a seed for reproducibility or let it randomize.
        # Let's try a few different strategies within the time budget.
        
        # Phase 1: Global search with DE
        # popsize=15, maxiter=100
        # We will run DE. If it finishes, great. If not, it returns the best found so far?
        # scipy's DE does not return partial results on timeout via callback easily in older versions?
        # Actually, `differential_evolution` in scipy does support `callback` which can return True to stop.
        # It returns the result object with the best x found so far.
        
        res_de = differential_evolution(
            objective_fast,
            bounds,
            maxiter=200,
            popsize=15,
            tol=1e-5,
            seed=42,
            callback=stop_callback,
            polish=False # We will polish manually, DE's internal polish can be slow/redundant
        )
        
        if res_de.fun < best_val:
            x_opt = np.maximum(res_de.x, 0)
            val = objective(x_opt)
            if val < best_val:
                best_val = val
                best_x = x_opt.copy()
        
    except Exception:
        pass
    
    # Phase 2: Local Polish with L-BFGS-B
    # Use the best result from DE as the starting point for L-BFGS-B
    if time.time() < DEADLINE - 10:
        try:
            # L-BFGS-B requires gradients or finite differences. 
            # For N=600, finite differences are 600 evals per step. 
            # 100 steps = 60k evals. 60k * 40k ops = 2.4B ops. Might be ~5-10s.
            # This is acceptable.
            
            res_lbfgs = minimize(
                objective,
                best_x,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            
            if res_lbfgs.fun < best_val:
                x_opt = np.maximum(res_lbfgs.x, 0)
                val = objective(x_opt)
                if val < best_val:
                    best_val = val
                    best_x = x_opt.copy()
                    
        except Exception:
            pass
            
    # Phase 3: Final quick polish if time remains
    if time.time() < DEADLINE - 5:
        try:
            res_lbfgs2 = minimize(
                objective,
                best_x,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 50, 'ftol': 1e-14, 'gtol': 1e-12}
            )
            if res_lbfgs2.fun < best_val:
                x_opt = np.maximum(res_lbfgs2.x, 0)
                val = objective(x_opt)
                if val < best_val:
                    best_val = val
                    best_x = x_opt.copy()
        except Exception:
            pass

    # Ensure nonnegativity
    best_x = np.maximum(best_x, 0)
    
    return list(best_x)
# EVOLVE-BLOCK-END
