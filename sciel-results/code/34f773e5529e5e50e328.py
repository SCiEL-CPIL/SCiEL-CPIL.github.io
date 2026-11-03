"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import minimize
import time

N = 400

# EVOLVE-BLOCK-START

def _compute_score(x):
    """
    Compute the score: maximum absolute autoconvolution divided by squared integral.
    Lower is better.
    """
    # Integral (sum) of x
    integral = np.sum(x)
    if abs(integral) < 1e-12:
        return np.inf
    
    # Autoconvolution: c[k] = sum_i x[i] * x[i+k] for k in -(N-1)..(N-1)
    # Using FFT for efficiency
    n_fft = 1 << (2 * N - 1).bit_length()
    fft_x = np.fft.rfft(x, n_fft)
    autoconv = np.fft.irfft(fft_x * np.conj(fft_x), n_fft)
    
    # The linear autoconvolution is in autoconv[0:N] (positive lags) 
    # and autoconv[n_fft-N+1:n_fft] (negative lags, reversed)
    # Actually, for real input, irfft of |X|^2 gives circular convolution.
    # Since we zero-padded to 2N-1, the first N elements are the linear autoconvolution for lags 0..N-1.
    # The negative lags are symmetric: c[-k] = c[k].
    # So we just need to check c[0:N].
    c = autoconv[:N]
    
    max_abs = np.max(np.abs(c))
    
    return max_abs / (integral * integral)

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    # Initial guess: alternating signs or random
    # Let's try a few initializations and pick the best
    best_x = None
    best_score = np.inf
    
    # Try a few different starting points
    starts = []
    
    # 1. All ones (baseline)
    starts.append(np.ones(N))
    
    # 2. Random
    np.random.seed(42)
    starts.append(np.random.randn(N))
    
    # 3. Alternating
    starts.append(np.array([1.0 if i % 2 == 0 else -1.0 for i in range(N)]))
    
    # 4. Block signs
    for block_size in [4, 8, 16, 32]:
        x = np.ones(N)
        for i in range(0, N, block_size):
            if (i // block_size) % 2 == 1:
                x[i:i+block_size] = -1.0
        starts.append(x)
        
    # 5. Random block signs
    for block_size in [4, 8, 16]:
        x = np.ones(N)
        for i in range(0, N, block_size):
            if np.random.rand() < 0.5:
                x[i:i+block_size] = -1.0
        starts.append(x)
        
    # 6. Spectral/Chirp-like initializations (from inspiration)
    x = np.arange(N)
    starts.append(np.cos(2 * np.pi * x / N))  # Simple wave
    starts.append(np.cos(4 * np.pi * x / N))  # Higher freq wave
    starts.append(np.sin(2 * np.pi * x / N))
    
    # 7. Gaussian window
    mu = N / 2
    sigma = N / 4
    starts.append(np.exp(-0.5 * ((x - mu) / sigma) ** 2))
    
    # 8. Chirp
    starts.append(np.cos(np.pi * x**2 / N))
    
    # Evaluate initial scores
    for x in starts:
        if time.time() > deadline:
            break
        score = _compute_score(x)
        if score < best_score:
            best_score = score
            best_x = x.copy()
    
    # Optimization strategy:
    # 1. Use a smooth approximation of the max absolute value to make it differentiable.
    # 2. Use L-BFGS-B to optimize.
    # 3. Use multiple betas (temperature) for the soft-max to refine.
    
    if best_x is not None:
        def _soft_max_abs(ac, beta):
            abs_ac = np.abs(ac)
            m = np.max(abs_ac)
            if m < 1e-15:
                return 0.0
            scaled = beta * (abs_ac - m)
            scaled = np.clip(scaled, -50, 50)
            log_sum = np.log(np.sum(np.exp(scaled)))
            return m + log_sum / beta
        
        def _objective_smooth(x, beta):
            integral = np.sum(x)
            if abs(integral) < 1e-12:
                return 1e10
            n_fft = 1 << (2 * N - 1).bit_length()
            fft_x = np.fft.rfft(x, n_fft)
            autoconv = np.fft.irfft(fft_x * np.conj(fft_x), n_fft)
            c = autoconv[:N]
            sm = _soft_max_abs(c, beta)
            return sm / (integral * integral)
        
        # Refine best_x with gradient descent using smooth objective
        # Start with lower beta (smoother) and increase
        current_x = best_x.copy()
        
        for beta in [20, 50, 100, 200]:
            if time.time() > deadline - 10:
                break
                
            def obj(x):
                return _objective_smooth(x, beta)
            
            try:
                res = minimize(obj, current_x, method='L-BFGS-B', 
                              options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8})
                if res.fun < best_score:
                    best_score = res.fun
                    best_x = res.x
                current_x = res.x
            except Exception:
                pass
        
        # Try local search / coordinate descent on the best solution
        # This can help escape local minima of the smooth approximation
        # or refine the "hard" max absolute value constraint.
        
        def _local_search(x, max_iter=1000):
            x = x.copy()
            current_val = _compute_score(x)
            n_fft = 1 << (2 * N - 1).bit_length()
            
            # Simple coordinate descent with random steps
            rng = np.random.default_rng(42)
            step_size = 0.5
            
            for i in range(max_iter):
                if time.time() > deadline - 5:
                    break
                idx = rng.integers(0, N)
                orig = x[idx]
                
                # Try a few perturbations
                for delta in [step_size, -step_size, step_size*0.5, -step_size*0.5]:
                    x[idx] = orig + delta
                    val = _compute_score(x)
                    if val < current_val:
                        current_val = val
                        break
                    x[idx] = orig
                else:
                    # No improvement, shrink step size
                    step_size *= 0.9
                    if step_size < 1e-4:
                        break
                
            return x, current_val
        
        # Run local search on the best solution
        if time.time() < deadline - 10:
            x_ls, val_ls = _local_search(best_x, max_iter=500)
            if val_ls < best_score:
                best_score = val_ls
                best_x = x_ls
        
        # Also try local search from a few other good initializations
        # (Skip for now to save time, but could be added if budget allows)
        
        # Final polish with high beta gradient step
        if time.time() < deadline - 5:
            def obj_final(x):
                return _objective_smooth(x, 500)
            
            try:
                res = minimize(obj_final, best_x, method='L-BFGS-B', 
                              options={'maxiter': 50, 'ftol': 1e-12, 'gtol': 1e-8})
                if res.fun < best_score:
                    best_score = res.fun
                    best_x = res.x
            except Exception:
                pass

    # Final result: signed step heights
    # The task says "sequence of signed step heights"
    # We should return the optimized x
    if best_x is not None:
        return best_x
    else:
        return np.ones(N)

# EVOLVE-BLOCK-END
