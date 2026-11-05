import numpy as np
import time
from scipy.optimize import minimize

N = 400
_DEADLINE = 145.0

def _evaluate(h):
    """
    Compute the task value.
    value = abs(2 * N * np.max(np.convolve(h, h)) / np.sum(h) ** 2)
    """
    s = np.sum(h)
    if s == 0 or not np.isfinite(s):
        return np.inf
    # Use FFT for efficient convolution
    # np.convolve(h, h) is equivalent to ifft(fft(h) * fft(h))
    # We need the full convolution, which has length 2N-1
    # FFT size must be at least 2N-1. Let's use 2*N for efficiency (power of 2 preferred)
    fft_size = 2 * N
    H = np.fft.fft(h, n=fft_size)
    conv = np.fft.ifft(H * H).real
    # The full convolution from np.convolve(h,h) corresponds to the first 2N-1 elements 
    # of the circular convolution if we zero-pad sufficiently.
    # Specifically, conv[k] for k in 0..2N-2.
    max_conv = np.max(conv[:2*N-1])
    return abs(2.0 * N * max_conv / (s * s))

def _generate_h(params):
    """
    Generate h from parameters: [amplitude, center, width, chirp_rate]
    h[n] = amplitude * exp(-((n - center)/width)^2) * cos(2*pi*chirp_rate*n^2/N)
    """
    if len(params) < 4:
        return np.ones(N)
    
    amp = params[0]
    center = params[1]
    width = max(1.0, abs(params[2])) # Ensure width is positive and not zero
    chirp_rate = params[3]
    
    n = np.arange(N)
    # Gaussian window
    window = np.exp(-((n - center) / width) ** 2)
    # Chirp
    phase = 2.0 * np.pi * chirp_rate * n * n / N
    h = amp * window * np.cos(phase)
    
    return h

def _objective(params):
    """
    Objective function for scipy.optimize.
    We want to minimize the value.
    """
    h = _generate_h(params)
    if not np.all(np.isfinite(h)):
        return 1e10
    val = _evaluate(h)
    if not np.isfinite(val):
        return 1e10
    return val

def _fast_local_optimize(h_init, time_budget, rng):
    """
    Perform local optimization using L-BFGS-B on the parameter space.
    """
    # Initialize parameters based on h_init or random
    # We will try to fit the initial h to our parametric form or just start random
    # Since h_init is already good, let's try to extract params or just use random good params
    # For simplicity and robustness, we start with a few random param sets near the "good" region
    # "Good" region: amp ~ 1, center ~ N/2, width ~ N/4 to N/2, chirp_rate ~ 1
    
    best_val = _evaluate(h_init)
    best_h = h_init.copy()
    
    t_start = time.time()
    
    # Define bounds for parameters
    # amp: [0.1, 2.0]
    # center: [0, N-1]
    # width: [1.0, N]
    # chirp_rate: [-5.0, 5.0]
    bounds = [(0.1, 2.0), (0.0, N-1.0), (1.0, float(N)), (-5.0, 5.0)]
    
    # Try a few random initializations for the optimizer
    for i in range(5):
        if time.time() - t_start > time_budget:
            break
        
        # Random init within bounds
        p0 = [
            rng.uniform(0.5, 1.5),
            rng.uniform(N*0.2, N*0.8),
            rng.uniform(N*0.1, N*0.5),
            rng.uniform(-2.0, 2.0)
        ]
        
        # If we have a good h_init, we might try to initialize center/width closer to its mass
        if i == 0:
            # Estimate center and width from h_init
            s = np.sum(h_init)
            if abs(s) > 1e-10:
                # Use weighted average for center
                n = np.arange(N)
                center_est = np.sum(n * h_init) / s
                # Use RMS for width
                width_est = np.sqrt(np.sum((n - center_est)**2 * h_init**2) / (np.sum(h_init**2) + 1e-10))
                p0[1] = np.clip(center_est, 0, N-1)
                p0[2] = np.clip(width_est, 1.0, N)
        
        try:
            res = minimize(_objective, p0, method='L-BFGS-B', bounds=bounds, 
                           options={'maxiter': 50, 'ftol': 1e-6, 'gtol': 1e-5})
            
            if res.success or res.fun < best_val:
                h_opt = _generate_h(res.x)
                v_opt = _evaluate(h_opt)
                if v_opt < best_val and np.all(np.isfinite(h_opt)):
                    best_val = v_opt
                    best_h = h_opt.copy()
        except Exception:
            pass

    return best_h, best_val

def construct():
    t0 = time.time()
    deadline = t0 + _DEADLINE
    
    best_h = np.ones(N)
    best_val = _evaluate(best_h)  # 2.0
    
    rng = np.random.default_rng(42)
    
    # Phase 1: Global Search via Random Parametric Initialization
    # We sample random parameters for the Gaussian-Chirp model
    # This explores the space of "smooth, windowed chirps"
    
    evaluated = 0
    # Number of random starts to try before moving to local opt
    # Each evaluation is fast (FFT), so we can do many
    max_random_starts = 200
    
    for i in range(max_random_starts):
        if time.time() > deadline - 30:
            break
            
        # Sample parameters
        amp = rng.uniform(0.5, 1.5)
        center = rng.uniform(N*0.2, N*0.8)
        width = rng.uniform(N*0.1, N*0.6)
        chirp_rate = rng.uniform(-3.0, 3.0)
        
        params = [amp, center, width, chirp_rate]
        h_candidate = _generate_h(params)
        
        v_candidate = _evaluate(h_candidate)
        evaluated += 1
        
        if v_candidate < best_val:
            best_val = v_candidate
            best_h = h_candidate.copy()
            
            # If we find a significantly better one, run a quick local optimization immediately
            # This helps escape local minima of the random search
            if time.time() < deadline - 15:
                h_ref, v_ref = _fast_local_optimize(h_candidate, time_budget=2.0, rng=rng)
                if v_ref < best_val:
                    best_val = v_ref
                    best_h = h_ref.copy()

    # Phase 2: Thorough Local Optimization on the Best Found
    if time.time() < deadline - 5:
        budget = min(30.0, deadline - time.time() - 5)
        h_final, v_final = _fast_local_optimize(best_h, time_budget=budget, rng=rng)
        if v_final < best_val:
            best_val = v_final
            best_h = h_final.copy()

    # Ensure validity
    if not np.all(np.isfinite(best_h)) or np.sum(best_h) == 0:
        return np.ones(N)
    
    return best_h
