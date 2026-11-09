import numpy as np
import time
from scipy.optimize import minimize

N = 600

def compute_score_fft(h):
    """
    Computes the normalized maximum of the autoconvolution using FFT.
    Metric: max(h*h) / (||h||_1 * ||h||_inf)
    """
    h = np.asarray(h, dtype=float)
    # Ensure non-negative
    h = np.maximum(h, 0)
    
    if np.all(h == 0):
        return float('inf')
        
    area = np.sum(h)
    max_val = np.max(h)
    
    if area == 0 or max_val == 0:
        return float('inf')
        
    # Use FFT for convolution
    # Zero-pad to length 2*N to avoid circular convolution artifacts
    f = np.zeros(2 * N)
    f[:N] = h
    fft_f = np.fft.fft(f)
    conv = np.fft.ifft(fft_f * fft_f).real
    
    # The full convolution has length 2N-1. 
    # np.fft.ifft gives length 2N. The first N elements correspond to lags 0 to N-1?
    # Actually, linear convolution of two N-length sequences has length 2N-1.
    # The FFT result wraps around. 
    # Let's just use np.convolve if N is small enough? 
    # N=600, N^2 = 360,000. This is very fast in NumPy (C implementation).
    # Let's stick to np.convolve for simplicity and correctness, it's likely fast enough.
    # 360k multiplications/additions is negligible.
    # Let's re-evaluate: 120s / 360k ops = 333 seconds per op if it was pure python. 
    # But it's vectorized C. It's microseconds.
    # So we can do thousands of evaluations.
    
    conv = np.convolve(h, h, mode='full')
    max_conv = np.max(conv)
    
    return max_conv / (area * max_val)

def construct():
    start_time = time.time()
    deadline = start_time + 110.0 # Safety margin
    
    # Initial candidates
    idx = np.arange(N)
    
    # 1. Constant
    h1 = np.ones(N)
    
    # 2. Triangle
    h2 = 1.0 - np.abs(2.0 * idx / N - 1.0)
    h2 = np.maximum(h2, 0)
    
    # 3. Sine squared
    h3 = np.sin(np.pi * idx / N) ** 2
    
    # 4. Gaussian
    center = N / 2.0
    sigma = N / 4.0
    x = (idx - center) / sigma
    h4 = np.exp(-0.5 * x**2)
    
    # 5. Two Spikes
    h5 = np.zeros(N)
    h5[:N//4] = 1.0
    h5[3*N//4:] = 1.0
    
    candidates = [h1, h2, h3, h4, h5]
    
    best_h = None
    best_score = float('inf')
    
    for h in candidates:
        if np.max(h) > 0:
            h_norm = h / np.max(h)
            score = compute_score_fft(h_norm)
            if score < best_score:
                best_score = score
                best_h = h_norm.copy()
        
        if time.time() > deadline:
            break
            
    if best_h is None:
        return [1.0] * N
        
    # Local Search: Hill Climbing with adaptive step size
    # We will try to perturb the best candidate found.
    current_h = best_h
    current_score = best_score
    
    # Define a perturbation function
    def perturb(h, scale):
        # Add random noise, clip to [0, 1]
        noise = np.random.normal(0, scale, N)
        h_new = h + noise
        h_new = np.clip(h_new, 0, 1)
        # Ensure max is 1? The metric is scale invariant.
        # But if we clip to 1, the max might be < 1.
        # The metric is max_conv / (area * max_val).
        # If we scale h by c, max_conv scales by c^2, area by c, max_val by c.
        # Ratio is invariant.
        # So we can work in [0,1] and not worry about normalization for comparison,
        # as long as we don't make it all zeros.
        if np.max(h_new) == 0:
            return h, current_score
        return h_new, compute_score_fft(h_new)
    
    np.random.seed(42)
    step_size = 0.1
    no_improve_count = 0
    max_no_improve = 50
    
    while time.time() < deadline:
        h_new, score_new = perturb(current_h, step_size)
        
        if score_new < current_score:
            current_h = h_new
            current_score = score_new
            no_improve_count = 0
            # Increase step size slightly if improving? Or keep stable
            step_size = min(step_size * 1.05, 0.5)
        else:
            no_improve_count += 1
            # Decrease step size
            step_size *= 0.95
            
            if no_improve_count > max_no_improve:
                # Restart from a slightly different point or break
                # Try a larger jump
                h_jump, score_jump = perturb(current_h, 0.5)
                if score_jump < current_score:
                    current_h = h_jump
                    current_score = score_jump
                    no_improve_count = 0
                    step_size = 0.2
                else:
                    # If stuck, try a new random candidate from the initial set?
                    # Or just break to save time for other strategies?
                    # Let's try a few restarts
                    no_improve_count = 0
                    step_size = 0.1
                    # Random restart from a random point in [0,1]^N
                    if np.random.rand() < 0.1:
                        current_h = np.random.rand(N)
                        current_score = compute_score_fft(current_h)
                        step_size = 0.2
                        no_improve_count = 0

    return list(current_h)
