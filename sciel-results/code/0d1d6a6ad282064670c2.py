import numpy as np
import time

N = 600
TIME_LIMIT = 115

def compute_score(f):
    """Compute the normalized maximum of the autoconvolution."""
    f = np.asarray(f, dtype=np.float64)
    if np.any(f < -1e-15):
        return np.inf
    f = np.maximum(f, 0.0)
    s = np.sum(f)
    if s < 1e-15:
        return np.inf
    # Use FFT for fast convolution
    conv = np.fft.irfft(np.fft.rfft(f) ** 2, n=2*N-1)
    return np.max(conv) / (s * s)

def construct():
    start_time = time.time()
    
    # Initialize with uniform distribution
    f = np.ones(N) / N
    current_score = compute_score(f)
    
    # Simulated Annealing Parameters
    T = 0.01  # Initial temperature
    cooling_rate = 0.995
    min_T = 1e-6
    max_iters = 5000
    
    best_f = f.copy()
    best_score = current_score
    
    rng = np.random.default_rng(42)
    
    for i in range(max_iters):
        if time.time() - start_time > TIME_LIMIT - 5:
            break
            
        # Generate a perturbation
        # Randomly select a subset of indices to perturb
        # Or add noise to all and project
        noise = rng.normal(0, 0.01, N)
        f_new = f + noise
        f_new = np.maximum(f_new, 0)
        
        # Project to simplex (sum=1)
        s = np.sum(f_new)
        if s < 1e-15:
            f_new = np.ones(N) / N
        else:
            f_new = f_new / s
            
        new_score = compute_score(f_new)
        
        # Acceptance criterion
        if new_score < current_score:
            f = f_new
            current_score = new_score
            if new_score < best_score:
                best_score = new_score
                best_f = f_new.copy()
        else:
            # Metropolis criterion
            delta = new_score - current_score
            if delta < 0:
                accept = True
            else:
                accept = np.random.rand() < np.exp(-delta / T)
                
            if accept:
                f = f_new
                current_score = new_score
            else:
                # Keep f unchanged, but maybe cool down faster?
                pass
        
        T *= cooling_rate
        if T < min_T:
            T = min_T
            # Restart with lower temp or break?
            # Let's just continue with low temp
            
    return [float(x) for x in best_f]

if __name__ == "__main__":
    f = construct()
    print(f"Score: {compute_score(f)}")
