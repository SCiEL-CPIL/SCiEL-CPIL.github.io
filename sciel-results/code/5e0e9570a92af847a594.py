import numpy as np
import time

N = 400

def _compute_metric(h):
    """Compute the upper bound on C3 - max abs autocorrelation at non-zero lags / N."""
    H = np.fft.fft(h)
    R = np.fft.ifft(np.abs(H) ** 2).real
    max_abs = np.max(np.abs(R[1:]))
    return max_abs / N

def construct():
    start_time = time.time()
    deadline = start_time + 145
    
    np.random.seed(42)
    
    # Start with all ones
    h = np.ones(N, dtype=np.float64)
    best_h = h.copy()
    best_score = _compute_metric(best_h)
    
    # Local search with random restarts
    for restart in range(20):
        if time.time() > deadline - 10:
            break
        
        # Random perturbation of best solution
        h_current = best_h.copy()
        if restart > 0:
            # Add some noise
            h_current += np.random.normal(0, 0.1, N)
        
        # Perform local search
        for step in range(500):
            if time.time() > deadline - 5:
                break
            
            # Try flipping/perturbing a random element
            idx = np.random.randint(0, N)
            h_trial = h_current.copy()
            h_trial[idx] = -h_trial[idx]  # Flip sign
            
            score = _compute_metric(h_trial)
            if score < best_score - 1e-12:
                best_score = score
                best_h = h_trial
                h_current = h_trial
            elif score < _compute_metric(h_current) - 1e-12:
                # Accept even if not globally better, for diversity
                h_current = h_trial
    
    return best_h
