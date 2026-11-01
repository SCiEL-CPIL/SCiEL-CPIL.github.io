import numpy as np
import time
from scipy.optimize import minimize

N = 400
DEADLINE = 120.0

def _compute_autocorr_max(h):
    """
    Compute the maximum absolute value of the autocorrelation sidelobes.
    h is a 1D array of +/- 1.
    """
    n_fft = 2 * N
    H = np.fft.fft(h, n=n_fft)
    P = np.real(np.fft.ifft(H * np.conj(H)))
    sidelobes = P[1:N]
    return np.max(np.abs(sidelobes))

def _smooth_objective(x):
    """
    Sum of squares of autocorrelation sidelobes.
    x is a 1D array of floats.
    """
    n_fft = 2 * N
    H = np.fft.fft(x, n=n_fft)
    P = np.real(np.fft.ifft(H * np.conj(H)))
    sidelobes = P[1:N]
    return np.sum(sidelobes ** 2)

def construct():
    start_time = time.time()
    best_h = np.ones(N, dtype=np.int8)
    best_score = _compute_autocorr_max(best_h)
    
    np.random.seed(42)
    
    # Number of continuous optimization attempts
    max_attempts = 15
    
    for attempt in range(max_attempts):
        if time.time() - start_time > DEADLINE - 15.0:
            break
            
        # Random continuous initialization
        x0 = np.random.uniform(-1, 1, N)
        
        # Define bounds
        bounds = [(-1, 1)] * N
        
        # Run L-BFGS-B
        try:
            res = minimize(_smooth_objective, x0, method='L-BFGS-B', bounds=bounds, 
                           options={'maxiter': 300, 'ftol': 1e-5, 'gtol': 1e-4})
            
            # Discretize the result
            h_candidate = np.sign(res.x).astype(np.int8)
            # Handle zeros
            h_candidate[h_candidate == 0] = 1
            
            score_candidate = _compute_autocorr_max(h_candidate)
            
            if score_candidate < best_score:
                best_score = score_candidate
                best_h = h_candidate.copy()
                
            # Local search refinement
            current_h = h_candidate.copy()
            current_score = score_candidate
            
            # Hill climbing
            # We allocate a fixed number of steps, but check time
            # 2000 steps * ~0.1ms/step = 0.2s. 
            # 15 attempts * 0.2s = 3s. 
            # Plus optimization time.
            # Optimization might take 1-2s per attempt.
            # Total ~ 15 * 3s = 45s. Safe.
            
            for _ in range(2000):
                if time.time() - start_time > DEADLINE - 5.0:
                    break
                
                idx = np.random.randint(N)
                old_val = current_h[idx]
                current_h[idx] = -old_val
                
                new_score = _compute_autocorr_max(current_h)
                
                if new_score < current_score:
                    current_score = new_score
                    if current_score < best_score:
                        best_score = current_score
                        best_h = current_h.copy()
                else:
                    current_h[idx] = old_val
                        
        except Exception:
            pass
            
    return best_h
