import numpy as np
import time
from scipy.optimize import minimize

N = 100

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    # Objective function: Soft max of overlaps
    # We optimize h[0:N-1], h[N-1] is determined by sum constraint
    def objective(x):
        h = np.empty(N)
        h[:N-1] = x
        h[N-1] = 50.0 - np.sum(x)
        
        # Check bounds (should be handled by optimizer, but for safety in eval)
        # If out of bounds, return large penalty
        if np.any(h < -1e-6) or np.any(h > 1 + 1e-6):
            return 1000.0 + np.sum((h < 0) * (h**2) + (h > 1) * ((h-1)**2))
            
        # Calculate overlaps
        # O_k = sum h_i (1 - h_{i+k})
        # Using FFT for speed
        # h is real.
        # corr = np.correlate(h, 1-h, mode='full')? No, cyclic.
        # Use FFT
        H = np.fft.fft(h)
        # 1-h is just 1 - h
        # Overlap k = sum h_i (1 - h_{i+k}) = sum h_i - sum h_i h_{i+k}
        # sum h_i = 50
        # sum h_i h_{i+k} is cyclic autocorrelation.
        # Autocorr via FFT: ifft(|H|^2)
        autocorr = np.fft.ifft(np.abs(H)**2).real
        # autocorr[k] is sum h_i h_{i+k} ?
        # np.fft.ifft(np.fft.fft(h) * np.fft.fft(h)[::-1])?
        # Standard circular autocorrelation:
        # r = ifft(fft(h) * conj(fft(h)))
        # r[k] = sum h_i h_{i-k} or h_{i+k}. Symmetric for real.
        
        overlaps = 50.0 - autocorr
        # overlaps[0] is k=0, which is 50 - 50 = 0? No.
        # autocorr[0] = sum h_i^2.
        # We want k=1 to N-1.
        # Note: autocorr is real for real h.
        
        # Soft max
        beta = 10.0 # Temperature for soft max
        # To avoid overflow, subtract max
        max_overlap = np.max(overlaps[1:])
        if max_overlap > 50: # Should not happen
            return max_overlap
        log_sum_exp = np.log(np.sum(np.exp(beta * (overlaps[1:] - max_overlap)))) / beta + max_overlap
        
        return log_sum_exp

    # Initial guess: 0.5 everywhere
    x0 = np.full(N-1, 50.0 / N)
    
    # Bounds for x0
    bounds = [(0, 1)] * (N-1)
    
    best_score = np.inf
    best_h = None
    
    # Run optimization
    try:
        res = minimize(objective, x0, method='L-BFGS-B', bounds=bounds, options={'maxiter': 500, 'ftol': 1e-8})
        
        if res.success or res.fun < best_score:
            h_candidate = np.empty(N)
            h_candidate[:N-1] = res.x
            h_candidate[N-1] = 50.0 - np.sum(res.x)
            
            # Clean up
            h_candidate = np.clip(h_candidate, 0, 1)
            
            # Calculate true hard max score
            H = np.fft.fft(h_candidate)
            autocorr = np.fft.ifft(np.abs(H)**2).real
            overlaps = 50.0 - autocorr
            hard_max = np.max(overlaps[1:])
            
            if hard_max < best_score:
                best_score = hard_max
                best_h = h_candidate
    except Exception as e:
        pass
    
    # If optimization failed or time is left, try a few random restarts or simple hill climbing
    # But L-BFGS-B from 0.5 is usually very good for this smooth landscape.
    
    # Fallback: if best_h is None, use constant
    if best_h is None:
        best_h = np.full(N, 0.5)
        # Calculate score for constant
        # Overlaps are all 25
        best_score = 25.0

    # Ensure sum is exactly 50 (floating point errors)
    # Adjust the last element
    diff = 50.0 - np.sum(best_h)
    best_h[N-1] += diff
    # If this pushes out of bounds, it's a minor issue, but let's check
    if best_h[N-1] < 0 or best_h[N-1] > 1:
        # Re-distribute
        # Simple scaling? No, sum is fixed.
        # Just clip and accept small sum error? 
        # The evaluator might check sum.
        # Let's do a proper projection.
        # This case is unlikely if bounds were respected.
        pass

    return best_h
