import numpy as np
import time

N = 400

def _compute_metric(h):
    """Compute the upper bound on C3: max abs autocorrelation at non-zero lags / N."""
    H = np.fft.fft(h)
    R = np.fft.ifft(np.abs(H) ** 2).real
    # R[0] is the DC term, R[1:] and R[-(N-1):] are non-zero lags
    max_abs = np.max(np.abs(R[1:]))
    return max_abs / N

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    np.random.seed(42)
    
    # Start with all ones
    h = np.ones(N, dtype=np.float64)
    best_h = h.copy()
    best_score = _compute_metric(best_h)
    
    # Local search with random restarts and coordinate updates
    for restart in range(50):
        if time.time() > deadline - 5:
            break
        
        # For restarts > 0, perturb from best solution
        h_current = best_h.copy()
        if restart > 0:
            # Add random noise
            noise_scale = 0.5 * (1.0 / (restart + 1))
            h_current += np.random.randn(N) * noise_scale
        
        # Coordinate descent: try flipping signs or adjusting values
        improved = True
        while improved and time.time() < deadline - 3:
            improved = False
            # Random subset of coordinates for efficiency
            num_coords = min(N, 50)
            coords = np.random.choice(N, num_coords, replace=False)
            for idx in coords:
                if time.time() > deadline - 3:
                    break
                
                old_val = h_current[idx]
                # Try a few candidate values
                candidates = [1.0, -1.0, 0.0]
                if old_val != 1.0 and old_val != -1.0 and old_val != 0.0:
                    candidates.append(old_val)
                
                for cval in candidates:
                    if cval == old_val:
                        continue
                    h_current[idx] = cval
                    score = _compute_metric(h_current)
                    if score < best_score - 1e-12:
                        best_score = score
                        best_h = h_current.copy()
                        improved = True
                        break
                    else:
                        h_current[idx] = old_val
                
                if time.time() > deadline - 3:
                    break
        
        # If current is better, update best
        current_score = _compute_metric(h_current)
        if current_score < best_score - 1e-12:
            best_score = current_score
            best_h = h_current.copy()
    
    return best_h
