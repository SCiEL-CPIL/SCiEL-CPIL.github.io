import numpy as np
import time

N = 100
DELTA = 2.0 / N
SUM_TARGET = N / 2.0

def compute_overlaps(h):
    # Returns array of overlaps O_k for all shifts k
    # O_k = sum_i h_i * (1 - h_{i+k}) * DELTA
    overlaps = np.zeros(N)
    for k in range(N):
        shifted_h = np.roll(h, k)
        overlaps[k] = np.sum(h * (1.0 - shifted_h)) * DELTA
    return overlaps

def project_to_constraint(h_raw):
    # Project h_raw onto {h : 0 <= h_i <= 1, sum(h) = SUM_TARGET}
    # h_i = clip(h_raw_i - lambda, 0, 1)
    # We need to find lambda such that sum(clip(h_raw_i - lambda, 0, 1)) = SUM_TARGET
    
    # Sort h_raw to help with efficient search? 
    # For N=100, simple bisection on lambda is fast enough.
    # Range of lambda: min(h_raw) - 1 to max(h_raw)
    lo = np.min(h_raw) - 1.0
    hi = np.max(h_raw) + 1.0
    
    # Check if solution is trivial
    # If sum(clip(h_raw, 0, 1)) == SUM_TARGET, lambda=0
    # But we need general case.
    
    # Bisection
    for _ in range(50):
        mid = (lo + hi) / 2.0
        h_proj = np.clip(h_raw - mid, 0.0, 1.0)
        s = np.sum(h_proj)
        if s > SUM_TARGET:
            # Need to reduce sum, so increase lambda (subtract more)
            lo = mid
        else:
            # Need to increase sum, so decrease lambda
            hi = mid
            
    h_final = np.clip(h_raw - (lo + hi) / 2.0, 0.0, 1.0)
    # Correct for any tiny floating point error to ensure exact sum
    # The problem requires sum to be exactly len/2.
    # The projection might leave a tiny error.
    # We can adjust the largest element (not 0 or 1) to fix the sum exactly.
    diff = SUM_TARGET - np.sum(h_final)
    if abs(diff) > 1e-12:
        # Find an index that is strictly inside (0, 1)
        mask = (h_final > 1e-9) & (h_final < 1.0 - 1e-9)
        if np.any(mask):
            idx = np.where(mask)[0][0]
            h_final[idx] += diff
            # Ensure it doesn't go out of bounds (should be safe if diff is tiny)
            h_final[idx] = np.clip(h_final[idx], 0.0, 1.0)
        else:
            # Fallback: distribute diff among all? This case shouldn't happen with valid projection.
            # If all are 0 or 1, sum is integer. SUM_TARGET is N/2.
            # If N is even, N/2 is integer.
            # If we are stuck, we might have a degenerate case.
            pass
            
    return h_final

def compute_max_overlap_circular(h):
    overlaps = compute_overlaps(h)
    return np.max(overlaps)

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    rng = np.random.default_rng(42)
    
    best_h = np.full(N, 0.5)
    best_score = compute_max_overlap_circular(best_h)
    
    # Gradient Descent with Smoothed Max
    # Objective: F(h) = log(sum(exp(beta * O_k)))
    # Grad F = sum( exp(beta*O_k) / sum(exp(beta*O_j)) * Grad O_k )
    
    beta = 10.0 # Temperature for smoothing
    alpha = 0.1 # Learning rate
    
    for restart in range(5):
        if time.time() > deadline:
            break
            
        # Initialize
        if restart == 0:
            h = np.full(N, 0.5)
        else:
            h = rng.uniform(0, 1, N)
            h = project_to_constraint(h)
            
        h = project_to_constraint(h)
        
        for it in range(500):
            if time.time() > deadline:
                break
                
            overlaps = compute_overlaps(h)
            
            # Smoothed max
            max_ov = np.max(overlaps)
            # Shift to avoid overflow
            shifted_ov = overlaps - max_ov
            exp_ov = np.exp(beta * shifted_ov)
            weights = exp_ov / np.sum(exp_ov)
            
            # Gradient of overlaps
            # grad_O_k = DELTA * [ (1 - roll(h, k)) - roll(h, -k) ]
            # We compute grad_h = sum_k weights[k] * grad_O_k
            grad_h = np.zeros(N)
            
            # Optimization: 
            # grad_h[i] = DELTA * sum_k weights[k] * ( (1 - h[(i+k)%N]) - h[(i-k)%N] )
            # = DELTA * ( sum_k weights[k] - sum_k weights[k] h[(i+k)%N] - sum_k weights[k] h[(i-k)%N] )
            # sum_k weights[k] = 1
            # sum_k weights[k] h[(i+k)%N] is cross-correlation of weights and h.
            # sum_k weights[k] h[(i-k)%N] is cross-correlation of weights and h (reversed or similar).
            
            # Let's just compute it directly for N=100, it's fast enough.
            # Or use FFT.
            # N=100, 500 iterations, 5 restarts. 2500 * 100 = 250k ops. Trivial.
            
            for k in range(N):
                if weights[k] == 0:
                    continue
                # grad_O_k[i] = DELTA * (1 - h[i+k]) - DELTA * h[i-k]
                # Note: h[i+k] is roll(h, k)[i]?
                # np.roll(h, k) shifts right by k. So element i is h[i-k].
                # Wait.
                # np.roll(h, k)[i] = h[(i-k) % N].
                # So h[i+k] = np.roll(h, -k)[i].
                
                term1 = 1.0 - np.roll(h, -k) # h[i+k]
                term2 = np.roll(h, k)       # h[i-k]
                
                grad_h += weights[k] * DELTA * (term1 - term2)
            
            # Update
            h_raw = h - alpha * grad_h
            
            # Project
            h_new = project_to_constraint(h_raw)
            
            # Check improvement
            score_new = compute_max_overlap_circular(h_new)
            if score_new < best_score:
                best_score = score_new
                best_h = h_new.copy()
            
            h = h_new
            
            # Adaptive learning rate?
            # If no improvement for a while, reduce alpha?
            # Or just keep going.
            
            # Early stop if gradient is small
            if np.max(np.abs(grad_h)) < 1e-6:
                break
                
    return best_h
