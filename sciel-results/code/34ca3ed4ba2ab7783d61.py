"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time

def _max_overlap(h):
    """Compute the maximum overlap of h with shifted (1-h) over all shifts."""
    N = len(h)
    overlaps = np.zeros(N)
    for k in range(N):
        shifted_1mh = 1.0 - np.roll(h, -k)
        overlaps[k] = np.dot(h, shifted_1mh)
    return np.max(overlaps)

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    # Start with uniform solution (known to be optimal for max overlap = N/4)
    best_h = np.full(N, 0.5)
    best_val = _max_overlap(best_h)
    
    # Try to improve using gradient-free optimization
    # Since the uniform solution gives all overlaps = N/4 (the average),
    # it's theoretically optimal. But let's verify with a local search.
    
    np.random.seed(42)
    
    # Try a few random perturbations to confirm optimality
    for trial in range(50):
        if time.time() > deadline - 10.0:
            break
        # Random perturbation maintaining sum = N/2
        h = best_h + np.random.randn(N) * 0.01
        h = np.clip(h, 0, 1)
        # Rescale to maintain sum = N/2
        s = np.sum(h)
        if s > 0:
            h = h * (N / 2.0) / s
            h = np.clip(h, 0, 1)
            # Re-adjust sum
            diff = N / 2.0 - np.sum(h)
            if diff > 0:
                # Add to elements that can accept
                idx = np.where(h < 1.0)[0]
                if len(idx) > 0:
                    h[idx] += diff / len(idx)
            else:
                idx = np.where(h > 0.0)[0]
                if len(idx) > 0:
                    h[idx] += diff / len(idx)
        
        val = _max_overlap(h)
        if val < best_val - 1e-10:
            best_val = val
            best_h = h.copy()
    
    # Final polish: try to equalize all overlaps using a simple iterative method
    if time.time() < deadline - 5.0:
        h = best_h.copy()
        for it in range(100):
            if time.time() > deadline - 5.0:
                break
            # Compute all overlaps
            overlaps = np.zeros(N)
            for k in range(N):
                shifted_1mh = 1.0 - np.roll(h, -k)
                overlaps[k] = np.dot(h, shifted_1mh)
            
            # Find the shift with max overlap and try to reduce it
            k_max = np.argmax(overlaps)
            # Gradient of overlap w.r.t. h[i] for shift k:
            # d/dh[i] sum_j h[j]*(1-h[(j+k)%N]) = (1-h[(i+k)%N]) - h[(i-k)%N]
            grad = np.zeros(N)
            for i in range(N):
                grad[i] = (1.0 - h[(i + k_max) % N]) - h[(i - k_max) % N]
            
            # Take a small step in negative gradient direction
            step = 0.001
            h_new = h - step * grad
            h_new = np.clip(h_new, 0, 1)
            # Rescale to maintain sum
            s = np.sum(h_new)
            if s > 0:
                h_new = h_new * (N / 2.0) / s
                h_new = np.clip(h_new, 0, 1)
            
            val = _max_overlap(h_new)
            if val < best_val - 1e-12:
                best_val = val
                best_h = h_new.copy()
            else:
                h = h_new
    
    # Ensure exact sum
    best_h = np.clip(best_h, 0, 1)
    s = np.sum(best_h)
    if s > 0:
        best_h = best_h * (N / 2.0) / s
        best_h = np.clip(best_h, 0, 1)
    
    # Final correction for exact sum
    diff = N / 2.0 - np.sum(best_h)
    if abs(diff) > 1e-15:
        idx = np.argmax(best_h < 1.0) if diff > 0 else np.argmin(best_h > 0.0)
        best_h[idx] += diff
        best_h = np.clip(best_h, 0, 1)
    
    return best_h
# EVOLVE-BLOCK-END
