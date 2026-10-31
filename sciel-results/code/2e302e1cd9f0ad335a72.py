"""Rewritten program for Erdos minimum overlap task."""
import numpy as np

N = 100

def _max_overlap(h):
    """Compute the maximum overlap of h with shifted 1-h over all shifts."""
    one_minus_h = 1.0 - h
    max_ov = 0.0
    # We can compute all overlaps at once using FFT or just loop since N is small
    # Looping is O(N^2) = 10000 ops, which is very fast.
    for s in range(N):
        shifted = np.roll(one_minus_h, s)
        ov = np.dot(h, shifted)
        if ov > max_ov:
            max_ov = ov
    return max_ov

def _construct_h(w, c):
    """
    Construct h given width w and center c.
    h[i] = 0.5 + 0.5 * tanh((i - c) / w)
    """
    i = np.arange(N)
    if w < 1e-6:
        w = 1e-6
    h = 0.5 + 0.5 * np.tanh((i - c) / w)
    return h

def _solve_c(w):
    """
    Find c such that sum(h) = N/2.
    """
    target_sum = N / 2.0
    
    # Binary search for c in [0, N]
    # sum(h) is decreasing with c.
    lo = 0.0
    hi = float(N)
    
    # 50 iterations of binary search gives precision ~ N / 2^50, which is plenty.
    for _ in range(50):
        mid = (lo + hi) / 2.0
        h_mid = _construct_h(w, mid)
        s_mid = np.sum(h_mid)
        if s_mid > target_sum:
            lo = mid
        else:
            hi = mid
            
    c_opt = (lo + hi) / 2.0
    return c_opt

def _objective(w):
    """
    Objective function for a given width w.
    Returns the maximum overlap.
    """
    # Clamp w to reasonable range
    if w < 1e-3:
        w = 1e-3
    if w > N:
        w = N
        
    c = _solve_c(w)
    h = _construct_h(w, c)
    
    # Ensure sum is exactly N/2
    current_sum = np.sum(h)
    diff = N / 2.0 - current_sum
    
    if abs(diff) > 1e-10:
        # Simple linear correction
        h = h + diff / N
        h = np.clip(h, 0.0, 1.0)
        
        # Iterative fix for exact sum if clipping occurred
        for _ in range(10):
            current_sum = np.sum(h)
            diff = N / 2.0 - current_sum
            if abs(diff) < 1e-12:
                break
            if diff > 0:
                mask = h < 1.0
                if not np.any(mask): break
                indices = np.where(mask)[0]
                step = diff / len(indices)
                for idx in indices:
                    h[idx] = min(1.0, h[idx] + step)
            else:
                mask = h > 0.0
                if not np.any(mask): break
                indices = np.where(mask)[0]
                step = diff / len(indices)
                for idx in indices:
                    h[idx] = max(0.0, h[idx] + step)

    return _max_overlap(h), h

def construct():
    """
    Construct the sequence of step heights.
    """
    # Global search for w
    # The optimal w is likely small (sharp transition) but not infinitesimal.
    # Let's search w in [0.1, 10.0]
    
    best_w = None
    best_val = float('inf')
    best_h = None
    
    # Coarse grid search
    # w from 0.1 to 10.0
    w_coarse = np.linspace(0.1, 10.0, 100)
    for w in w_coarse:
        val, h = _objective(w)
        if val < best_val:
            best_val = val
            best_w = w
            best_h = h
    
    # Fine grid search around the best coarse w
    if best_w is not None:
        # Determine range for fine search
        # Check neighbors in coarse grid to estimate step size
        idx = np.argmin(np.abs(w_coarse - best_w))
        step = w_coarse[1] - w_coarse[0] if len(w_coarse) > 1 else 0.1
        w_fine = np.linspace(max(0.01, best_w - 5*step), min(N, best_w + 5*step), 100)
        
        for w in w_fine:
            val, h = _objective(w)
            if val < best_val:
                best_val = val
                best_w = w
                best_h = h

    return best_h

# EVOLVE-BLOCK-START
# (The construct function is defined above, but we need to keep the interface)
# The task says "Keep the function names and signatures of the seed program."
# The seed program has construct() -> sequence of step heights.
# So we just need to make sure construct() is defined and returns the right thing.
# EVOLVE-BLOCK-END
