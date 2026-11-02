"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

N = 50

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit
    
    # Strategy: Use a sparse vector to concentrate L2 energy
    # while keeping autoconvolution peak manageable.
    # A single spike gives high ratio; let's try different sparsity patterns.
    
    best_value = 0.0
    best_heights = [1.0] * N  # Fallback to seed
    
    def compute_ratio(heights):
        h = np.array(heights, dtype=float)
        # Step function on [-1/4, 1/4] with N intervals
        # L2 norm squared = (1/N) * sum(h_i^2)
        # L1 norm = (1/N) * sum(h_i)
        # Autoconvolution: need to compute (h*h)(t) for step function
        # The peak is at t=0: (1/N) * sum(h_i^2)
        # But other lags may have larger values if there's structure
        
        # For step function autoconvolution, the value at lag k (in units of interval width)
        # is (1/N) * sum_i h_i * h_{i-k} for valid i
        # Actually for continuous autoconvolution of step function:
        # (h*h)(t) = integral h(s)h(t-s) ds
        # For step function with heights h_0..h_{N-1} on intervals of width w=1/N:
        # This is a piecewise linear function. The maximum occurs at t=0 typically.
        
        # Compute autoconvolution values at integer lags (multiples of w)
        # At lag 0: (1/N) * sum(h_i^2)
        # At lag k: (1/N) * sum_{i} h_i * h_{i-k} where indices are valid
        # But for continuous, the peak might be between integer lags.
        
        # For safety, compute the discrete autoconvolution and take max
        h_conv = np.correlate(h, h, mode='full')
        # The continuous autoconvolution peak is bounded by the discrete one scaled appropriately
        # For step functions, the continuous autoconvolution at integer lags equals
        # (1/N) * discrete autoconvolution at that lag
        # The maximum of the continuous autoconvolution is at most the max of discrete / N
        # Actually, for non-negative step functions, the autoconvolution is maximized at t=0
        # and equals (1/N) * sum(h_i^2)
        
        l2_sq = (1.0 / N) * np.sum(h**2)
        l1 = (1.0 / N) * np.sum(h)
        
        if l1 == 0:
            return 0.0
        
        # Autoconvolution supremum: for non-negative step functions, peak is at t=0
        # (h*h)(0) = (1/N) * sum(h_i^2) = l2_sq
        # But we should verify no other point exceeds this.
        # For safety, compute the full discrete autoconvolution and scale
        max_auto = np.max(h_conv) / N  # This is an upper bound
        
        # Actually, for non-negative h, (h*h)(t) <= (h*h)(0) for all t
        # This is a known property for autoconvolution of non-negative functions
        # So supremum = l2_sq
        sup_auto = l2_sq
        
        ratio = l2_sq / (l1 * sup_auto)
        return ratio
    
    # Try various sparse patterns
    # Pattern 1: Single spike
    for pos in range(N):
        h = [0.0] * N
        h[pos] = 1.0
        val = compute_ratio(h)
        if val > best_value:
            best_value = val
            best_heights = h[:]
        if time.time() > deadline:
            break
    
    # Pattern 2: Two spikes
    if time.time() < deadline:
        for i in range(N):
            for j in range(i+1, N):
                h = [0.0] * N
                h[i] = 1.0
                h[j] = 1.0
                val = compute_ratio(h)
                if val > best_value:
                    best_value = val
                    best_heights = h[:]
            if time.time() > deadline:
                break
    
    # Pattern 3: Try different amplitudes for single spike (scale invariance check)
    # Since ratio scales as 1/height for single spike, use height=1.0 as baseline
    
    # Pattern 4: Try a few more structured patterns
    if time.time() < deadline:
        # Alternating pattern
        h = [1.0 if i % 2 == 0 else 0.0 for i in range(N)]
        val = compute_ratio(h)
        if val > best_value:
            best_value = val
            best_heights = h[:]
        
        # Block pattern
        for block_size in [1, 2, 5, 10]:
            h = [1.0] * block_size + [0.0] * (N - block_size)
            val = compute_ratio(h)
            if val > best_value:
                best_value = val
                best_heights = h[:]
    
    return best_heights
# EVOLVE-BLOCK-END
