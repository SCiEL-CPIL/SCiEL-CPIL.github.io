import numpy as np
import time

N = 600
dx = (1.0 / 4.0 - (-1.0 / 4.0)) / N  # 1/1200

def _autoconv_max(heights):
    """Compute max of autoconvolution of step function with given heights.
    Uses FFT for efficiency. The autoconvolution is computed as:
    c[k] = sum_i h[i] * h[i+k] * dx for k >= 0, and c[-k] = c[k].
    We use FFT to compute the full convolution and take the max over valid range.
    """
    heights = np.asarray(heights, dtype=float)
    N = len(heights)
    # Use FFT to compute autoconvolution
    n_fft = 2 * N
    h_padded = np.zeros(n_fft)
    h_padded[:N] = heights
    H = np.fft.fft(h_padded, n=n_fft)
    # Autoconvolution: ifft(H * conj(H)) gives the autocorrelation
    # Actually for real signals, autoconvolution = ifft(fft(h) * fft(h))
    # But we want the circular convolution, so we need to be careful
    # The linear autoconvolution of h with itself:
    conv = np.fft.ifft(H * H.conj()).real[:N] * dx
    # conv[k] for k=0..N-1 is the autoconvolution at lag k*dx
    # The maximum is at k=0 for nonnegative h, but let's check all
    return np.max(conv)

def _objective(heights):
    """Objective: normalized max of autoconvolution.
    Normalize by (sum h_i * dx)^2 to make it scale-invariant.
    """
    heights = np.maximum(heights, 0.0)  # Ensure nonnegative
    integral = np.sum(heights) * dx
    if integral < 1e-15:
        return 1e10
    max_conv = _autoconv_max(heights)
    return max_conv / (integral * integral)

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s
    
    # Strategy: Two-block disjoint rectangles
    # Represent f as sum of two rectangular blocks with different amplitudes
    # Block 1: positions [a1, b1], amplitude A1
    # Block 2: positions [a2, b2], amplitude A2
    # where [a1,b1] and [a2,b2] are disjoint intervals within [0, N-1]
    
    # The autoconvolution of two rectangles has a known analytic form
    # For a single rectangle of width w and amplitude A on domain of measure L:
    # (f*f)(x) is a trapezoid with max A^2 * w at x=0
    # For two disjoint rectangles, the autoconvolution is the sum of:
    # - self-convolutions of each block
    # - cross-convolutions between the two blocks
    
    # Let's use a parameterized approach:
    # x = [start1, width1, amp1, start2, width2, amp2]
    # All in units of dx (i.e., index space)
    
    def _blocks_to_heights(params):
        s1, w1, a1, s2, w2, a2 = params
        s1 = int(round(np.clip(s1, 0, N - 1)))
        w1 = int(round(np.clip(w1, 1, N)))
        a1 = max(0.0, a1)
        s2 = int(round(np.clip(s2, 0, N - 1)))
        w2 = int(round(np.clip(w2, 1, N)))
        a2 = max(0.0, a2)
        
        # Ensure blocks are within bounds
        w1 = min(w1, N - s1)
        w2 = min(w2, N - s2)
        
        heights = np.zeros(N)
        if w1 > 0 and a1 > 0:
            heights[s1:s1 + w1] = a1
        if w2 > 0 and a2 > 0:
            heights[s2:s2 + w2] = a2
        return heights
    
    def _objective_blocks(params):
        heights = _blocks_to_heights(params)
        return _objective(heights)
    
    # Grid search over block configurations
    best_val = 1e10
    best_params = None
    
    # Try various configurations of two blocks
    # Parameterize by: relative positions and widths
    # Let total support be split into two blocks
    
    # Strategy: Try symmetric and asymmetric splits
    # For each split, try different amplitudes
    
    # First, do a coarse grid search
    num_coarse = 30
    t0 = time.time()
    
    for i in range(num_coarse):
        if time.time() > deadline - 10:
            break
        # Block 1: start at 0, width w1
        # Block 2: start at s2, width w2
        # Try different splits
        
        # Symmetric: two equal-width blocks at opposite ends
        w = N // (2 * (i + 1)) if i < 5 else max(1, N // 20)
        gap = (N - 2 * w) // 2
        
        # Block 1 at [0, w), Block 2 at [N-w, N)
        s1, w1 = 0, w
        s2, w2 = N - w, w
        
        # Try different amplitude ratios
        for ratio in [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0]:
            a1 = 1.0
            a2 = ratio
            params = [s1, w1, a1, s2, w2, a2]
            val = _objective_blocks(params)
            if val < best_val:
                best_val = val
                best_params = params.copy()
        
        # Also try middle split
        w1 = N // 3
        w2 = N // 3
        s1 = 0
        s2 = N - w2
        for ratio in [0.5, 0.75, 1.0]:
            a1 = 1.0
            a2 = ratio
            params = [s1, w1, a1, s2, w2, a2]
            val = _objective_blocks(params)
            if val < best_val:
                best_val = val
                best_params = params.copy()
    
    # Refine with local optimization around best found
    if best_params is not None:
        from scipy.optimize import minimize
        
        # Continuous parameterization for refinement
        def _objective_continuous(x):
            s1, w1, a1, s2, w2, a2 = x
            s1 = np.clip(s1, 0, N - 1)
            w1 = np.clip(w1, 1, N)
            a1 = max(0.0, a1)
            s2 = np.clip(s2, 0, N - 1)
            w2 = np.clip(w2, 1, N)
            a2 = max(0.0, a2)
            
            heights = np.zeros(N)
            i1 = int(round(s1))
            i2 = int(round(s2))
            iw1 = int(round(w1))
            iw2 = int(round(w2))
            iw1 = min(iw1, N - i1)
            iw2 = min(iw2, N - i2)
            
            if iw1 > 0 and a1 > 0:
                heights[i1:i1 + iw1] = a1
            if iw2 > 0 and a2 > 0:
                heights[i2:i2 + iw2] = a2
            return _objective(heights)
        
        bounds = [(0, N), (1, N), (0, 5), (0, N), (1, N), (0, 5)]
        try:
            res = minimize(
                _objective_continuous,
                best_params,
                method='Nelder-Mead',
                bounds=bounds,
                options={'maxiter': 2000, 'xatol': 1e-4, 'fatol': 1e-8}
            )
            if res.fun < best_val:
                best_val = res.fun
                best_params = res.x.copy()
        except Exception:
            pass
    
    # Convert best params to heights
    if best_params is not None:
        heights = _blocks_to_heights(best_params)
    else:
        # Fallback: constant function
        heights = np.ones(N)
    
    # Ensure nonnegative
    heights = np.maximum(heights, 0.0)
    
    return list(heights)
