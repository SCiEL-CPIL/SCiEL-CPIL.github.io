"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

N = 600

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    # Initial guess: all ones
    h = np.ones(N)
    
    # Compute autoconvolution (linear, not circular)
    def autoconv(h):
        # Use FFT for efficiency
        n = len(h)
        size = 2 * n
        H = np.fft.rfft(h, n=size)
        conv = np.fft.irfft(H * H, n=size)[:2*n-1]
        return conv
    
    def score_func(h):
        conv = autoconv(h)
        # Normalized maximum: max of autoconv / sum of squares
        # This is always >= 1 for nonnegative, but let's try
        s = np.sum(h**2)
        if s < 1e-12:
            return 1e10
        return np.max(conv) / s
    
    # Try a few different initializations and local optimizations
    best_h = h.copy()
    best_score = score_func(h)
    
    # Try parabolic shape
    x = np.linspace(-1, 1, N)
    h2 = 1.0 + 0.5 * (1 - x**2)
    h2 = np.maximum(h2, 0.0)
    s2 = score_func(h2)
    if s2 < best_score:
        best_score = s2
        best_h = h2.copy()
    
    # Try inverted parabolic
    h3 = 1.0 - 0.5 * x**2
    h3 = np.maximum(h3, 0.0)
    s3 = score_func(h3)
    if s3 < best_score:
        best_score = s3
        best_h = h3.copy()
    
    # Gradient descent with smoothing
    if time.time() < deadline - 20:
        lr = 0.01
        for it in range(500):
            if time.time() > deadline - 5:
                break
            conv = autoconv(best_h)
            s = np.sum(best_h**2)
            # Gradient of max(conv)/s with respect to h
            # Simplified: just try to reduce the peak
            idx = np.argmax(conv)
            # The gradient of conv[idx] w.r.t. h is h[idx - j] for various j
            # This is complex, so use a simple heuristic
            # Reduce heights near the peak of autoconv
            # The peak is at lag 0, which involves all h_i^2
            # Try to reduce variance in h
            mean_h = np.mean(best_h)
            var_h = np.var(best_h)
            if var_h > 1e-6:
                # Pull toward mean
                best_h = best_h - lr * (best_h - mean_h)
                best_h = np.maximum(best_h, 0.0)
                new_score = score_func(best_h)
                if new_score < best_score:
                    best_score = new_score
                else:
                    best_h = best_h + lr * (best_h - mean_h)  # revert
                    best_h = np.maximum(best_h, 0.0)
            else:
                break
    
    # Return as list of floats
    return [float(x) for x in best_h]
# EVOLVE-BLOCK-END
