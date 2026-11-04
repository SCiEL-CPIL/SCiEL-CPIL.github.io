import numpy as np
from scipy.optimize import minimize
import time

N = 600

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    w = 1.0 / (4 * N)  # interval width: total length 1/2, N intervals
    
    def autoconv_peak_normalized(h):
        """Compute normalized max of autoconvolution."""
        # h is array of N nonnegative heights
        # Autoconvolution of step function: 
        # For step function on [-1/4, 1/4] with N intervals of width w,
        # the autoconvolution is computed via FFT or direct convolution
        
        # Pad to handle full convolution length
        # The autoconvolution has support on [-1/2, 1/2], length 2N intervals of width w
        # Use FFT for efficiency
        n_fft = 2 * N
        h_ext = np.zeros(n_fft)
        h_ext[:N] = h
        
        # Convolve h with itself
        conv = np.fft.irfft(np.fft.rfft(h_ext) ** 2, n=n_fft)
        
        # The autoconvolution values: conv[k] corresponds to the integral contribution
        # Actually, for step functions, the autoconvolution at a point is:
        # (f*f)(t) = sum over pairs where the overlap gives the product
        # More precisely, if f(x) = h[i] for x in [x_i, x_{i+1}], then
        # (f*f)(t) = integral f(s) f(t-s) ds
        # This is a piecewise linear function. The peaks occur at the midpoints.
        
        # Let me compute this more carefully.
        # The autoconvolution of a step function can be computed as:
        # For each pair (i,j), the contribution to (f*f) from h[i]*h[j] 
        # is a triangle function with base 2w centered at (x_i_center + x_j_center)
        
        # Actually, let me use the discrete convolution approach:
        # The autoconvolution sampled at the grid points (midpoints of intervals)
        # can be approximated by the discrete autoconvolution.
        
        # Let's compute the continuous autoconvolution properly.
        # f(x) = h[i] for x in [a_i, a_i + w] where a_i = -1/4 + i*w
        # (f*f)(t) = integral_{-infty}^{infty} f(s) f(t-s) ds
        # = sum_i sum_j h[i] h[j] * measure({s: s in [a_i, a_i+w] and t-s in [a_j, a_j+w]})
        # = sum_i sum_j h[i] h[j] * max(0, min(a_i+w, t-a_j) - max(a_i, t-a_j-w))
        
        # This is complex. Let me use FFT-based approach with proper scaling.
        # The autoconvolution of a piecewise constant function can be computed
        # by convolving the step function with itself.
        
        # Using the fact that for step functions, the autoconvolution is a 
        # piecewise linear function. Let me compute it on a fine grid.
        
        # Alternative: use the discrete approximation
        # The integral of f = w * sum(h)
        # The autoconvolution peak: for uniform, it's w * N * w = w^2 * N^2 / 2? 
        # No, let me just compute numerically.
        
        # Compute autoconvolution on a grid of 2N points
        # The autoconvolution (f*f)(t) for t in [-1/2, 1/2]
        # Sample at 2N equally spaced points
        
        # Actually, let me use a simpler approach: 
        # The autoconvolution of a step function with heights h[i] and width w
        # can be computed as follows:
        # Create a dense representation and convolve
        
        # For efficiency, use the fact that the autoconvolution of a step function
        # is a "splat" convolution. Let me compute it directly.
        
        # The autoconvolution at point t is:
        # (f*f)(t) = sum_i h[i] * (integral of f(t-s) ds over s in interval i)
        # = sum_i h[i] * (sum_j h[j] * length of intersection of [a_i, a_i+w] with [t-a_j-w, t-a_j])
        
        # This is O(N^2) which is 360000, acceptable for 600.
        
        a = -0.25 + w * np.arange(N)  # left edges of intervals
        
        # Compute autoconvolution at 2N sample points
        t_samples = -0.5 + (0.5 / (2 * N)) * np.arange(2 * N + 1)
        # Actually, let me sample at the critical points: the autoconvolution is
        # piecewise linear with breakpoints at t = a_i + a_j and t = a_i + a_j + w
        # for all i, j. There are O(N^2) breakpoints.
        
        # For simplicity, sample on a fine grid
        n_samples = 4 * N
        t_grid = np.linspace(-0.5, 0.5, n_samples, endpoint=False)
        dt = 1.0 / n_samples
        
        # Compute (f*f)(t) for each t in t_grid
        # (f*f)(t) = sum_i h[i] * integral_{s in interval_i} f(t-s) ds
        # For s in [a_i, a_i+w], t-s ranges from [t-a_i-w, t-a_i]
        # f(t-s) = h[j] when t-s in [a_j, a_j+w], i.e., s in [t-a_j-w, t-a_j]
        # So integral = sum_j h[j] * length of intersection of [a_i, a_i+w] and [t-a_j-w, t-a_j]
        
        # This is O(N^2 * n_samples) which is 600*600*2400 = 864M, too slow.
        
        # Let me use FFT instead.
        # The autoconvolution can be computed via FFT if I represent f on a fine grid.
        
        # Represent f on a grid of M points covering [-1/4, 1/4]
        M = N * 4  # 4 subintervals per step for better accuracy
        x = np.linspace(-0.25, 0.25, M, endpoint=False)
        dx = 0.5 / M
        
        # Interpolate h onto the fine grid
        # Each step interval i covers [a_i, a_i+w], which is M/N fine grid points
        f_fine = np.zeros(M)
        for i in range(N):
            idx_start = int(i * (M / N))
            idx_end = int((i + 1) * (M / N))
            f_fine[idx_start:idx_end] = h[i]
        
        # Autoconvolution via FFT
        # Pad to length 2M
        n_fft = 2 * M
        f_pad = np.zeros(n_fft)
        f_pad[:M] = f_fine
        
        conv = np.fft.irfft(np.fft.rfft(f_pad) ** 2, n=n_fft)
        
        # The autoconvolution values: conv[k] * dx gives the actual integral
        # conv[k] is the discrete convolution sum, so (f*f)(t_k) ≈ conv[k] * dx
        # where t_k = k * dx - 0.25 (shifted)
        
        # Actually, the FFT convolution gives:
        # conv[k] = sum_m f_fine[m] * f_fine[(k-m) mod n_fft]
        # This approximates (1/dx) * (f*f)(t_k) where t_k = (k - M + 1) * dx
        # Hmm, let me be more careful.
        
        # The continuous autoconvolution (f*f)(t) = integral f(s) f(t-s) ds
        # Discrete approximation: (1/dx) * sum_m f_fine[m] * f_fine[m'] where 
        # t - s_m = s_{m'}, i.e., m' = m + t/dx
        # So (f*f)(t) ≈ dx * sum_m f_fine[m] * f_fine[m + t/dx]
        
        # The FFT gives conv[k] = sum_m f_fine[m] * f_fine[(k-m) mod n_fft]
        # For k in [0, M], this corresponds to t = (k - M + 1) * dx approximately
        
        # Let me just compute the max of the autoconvolution
        # The peak should be near the center
        
        # Scale: the FFT convolution sum needs to be multiplied by dx to get the integral
        conv_values = conv * dx
        
        # The autoconvolution is supported on [-0.5, 0.5]
        # conv[k] for k=0 corresponds to t = -0.5 + something
        # Let me just find the max
        peak = np.max(conv_values)
        
        # Normalization: divide by (integral of f)^2
        integral_f = np.sum(h) * w
        normalized_peak = peak / (integral_f ** 2) if integral_f > 0 else 0
        
        return normalized_peak
    
    # Initial guess: uniform
    h0 = np.ones(N)
    
    # Ensure nonnegativity
    def objective(h):
        h_nonneg = np.maximum(h, 0)
        return autoconv_peak_normalized(h_nonneg)
    
    # Use Nelder-Mead or L-BFGS-B
    # Since we have 600 variables, use L-BFGS-B with bounds
    bounds = [(0, None)] * N
    
    best_h = h0.copy()
    best_score = objective(best_h)
    
    # Try a few optimization runs
    n_restarts = 5
    for restart in range(n_restarts):
        if time.time() > deadline:
            break
        
        if restart == 0:
            h_init = h0.copy()
        else:
            # Add some noise
            h_init = h0.copy()
            noise = np.random.randn(N) * 0.1
            h_init = np.maximum(h_init + noise, 0)
        
        try:
            result = minimize(objective, h_init, method='L-BFGS-B', bounds=bounds, 
                            options={'maxiter': 200, 'ftol': 1e-8, 'gtol': 1e-6})
            
            if result.fun < best_score:
                best_score = result.fun
                best_h = np.maximum(result.x, 0)
        except Exception as e:
            pass
    
    # Convert back to list
    return list(best_h)
# EVOLVE-BLOCK-END
