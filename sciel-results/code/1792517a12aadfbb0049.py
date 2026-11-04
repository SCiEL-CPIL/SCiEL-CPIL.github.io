import numpy as np
from scipy.signal import fftconvolve

N = 400
w = 1.0 / 200.0  # interval width

def construct():
    # Start with all ones
    s = np.ones(N)
    
    # Compute autoconvolution and its L-infinity norm
    # The autoconvolution of the step function can be computed from the discrete autocorrelation
    # For a step function with values s_i on intervals of width w:
    # (f*f)(t) is piecewise linear, and its values at the interval boundaries and midpoints
    # can be determined from the discrete autocorrelation
    
    # Discrete autocorrelation: r[k] = sum_{i=0}^{N-1-k} s[i] * s[i+k] for k = 0, ..., N-1
    # The continuous autoconvolution at shift t = k*w + delta is:
    # (f*f)(t) = w * (r[k] - 2*delta/w * something) ... 
    
    # Actually, let me think again. f(x) = sum_i s_i * 1_{[i*w, (i+1)*w]}(x + offset)
    # The support is [-1/4, 1/4], so x_i = -1/4 + i*w for i=0..N-1
    
    # (f*f)(t) = integral f(x) f(t-x) dx
    # For t in [0, N*w], this is a piecewise linear function
    # The slope changes at each multiple of w
    
    # Let me just compute it directly for the maximum
    
    # Simpler: use the fact that for a step function, the autoconvolution's maximum
    # is related to the discrete autocorrelation. The peak of the autoconvolution
    # at t=0 is w * sum(s[i]^2) = w * N = 2 for all ones.
    
    # For minimizing the L-infinity norm, we want to reduce the peaks.
    
    # Let me use a gradient descent approach on the sign variables (relaxed to [-1,1])
    
    def compute_autoconv_max(s):
        # Compute the discrete autocorrelation
        # r[k] = sum_{i} s[i] * s[i+k]
        # Using FFT for efficiency
        n_fft = 2 * N
        S = np.fft.rfft(s, n_fft)
        r = np.fft.irfft(S * np.conj(S), n_fft)[:N]
        
        # The continuous autoconvolution is piecewise linear
        # At t = k*w (for integer k), the value is w * r[k]
        # Between k*w and (k+1)*w, it's linear
        # The maximum absolute value occurs at the endpoints of each piece
        # which are at t = k*w for k = 0, 1, ..., N-1 (and negative k by symmetry)
        
        # Actually, the autoconvolution is symmetric, so we only need k >= 0
        # The values at the breakpoints are w * r[k] for k = 0, ..., N-1
        # But we also need to check the midpoints? No, since it's linear between breakpoints,
        # the extremum is at the endpoints.
        
        # Wait, I need to be more careful. The autoconvolution of two step functions
        # is a piecewise linear function that is trapezoidal in shape for each pair of intervals.
        
        # Let me just evaluate at all the breakpoints: t = k*w for k = -(N-1), ..., N-1
        # By symmetry, (f*f)(t) = (f*f)(-t), so we just need k = 0, ..., N-1
        
        # The value at t = k*w is:
        # sum_{i: i*w >= 0 and (i+1)*w <= (N-1)*w and ...} 
        # Actually, it's simpler: the overlap of interval i and interval j when shifted by k*w
        # is w if |i-j| = k, 0 otherwise (for the discrete case)
        
        # For the continuous case, the autoconvolution at t = k*w is exactly w * r[k]
        # where r[k] is the discrete autocorrelation
        
        # But the maximum might occur between breakpoints? No, since it's linear,
        # the max is at the breakpoints.
        
        max_val = np.max(np.abs(r)) * w
        return max_val
    
    # Subgradient descent on relaxed signs
    # The objective is max_k |w * r[k]| where r[k] = sum_i s[i] s[i+k]
    
    # This is non-smooth, so we use subgradient method
    # The subgradient of max_k |r[k]| with respect to s[i] involves the k that achieves the max
    
    best_s = s.copy()
    best_val = compute_autoconv_max(s)
    
    for iteration in range(5000):
        # Compute autocorrelation
        n_fft = 2 * N
        S = np.fft.rfft(s, n_fft)
        r = np.fft.irfft(S * np.conj(S), n_fft)[:N]
        
        # Find the k that achieves the maximum absolute value
        k_max = np.argmax(np.abs(r))
        r_max = r[k_max]
        
        # Subgradient of |r[k_max]| with respect to s
        # r[k_max] = sum_{i=0}^{N-1-k_max} s[i] * s[i+k_max]
        # d|r[k_max]|/ds[i] = sign(r[k_max]) * (s[i+k_max] if i < N-k_max else 0)
        #                    + sign(r[k_max]) * (s[i-k_max] if i >= k_max else 0)
        
        sg = np.zeros(N)
        sign_r = np.sign(r_max)
        
        for i in range(N):
            grad = 0.0
            if i < N - k_max:
                grad += s[i + k_max]
            if i >= k_max:
                grad += s[i - k_max]
            sg[i] = sign_r * grad
        
        # Update: s = s - lr * sg, then project to [-1, 1]
        lr = 0.1 / (1 + iteration)
        s = s - lr * sg
        s = np.clip(s, -1, 1)
        
        # Check current value
        current_val = compute_autoconv_max(s)
        if current_val < best_val:
            best_val = current_val
            best_s = s.copy()
    
    # Round to ±1
    s_final = np.sign(best_s)
    # Handle zeros
    s_final[s_final == 0] = 1
    
    return s_final
