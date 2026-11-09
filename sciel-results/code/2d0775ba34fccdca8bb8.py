import numpy as np
import time

N = 600
H = 1.0 / (2 * N)  # Interval width

def compute_autoconvolution(f):
    """
    Compute the autoconvolution of the step function f.
    f is a vector of heights.
    Returns the autoconvolution values at grid points.
    """
    # Zero-pad to avoid circular convolution
    # Length of autoconvolution is 2*N - 1
    # We need FFT size at least 2*N
    n_fft = 2 * N
    F = np.fft.rfft(f, n=n_fft)
    # Autoconvolution in frequency domain is |F|^2
    # Inverse FFT gives the linear convolution (zero-padded)
    c = np.fft.irfft(F * np.conj(F), n=n_fft)
    # The result c has length n_fft.
    # The autoconvolution of f with f is centered at index 0 in circular sense,
    # but in linear convolution with zero padding, the peak is at index 0?
    # No, np.fft.irfft returns the sequence such that c[k] = sum f[i] f[k-i] (circular).
    # With zero padding, c[k] for k=0..N-1 corresponds to the linear convolution.
    # Specifically, c[0] is the sum of f[i]*f[i] = ||f||^2.
    # The symmetry is c[k] = c[-k].
    # We want the maximum of c[k] for k != 0.
    # Note: c is real and symmetric.
    # The peak is at c[0].
    # The sidelobes are c[1], c[2], ...
    return c

def construct():
    start_time = time.time()
    deadline = 140.0  # 10s margin
    
    # Initialize with edge concentration
    f = np.zeros(N)
    # Put mass in the first and last quarters
    f[:N//4] = 1.0
    f[3*N//4:] = 1.0
    
    # Normalize such that (f*f)(0) = 1
    # (f*f)(0) = sum(f^2) * H (if we consider continuous integral)
    # But the ratio is scale invariant.
    # Let's just work with the discrete sum for the ratio.
    # Ratio = max(c[1:]) / c[0]
    
    best_f = f.copy()
    best_ratio = np.inf
    
    # Gradient Descent on the maximum sidelobe
    # Objective: minimize max(c[1:])
    # We use a smooth approximation or subgradient
    
    eta = 0.01
    step = 0
    
    while time.time() < deadline:
        # Compute autoconvolution
        c = compute_autoconvolution(f)
        
        # Main lobe peak
        main_peak = c[0]
        
        # Sidelobe peak
        # c is symmetric. We only need to look at c[1:N]
        # Note: c has length 2*N. The linear convolution part is c[0:N] and c[N:2N] is the wrap-around (which is zero if padding is sufficient? No, padding is 2N, so linear conv is in c[0:N-1] and c[N:2N-1] is the other side).
        # Actually, for n_fft = 2*N, the linear convolution of two length N signals fits in 2*N-1 points.
        # c[0] is index 0.
        # c[1] ... c[N-1] are positive lags.
        # c[N] ... c[2N-1] are negative lags (circularly).
        # Due to symmetry, c[k] = c[2N-k].
        # So we just need to check c[1:N].
        
        sidelobe_max = np.max(c[1:N])
        
        # Update best
        if main_peak > 0:
            ratio = sidelobe_max / main_peak
            if ratio < best_ratio:
                best_ratio = ratio
                best_f = f.copy()
        
        # Compute gradient of the maximum sidelobe
        # Find the index of the max sidelobe
        k_max = np.argmax(c[1:N]) + 1
        
        # Gradient of c[k_max] w.r.t f
        # c[k] = sum_i f[i] f[i+k]
        # d(c[k])/d(f[j]) = f[j+k] + f[j-k]
        
        grad = np.zeros(N)
        # f[j+k] part
        # For j=0..N-1, we need f[j+k]. If j+k >= N, it's 0.
        # We can use np.roll
        f_shifted_plus = np.roll(f, -k_max)
        # But we need to zero out the parts that wrapped around
        # Actually, the convolution is linear.
        # f[j+k] is non-zero only if j+k < N.
        # So f_shifted_plus[j] is f[j+k].
        # If j+k >= N, f_shifted_plus[j] is f[j+k-N], which is incorrect.
        # We must mask it.
        mask_plus = np.arange(N) + k_max < N
        f_shifted_plus[~mask_plus] = 0
        
        f_shifted_minus = np.roll(f, k_max)
        mask_minus = np.arange(N) - k_max >= 0
        f_shifted_minus[~mask_minus] = 0
        
        grad = f_shifted_plus + f_shifted_minus
        
        # Update f
        f = f - eta * grad
        
        # Project to non-negative
        f = np.maximum(f, 0)
        
        # Renormalize to keep main_peak = 1 (or constant)
        # This keeps the scale fixed
        norm = np.sqrt(np.sum(f**2))
        if norm > 1e-9:
            f = f / norm
        
        step += 1
        # Adaptive step size
        if step % 100 == 0:
            eta *= 0.9

    # Final calculation of the score value
    c = compute_autoconvolution(best_f)
    main_peak = c[0]
    sidelobe_max = np.max(c[1:N])
    
    # The "value" is the ratio
    if main_peak > 0:
        value = sidelobe_max / main_peak
    else:
        value = 1.0
        
    # The score is seed/value. Seed is 1.0.
    # The evaluator expects the "value" to be returned?
    # "score is seed/value"
    # "score is the upper bound on C1"
    # This implies the returned value IS the upper bound.
    # So we return `value`.
    
    # However, the prompt says "construct() -> sequence of nonnegative step heights"
    # And "score is the upper bound on C1"
    # Usually, the evaluator calls construct(), gets the heights, computes the score itself.
    # "The evaluator measures the raw value"
    # So I just need to return the heights that minimize the ratio.
    
    return list(best_f)

# EVOLVE-BLOCK-END
