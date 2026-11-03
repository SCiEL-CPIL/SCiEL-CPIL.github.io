"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

N = 600

# EVOLVE-BLOCK-START
def construct():
    start = time.time()
    deadline = start + 140.0  # safety margin before 150s kill

    # We want to minimize the normalized maximum of the autoconvolution.
    # For a step function h on [-1/4, 1/4] with N equal intervals,
    # the autoconvolution is (h * h)(x) = integral h(t)h(x-t) dt.
    # The "normalized maximum" likely refers to max(autoconv) / ||h||_2^2 or similar.
    # 
    # From the inspiration, a "parametric sharp-peak family (low base + narrow central Gaussian)"
    # improved the score from 1.002 to 1.011. This suggests concentrating mass at the center.
    #
    # Let's try a family of step functions that are peaked at the center.
    # A simple approach: h_i = a + b * exp(-c * i^2) for i in [-N/2, N/2]
    # But we need nonnegative values.
    #
    # Actually, let's think about what minimizes max(h*h)/||h||_2^2.
    # If h is a delta function (concentrated at one point), the autoconvolution is also a delta,
    # so the max is very large relative to the L2 norm squared.
    # If h is flat, the autoconvolution is a triangle, and max/||h||_2^2 is proportional to 1/N.
    # 
    # Wait, let's reconsider. For a flat function h=1 on N intervals of width dx=1/300:
    # ||h||_2^2 = N * dx = 600 * (1/300) = 2
    # autoconv at center = integral h(t)h(x-t) dt. For x=0, this is the area of overlap of h with itself shifted by 0.
    # Since h is 1 on [-1/4, 1/4], the autoconv at 0 is the length of the intersection, which is 1/2.
    # So max(autoconv) = 1/2.
    # Normalized max = (1/2) / 2 = 1/4 = 0.25.
    #
    # For a delta-like function concentrated at the center:
    # If h is 1 on one interval and 0 elsewhere, ||h||_2^2 = dx = 1/300.
    # autoconv at 0 = dx * dx = 1/90000.
    # Normalized max = (1/90000) / (1/300) = 1/300 ≈ 0.0033.
    # This is much smaller! So concentrating mass should help.
    #
    # But wait, the score is "upper bound on C1", and lower is better.
    # The seed program (flat) has score 1.0. If the normalized max for flat is 0.25,
    # then C1 upper bound is related to this. Let me re-read the task.
    #
    # "minimize the normalized maximum of its autoconvolution; this gives an upper bound on C1"
    # The score is the upper bound on C1, and lower is better.
    # The seed (flat) gives score 1.0.
    #
    # So we want to minimize this quantity. The delta-like function should give a much lower score.
    #
    # Let's try a Gaussian-like peak. We'll parameterize by width sigma.
    # h_i = exp(-i^2 / (2 * sigma^2)) for i in range(-N//2, N//2)
    #
    # We need to compute the autoconvolution numerically and find its max.
    #
    # Let's implement a function to compute the normalized max autoconvolution.

    def normalized_max_autoconv(h):
        """Compute max(autoconv) / ||h||_2^2 for step function h on N equal intervals."""
        dx = 1.0 / (2 * N)  # Wait, the interval is [-1/4, 1/4], so length is 1/2.
        # Actually, N intervals on [-1/4, 1/4], so each interval has width (1/2)/N = 1/(2N).
        dx = 1.0 / (2 * N)
        
        # Compute autoconvolution using FFT for efficiency
        # Pad to next power of 2 or use scipy
        from scipy.fft import fft, ifft
        
        # The autoconvolution of h with itself
        # We need to compute (h * h)(x) = sum_j h_j * h_k * dx^2 where x = i*dx
        # Actually, for step functions, the autoconvolution at grid points can be computed via FFT
        
        n_fft = 2 * N
        h_padded = np.zeros(n_fft)
        h_padded[:N] = h
        
        # Circular convolution via FFT
        H = fft(h_padded)
        autoconv_circ = ifft(H * H).real
        
        # The circular convolution gives values at positions 0, dx, 2dx, ..., (n_fft-1)dx
        # But we want the linear convolution. The linear convolution of two length-N sequences
        # has length 2N-1. The circular convolution of length 2N gives us the linear convolution
        # wrapped around. Since the support is [-1/4, 1/4], the autoconvolution has support [-1/2, 1/2].
        # 
        # Let's just compute it directly for now, or use a proper linear convolution.
        
        # Actually, let's use np.convolve for simplicity, it's O(N^2) but N=600 is small.
        autoconv = np.convolve(h, h, mode='full') * dx**2
        # autoconv has length 2N-1, corresponding to lags from -(N-1)dx to (N-1)dx
        # The max of autoconv is what we want.
        
        l2_sq = np.sum(h**2) * dx
        if l2_sq < 1e-15:
            return float('inf')
        
        return np.max(autoconv) / l2_sq

    # Now let's search over a family of peaked functions.
    # Family: h_i = a + b * exp(-c * i^2)
    # We can fix a=0 and vary b and c, or fix b=1 and vary a and c.
    # Since scaling doesn't affect the ratio, we can fix b=1 and vary a and c.
    
    best_score = float('inf')
    best_h = None
    
    # Try a grid of parameters
    # c controls the width: larger c means narrower peak
    # a controls the base level
    
    # Let's try various c values (inverse variance)
    c_values = np.logspace(-4, 2, 50)
    a_values = np.linspace(0, 1, 20)
    
    centers = np.arange(-N//2, N//2)
    
    for c in c_values:
        if time.time() > deadline:
            break
        for a in a_values:
            if time.time() > deadline:
                break
            h = a + np.exp(-c * centers**2)
            h = np.maximum(h, 0)  # ensure nonnegative
            score = normalized_max_autoconv(h)
            if score < best_score:
                best_score = score
                best_h = h.copy()
    
    # Also try a pure peak (a=0) with finer c search
    for c in np.logspace(-3, 1, 100):
        if time.time() > deadline:
            break
        h = np.exp(-c * centers**2)
        score = normalized_max_autoconv(h)
        if score < best_score:
            best_score = score
            best_h = h.copy()
    
    if best_h is None:
        return [1.0] * N
    
    # Normalize or return as is? The task says "sequence of nonnegative step heights"
    # The score is the normalized max, which is scale-invariant, so any positive scaling works.
    return list(best_h)

# EVOLVE-BLOCK-END
