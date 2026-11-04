"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np


def construct():
    """
    Construct signed step heights minimizing the maximum absolute autoconvolution
    divided by squared integral (C3' metric).
    
    Approach:
    - We want to minimize max_k |r(k)| / (sum x)^2 where r = x * x (autoconvolution)
    - This is equivalent to minimizing the peak of the autocorrelation relative to DC component.
    - Using ADMM with spectral masking:
      * Maintain signal x and dual spectral mask z
      * Iteratively:
        1. Update x to minimize ||x - z||^2 + penalty for large autocorrelation peaks
        2. Update z via proximal operator on the autocorrelation peak constraint
    
    For practical purposes within time limits, we use a simplified ADMM-like iteration:
    - Start with random sign pattern
    - Iteratively adjust signs to reduce peak autocorrelation
    - Use gradient descent on the peak value with subgradients
    
    Alternative simpler approach: Use Golay-like sequences or Barker codes adapted.
    But let's try a direct optimization:
    
    The key insight: if we can make the autocorrelation (aside from lag 0) as small as possible
    in absolute value, we minimize the max. This is related to low autocorrelation sequences.
    
    Let's use a spectral approach:
    - The autocorrelation r(k) = sum_i x[i] x[i+k]
    - In frequency domain, |X(f)|^2 is the periodogram, and r(k) are its Fourier coefficients
    - We want to minimize max_k |r(k)| subject to sum(x) being large (to maximize the denominator)
    
    Actually, the ratio is max_k |r(k)| / (sum x)^2.
    
    Let's try: start with a sequence that has good autocorrelation properties,
    then refine using local search / gradient-based methods.
    
    Given time constraints, I'll implement a practical ADMM-like solver:
    """
    
    # Initialize with a random sign sequence
    np.random.seed(42)
    x = np.random.choice([-1.0, 1.0], size=N)
    
    # We'll iterate to reduce the peak autocorrelation
    # Use a simplified ADMM approach
    
    alpha = 0.5  # step size
    rho = 10.0   # ADMM penalty parameter
    
    best_x = x.copy()
    best_score = np.inf
    
    for iteration in range(50):
        # Compute autocorrelation
        r = np.correlate(x, x, mode='full')
        # r has length 2N-1, centered at index N-1
        # r[N-1+k] is the autocorrelation at lag k
        
        # Find the max absolute value excluding the DC term (lag 0)
        # Actually, we want max over all lags including 0? Let's re-read.
        # "maximum absolute autoconvolution" - this likely includes all lags
        # But the DC term (lag 0) is sum(x^2) = N for binary, which is large
        # The denominator is (sum x)^2
        
        # Let's compute the metric
        sum_x = np.sum(x)
        if sum_x == 0:
            score = np.inf
        else:
            max_abs_r = np.max(np.abs(r))
            score = max_abs_r / (sum_x ** 2)
        
        if score < best_score:
            best_score = score
            best_x = x.copy()
        
        # If score is already very good, break
        if best_score < 1.0:
            break
        
        # Gradient-based update: try to reduce the peak
        # Find the lag with maximum absolute autocorrelation
        lag_idx = np.argmax(np.abs(r))
        lag = lag_idx - (N - 1)
        
        # Compute subgradient for the peak autocorrelation
        # d/dx max|r| is tricky; use the sign of the peak term
        if r[lag_idx] > 0:
            sign = 1.0
        else:
            sign = -1.0
        
        # The autocorrelation at lag k: r(k) = sum_i x[i] x[i+k]
        # Gradient w.r.t. x[j]: sign * (x[j+k] + x[j-k]) (with boundary handling)
        grad = np.zeros(N)
        for j in range(N):
            # Contribution from x[j] in r(k): x[j] appears in terms where i=j or i+k=j
            # r(k) = sum_i x[i] x[i+k]
            # d r(k)/d x[j] = x[j+k] (if j+k < N) + x[j-k] (if j-k >= 0)
            val = 0.0
            if j + lag < N:
                val += x[j + lag]
            if j - lag >= 0:
                val += x[j - lag]
            grad[j] = sign * val
        
        # Update x in direction that reduces the peak
        # x_new = x - alpha * grad
        # But we want to keep x as signed values, so project back
        x_new = x - alpha * grad
        # Project to [-1, 1]
        x_new = np.clip(x_new, -1.0, 1.0)
        
        # Optionally, round to signs to maintain binary nature
        # But continuous might be better for the metric
        x = x_new
    
    # Return the best found
    return best_x
# EVOLVE-BLOCK-END
