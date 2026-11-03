"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

N = 600

def _autoconvolve_peak(x):
    """Return the maximum value of the autoconvolution of x (zero-padded)."""
    # Use FFT-based convolution for speed
    n = len(x)
    # Zero-pad to avoid circular convolution
    f = np.fft.rfft(x, n=2 * n)
    conv = np.fft.irfft(f * f, n=2 * n)
    return np.max(conv)

def _smooth_max(values, sigma=1.0):
    """Log-sum-exp based smooth maximum."""
    m = np.max(values)
    return m + sigma * np.log(np.sum(np.exp((values - m) / sigma)))

def _score(x):
    """Compute the upper bound on C1 from the step function x.
    
    The autoconvolution of a step function on [-1/4, 1/4] with N intervals:
    Each interval has width h = 1/2 / N.
    The autoconvolution at lag k*h is sum_i x_i x_{i+k} * h (approximately, 
    but we need to be careful about the exact formulation).
    
    Actually, for a step function f(t) = x_i on interval i, the autoconvolution
    (f*f)(s) = integral f(t) f(s-t) dt.
    
    With N equal intervals on [-1/4, 1/4], interval width h = (1/4 - (-1/4))/N = 1/(2N).
    
    The autoconvolution is a piecewise quadratic function. Its maximum occurs at
    some point. For a step function, the autoconvolution at the center (s=0) is
    sum_i x_i^2 * h (since when s=0, f(t)f(-t) and the overlap is full interval width).
    
    Wait, let me think more carefully. The autoconvolution (f*f)(s) = ∫_{-∞}^{∞} f(t)f(s-t) dt.
    
    For a step function on [-1/4, 1/4] with N intervals of width h = 1/(2N):
    f(t) = x_i for t in [-1/4 + i*h, -1/4 + (i+1)*h), i = 0, ..., N-1.
    
    (f*f)(s) = ∫ f(t) f(s-t) dt
    
    This is a convolution of the step function with itself. The result is a piecewise
    quadratic function. The maximum of this function gives the upper bound on C1.
    
    For computational efficiency, we can compute the autoconvolution at the grid points
    and take the max. But actually, the max of the autoconvolution of a step function
    occurs at one of the "critical points" which are at positions that are multiples of h.
    
    Actually, for a step function, the autoconvolution is a piecewise quadratic function
    whose breakpoints occur at s = (j - i)*h for various i, j. The maximum will occur
    either at a breakpoint or where the derivative is zero within a piece.
    
    For simplicity and speed, let's use FFT to compute the autoconvolution at fine grid points.
    """
    h = 1.0 / (2 * N)
    # The autoconvolution of the step function. 
    # If we treat x as a sequence and compute linear convolution, the result at index k
    # corresponds to the integral over a region of width h.
    # Specifically, (f*f)(k*h) ≈ h * sum_i x_i * x_{i+k} for the discrete approximation,
    # but this is not exactly right for the continuous case.
    
    # Let's use the fact that the autoconvolution of a step function can be computed
    # exactly. The autoconvolution at lag m*h (where m can be from -(N-1) to N-1) is:
    # sum over all pairs (i, j) such that the overlap of interval i and interval (j) 
    # shifted by m*h has length... this is getting complex.
    
    # Simpler approach: use FFT with proper scaling.
    # The continuous autoconvolution at s = m*h is:
    # (f*f)(m*h) = sum_{i,j} x_i x_j * length of intersection of interval_i and (interval_j shifted by m*h)
    # 
    # For step functions on a regular grid, the autoconvolution at the grid points
    # m*h is: h * sum_i x_i * x_{i+m} (with x_i = 0 outside [0,N-1])... 
    # No, that's not right either.
    
    # Let me use a direct FFT-based approach with proper scaling.
    # The autoconvolution of the step function can be computed as:
    # 1. Treat x as a vector
    # 2. Compute linear convolution of x with itself: conv = np.convolve(x, x, 'full')
    # 3. The result conv[k] = sum_i x[i] * x[k-i] for valid i, k-i
    # 4. Multiply by h to get the actual autoconvolution values at the grid points
    # 5. But the max of the continuous autoconvolution may be higher than at grid points
    
    # For a step function, the autoconvolution is piecewise quadratic. Between grid points,
    # it can bulge. The maximum between two adjacent grid points can be found analytically.
    # But for an upper bound, taking the max at grid points gives a lower bound on the true max.
    # We need the true max for the score.
    
    # Let's compute the autoconvolution at fine grid points for accuracy.
    n_fine = 2 * N
    # Sample the step function at fine grid
    # Actually, let's just use the FFT approach with the discrete approximation
    # and note that for step functions, the discrete convolution with proper scaling
    # gives the autoconvolution at the cell centers, and the true max is slightly higher.
    
    # For the purpose of this task, let's use the discrete autoconvolution scaled by h.
    # This should give a reasonable approximation of the upper bound on C1.
    
    conv = np.convolve(x, x, mode='full')
    peak = np.max(conv) * h
    
    # The actual maximum of the continuous autoconvolution is at least this value.
    # It could be slightly higher between grid points. For a tighter bound, we could
    # evaluate at finer points, but this should be a good approximation.
    
    return peak

def construct():
    """Construct a nonnegative step function with 600 intervals on [-1/4, 1/4]
    to minimize the normalized maximum of its autoconvolution.
    
    Returns: list of 600 nonnegative step heights.
    """
    start_time = time.time()
    deadline = start_time + 115.0  # Leave 5s safety margin
    
    # Start with uniform function
    x = np.ones(N)
    best_x = x.copy()
    best_score = _score(x)
    
    # Use spectral parameterization with coordinate descent
    # Transform to frequency domain
    X = np.fft.rfft(x)
    
    # We want to minimize the peak of the autoconvolution.
    # In the frequency domain, the autoconvolution corresponds to |X|^2 (with scaling).
    # Minimizing the peak of the autoconvolution is related to minimizing the
    # L-infinity norm of the inverse FFT of |X|^2.
    
    # Strategy: Use subgradient-like updates in the time domain, or
    # adjust frequency coefficients to flatten the autoconvolution peak.
    
    # Let's try a simple approach: iterative clipping and renormalization.
    # The idea is to reduce peaks in the autoconvolution by adjusting x.
    
    # Alternative: Use the fact that we want the autoconvolution to be as flat as possible.
    # For a given L2 norm, the flat autoconvolution corresponds to a specific shape.
    
    # Let's try: start with a raised cosine or similar smooth shape, then refine.
    
    # Raised cosine in time domain
    t = np.linspace(-1/4, 1/4, N, endpoint=False)
    x_rc = 0.5 * (1 + np.cos(2 * np.pi * t))  # This is a raised cosine
    x_rc = np.maximum(x_rc, 0)
    
    # Normalize to have the same L2 norm as the uniform function
    norm_orig = np.sum(x) * (1.0 / (2 * N))  # Integral of uniform function
    norm_rc = np.sum(x_rc) * (1.0 / (2 * N))
    if norm_rc > 0:
        x_rc = x_rc * (norm_orig / norm_rc)
    
    score_rc = _score(x_rc)
    if score_rc < best_score:
        best_score = score_rc
        best_x = x_rc.copy()
    
    # Try a triangle (Bartlett) window
    x_tri = 1.0 - 2 * np.abs(t)
    x_tri = np.maximum(x_tri, 0)
    norm_tri = np.sum(x_tri) * (1.0 / (2 * N))
    if norm_tri > 0:
        x_tri = x_tri * (norm_orig / norm_tri)
    
    score_tri = _score(x_tri)
    if score_tri < best_score:
        best_score = score_tri
        best_x = x_tri.copy()
    
    # Now do iterative refinement: subgradient descent on the autoconvolution peak
    x = best_x.copy()
    h = 1.0 / (2 * N)
    
    # Learning rate
    lr = 0.01
    
    # Number of iterations based on time budget
    max_iter = 2000
    for it in range(max_iter):
        if time.time() > deadline - 5.0:
            break
        
        # Compute autoconvolution
        conv = np.convolve(x, x, mode='full')
        conv_scaled = conv * h
        
        # Find the peak location
        peak_idx = np.argmax(conv_scaled)
        peak_val = conv_scaled[peak_idx]
        
        # Compute subgradient: the gradient of the peak w.r.t. x
        # If the peak is at index k, then d(peak)/dx_i = h * x_{k-i} (approximately)
        # This is because conv[k] = sum_i x[i] * x[k-i], so d(conv[k])/dx[i] = x[k-i]
        
        k = peak_idx
        # The gradient of conv[k] w.r.t. x is: x shifted
        grad = np.zeros(N)
        # For i in range(N): grad[i] = x[k-i] if 0 <= k-i < N
        for i in range(N):
            j = k - i
            if 0 <= j < N:
                grad[i] = x[j]
        
        grad = h * grad
        
        # Update x in the direction that reduces the peak
        x_new = x - lr * grad
        x_new = np.maximum(x_new, 0)  # Enforce nonnegativity
        
        # Renormalize to maintain the same integral
        integral = np.sum(x_new) * h
        if integral > 0:
            x_new = x_new * (norm_orig / integral)
        
        score_new = _score(x_new)
        if score_new < best_score:
            best_score = score_new
            best_x = x_new.copy()
            x = x_new
            # Decay learning rate slightly
            lr *= 0.999
        else:
            # If score didn't improve, try smaller step
            lr *= 0.9
            x = best_x.copy()
    
    return [float(v) for v in best_x]
