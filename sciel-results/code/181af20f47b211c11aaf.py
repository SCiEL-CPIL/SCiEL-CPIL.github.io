import numpy as np
import time
from scipy.optimize import minimize

N = 600
W = 1.0 / 1200.0
# Domain is [-0.25, 0.25]
# Intervals are [ -0.25 + i*W, -0.25 + (i+1)*W )

def _compute_autoconv(h):
    """
    Compute the autoconvolution of a step function h on N intervals.
    Returns an array of length N representing the autoconvolution at the centers of the intervals.
    Note: The true max might be at boundaries, but for smooth functions, centers are close.
    Actually, for a step function, the autoconvolution is piecewise quadratic.
    The maximum over the continuous domain is either at a center or a boundary.
    Due to symmetry and the nature of autoconvolution, the peak is usually at x=0 (center of the domain).
    Let's compute the values at all interval centers and boundaries to be safe?
    For efficiency, we'll compute at centers first. 
    (h*h)(x) for step functions:
    If we define discrete conv c_k = sum_j h_j h_{k-j}, 
    the continuous autoconv at the center of interval k is approx W * c_k.
    Wait, the scaling is:
    h(t) = h_i for t in I_i.
    (h*h)(x) = sum_i sum_j h_i h_j * Length(I_i \cap (x - I_j)).
    This is exactly W * (h * h)_discrete[k] if x is at the center of I_k?
    Let's verify.
    If h is constant 1, c_k = N for k=0..N-1 (with wrap around? No, linear conv).
    Linear conv length is 2N-1.
    The autoconv of a box of width L is a triangle of base 2L and height L.
    Here L = N*W = 0.5.
    Height should be 0.5.
    Sum of h_i^2 * W = N * 1 * W = 0.5.
    Ratio = 1.
    
    Discrete conv:
    c = np.convolve(h, h, mode='full')
    The peak is at index N-1 (center).
    c[N-1] = sum h_i^2 = N.
    The continuous value at peak is c[N-1] * W = N * W = 0.5.
    So max_conv = max(c) * W is NOT correct for all shifts?
    Actually, the continuous autoconvolution at the center of interval k (relative to the start)
    corresponds to the discrete convolution value at that index?
    Yes, if we align the grids properly.
    Let's just use FFT to compute the continuous autoconvolution sampled at the interval centers.
    """
    # Pad h to length 2N to perform linear convolution via FFT
    # Or use np.convolve
    # For speed, use FFT
    n_fft = 1024 # Next power of 2 >= 2*600 = 1200? 1024 is too small. 2048.
    n_fft = 2048
    F = np.fft.rfft(h, n_fft)
    conv = np.fft.irfft(F * F, n_fft)
    # conv[k] is the sum h_j h_{k-j} for the circular convolution.
    # Since we padded with zeros to 2048, and N=600, the linear convolution is contained in conv[0:1199].
    # The peak is at index 599 (center of the box if centered at 0).
    # The interval centers are at x_i = -0.25 + (i+0.5)W.
    # The autoconvolution is symmetric around 0.
    # We need the max over the entire domain [-0.25, 0.25].
    # The values at the interval centers are sufficient approximation for smooth h.
    # For step functions, the max is at the boundaries of the intervals?
    # Actually, the autoconvolution of a step function is continuous and piecewise quadratic.
    # The derivative changes at the boundaries.
    # The maximum is either at x=0 or at a boundary.
    # Given symmetry, x=0 is a candidate.
    # Let's extract the values at the centers of the N intervals.
    # The center of interval i is at shift i-0.5 relative to the start of the conv?
    # Let's map the continuous x to the conv index.
    # x ranges from -0.25 to 0.25.
    # The conv array `conv` corresponds to shifts from -0.25 to 0.25 (approx).
    # Specifically, conv[k] for k in [0, 2N-1] corresponds to the sum for shift k-N+1?
    # Let's just take the max of the central part.
    # The valid range of the linear convolution is indices [0, 2*N-2].
    # The center is at N-1.
    # The interval centers correspond to indices N-1 - (N-1) ... ?
    # Let's just compute the ratio using the max of the discrete convolution * W.
    # This is an upper bound or approximation?
    # For a box function, max(c) * W = N * W = L. Correct.
    # For a general function, max(c) * W approximates the peak.
    # To be more accurate, we could check boundaries, but let's start with centers.
    
    # The relevant part of the convolution is conv[0:2*N-1]
    # The peak is likely near the center.
    # We take the max of the entire valid convolution range to be safe.
    max_conv = np.max(conv[:2*N-1]) * W
    return max_conv

def _objective(u, alpha=50.0):
    """
    Objective function for L-BFGS-B.
    u: log-heights
    alpha: soft-max parameter
    """
    h = np.exp(u)
    l2_sq = np.sum(h * h) * W
    if l2_sq < 1e-15:
        return 1e10
    
    max_conv = _compute_autoconv(h)
    
    # Soft max surrogate for max
    # We need the values of the autoconvolution at the grid points to apply soft max?
    # _compute_autoconv returns the max. We need the vector to do soft max.
    # Let's modify _compute_autoconv to return the vector.
    
    # Re-implementing inside for speed/clarity in this step
    n_fft = 2048
    F = np.fft.rfft(h, n_fft)
    conv = np.fft.irfft(F * F, n_fft)
    conv_valid = conv[:2*N-1] * W
    
    # Apply soft max
    # Scale to prevent overflow
    m = np.max(conv_valid)
    if m == 0:
        return 0.0
    # log-sum-exp
    soft_max_val = m + np.log(np.sum(np.exp(conv_valid / m - 1.0))) * m / alpha  # Approximation?
    # Better: 
    # soft_max = (1/alpha) * log(sum(exp(alpha * x)))
    # To avoid overflow: let M = max(x). soft_max = M + (1/alpha) * log(sum(exp(alpha*(x-M))))
    M = np.max(conv_valid)
    if M < 1e-15:
        return 0.0
    exp_terms = np.exp(alpha * (conv_valid - M))
    soft_max_val = M + np.log(np.sum(exp_terms)) / alpha
    
    return soft_max_val / l2_sq

def _objective_grad(u, alpha=50.0):
    """
    Gradient of the objective function.
    This is complex due to the max/soft-max and the ratio.
    For now, let's use numerical gradients or a simpler objective.
    """
    # Using numerical gradient is safer for implementation.
    # L-BFGS-B can use finite differences if jac=None.
    pass

def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin
    
    # Initialize u (log heights)
    # Start with a Gaussian-like shape
    x = np.linspace(-0.25, 0.25, N, endpoint=False) + W/2
    # Gaussian
    sigma = 0.1
    h_init = np.exp(-x**2 / (2*sigma**2))
    # Normalize to sum of squares = 1? No, ratio is scale invariant.
    # But for numerical stability, keep values in reasonable range.
    u_init = np.log(h_init + 1e-6)
    
    best_h = h_init
    best_score = _objective(u_init)
    
    # Optimization using L-BFGS-B
    # We minimize the objective.
    # Since the objective is scale invariant, we can fix the norm?
    # Or just let it optimize u.
    # The objective is homogeneous of degree 0 in h.
    # u -> u + c gives h -> h * exp(c). Ratio is unchanged.
    # So there is a flat direction. L-BFGS-B might struggle.
    # We can constrain sum(u) = 0 or similar?
    # Or just use Nelder-Mead which handles flat directions better?
    # Let's try Nelder-Mead on u with the soft-max objective.
    
    n_restarts = 5
    max_iters = 500
    
    for restart in range(n_restarts):
        if time.time() > deadline - 30.0:
            break
        
        if restart == 0:
            u_current = u_init.copy()
        elif restart == 1:
            # Cosine bump
            h_c = np.cos(np.pi * x / 0.5) # cos(2 pi x)
            h_c = np.clip(h_c, 1e-6, None)
            u_current = np.log(h_c)
        elif restart == 2:
            # Two bumps
            h_b = np.exp(-(x-0.1)**2 / 0.01) + np.exp(-(x+0.1)**2 / 0.01)
            h_b = np.clip(h_b, 1e-6, None)
            u_current = np.log(h_b)
        else:
            # Random perturbation of best
            u_current = best_h.copy()
            u_current += np.random.randn(N) * 0.1
            u_current = np.log(u_current + 1e-6) # Ensure positive base
        
        # Evaluate current
        score_current = _objective(u_current)
        if score_current < best_score:
            best_score = score_current
            best_h = np.exp(u_current)
        
        # Run Nelder-Mead
        try:
            result = minimize(
                _objective, 
                u_current,
                args=(50.0,),
                method='Nelder-Mead',
                options={'maxiter': max_iters, 
                         'xatol': 1e-8, 'fatol': 1e-8,
                         'adaptive': True}
            )
            
            if result.fun < best_score:
                best_score = result.fun
                best_h = np.exp(result.x)
        except Exception:
            pass
        
        # Check time
        if time.time() > deadline - 40.0:
            break
    
    # Local refinement on h directly (with clipping)
    # Coordinate descent
    if time.time() < deadline - 10.0:
        h = best_h.copy()
        step_size = np.max(h) * 0.01
        
        for iteration in range(100):
            if time.time() > deadline - 5.0:
                break
            
            improved = False
            # Random subset of indices
            idxs = np.random.choice(N, size=50, replace=False)
            
            for idx in idxs:
                h_old = h[idx]
                
                # Try increase
                h[idx] = h_old + step_size
                r_plus = _objective(np.log(h + 1e-6)) # Use log objective to maintain positivity in calc?
                # Actually _objective takes u. If we change h, we should change u.
                # Let's just compute ratio directly for speed in CD?
                # _compute_ratio_fast
                r_plus = _compute_autoconv(h) / (np.sum(h*h)*W)
                
                # Try decrease
                h[idx] = h_old - step_size
                if h[idx] < 0: h[idx] = 0
                r_minus = _compute_autoconv(h) / (np.sum(h*h)*W)
                
                h[idx] = h_old
                
                if r_plus < best_score:
                    h[idx] = h_old + step_size
                    best_score = r_plus
                    improved = True
                elif r_minus < best_score:
                    h[idx] = h_old - step_size
                    if h[idx] < 0: h[idx] = 0
                    best_score = r_minus
                    improved = True
            
            if not improved:
                step_size *= 0.7
                if step_size < 1e-8:
                    break
    
    return best_h.tolist()
