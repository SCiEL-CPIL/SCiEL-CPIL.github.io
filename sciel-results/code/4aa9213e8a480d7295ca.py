"""Second autocorrelation inequality - optimize bump shape for 50 nonnegative heights."""
import numpy as np
import time

N = 50

# EVOLVE-BLOCK-START
def construct():
    """Return 50 nonnegative step heights maximizing the ratio."""
    deadline = time.time() + 110.0  # safety margin before 120s limit
    
    def compute_score(h):
        """Compute ||f||^2_2 / (||f||_1 * ||f*f||_inf) for step function with heights h."""
        h = np.asarray(h, dtype=float)
        if np.any(h < -1e-12):
            return 0.0
        h = np.maximum(h, 0.0)
        
        l1 = np.sum(h)
        if l1 < 1e-12:
            return 0.0
        
        # Step width
        w = 1.0 / (2.0 * N)
        
        # ||f||_2^2 = sum h[i]^2 * w
        l2_sq = np.sum(h * h) * w
        
        # Compute ||f*f||_inf: the supremum of the autoconvolution of the step function.
        # For a step function on [-1/4, 1/4] with N steps of width w,
        # (f*f)(t) is piecewise linear with breakpoints at t = k*w.
        # For nonnegative f, the maximum is at t=0.
        # (f*f)(0) = integral f(x) f(-x) dx
        
        # Compute (f*f)(0) = sum_i sum_j h[i] h[j] * overlap(i, j)
        # where overlap(i, j) = length([x_i, x_{i+1}] ∩ [-x_{j+1}, -x_j])
        # x_i = -0.25 + i*w
        # -x_{j+1} = 0.25 - (j+1)*w, -x_j = 0.25 - j*w
        
        # Vectorized computation:
        # For each pair (i, j), compute the overlap.
        # a_i = -0.25 + i*w, b_i = -0.25 + (i+1)*w
        # c_j = 0.25 - (j+1)*w, d_j = 0.25 - j*w
        # overlap = max(0, min(b_i, d_j) - max(a_i, c_j))
        
        i_idx = np.arange(N)
        j_idx = np.arange(N)
        I, J = np.meshgrid(i_idx, j_idx, indexing='ij')
        
        a = -0.25 + I * w
        b = -0.25 + (I + 1) * w
        c = 0.25 - (J + 1) * w
        d = 0.25 - J * w
        
        overlap = np.maximum(0, np.minimum(b, d) - np.maximum(a, c))
        
        ff0 = np.sum(h[:, None] * h[None, :] * overlap)
        
        # The supremum of (f*f)(t) is at t=0 for nonnegative step functions
        # (this is a known property: the autoconvolution of a nonnegative function
        # achieves its maximum at t=0)
        ff_inf = ff0
        
        if ff_inf < 1e-12:
            return 0.0
        
        return l2_sq / (l1 * ff_inf)
    
    # Try various bump shapes
    best_h = None
    best_score = 0.0
    
    x = np.arange(N)
    center = (N - 1) / 2.0
    
    # Test various shapes
    shapes = []
    
    # Raised cosine
    t = (x - center) / center
    shapes.append(('cos', 0.5 * (1 + np.cos(np.pi * t))))
    
    # Triangle
    shapes.append(('tri', np.maximum(1.0 - np.abs(t), 0)))
    
    # Various Gaussians
    for sigma_frac in [0.2, 0.25, 0.3, 0.35, 0.4, 0.5]:
        sigma = sigma_frac * N
        shapes.append((f'gauss_{sigma_frac}', np.exp(-0.5 * ((x - center) / sigma) ** 2)))
    
    # Power bump: (1 - t^2)^p
    for p in [0.5, 1.0, 2.0, 3.0]:
        shapes.append((f'pow_{p}', np.maximum(1 - t*t, 0) ** p))
    
    for name, h_shape in shapes:
        h = h_shape.copy()
        h = np.maximum(h, 0)
        s = np.sum(h)
        if s < 1e-12:
            continue
        h = h / s  # normalize
        sc = compute_score(h)
        if sc > best_score:
            best_score = sc
            best_h = h.copy()
    
    if best_h is None:
        return [1.0] * N
    
    return list(best_h)
# EVOLVE-BLOCK-END
