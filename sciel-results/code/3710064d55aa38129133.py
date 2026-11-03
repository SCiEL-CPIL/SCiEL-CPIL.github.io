"""Structural rewrite: optimize a nonnegative 600-step function on [-1/4,1/4]
to minimize the normalized max of its autoconvolution (upper bound on C1)."""
import time
import math
import numpy as np

N = 600
START = time.time()
DEADLINE = START + 145.0  # safety margin before 150s kill

# Domain: [-1/4, 1/4], N equal intervals
DX = (0.25 - (-0.25)) / N  # 0.5/600

def _autoconv_max_norm(h):
    """
    Compute the normalized maximum of the autoconvolution of step function h.
    h is an array of N nonnegative heights on equal intervals of [-1/4, 1/4].
    
    The autoconvolution (self-convolution) of the step function:
    (f*f)(t) = integral f(s) f(t-s) ds
    
    For a step function with heights h_i on intervals of width dx,
    the convolution is piecewise linear. We compute it on a fine grid.
    
    Normalization: The "normalized maximum" likely refers to
    max_t (f*f)(t) / (integral f)^2 * (some length factor)
    
    Actually, for the first autocorrelation inequality (C1 bound),
    we want to minimize max_t |hat(f)(t)|^2 / (integral f)^2 or similar.
    
    Let me reconsider: The autoconvolution of f is g(t) = ∫ f(s)f(t-s)ds.
    The "normalized maximum" is likely max_t g(t) / (∫f)^2 * L where L is some scale,
    or simply max_t g(t) / (∫f)^2.
    
    Given the task says "upper bound on C1" and the seed returns [1.0]*600:
    For f=1 on [-1/4,1/4], ∫f = 0.5, (f*f)(t) is a triangle peaking at 0.5 at t=0.
    max (f*f) = 0.5. (∫f)^2 = 0.25. Ratio = 2.
    
    The score is seed/value, so value is the raw upper bound. Lower is better.
    Seed value = 2.0 (for the constant function).
    
    Let me verify: f(s) = 1 for s in [-1/4, 1/4].
    (f*f)(t) = ∫_{-∞}^{∞} f(s)f(t-s) ds
    For t=0: ∫_{-1/4}^{1/4} 1*1 ds = 0.5
    For t=1/4: ∫ f(s)f(1/4-s) ds, s in [-1/4,1/4], 1/4-s in [-1/4,1/4] means s in [0,1/4], integral = 0.25
    So max is 0.5 at t=0.
    
    Normalized: 0.5 / (0.5)^2 = 2.0. Yes, seed value = 2.0.
    """
    h = np.asarray(h, dtype=np.float64)
    if h.shape[0] != N:
        return float('inf')
    if np.any(h < -1e-15):
        return float('inf')
    h = np.maximum(h, 0.0)
    
    total_mass = np.sum(h) * DX
    if total_mass < 1e-15:
        return float('inf')
    
    # Compute autoconvolution via FFT for efficiency
    # Convolution of two length-N arrays has length 2N-1
    # Use FFT-based convolution
    size = 2 * N
    h_pad = np.zeros(size)
    h_pad[:N] = h * DX  # scale by dx for integral
    
    # FFT convolution
    H = np.fft.rfft(h_pad, n=2*size)
    conv = np.fft.irfft(H * H, n=2*size)
    # The convolution is valid in the first 2N-1 samples (indices 0 to 2N-2)
    conv = conv[:2*N-1]
    
    max_conv = np.max(conv)
    normalized = max_conv / (total_mass * total_mass)
    return float(normalized)


def _try_optimization(h0, max_iter=5000, tol=1e-10):
    """Optimize using projected gradient descent with adaptive step size."""
    h = np.array(h0, dtype=np.float64).copy()
    h = np.maximum(h, 0.0)
    
    best_val = _autoconv_max_norm(h)
    best_h = h.copy()
    
    # Use subgradient method since max is not differentiable everywhere
    # Compute gradient of the autoconvolution at the max point
    
    lr = 0.01
    for it in range(max_iter):
        if time.time() > DEADLINE:
            break
        
        val = _autoconv_max_norm(h)
        if val < best_val:
            best_val = val
            best_h = h.copy()
        
        # Compute gradient: need d/dh_i of max_t conv(t)
        # conv(t) = sum_j h_j * (autocorr_kernel)(t - j*dx)
        # At the max point t*, gradient w.r.t. h_i is the kernel value at t* - i*dx
        
        # For simplicity, use finite difference or analytical gradient
        # The autoconvolution at point t is:
        # conv(t) = sum_i sum_j h_i h_j * K((t - (i+j)*dx)/dx) ... this is complex
        
        # Simpler: use FFT-based gradient
        # conv = ifft(fft(h)^2)
        # d conv / d h_i = 2 * ifft(fft(h) * e_{-i}) where e_i is the impulse
        
        # Actually for real signals: d(conv)/dh = 2 * correlate(h, h) shifted
        # The gradient of conv(t) w.r.t. h_i is 2 * sum_j h_j * delta(t - i*dx - j*dx)
        # = 2 * h_i at t = i*dx (for the discrete case)
        
        # Let's just use a simple approach: perturb and use subgradient
        # Or better: use the fact that for the max, we can compute the active constraint
        
        # Practical approach: use Nelder-Mead or coordinate descent
        # Given time constraints, let's use a simple projected gradient with
        # the gradient computed via FFT
        
        # Compute conv and find argmax
        size = 2 * N
        h_pad = np.zeros(size)
        h_pad[:N] = h * DX
        H = np.fft.rfft(h_pad, n=2*size)
        conv = np.fft.irfft(H * H, n=2*size)[:2*N-1]
        
        t_max = np.argmax(conv)
        
        # Gradient of conv at t_max w.r.t. h:
        # conv(t) = sum_i h_i * (h * dx) convolved... 
        # Actually conv(t) = sum_i sum_j h_i h_j dx^2 * K(t - (i+j)dx)
        # This is getting complicated. Let's use a simpler optimization.
        
        # Switch to coordinate descent / Nelder-Mead for better results
        break
    
    return best_h, best_val


def construct():
    """
    Construct a nonnegative step function with 600 equal intervals on [-1/4,1/4]
    to minimize the normalized maximum of its autoconvolution.
    
    Returns: list of N nonnegative step heights.
    """
    t_start = time.time()
    
    # Initial candidates
    candidates = []
    
    # 1. Constant function (seed)
    candidates.append(np.ones(N))
    
    # 2. Triangular / tent function
    tri = np.zeros(N)
    mid = N // 2
    for i in range(N):
        tri[i] = 1.0 - abs(i - mid) / mid
    candidates.append(tri)
    
    # 3. Gaussian-like
    x = np.linspace(-1, 1, N)
    gauss = np.exp(-x**2 * 2)
    candidates.append(gauss)
    
    # 4. Uniform on a subinterval (concentrated mass)
    for frac in [0.3, 0.5, 0.7]:
        k = int(N * frac)
        start = (N - k) // 2
        c = np.zeros(N)
        c[start:start+k] = 1.0
        candidates.append(c)
    
    # 5. Sinc-like (zero at edges)
    x = np.linspace(-1, 1, N)
    sinc = np.sinc(x)  # sin(pi*x)/(pi*x)
    sinc = np.maximum(sinc, 0)
    candidates.append(sinc)
    
    # 6. Cosine bump
    x = np.linspace(-1, 1, N)
    cos_bump = np.where(np.abs(x) < 1, 0.5*(1+np.cos(np.pi*x)), 0)
    candidates.append(cos_bump)
    
    # Evaluate all candidates
    best_val = float('inf')
    best_h = None
    for c in candidates:
        c = np.maximum(c, 0)
        val = _autoconv_max_norm(c)
        if val < best_val:
            best_val = val
            best_h = c.copy()
    
    if time.time() > DEADLINE:
        return [float(x) for x in best_h]
    
    # Local optimization via projected gradient descent
    # Use a simpler approach: iterative refinement
    h = best_h.copy()
    val = best_val
    
    # Try to improve with a few optimization strategies
    # Strategy: minimize max autoconvolution by spreading mass more evenly
    # or concentrating it to reduce peak
    
    # Use scipy if available for better optimization
    try:
        from scipy.optimize import minimize
        import numpy as np
        
        def objective(h_flat):
            h = np.maximum(h_flat, 0)
            return _autoconv_max_norm(h)
        
        # Projected gradient via scipy
        best_scipy_val = best_val
        best_scipy_h = best_h.copy()
        
        # Try a few starting points
        for start_h in candidates[:4]:
            start_h = np.maximum(start_h, 0).astype(np.float64)
            try:
                res = minimize(
                    objective,
                    start_h,
                    method='L-BFGS-B',
                    options={'maxiter': 200, 'ftol': 1e-12}
                )
                if res.success or res.fun < best_scipy_val:
                    h_opt = np.maximum(res.x, 0)
                    v = _autoconv_max_norm(h_opt)
                    if v < best_scipy_val:
                        best_scipy_val = v
                        best_scipy_h = h_opt.copy()
            except Exception:
                pass
            
            if time.time() > DEADLINE - 5:
                break
        
        if best_scipy_val < best_val:
            best_val = best_scipy_val
            best_h = best_scipy_h
    
    except ImportError:
        pass
    
    if time.time() > DEADLINE:
        return [float(x) for x in best_h]
    
    # Additional local refinement: coordinate descent
    h = best_h.copy()
    val = _autoconv_max_norm(h)
    
    # Simple coordinate ascent/descent for a few iterations
    for coord_iter in range(3):
        if time.time() > DEADLINE - 2:
            break
        improved = False
        for i in range(0, N, 3):  # every 3rd coordinate for speed
            if time.time() > DEADLINE - 2:
                break
            # Try adjusting h[i]
            old_h = h[i]
            for delta in [-0.5, -0.25, -0.1, 0.1, 0.25, 0.5]:
                new_h = max(0, old_h + delta)
                h[i] = new_h
                new_val = _autoconv_max_norm(h)
                if new_val < val:
                    val = new_val
                    improved = True
                    break
            h[i] = old_h
        
        if not improved:
            break
    
    # Final best
    final_val = _autoconv_max_norm(best_h)
    if final_val < best_val:
        best_val = final_val
    
    # Ensure nonnegativity and correct length
    best_h = np.maximum(best_h, 0)
    assert len(best_h) == N
    
    return [float(x) for x in best_h]
