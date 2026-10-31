"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution

def _compute_c3_bound(heights):
    """
    Compute the C3 autoconvolution upper bound for a signed step function.
    
    The step function has N equal intervals on [-1/4, 1/4].
    We interpret C3 as related to the autoconvolution:
    For f a step function, (f*f)(0) = sum_i heights[i]^2 * dx
    The C3 bound likely involves ||f*f||_1 or similar.
    
    A common bound: ||f*f||_1 <= C * ||f||_1^2
    Or for the "third autocorrelation": involves f*f*f
    
    Let's use: C3 = ||f*f||_1 / (||f||_1^2) as a normalized autoconvolution bound.
    Lower is better.
    """
    dx = 0.5 / N  # interval length: [-1/4, 1/4] has length 1/2
    n = len(heights)
    
    # Compute autoconvolution via FFT
    # f*f at 0 = sum_i heights[i]^2 * dx
    # ||f*f||_1 = integral of |f*f(x)| dx
    
    # For step functions, f*f is a piecewise linear function
    # Use FFT for efficient computation
    # Pad to avoid circular convolution
    nfft = 1 << (2 * n - 1).bit_length()
    f_padded = np.zeros(nfft)
    f_padded[:n] = heights
    
    conv = np.fft.irfft(np.fft.rfft(f_padded) ** 2) * dx
    # conv has length nfft, but we only care about the valid part
    # The autoconvolution is defined on [-1/2, 1/2]
    
    # ||f*f||_1
    l1_conv = np.sum(np.abs(conv)) * dx
    
    # ||f||_1
    l1_f = np.sum(np.abs(heights)) * dx
    
    if l1_f == 0:
        return 0.0
    
    # C3 bound: ||f*f||_1 / ||f||_1^2
    # This is dimensionless and measures how "spread out" the autoconvolution is
    c3 = l1_conv / (l1_f ** 2)
    
    return c3

def construct():
    """
    Construct a signed step function with 400 intervals on [-1/4, 1/4]
    to minimize the C3 autoconvolution upper bound.
    
    Returns: sequence of signed step heights
    """
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill
    
    # Strategy: use smooth band-limited signals and local optimization
    # Start with various initializations and optimize
    
    best_heights = np.ones(N)
    best_score = _compute_c3_bound(best_heights)
    
    # Try different initialization strategies
    rng = np.random.default_rng(42)
    
    # Strategy 1: Random smooth signals (sum of sines)
    for trial in range(20):
        if time.time() > deadline:
            break
        
        # Generate a smooth oscillating signal
        n_harmonics = rng.integers(2, 8)
        freqs = rng.uniform(0.5, 5.0, n_harmonics)
        phases = rng.uniform(0, 2 * np.pi, n_harmonics)
        amps = rng.uniform(0.5, 1.0, n_harmonics)
        
        x = np.linspace(-1/4, 1/4, N, endpoint=False) + 0.5 / (2 * N)
        heights = np.zeros(N)
        for f, p, a in zip(freqs, phases, amps):
            heights += a * np.sin(2 * np.pi * f * x + p)
        
        # Sign the heights (keep sign, scale)
        heights = np.sign(heights) * np.abs(heights)
        
        score = _compute_c3_bound(heights)
        if score < best_score:
            best_score = score
            best_heights = heights.copy()
    
    # Strategy 2: Local optimization from best candidate
    if time.time() < deadline:
        def objective(h):
            return _compute_c3_bound(h)
        
        # Use Nelder-Mead or L-BFGS for local refinement
        try:
            result = minimize(
                objective,
                best_heights,
                method='Nelder-Mead',
                options={'maxiter': 5000, 'xatol': 1e-6, 'fatol': 1e-8}
            )
            if result.fun < best_score:
                best_score = result.fun
                best_heights = result.x
        except Exception:
            pass
    
    # Strategy 3: Try sign-flip patterns (block signs)
    if time.time() < deadline:
        for n_blocks in [2, 4, 8, 10, 16, 20, 40, 50, 100, 200]:
            if time.time() > deadline:
                break
            
            block_size = N // n_blocks
            for sign_pattern in range(min(2 ** min(n_blocks, 8), 256)):
                if time.time() > deadline:
                    break
                
                heights = np.ones(N)
                for b in range(n_blocks):
                    if (sign_pattern >> b) & 1:
                        heights[b * block_size:(b + 1) * block_size] = -1.0
                
                score = _compute_c3_bound(heights)
                if score < best_score:
                    best_score = score
                    best_heights = heights.copy()
    
    # Final polish: gradient-based optimization
    if time.time() < deadline - 10:
        def objective(h):
            return _compute_c3_bound(h)
        
        def gradient(h):
            # Numerical gradient
            eps = 1e-7
            grad = np.zeros_like(h)
            for i in range(len(h)):
                h_plus = h.copy()
                h_plus[i] += eps
                h_minus = h.copy()
                h_minus[i] -= eps
                grad[i] = (objective(h_plus) - objective(h_minus)) / (2 * eps)
            return grad
        
        try:
            result = minimize(
                objective,
                best_heights,
                jac=gradient,
                method='L-BFGS-B',
                options={'maxiter': 1000, 'gtol': 1e-6}
            )
            if result.fun < best_score:
                best_score = result.fun
                best_heights = result.x
        except Exception:
            pass
    
    return best_heights
# EVOLVE-BLOCK-END
