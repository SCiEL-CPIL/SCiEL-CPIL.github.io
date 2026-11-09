"""Rewritten program for the third autocorrelation inequality task."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize, LinearConstraint


def construct():
    """
    Construct a signed step function with 400 equal intervals on [-1/4, 1/4]
    to improve the autoconvolution upper bound C3.
    
    The score is the upper bound on C3 (lower is better).
    We model the step function as a sparse combinatorial pattern of alternating
    positive and negative blocks to exploit orthogonality in the autoconvolution spectrum.
    
    Approach:
    1. Initialize with a high-contrast deterministic sequence (Thue-Morse based) to break symmetry.
    2. Use the fact that the autoconvolution upper bound C3 relates to the L1 norm of the 
       autoconvolution or similar spectral quantity.
    3. Optimize the heights to minimize the relevant bound.
    
    The autoconvolution of a step function f with itself is (f*f)(t) = ∫ f(τ)f(t-τ) dτ.
    The upper bound on C3 typically involves the integral of |f*f| or related quantities.
    
    For a step function with heights h_i on intervals of width Δ = (1/2)/400 = 1/800,
    the autoconvolution can be computed efficiently.
    
    We aim to minimize the L1 norm of the autoconvolution, which gives an upper bound on C3.
    """
    # Initialize with Thue-Morse sequence for high contrast and low autocorrelation
    def thue_morse(n):
        """Generate Thue-Morse sequence of length n."""
        seq = np.zeros(n, dtype=np.int8)
        for i in range(1, n):
            seq[i] = seq[i >> 1] ^ (i & 1)
        return seq
    
    # Create initial signed pattern: +1 and -1 based on Thue-Morse
    tm = thue_morse(N)
    heights = np.where(tm == 0, 1.0, -1.0)
    
    # Compute the autoconvolution upper bound
    def compute_c3_bound(h):
        """
        Compute an upper bound on C3 for the given step function heights.
        
        The step function f(x) has height h_i on interval [x_i, x_{i+1}] where
        x_i = -1/4 + i * Δ, Δ = 1/800.
        
        The autoconvolution (f*f)(t) = ∫ f(τ)f(t-τ) dτ.
        
        For step functions, we can compute this on a grid.
        The upper bound on C3 is related to the L1 norm of the autoconvolution
        or the integral of the absolute value.
        
        A common bound: C3 ≤ ∫|f*f| dx / (∫|f| dx)^3 * (some constant)
        
        For simplicity, we'll use the L1 norm of the autoconvolution as our metric,
        normalized appropriately.
        
        Actually, let's think about this more carefully. The task says "autoconvolution 
        upper bound" and "score is the upper bound on C3". Let's compute the actual
        autoconvolution and use its L1 norm as the bound.
        """
        h = np.asarray(h, dtype=float)
        n = len(h)
        delta = 1.0 / 800.0  # width of each interval
        
        # Compute autoconvolution using FFT for efficiency
        # Pad to avoid circular convolution
        size = 2 * n
        h_pad = np.zeros(size)
        h_pad[:n] = h
        
        # Autoconvolution via FFT
        fft_h = np.fft.fft(h_pad, size)
        autoconv = np.fft.ifft(fft_h * fft_h).real
        
        # The autoconvolution on the original domain
        # Take the relevant part (indices 0 to n-1 represent t from 0 to (n-1)*delta)
        # But we need the full autoconvolution on [-1/4, 1/4]
        
        # Actually, the autoconvolution of f with itself, where f is supported on [-1/4, 1/4],
        # has support on [-1/2, 1/2].
        
        # Let's compute it properly:
        # f*f(t) = ∫ f(τ) f(t-τ) dτ
        # For step functions, this is a piecewise linear function.
        
        # Simpler approach: compute the autoconvolution on a fine grid
        # and integrate |f*f| to get the L1 norm.
        
        # Using the discrete autoconvolution scaled by delta
        # The discrete autoconvolution c_k = sum_i h_i * h_{k-i} * delta^2
        # for the appropriate range
        
        # Let's use scipy.signal.fftconvolve for accuracy
        from scipy.signal import fftconvolve
        
        # The autoconvolution of the step function
        # Each interval has height h_i and width delta
        # So the function is sum_i h_i * indicator[x in interval_i]
        # The autoconvolution is sum_i sum_j h_i * h_j * (overlap of interval_i and shifted interval_j)
        
        # For step functions, the autoconvolution at point t is:
        # (f*f)(t) = sum_i h_i * (measure of {τ: τ in interval_i AND t-τ in support(f)})
        
        # This is getting complex. Let's use a simpler metric:
        # The L1 norm of the discrete autoconvolution, scaled appropriately.
        
        conv = fftconvolve(h, h, mode='full')
        # Scale by delta^2 since we're integrating
        conv_scaled = conv * (delta ** 2)
        
        # The L1 norm of the autoconvolution
        l1_norm = np.sum(np.abs(conv_scaled))
        
        return l1_norm
    
    # Optimize the heights to minimize the C3 bound
    # Use gradient-based optimization with constraints
    
    # Initial guess
    x0 = heights.copy()
    
    # Define objective
    def objective(x):
        return compute_c3_bound(x)
    
    # Constraints: heights should be in [-1, 1] for a "signed step function"
    # The problem says "signed step function", so heights can be any real number,
    # but typically we'd want them bounded. Let's allow [-1, 1].
    
    bounds = [(-1.0, 1.0)] * N
    
    # Try to optimize using L-BFGS-B
    try:
        result = minimize(objective, x0, method='L-BFGS-B', bounds=bounds,
                         options={'maxiter': 100, 'ftol': 1e-8})
        if result.success:
            x_opt = result.x
        else:
            x_opt = x0
    except Exception:
        x_opt = x0
    
    # Ensure the result is a valid step function
    # Round to reasonable precision
    x_opt = np.round(x_opt, 6)
    
    return x_opt
# EVOLVE-BLOCK-END
