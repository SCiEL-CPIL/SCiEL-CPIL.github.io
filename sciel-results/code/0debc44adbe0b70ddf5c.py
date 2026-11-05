"""
Strategy: Spectral Shaping via Low-Frequency Fourier Coefficients with Multi-Start Nelder-Mead

This strategy fundamentally changes the representation of the signal h.
Instead of optimizing the 400 time-domain samples directly (which is a high-dimensional,
non-convex landscape with many local minima due to the max operation on the convolution),
we parameterize the signal by its low-frequency Fourier coefficients.

Key Insights:
1.  **Low-Frequency Bias**: The objective function involves the global sum of h and the
    autoconvolution. High-frequency components tend to create sharp peaks in the
    autoconvolution (sidelobes) or cancel out in the sum. By restricting the signal
    to a low-dimensional subspace of smooth, low-frequency shapes (e.g., the first
    K cosine/sine coefficients), we implicitly regularize the signal to be smooth.
    Smooth signals have autoconvolutions that are also smooth and have lower peak
    sidelobes relative to the main lobe compared to noisy or high-frequency signals.
2.  **Dimensionality Reduction**: We reduce the optimization from 400 dimensions to
    approximately 20-40 dimensions. This makes the optimization landscape much more
    tractable for standard local optimizers like Nelder-Mead.
3.  **Multi-Start**: Since the problem is still non-convex (due to the max and division),
    we perform multiple random restarts in the Fourier coefficient space. Each restart
    explores a different "shape" of the low-frequency signal.

This is structurally different from the current approach (Zadoff-Chu + Powell on 400D)
because it does not rely on a specific structured waveform like ZC, nor does it operate
in the full time-domain. It treats the signal as a smooth curve defined by a few
spectral parameters.
"""

import numpy as np
import time
from scipy.optimize import minimize

N = 400
_DEADLINE = 145.0

def _evaluate(h):
    """Compute the task value."""
    s = np.sum(h)
    if s == 0 or not np.isfinite(s):
        return np.inf
    conv = np.convolve(h, h)
    m = np.max(conv)
    return abs(2.0 * N * m / (s * s))

def _h_from_coeffs(coeffs, K):
    """
    Reconstruct h from Fourier coefficients.
    coeffs[0] is the DC offset.
    coeffs[1:2K+1] are cosine and sine coefficients for frequencies 1..K.
    We use a real-valued basis:
    h[n] = c0 + sum_{k=1}^K (c_k * cos(2*pi*k*n/N) + s_k * sin(2*pi*k*n/N))
    """
    n = np.arange(N)
    h = np.zeros(N)
    h += coeffs[0]
    for k in range(1, K + 1):
        c_k = coeffs[2*k - 1]
        s_k = coeffs[2*k]
        h += c_k * np.cos(2.0 * np.pi * k * n / N)
        h += s_k * np.sin(2.0 * np.pi * k * n / N)
    return h

def construct():
    t0 = time.time()
    deadline = t0 + _DEADLINE
    
    best_h = np.ones(N)
    best_val = 2.0
    
    # We will optimize over the first K Fourier coefficients.
    # Let's try a range of K values. Higher K allows more complex shapes but increases dimensionality.
    # K=10 -> 21 variables. K=20 -> 41 variables.
    K_values = [5, 10, 15, 20, 30]
    
    # Number of restarts per K value
    # We have ~140s. 
    # Each Nelder-Mead run on ~20-40 dims might take 0.1-1s depending on convergence.
    # Let's allocate time dynamically.
    
    rng = np.random.default_rng(42)
    
    # Pre-compute basis vectors for speed? 
    # For small K, direct computation is fast enough. 
    # But we can pre-compute the basis matrix B (N x (2K+1)) to make _h_from_coeffs a matrix-vector product.
    
    basis_cache = {}
    def get_basis(K):
        if K not in basis_cache:
            n = np.arange(N)
            B = np.zeros((N, 2*K + 1))
            B[:, 0] = 1.0
            for k in range(1, K + 1):
                B[:, 2*k - 1] = np.cos(2.0 * np.pi * k * n / N)
                B[:, 2*k] = np.sin(2.0 * np.pi * k * n / N)
            basis_cache[K] = B
        return basis_cache[K]
    
    def _obj_from_coeffs(coeffs, K):
        B = get_basis(K)
        h = B @ coeffs
        return _evaluate(h)
    
    # Multi-start optimization
    # We will iterate through K values and perform several restarts for each.
    # To ensure we use time effectively, we check the deadline frequently.
    
    # Strategy for restarts:
    # 1. Try a "flat" start (all zeros except DC=1)
    # 2. Try random Gaussian starts with small variance
    
    total_restarts = 50 # Total budget for restarts across all K
    
    # Distribute restarts among K values
    # Smaller K is faster, so we can do more restarts there.
    # Let's just cycle through K values.
    
    restart_idx = 0
    while restart_idx < total_restarts and time.time() < deadline - 5:
        K = K_values[restart_idx % len(K_values)]
        dim = 2 * K + 1
        
        # Generate initial guess
        if restart_idx == 0:
            # Start with DC=1, others 0
            x0 = np.zeros(dim)
            x0[0] = 1.0
        else:
            # Random start
            # Scale matters. The objective is scale-invariant, but the optimizer path depends on scale.
            # Let's keep coefficients in a reasonable range, e.g., [-1, 1].
            x0 = rng.uniform(-0.5, 0.5, size=dim)
            # Ensure DC is not zero to avoid division by zero early on
            if abs(x0[0]) < 0.1:
                x0[0] = 0.1 if x0[0] >= 0 else -0.1
        
        try:
            res = minimize(
                _obj_from_coeffs,
                x0,
                args=(K,),
                method='Nelder-Mead',
                options={
                    'maxiter': 500,
                    'xatol': 1e-6,
                    'fatol': 1e-6,
                    'adaptive': True,
                    'disp': False
                }
            )
            
            if res.fun < best_val and np.isfinite(res.fun):
                best_val = res.fun
                B = get_basis(K)
                best_h = B @ res.x
                
        except Exception:
            pass
        
        restart_idx += 1
        
        # Check time
        if time.time() > deadline - 10:
            break

    # Final check
    final_val = _evaluate(best_h)
    if not np.isfinite(final_val) or final_val >= 2.0:
        return np.ones(N)
        
    return best_h
