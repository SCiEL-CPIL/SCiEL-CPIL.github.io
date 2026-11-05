"""
Strategy: Fourier-Domain Low-Dimensional Parameterization with Differential Evolution

Instead of optimizing the 400 time-domain samples directly (which is a high-dimensional, 
non-smooth problem due to the max operator on the convolution), we parameterize the 
signal h as the inverse Discrete Fourier Transform (IDFT) of a sparse set of low-frequency 
coefficients. 

By restricting the signal to the first K Fourier modes (plus their conjugate counterparts 
to ensure real-valued output), we reduce the search space to a low-dimensional manifold 
(roughly 2*K real variables). Signals in this subspace are smooth and band-limited, which 
inherently suppresses high-frequency artifacts in the autocorrelation. 

We use Differential Evolution (DE) to globally optimize these spectral coefficients. 
DE is robust to the non-convexity of the objective function and does not require gradients. 
This approach fundamentally changes the representation from "time-domain spikes/levels" to 
"smooth spectral shapes," which is a different design family than the time-domain Zadoff-Chu 
windowing.

Key structural differences:
1. Representation: h = IDFT(c) where c is sparse (only low frequencies non-zero).
2. Optimization: Differential Evolution (global, population-based) instead of Powell (local, derivative-free).
3. Search Space: Low-dimensional (e.g., 20-50 dims) spectral coefficients instead of 400 time-domain samples.
"""

import numpy as np
import time
from scipy.optimize import differential_evolution

N = 400
_DEADLINE = 145.0

def _evaluate(h):
    """Compute the task value."""
    s = np.sum(h)
    if s == 0 or not np.isfinite(s):
        return np.inf
    # Ensure h is real
    h = np.real(h)
    conv = np.convolve(h, h)
    m = np.max(conv)
    return abs(2.0 * N * m / (s * s))

def _h_from_coeffs(coeffs, K):
    """
    Reconstruct h from the first K complex Fourier coefficients.
    coeffs: complex array of length K+1 (indices 0..K).
    We use the symmetry c[N-k] = conj(c[k]) to ensure h is real.
    """
    c = np.zeros(N, dtype=complex)
    # Place coefficients in the positive frequency bin
    # Note: np.fft.ifft expects c[0], c[1], ..., c[N-1]
    # We will fill c[0]..c[K] and c[N-K]..c[N-1]
    
    for k in range(K + 1):
        val = coeffs[k]
        c[k] = val
        if k > 0:
            c[N - k] = np.conj(val)
            
    h = np.fft.ifft(c) * N # Scale by N because ifft includes 1/N factor
    return np.real(h)

def construct():
    t0 = time.time()
    deadline = t0 + _DEADLINE
    
    # We will optimize the real and imaginary parts of the first K Fourier coefficients.
    # Let's try a few different K values (dimensionality of the subspace).
    # K=10 means we use frequencies 0..10. Total real variables = 2*(K+1) = 22.
    # K=20 means 42 variables.
    
    best_h = np.ones(N)
    best_val = 2.0
    
    # Define the objective function for the optimizer
    def _objective(x):
        """
        x: 1D array of real numbers, length 2*(K+1).
        First K+1 elements are real parts of c[0]..c[K].
        Next K+1 elements are imag parts of c[0]..c[K].
        """
        if not np.all(np.isfinite(x)):
            return 1e10
        
        K = len(x) // 2 - 1
        # Reconstruct complex coefficients
        c = np.zeros(K + 1, dtype=complex)
        real_parts = x[:K+1]
        imag_parts = x[K+1:]
        c = real_parts + 1j * imag_parts
        
        h = _h_from_coeffs(c, K)
        
        # The objective is scale invariant, but DE needs a bounded domain.
        # We can normalize h inside the evaluation to prevent overflow/underflow,
        # but the task value is invariant to scaling h.
        # However, _evaluate handles the scaling via sum(h)^2.
        val = _evaluate(h)
        return val

    # Try a range of K values
    K_candidates = [5, 10, 20, 40]
    
    for K in K_candidates:
        if time.time() > deadline - 20:
            break
            
        n_vars = 2 * (K + 1)
        
        # Define bounds for the coefficients.
        # The magnitude of coefficients determines the amplitude of h.
        # Since the objective is scale-invariant, we can fix a reasonable range.
        # If all coeffs are 0, sum(h)=0, which is invalid.
        # We must ensure the DC component (c[0]) is not too small or the signal doesn't vanish.
        # Let's use bounds [-1, 1] for all real/imag parts.
        bounds = [(-1.0, 1.0)] * n_vars
        
        # DE settings
        # maxiter is the number of generations.
        # popsize is the population size multiplier.
        # Since we have limited time, we start with modest settings.
        
        try:
            # Use a smaller population if K is large to save time
            popsize = max(10, 20 - K//2)
            
            res = differential_evolution(
                _objective,
                bounds,
                maxiter=50,
                popsize=popsize,
                tol=1e-5,
                seed=42,
                polish=True, # Use L-BFGS-B at the end for local refinement
                options={'maxiter': 100, 'ftol': 1e-8}
            )
            
            if res.fun < best_val:
                best_val = res.fun
                # Reconstruct best_h
                K_used = K
                c_best = np.zeros(K_used + 1, dtype=complex)
                real_parts = res.x[:K_used+1]
                imag_parts = res.x[K_used+1:]
                c_best = real_parts + 1j * imag_parts
                best_h = _h_from_coeffs(c_best, K_used)
                
        except Exception as e:
            pass

    # If DE didn't find anything better than seed, return seed
    final_val = _evaluate(best_h)
    if not np.isfinite(final_val) or final_val >= 2.0:
        return np.ones(N)
        
    return best_h
