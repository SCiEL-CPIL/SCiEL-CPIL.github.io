"""
Strategy: Zadoff-Chu Sequence Initialization with Local Gradient Refinement

The current program uses a low-rank Fourier parameterization. This approach is
structurally different: it initializes the step heights using a Zadoff-Chu sequence,
which is a chirp-like sequence known to have near-constant autocorrelation sidelobes.
This property makes it an excellent starting point for minimizing the peak of the
autoconvolution. 

We then perform a local gradient-based refinement (L-BFGS-B) directly on the 400
heights to further minimize the objective function. The Zadoff-Chu structure provides
a strong global optimum that is difficult to reach with random initialization, and the
local refinement polishes the result.

The value to minimize is:
    V(h) = abs(2 * N * max(conv(h, h)) / sum(h)^2)

By starting with a Zadoff-Chu sequence, we avoid the local minima traps that can
occur with random or smooth-window initializations, and we leverage the inherent
low-sidelobe properties of the chirp.
"""

import numpy as np
import time
from scipy.optimize import minimize

N = 400
_DEADLINE = 145.0  # Wall-clock limit with safety margin


def _evaluate(h):
    """Compute the task value."""
    s = np.sum(h)
    if s == 0 or not np.isfinite(s):
        return np.inf
    conv = np.convolve(h, h)
    return abs(2.0 * N * np.max(conv) / (s * s))


def _softmax_objective(h, beta=100.0):
    """
    Differentiable surrogate for the objective function.
    Uses a softmax approximation of the max to ensure differentiability.
    """
    s = np.sum(h)
    if s == 0 or not np.isfinite(s):
        return 1e10
    
    conv = np.convolve(h, h)
    m = np.max(conv)
    
    # Softmax approximation of max
    conv_shifted = conv - m
    weights = np.exp(beta * conv_shifted)
    soft_max = m + np.log(np.sum(weights)) / beta
    
    val = 2.0 * N * soft_max / (s * s)
    if not np.isfinite(val):
        return 1e10
    return val


def _generate_zadoff_chu(N, root):
    """
    Generate a Zadoff-Chu sequence of length N with the given root.
    The sequence is defined as:
        z[n] = exp(-j * pi * root * n * (n + 1) / N)  for n = 0, ..., N-1
    """
    n = np.arange(N)
    # Ensure integer arithmetic for the phase
    phase = -np.pi * root * n * (n + 1) / N
    z = np.exp(1j * phase)
    return z


def construct():
    """
    Construct step heights to minimize the autocorrelation inequality value.
    """
    t0 = time.time()
    deadline = t0 + _DEADLINE
    
    best_h = np.ones(N)
    best_val = 2.0
    
    # Try different Zadoff-Chu roots
    # The root should be coprime to N for optimal properties
    # N = 400 = 16 * 25, so roots coprime to 400 are those not divisible by 2 or 5
    candidate_roots = [1, 3, 7, 9, 11, 13, 17, 19, 21, 23, 27, 29, 31, 33, 37, 39]
    
    for root in candidate_roots:
        if time.time() > deadline - 20:
            break
        
        # Generate the Zadoff-Chu sequence
        z = _generate_zadoff_chu(N, root)
        
        # The heights must be real. We can use the real part, or a combination.
        # Using the real part (cosine) is a common choice.
        h_init = np.real(z)
        
        # Normalize to have sum 1 (or any non-zero sum) to avoid scaling issues
        s_init = np.sum(h_init)
        if abs(s_init) > 1e-10:
            h_init = h_init / s_init
        
        # Evaluate the initial value
        val_init = _evaluate(h_init)
        
        if val_init < best_val:
            best_val = val_init
            best_h = h_init.copy()
        
        # Local refinement using L-BFGS-B
        if time.time() < deadline - 10:
            def _obj_refine(h_flat):
                if time.time() > deadline - 5:
                    raise ValueError("Timeout")
                return _softmax_objective(h_flat, beta=100.0)
            
            try:
                res_refine = minimize(
                    _obj_refine,
                    h_init,
                    method='L-BFGS-B',
                    options={'maxiter': 50, 'ftol': 1e-12, 'gtol': 1e-8, 'maxfun': 100}
                )
                
                if res_refine.success:
                    h_ref = res_refine.x
                    val_ref = _evaluate(h_ref)
                    
                    if val_ref < best_val:
                        best_val = val_ref
                        best_h = h_ref.copy()
                        
            except Exception:
                pass

    # Final fallback check
    final_val = _evaluate(best_h)
    if not np.isfinite(final_val) or final_val >= 2.0:
        best_h = np.ones(N)
        
    return best_h
