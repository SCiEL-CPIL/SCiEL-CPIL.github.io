"""
Strategy: Low-Frequency Fourier Parameterization.

The objective function V = |2 * N * max(conv(h, h)) / sum(h)^2| is scale-invariant.
We can fix the sum of h to 1 (or any non-zero constant) without loss of generality.
Let h = c * (1 + u), where sum(u) = 0.
Then sum(h) = c * N.
conv(h, h) = c^2 * conv(1+u, 1+u).
V = |2 * N * c^2 * max(conv(1+u, 1+u)) / (c^2 * N^2)| = |2 * max(conv(1+u, 1+u)) / N|.

So we just need to minimize max(conv(1+u, 1+u)) subject to sum(u)=0.
However, the max can be negative if the peak is negative? No, conv(0,0) = sum(u_i^2) >= 0.
Wait, max(conv) includes the lag 0 term which is sum(h_i^2).
Actually, the prompt says "maximum is taken over its signed entries".
For h = ones(N), conv(0) = N, other terms are N-lag. Max is N.
V = 2 * N * N / N^2 = 2.

If we represent h using a few low-frequency Fourier modes, we capture the main variations
that affect the autocorrelation peak. High-frequency noise tends to increase the lag-0
peak (energy) without significantly shifting the location of the max, or it creates
many small peaks. The global max is likely dominated by the DC and low-frequency terms.

We parameterize h as:
h[k] = A_0 + sum_{m=1}^{M} (A_m cos(2 pi m k / N) + B_m sin(2 pi m k / N))

We optimize the coefficients {A_0, ..., B_M} to minimize V.
Since V is scale invariant, we can fix A_0 = 1 (or any non-zero value) to remove the
gauge freedom, or just let the optimizer find the scale. Fixing A_0=1 is safer to avoid
convergence to 0.

We will use scipy.optimize.minimize with L-BFGS-B on this reduced parameter space.
We will perform multiple restarts with different random initializations of the
higher-frequency coefficients (keeping low frequencies small initially) to explore
the basin of attraction.
"""

import numpy as np
import time
from scipy.optimize import minimize

N = 400
_DEADLINE = 140.0

def _evaluate(h):
    """Compute the task value."""
    s = np.sum(h)
    if s == 0 or not np.isfinite(s):
        return np.inf
    conv = np.convolve(h, h)
    return abs(2.0 * N * np.max(conv) / (s * s))

def _h_from_coeffs(coeffs):
    """
    Reconstruct h from Fourier coefficients.
    coeffs[0] = A_0 (DC)
    coeffs[1] = A_1, coeffs[2] = B_1
    coeffs[3] = A_2, coeffs[4] = B_2
    ...
    """
    n = np.arange(N)
    h = np.full(N, coeffs[0])
    for m in range(1, len(coeffs)//2 + 1):
        a = coeffs[2*m - 1]
        b = coeffs[2*m]
        h += a * np.cos(2 * np.pi * m * n / N) + b * np.sin(2 * np.pi * m * n / N)
    return h

def _objective(coeffs):
    """
    Objective function for the optimizer.
    We add a small penalty if sum(h) is close to 0 to keep it well-conditioned,
    though scale invariance means the value doesn't change, the gradient might.
    Actually, just evaluating V is enough.
    """
    h = _h_from_coeffs(coeffs)
    if not np.all(np.isfinite(h)):
        return 1e10
    s = np.sum(h)
    if abs(s) < 1e-6:
        return 1e10
    conv = np.convolve(h, h)
    return abs(2.0 * N * np.max(conv) / (s * s))

def construct():
    """
    Construct step heights using low-frequency Fourier optimization.
    """
    t0 = time.time()
    deadline = t0 + _DEADLINE
    
    best_val = _evaluate(np.ones(N))
    best_h = np.ones(N)
    
    # Number of frequency modes to include
    # Higher modes allow more flexibility but increase dimensionality.
    # M=10 means 21 parameters. M=20 means 41 parameters.
    # Let's try M=15 (31 parameters).
    M = 15
    
    rng = np.random.default_rng(42)
    
    # We will run several independent optimizations.
    # Each starts with a random set of coefficients.
    n_restarts = 20
    
    for i in range(n_restarts):
        if time.time() > deadline - 10:
            break
            
        # Initialize coefficients
        # A_0 should be non-zero. Let's set it to 1.0.
        # Other coefficients start small to avoid wild oscillations initially.
        coeffs = np.zeros(2 * M + 1)
        coeffs[0] = 1.0
        
        # Add random small perturbations to higher modes
        for j in range(1, 2 * M + 1):
            # Amplitude decays with frequency to favor smooth solutions
            amp = 0.1 / (1 + j/10.0)
            coeffs[j] = rng.uniform(-amp, amp)
            
        # If we want to explore more aggressively, we can add larger perturbations
        # in later restarts
        if i > n_restarts // 2:
            for j in range(1, 2 * M + 1):
                amp = 0.5 / (1 + j/20.0)
                coeffs[j] = rng.uniform(-amp, amp)

        try:
            res = minimize(
                _objective,
                coeffs,
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            
            if res.success or res.fun < best_val:
                h_opt = _h_from_coeffs(res.x)
                if np.all(np.isfinite(h_opt)) and np.sum(h_opt) != 0:
                    v_opt = _evaluate(h_opt)
                    if v_opt < best_val:
                        best_val = v_opt
                        best_h = h_opt.copy()
                        
        except Exception:
            pass

    # Final polish with a few greedy steps on the full 400-dim vector
    # This can catch improvements that the Fourier basis missed (high freq details)
    if time.time() < deadline - 5:
        h_curr = best_h.copy()
        v_curr = best_val
        step_size = 0.05
        for _ in range(5000):
            if time.time() > deadline - 2:
                break
            idx = rng.integers(0, N)
            h_new = h_curr.copy()
            h_new[idx] += rng.normal(0, step_size)
            v_new = _evaluate(h_new)
            if v_new < v_curr:
                v_curr = v_new
                h_curr = h_new
                step_size *= 0.95
            else:
                step_size = min(step_size * 1.1, 0.2)
        
        if v_curr < best_val:
            best_val = v_curr
            best_h = h_curr

    if not np.all(np.isfinite(best_h)) or np.sum(best_h) == 0:
        return np.ones(N)
        
    return best_h

# EVOLVE-BLOCK-END
