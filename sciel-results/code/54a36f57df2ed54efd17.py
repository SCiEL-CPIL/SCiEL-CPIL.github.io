"""
Strategy: Represent the step function as a smooth, nonnegative envelope (raised cosine)
multiplied by a periodic square-wave-like modulation with adjustable harmonic phases.
Use scipy.optimize.differential_evolution to minimize the autoconvolution peak.

The step function is defined on 600 equal intervals on [-1/4, 1/4].
We parameterize the heights as:
  h_i = envelope(x_i) * (0.5 + 0.5 * sign(sin(2*pi*k*x_i + phi)))
But to keep it smooth and differentiable, we use a soft version:
  h_i = envelope(x_i) * (0.5 + 0.5 * tanh(beta * sin(2*pi*k*x_i + phi)))

where x_i is the center of interval i, envelope is a raised cosine,
and k, phi, beta are optimization parameters.

Actually, let's use a simpler but effective parameterization:
  h_i = A * (1 + r * cos(2*pi*m*x_i + phi))

where A, r, m, phi are parameters. We constrain r in [0, 1) to ensure nonnegativity
when A is chosen appropriately. Actually, let's just use:
  h_i = max(0, A * (1 + r * cos(2*pi*m*x_i + phi)))

This is nonnegative by construction. We optimize A, r, m, phi.

But the task says "nonnegative step function", so we need all h_i >= 0.

Let me use a cleaner approach:
  h_i = a + b * cos(2*pi*m*x_i + phi)

with constraint a >= |b| to ensure nonnegativity. We can parameterize:
  a = c * (1 + eps), b = c * eps * cos(alpha) ... no, let's just use:
  a = t * (1 + r), b = t * r, where t > 0, 0 <= r < 1.
  Then h_i = t * (1 + r + r*cos(...)) >= t*(1+r-r) = t > 0.

Wait, that's not right. Let me just use:
  h_i = t * (1 + r * cos(2*pi*m*x_i + phi))
with 0 <= r < 1, t > 0. Then min h_i = t*(1-r) > 0.

We optimize t, r, m, phi. The autoconvolution peak is what we minimize.

Actually, the "normalized maximum of its autoconvolution" - let me think about what this means.
The autoconvolution of a step function on [-1/4, 1/4] with 600 intervals.
The autoconvolution at lag k is sum_i h_i * h_{i+k} (with appropriate bounds).
The "normalized maximum" is likely max_k (autoconv_k) / (sum h_i)^2 or similar.

Let me just compute the autoconvolution as the convolution of h with h, take the max,
and normalize by (sum h)^2 to make it scale-invariant.

Let me implement this and use differential_evolution.
"""

import numpy as np
import time
from scipy.optimize import differential_evolution

N = 600
DEADLINE = 120.0  # seconds

def _compute_autoconv_peak(heights):
    """Compute the normalized maximum of the autoconvolution."""
    h = np.asarray(heights, dtype=np.float64)
    # Autoconvolution: (h * h)(k) = sum_i h[i] * h[i+k]
    # Using FFT for efficiency
    n = len(h)
    # Pad to avoid circular convolution
    size = 2 * n
    f = np.fft.rfft(h, n=size)
    autoconv = np.fft.irfft(f * f, n=size)
    # The autoconvolution of h with itself has length 2*n-1, centered at n-1
    # The peak is the maximum of the autoconvolution
    peak = np.max(autoconv)
    # Normalize by (sum h)^2
    total = np.sum(h)
    if total > 1e-12:
        return peak / (total * total)
    return 0.0

def _objective(params):
    """Objective function for optimization: normalized autoconvolution peak."""
    t, r, m, phi = params
    # Ensure nonnegativity: r in [0, 1), t > 0
    r = np.clip(r, 0.0, 0.999)
    t = max(t, 1e-6)
    
    # Interval centers on [-1/4, 1/4]
    x = np.linspace(-0.25, 0.25, N, endpoint=False) + (0.5 / N)
    
    # Step function heights
    h = t * (1.0 + r * np.cos(2.0 * np.pi * m * x + phi))
    
    return _compute_autoconv_peak(h)

def construct():
    """Construct the step function heights."""
    start_time = time.time()
    
    # Initial guess: uniform
    best_heights = np.ones(N)
    best_score = _compute_autoconv_peak(best_heights)
    
    # Define bounds for differential_evolution
    # t: scale, r: modulation depth, m: frequency, phi: phase
    bounds = [
        (0.1, 5.0),    # t: scale
        (0.0, 0.99),   # r: modulation depth
        (1.0, 10.0),   # m: frequency
        (0.0, 2.0 * np.pi),  # phi: phase
    ]
    
    # Check time
    elapsed = time.time() - start_time
    if elapsed > DEADLINE - 30:
        return list(best_heights)
    
    try:
        result = differential_evolution(
            _objective,
            bounds,
            seed=42,
            maxiter=50,
            tol=1e-6,
            polish=True,
            popsize=15,
        )
        
        elapsed = time.time() - start_time
        if elapsed < DEADLINE - 10:
            if result.fun < best_score:
                t, r, m, phi = result.x
                r = np.clip(r, 0.0, 0.999)
                t = max(t, 1e-6)
                x = np.linspace(-0.25, 0.25, N, endpoint=False) + (0.5 / N)
                best_heights = t * (1.0 + r * np.cos(2.0 * np.pi * m * x + phi))
                best_score = result.fun
    except Exception:
        pass
    
    # Final check
    elapsed = time.time() - start_time
    if elapsed < DEADLINE - 5:
        # Try a few more random restarts with different parameters
        for seed in [1, 2, 3, 4, 5]:
            if time.time() - start_time > DEADLINE - 10:
                break
            try:
                result = differential_evolution(
                    _objective,
                    bounds,
                    seed=seed,
                    maxiter=20,
                    tol=1e-5,
                    polish=False,
                    popsize=10,
                )
                if result.fun < best_score:
                    t, r, m, phi = result.x
                    r = np.clip(r, 0.0, 0.999)
                    t = max(t, 1e-6)
                    x = np.linspace(-0.25, 0.25, N, endpoint=False) + (0.5 / N)
                    best_heights = t * (1.0 + r * np.cos(2.0 * np.pi * m * x + phi))
                    best_score = result.fun
            except Exception:
                pass
    
    # Ensure nonnegativity
    best_heights = np.maximum(best_heights, 0.0)
    
    return list(best_heights)
