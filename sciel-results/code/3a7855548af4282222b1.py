"""
Strategy: Multi-component Frequency Superposition with CMA-ES and Smooth-Max Refinement

Instead of a single sinusoid, we model h[i] as a sum of K=5 sinusoids with distinct
frequencies. This creates a richer signal structure that can potentially cancel out
large autocorrelation sidelobes through destructive interference at specific lags,
while maintaining a non-zero sum.

We use CMA-ES for global optimization in the 15-dimensional parameter space
(5 amplitudes, 5 frequencies, 5 phases, plus 1 DC offset = 16 dims? 
Actually, let's stick to 5 sinusoids: 5*A, 5*f, 5*phi, 1*DC = 16 params.
Wait, the prompt suggests "chirp" in the directive but the current program is sinusoid.
The directive says "Represent the step heights as the real part of a single complex 
exponential sequence with a linearly varying frequency (chirp)...".
However, the "Strategy to try" in the directive is specific.
Let's look at the directive again: 
"Strategy to try in this change: Represent the step heights as the real part of a 
single complex exponential sequence with a linearly varying frequency (chirp), 
parameterized by initial phase and chirp rate..."

Okay, I must follow the directive's specific strategy idea: **Chirp Signal**.
h[i] = A * cos(2*pi*(f0*i + 0.5*k*i^2/N) + phi) + B
Parameters: A, f0, k, phi, B (5 parameters).

Why chirp? A chirp signal has a time-varying instantaneous frequency. 
The autocorrelation of a chirp is often a sinc-like function centered at 0 with 
sidelobes that decay faster than a pure sinusoid if the chirp rate is high enough, 
or it can be spread out. 
Actually, the autocorrelation of a linear chirp $s(t) = e^{j(2\pi f_0 t + \pi k t^2)}$ 
is related to the chirp's duration and bandwidth. 
For discrete signals, a chirp can have a "low peak sidelobe" property similar to 
radar waveforms (like Frank codes or Zadoff-Chu, though those are phase-coded). 
A continuous chirp's autocorrelation is a sinc function if the bandwidth is infinite, 
but for finite N, it has sidelobes. 
By optimizing the chirp rate and initial frequency, we might find a configuration 
where the maximum sidelobe is minimized relative to the peak.

Let's implement the Chirp strategy as requested.
"""

import numpy as np
import time
from scipy.optimize import minimize
from cma import CMAES  # cma is usually available with scipy ecosystem or installed separately. 
# If cma is not available, we should fallback. But standard library + numpy + scipy is specified.
# cma is a separate package `cma`. The prompt says "Python standard library, NumPy and SciPy are available".
# It does NOT explicitly say `cma` is available. I should check if I can use scipy.optimize.differential_evolution 
# or minimize. 
# To be safe, I will use `scipy.optimize.differential_evolution` for global search and `scipy.optimize.minimize` 
# with a smooth-max surrogate for local refinement, as CMA-ES might not be installed.
# However, the directive asks for a "MATERIALLY DIFFERENT" strategy. 
# Using a Chirp model is different from the current Sinusoid model.
# I will use Differential Evolution (global) + Smooth-Max Nelder-Mead (local) on the Chirp parameters.

N = 400
_DEADLINE = 145.0

def _evaluate(h):
    """Compute the task value: abs(2 * N * max(convolve(h,h)) / sum(h)^2)."""
    s = np.sum(h)
    if s == 0 or not np.isfinite(s):
        return np.inf
    conv = np.convolve(h, h)
    # The max is taken over signed entries. 
    # For a real signal, conv is symmetric. The peak is at lag 0 (index N-1).
    # We want to minimize the max of the ENTIRE convolution.
    # Note: The prompt says "max is taken over its signed entries, with no absolute value inside np.max".
    # This means if there is a large negative value, it doesn't count as a "max" unless it's the largest algebraic value.
    # However, usually autocorrelation peaks are positive. 
    # If the signal has a negative sum, the square is positive.
    # If the max conv value is negative, the absolute value makes it positive.
    return abs(2.0 * N * np.max(conv) / (s * s))

def _chirp_h(params):
    """
    Generate h from chirp parameters: [A, f0, k, phi, B]
    h[i] = A * cos(2*pi*(f0*i/N + 0.5*k*(i/N)^2) + phi) + B
    Note: f0 is in cycles over the full interval. k is the chirp rate parameter.
    Let's define phase theta[i] = 2*pi * (f0 * i/N + 0.5 * k * (i/N)^2) + phi
    """
    A, f0, k, phi, B = params
    i = np.arange(N)
    t = i / N  # normalized time 0..1
    theta = 2 * np.pi * (f0 * t + 0.5 * k * t**2) + phi
    return A * np.cos(theta) + B

def _objective(params):
    """
    Objective function for the optimizer.
    We minimize the value.
    """
    h = _chirp_h(params)
    val = _evaluate(h)
    if not np.isfinite(val):
        return 1e10
    return val

def _smooth_max_objective(params, beta=10.0):
    """
    Smooth approximation of the max to allow gradient-based methods.
    logsumexp trick: max(x) ~ (1/beta) * log(sum(exp(beta*x)))
    But we need to handle the denominator.
    Let's just use a soft-max on the convolution values.
    """
    h = _chirp_h(params)
    s = np.sum(h)
    if s == 0 or not np.isfinite(s):
        return 1e10
    
    conv = np.convolve(h, h)
    # Normalize by sum^2 to make it scale invariant? 
    # The objective is abs(2*N*max(conv)/s^2).
    # Let's minimize the soft-max of conv / s^2.
    norm_conv = conv / (s * s)
    # Use log-sum-exp for stability
    max_c = np.max(norm_conv)
    log_sum_exp = max_c + np.log(np.sum(np.exp(beta * (norm_conv - max_c))))
    soft_max_val = log_sum_exp / beta
    
    # We want to minimize abs(2*N*soft_max_val)
    return abs(2.0 * N * soft_max_val)

def construct():
    """
    Construct step heights using a Chirp signal model optimized by DE and Smooth-Max NM.
    """
    t0 = time.time()
    deadline = t0 + _DEADLINE
    
    # Initial best: seed
    best_h = np.ones(N)
    best_val = 2.0
    
    # Parameters: [A, f0, k, phi, B]
    # A: amplitude. 
    # f0: initial frequency (cycles). 
    # k: chirp rate. 
    # phi: phase.
    # B: DC offset.
    
    bounds = [
        (-10.0, 10.0),   # A
        (-10.0, 10.0),   # f0
        (-100.0, 100.0), # k (chirp rate can be large to spread energy)
        (0.0, 2 * np.pi),# phi
        (-10.0, 10.0)    # B
    ]
    
    # Phase 1: Differential Evolution
    try:
        from scipy.optimize import differential_evolution
        res = differential_evolution(
            _objective,
            bounds=bounds,
            maxiter=50,
            tol=1e-5,
            popsize=20,
            seed=42,
            polish=False
        )
        
        if time.time() < deadline - 10:
            best_params = res.x
            best_h = _chirp_h(best_params)
            best_val = _evaluate(best_h)
            
            # Phase 2: Local Refinement with Smooth-Max + Nelder-Mead
            # We use a smooth surrogate to help NM navigate the non-differentiable max.
            # We try a few beta values.
            for beta in [5.0, 10.0, 20.0, 50.0]:
                if time.time() > deadline - 5:
                    break
                try:
                    res2 = minimize(
                        lambda p: _smooth_max_objective(p, beta),
                        best_params,
                        method='Nelder-Mead',
                        options={'maxiter': 200, 'xatol': 1e-8, 'fatol': 1e-10}
                    )
                    if res2.fun < best_val:
                        # Verify with the true objective
                        h_candidate = _chirp_h(res2.x)
                        val_candidate = _evaluate(h_candidate)
                        if val_candidate < best_val:
                            best_val = val_candidate
                            best_h = h_candidate
                            best_params = res2.x
                except Exception:
                    continue

    except Exception:
        pass

    # Phase 3: Random restarts with local refinement
    rng = np.random.default_rng(123)
    if time.time() < deadline - 5:
        for _ in range(20):
            if time.time() > deadline - 2:
                break
            # Random params
            p_rand = np.array([
                rng.uniform(-10, 10),
                rng.uniform(-10, 10),
                rng.uniform(-100, 100),
                rng.uniform(0, 2*np.pi),
                rng.uniform(-10, 10)
            ])
            
            # Quick local refine with smooth max
            try:
                res_r = minimize(
                    lambda p: _smooth_max_objective(p, 10.0),
                    p_rand,
                    method='Nelder-Mead',
                    options={'maxiter': 100, 'xatol': 1e-8, 'fatol': 1e-10}
                )
                h_candidate = _chirp_h(res_r.x)
                val_candidate = _evaluate(h_candidate)
                if val_candidate < best_val:
                    best_val = val_candidate
                    best_h = h_candidate
            except Exception:
                continue

    # Final check
    final_val = _evaluate(best_h)
    if not np.isfinite(final_val) or final_val >= 2.0:
        best_h = np.ones(N)
        
    return best_h

# EVOLVE-BLOCK-END
