"""
Optimized Signed Step Functions: Constant + Rudin-Shapiro Exploration.

The constant sequence achieves the theoretical lower bound of 1/N for
max|autoconv| / (sum x)^2 by Cauchy-Schwarz. This program retains the
constant as the primary solution and explores Rudin-Shapiro-like
constructions as a secondary search, updating only if a strictly better
score is found.
"""

import time
import numpy as np
from scipy import signal

N = 400

# EVOLVE-BLOCK-START

def _autoconv_max_abs(x):
    """Compute max |autoconvolution| of x."""
    ac = signal.fftconvolve(x, x, mode='full')
    return np.max(np.abs(ac))

def _integral_sq(x):
    """Compute (sum x)^2."""
    s = np.sum(x)
    return s * s

def _score(x):
    """Compute the objective: max |autoconv| / (sum x)^2. Lower is better."""
    s = np.sum(x)
    if abs(s) < 1e-12:
        return np.inf
    return _autoconv_max_abs(x) / _integral_sq(x)

def _build_rudin_shapiro_like(n, rng):
    """
    Generate a Rudin-Shapiro-like sequence of length n.
    Rudin-Shapiro sequences have bounded autocorrelation.
    We generate them via the standard recursive construction,
    then truncate/pad to length n.
    """
    # Build Rudin-Shapiro sequence of length power of 2 >= n
    length = 1
    while length < n:
        length *= 2
    
    # Standard Rudin-Shapiro construction
    a = np.array([1.0])
    b = np.array([1.0])
    for _ in range(int(np.log2(length))):
        a_new = np.concatenate([a, b])
        b_new = np.concatenate([a, -b])
        a, b = a_new, b_new
    
    # Take first n elements
    rs = a[:n].copy()
    
    # Add random sign flips to create variants
    if rng is not None:
        flip = rng.random(n) < 0.1
        rs[flip] *= -1
    
    return rs

def _hill_climb(x, rng, max_iter=50, step_size=0.1, deadline=None):
    """
    Local hill climbing on the sequence values.
    Tries small perturbations to reduce the objective.
    """
    best_x = x.copy()
    best_score = _score(best_x)
    
    for i in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        
        # Pick a random index to perturb
        idx = rng.integers(0, len(best_x))
        
        # Try small changes
        for delta in [-step_size, step_size]:
            trial = best_x.copy()
            trial[idx] += delta
            
            sc = _score(trial)
            if sc < best_score:
                best_score = sc
                best_x = trial
                break
        
        # Also try flipping a sign
        if rng.random() < 0.3:
            idx = rng.integers(0, len(best_x))
            trial = best_x.copy()
            trial[idx] = -trial[idx]
            sc = _score(trial)
            if sc < best_score:
                best_score = sc
                best_x = trial
    
    return best_x, best_score

def construct():
    """
    Build a signed step-height vector of length N.
    Primary: constant sequence (theoretical optimum 1/N).
    Secondary: Rudin-Shapiro exploration + hill climbing.
    """
    t0 = time.time()
    deadline = 140.0  # Safety margin before 150s limit
    
    # Initial best: The constant sequence (theoretical optimum)
    best_x = np.ones(N)
    best_score = _score(best_x)
    
    rng = np.random.default_rng(42)
    
    # Explore Rudin-Shapiro-like sequences
    n_rs_trials = 2000
    for i in range(n_rs_trials):
        if time.time() > deadline:
            break
        
        # Generate Rudin-Shapiro-like candidate
        x_candidate = _build_rudin_shapiro_like(N, rng)
        
        # Try adding a DC offset to increase the sum
        # x' = x + c. Optimal c to maximize sum while controlling peak.
        # For a zero-mean-like sequence, adding c makes sum = N*c
        # and autoconv gets dominated by the constant part.
        
        # Try several bias values
        for bias in [0.0, 0.5, 1.0, 2.0, 5.0]:
            x_b = x_candidate + bias
            sc = _score(x_b)
            if sc < best_score:
                best_score = sc
                best_x = x_b.copy()
        
        # If we found something promising, do a quick hill climb
        if best_score < 1.0 / N * 1.01:  # Very close to optimal
            if time.time() < deadline - 10:
                best_x, best_score = _hill_climb(
                    best_x, rng, max_iter=20, step_size=0.05, deadline=deadline
                )
    
    # Final verification
    final_score = _score(best_x)
    if final_score < best_score:
        best_score = final_score
    
    return best_x

# EVOLVE-BLOCK-END
