"""
Search for a signed step function (sequence of ±1) of length N=400 that minimizes
the upper bound on C3 in the third autocorrelation inequality.

The task asks us to choose a signed step function with 400 equal intervals on [-1/4, 1/4].
The score is the upper bound on C3 (lower is better). The seed program returns all ones,
giving a score of 1.0 (the baseline).

Key insight: The "third autocorrelation inequality" relates to the autoconvolution of
the signal. For a signed sequence s of length N, the autoconvolution (or autocorrelation
at various lags) determines concentration. The C3 bound is related to the maximum of
|sum_{k} s_k s_{k+lag}| over non-zero lags, normalized appropriately.

The strategy: Use simulated annealing with block-flip moves to find a sequence that
minimizes the peak sidelobe level of the autocorrelation. We also incorporate a
gradient-descent-inspired local polish using a smooth surrogate (log-sum-exp) for
the max objective.

The evaluator measures the raw value (lower is better). Score = seed_value / value,
so 1.0 equals the seed and above 1.0 is an improvement.
"""

import numpy as np
import time

N = 400
DEADLINE = 145.0  # Safety margin before 150s kill


def _autocorr(s):
    """Compute the full autocorrelation of a real sequence s."""
    n = len(s)
    # Use FFT-based autocorrelation for speed
    f = np.fft.fft(s, n=2 * n)
    acf = np.fft.ifft(f * np.conj(f)).real[:n]
    return acf


def _score_from_acf(acf):
    """
    Compute the C3 upper bound from the autocorrelation.
    
    The third autocorrelation inequality involves the autoconvolution.
    For a step function on [-1/4, 1/4] with N intervals, the relevant quantity
    is the maximum absolute value of the non-trivial autocorrelation lags,
    normalized by N.
    
    The C3 bound is related to: max_{k != 0} |sum_i s_i s_{i+k}| / N
    which is the peak sidelobe of the autocorrelation.
    
    We use the peak sidelobe level (excluding lag 0) as our metric.
    """
    n = len(acf)
    # Exclude lag 0 (which is sum of squares = N for ±1 sequence)
    sidelobes = np.abs(acf[1:])
    if len(sidelobes) == 0:
        return 0.0
    peak = np.max(sidelobes)
    return peak / n


def _smooth_score(acf, beta=50.0):
    """
    Smooth surrogate for the max using log-sum-exp.
    This is differentiable and can guide gradient-based polish.
    """
    n = len(acf)
    sidelobes = np.abs(acf[1:])
    if len(sidelobes) == 0:
        return 0.0
    # Soft max via log-sum-exp
    m = np.max(sidelobes)
    soft_max = m + np.log(np.sum(np.exp(beta * (sidelobes - m)))) / beta
    return soft_max / n


def _greedy_polish(s, max_iter=20000, deadline=None):
    """
    Greedy local search: flip individual bits that reduce the score.
    """
    s = s.copy()
    n = len(s)
    best_score = _score_from_acf(_autocorr(s))
    
    for it in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        improved = False
        # Try flipping each position
        for i in range(n):
            s[i] = -s[i]
            acf = _autocorr(s)
            sc = _score_from_acf(acf)
            if sc < best_score - 1e-12:
                best_score = sc
                improved = True
            else:
                s[i] = -s[i]  # revert
        if not improved:
            break
    return s, best_score


def _anneal(s, t_start=1.0, t_end=0.01, n_iter=50000, deadline=None):
    """
    Simulated annealing with block-flip moves.
    Block flips of size 4-32 allow coordinated sign changes.
    """
    s = s.copy()
    n = len(s)
    best_s = s.copy()
    best_score = _score_from_acf(_autocorr(s))
    
    current_score = best_score
    current_temp = t_start
    
    for it in range(n_iter):
        if deadline is not None and time.time() > deadline:
            break
        
        # Annealing schedule
        progress = it / max(n_iter - 1, 1)
        current_temp = t_start * (t_end / t_start) ** progress
        
        # Choose a block size
        block_size = np.random.randint(4, 33)
        block_size = min(block_size, n)
        
        # Choose random start position
        start = np.random.randint(0, n - block_size + 1)
        
        # Save old values
        old_vals = s[start:start + block_size].copy()
        
        # Flip the block
        s[start:start + block_size] = -old_vals
        
        # Compute new score
        acf = _autocorr(s)
        new_score = _score_from_acf(acf)
        
        delta = new_score - current_score
        
        # Metropolis acceptance
        if delta < 0 or (current_temp > 0 and np.random.random() < np.exp(-delta / current_temp)):
            current_score = new_score
            if new_score < best_score:
                best_score = new_score
                best_s = s.copy()
        else:
            # Revert
            s[start:start + block_size] = old_vals
    
    return best_s, best_score


def _gradient_polish(s, max_iter=5000, deadline=None):
    """
    Local polish using a smooth surrogate and coordinate-wise gradient steps.
    For each position, determine the sign that minimizes the smooth score.
    """
    s = s.copy()
    n = len(s)
    beta = 30.0
    best_score = _score_from_acf(_autocorr(s))
    
    for it in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        improved = False
        for i in range(n):
            # Try both signs
            s[i] = 1.0
            acf = _autocorr(s)
            sc1 = _smooth_score(acf, beta)
            
            s[i] = -1.0
            acf = _autocorr(s)
            sc2 = _smooth_score(acf, beta)
            
            if sc2 < sc1:
                s[i] = -1.0
            else:
                s[i] = 1.0
            
            # Check actual score
            acf = _autocorr(s)
            actual = _score_from_acf(acf)
            if actual < best_score - 1e-12:
                best_score = actual
                improved = True
        if not improved:
            break
    
    return s, best_score


def construct():
    """
    Construct a signed step function (sequence of ±1) of length N=400
    that minimizes the C3 autoconvolution upper bound.
    
    Returns: numpy array of shape (N,) with values ±1
    """
    start_time = time.time()
    deadline = start_time + DEADLINE
    
    # Initialize with random sequence
    np.random.seed(42)
    s = np.random.choice([-1.0, 1.0], size=N)
    
    best_s = s.copy()
    best_score = _score_from_acf(_autocorr(s))
    
    # Phase 1: Simulated annealing with block flips
    s, score = _anneal(s, t_start=1.0, t_end=0.005, n_iter=30000, deadline=deadline)
    if score < best_score:
        best_score = score
        best_s = s.copy()
    
    # Phase 2: Greedy polish
    s, score = _greedy_polish(best_s, max_iter=10000, deadline=deadline)
    if score < best_score:
        best_score = score
        best_s = s.copy()
    
    # Phase 3: More annealing from the polished state
    if time.time() < deadline - 5:
        s, score = _anneal(best_s, t_start=0.3, t_end=0.002, n_iter=20000, deadline=deadline)
        if score < best_score:
            best_score = score
            best_s = s.copy()
        
        # Phase 4: Final greedy polish
        s, score = _greedy_polish(best_s, max_iter=10000, deadline=deadline)
        if score < best_score:
            best_score = score
            best_s = s.copy()
    
    # Phase 5: Gradient-based polish with smooth surrogate
    if time.time() < deadline - 3:
        s, score = _gradient_polish(best_s, max_iter=3000, deadline=deadline)
        if score < best_score:
            best_score = score
            best_s = s.copy()
    
    # Final check: ensure we're within deadline
    if time.time() > deadline - 2:
        pass  # Just return what we have
    
    # Ensure output is integer ±1
    best_s = np.sign(best_s)
    best_s[best_s == 0] = 1.0  # Handle any zeros (shouldn't happen)
    
    return best_s
