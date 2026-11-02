"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time


def _compute_c3_bound(x):
    """
    Compute the upper bound on C3 for a signed step function x.
    
    The third autocorrelation inequality involves the autoconvolution.
    For a function f, the autoconvolution is (f * f)(t) = ∫ f(s)f(t-s) ds.
    
    For a step function on [-1/4, 1/4] with N equal intervals, the autoconvolution
    at lag k (in interval units) is related to the autocorrelation.
    
    The C3 bound is typically related to the maximum absolute value of the
    normalized autoconvolution/autocorrelation away from zero lag.
    
    For a ±1 sequence of length N, the autocorrelation at lag k is:
        R(k) = sum_{i=0}^{N-1-k} x[i] * x[i+k]
    
    The normalized version is R(k) / (N-k) or similar.
    
    The upper bound on C3 is likely max_{k != 0} |R(k)| / R(0) or similar.
    Since R(0) = N for ±1 values, this is max_{k != 0} |R(k)| / N.
    """
    x = np.asarray(x)
    N_local = len(x)
    
    # Compute autocorrelation for all lags
    # R(k) = sum_{i=0}^{N-1-k} x[i] * x[i+k]
    max_abs = 0.0
    for k in range(1, N_local):
        corr = np.dot(x[:N_local-k], x[k:])
        abs_corr = abs(corr)
        if abs_corr > max_abs:
            max_abs = abs_corr
    
    # Normalize by N (since R(0) = N for ±1 values)
    return max_abs / N_local


def _flip_bit(x, idx):
    """Flip the sign at position idx."""
    x = x.copy()
    x[idx] = -x[idx]
    return x


def construct():
    """
    Search for a signed step function that minimizes the C3 upper bound.
    Uses simulated annealing / local search with time limit.
    """
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit
    
    # Initialize with random ±1 sequence
    rng = np.random.default_rng(42)
    best_x = rng.choice([-1, 1], size=N)
    best_score = _compute_c3_bound(best_x)
    
    current_x = best_x.copy()
    current_score = best_score
    
    # Simulated annealing
    T = 0.5
    T_min = 0.001
    cooling_rate = 0.999
    
    while time.time() < deadline:
        # Pick a random position to flip
        idx = rng.integers(0, N)
        new_x = _flip_bit(current_x, idx)
        new_score = _compute_c3_bound(new_x)
        
        delta = new_score - current_score
        
        # Accept if better or with probability exp(-delta/T)
        if delta < 0 or (T > T_min and rng.random() < np.exp(-delta / T)):
            current_x = new_x
            current_score = new_score
            
            if current_score < best_score:
                best_score = current_score
                best_x = current_x.copy()
        
        # Cool down
        T = max(T * cooling_rate, T_min)
        
        # Check time
        if time.time() > deadline - 10:
            break
    
    return best_x
# EVOLVE-BLOCK-END
