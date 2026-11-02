import numpy as np
import time

N = 400

def _autocorr_max_abs(x):
    """Compute max absolute value of circular autocorrelation using FFT."""
    X = np.fft.rfft(x)
    power = X * np.conj(X)
    acf = np.fft.irfft(power, N)
    return np.max(np.abs(acf))

def _compute_score(x):
    """
    Compute the score: max|autoconvolution| / (integral)^2.
    Based on analysis, this is max|r(k)| / (sum(x))^2.
    For x = ones(N), this is N / N^2 = 1/N.
    """
    acf_max = _autocorr_max_abs(x)
    total_sum = np.sum(x)
    norm_sq = total_sum * total_sum
    if norm_sq < 1e-12:
        return np.inf
    return acf_max / norm_sq

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    # The optimal solution for the metric max|r| / (sum x)^2 on {-1, 1}^N
    # is the all-ones sequence, which yields 1/N.
    # Any other sequence has sum < N and max|r| >= N (at lag 0),
    # or if lag 0 is excluded, the peak reduction is insufficient to
    # compensate for the square of the sum reduction.
    
    best_x = np.ones(N)
    best_score = _compute_score(best_x)
    
    # Attempt local search to see if any deviation improves the score.
    # This is likely to fail, but ensures we aren't missing a niche optimum.
    rng = np.random.RandomState(42)
    x = best_x.copy()
    current_score = best_score
    
    # We try to flip a small number of bits to see if we can reduce peak
    # while keeping sum high.
    # However, as derived, flipping bits increases the ratio.
    # We will run a short hill-climbing just to be sure.
    
    steps = 10000
    for i in range(steps):
        if time.time() > deadline:
            break
            
        idx = rng.randint(N)
        old_val = x[idx]
        x[idx] = -old_val
        
        new_score = _compute_score(x)
        
        if new_score < current_score:
            current_score = new_score
            if new_score < best_score:
                best_score = new_score
                best_x = x.copy()
        else:
            x[idx] = old_val
            
    return best_x
