import numpy as np
import time

N = 400

def _autoconv(x):
    """Compute full autoconvolution of x via FFT."""
    n = len(x)
    size = 1
    while size < 2 * n - 1:
        size *= 2
    X = np.fft.rfft(x, size)
    ac = np.fft.irfft(X * np.conj(X), size)[:2 * n - 1]
    return ac

def _score(x):
    """Compute max|autoconv| / sum(x^2). Lower is better."""
    # Based on the analysis that seed score is 1.0 for all-ones, 
    # and max_ac=N, the denominator must be N (sum of squares).
    s_sq = np.sum(x * x)
    if abs(s_sq) < 1e-12:
        return float('inf')
    ac = _autoconv(x)
    max_ac = np.max(np.abs(ac))
    return max_ac / s_sq

def construct():
    start_time = time.time()
    deadline = start_time + 140.0

    # Initialize with random +/- 1 sequence
    # All ones has score 1.0. Random has score ~0.1-0.2.
    # We will start with random to get a better initial score immediately.
    rng = np.random.default_rng(42)
    x = rng.choice([-1.0, 1.0], size=N)
    
    best_score = _score(x)
    x_best = x.copy()

    # Precompute FFT size
    fft_size = 1
    while fft_size < 2 * N - 1:
        fft_size *= 2

    # Local Search: Flip bits to reduce max |ac|
    # We focus on the peak of the autoconvolution.
    
    # To speed up, we can compute the "flip effect" on the current peak.
    # Let k_max be the index of max |ac|.
    # Flipping x[i] changes ac[k] by delta.
    # We want to pick i that minimizes the new max |ac|.
    # Approximation: Minimize |ac[k_max] + delta_{k_max}|
    
    max_iterations = 100000
    no_improve = 0
    max_no_improve = 2000
    
    for iter in range(max_iterations):
        if time.time() > deadline - 5:
            break
            
        ac = _autoconv(x)
        abs_ac = np.abs(ac)
        k_max = np.argmax(abs_ac)
        val_max = ac[k_max]
        
        cur_score = abs(val_max) / np.sum(x*x)
        
        if cur_score < best_score:
            best_score = cur_score
            x_best = x.copy()
            no_improve = 0
        else:
            no_improve += 1
            if no_improve > max_no_improve:
                break

        # Find best flip
        # delta_ac[k] for flipping x[i] is -2 * x[i] * (x[k-i] + x[i-k])
        # We only care about k_max primarily, but flipping might increase other peaks.
        # Simple heuristic: Flip the i that most reduces |ac[k_max]|.
        
        best_i = -1
        best_val = abs(val_max)
        
        # Vectorized calculation of potential improvement for k_max
        # term_i = x[k_max - i] + x[i - k_max] (with zeros out of bounds)
        # We need to handle boundaries.
        
        # Create shifted arrays
        # x_shifted_pos: x[k_max - i] for i in 0..N-1
        # x_shifted_neg: x[i - k_max] for i in 0..N-1
        
        # k_max is in [0, 2N-2]
        # For a given i, valid indices for x[k_max - i] are 0 <= k_max - i < N
        # i.e., k_max - N < i <= k_max
        
        # Let's just loop if N is small enough? N=400. 400 iterations per step is fast.
        # 100k steps * 400 ops = 40M ops. Should be fine in 10s.
        
        for i in range(N):
            # Calculate change in ac[k_max]
            # term1: x[k_max - i] if valid
            t1 = 0.0
            idx1 = k_max - i
            if 0 <= idx1 < N:
                t1 = x[idx1]
            
            # term2: x[i - k_max] if valid
            t2 = 0.0
            idx2 = i - k_max
            if 0 <= idx2 < N:
                t2 = x[idx2]
            
            # delta = -2 * x[i] * (t1 + t2)
            delta = -2.0 * x[i] * (t1 + t2)
            
            new_val = val_max + delta
            new_abs = abs(new_val)
            
            if new_abs < best_val:
                best_val = new_abs
                best_i = i
                
        if best_i != -1:
            x[best_i] *= -1.0
        else:
            # Stuck? Try random flip or break
            if no_improve > 100:
                # Perturb
                idx = rng.integers(N)
                x[idx] *= -1.0
            else:
                break

    # Final check
    final_score = _score(x_best)
    if final_score < best_score:
        best_score = final_score

    return x_best
