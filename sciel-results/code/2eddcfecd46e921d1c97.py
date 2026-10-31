"""Optimized program: minimize max|autoconv| / (sum(x))^2 via local sign-flip search."""
import numpy as np
import time

N = 400

def _autoconv_max_abs(x):
    """Compute max absolute value of full autoconvolution of x."""
    # Use FFT for efficiency
    n = len(x)
    size = 2 * n
    fft_x = np.fft.fft(x, size)
    acov = np.fft.ifft(fft_x * np.conj(fft_x)).real
    # Full autoconvolution: first n values are positive lags, last n-1 are negative
    # acov[0:n] = sum_i x[i]*x[i+k] for k=0..n-1
    # acov[n:2n-1] = sum_i x[i]*x[i-k] for k=1..n-1
    return np.max(np.abs(acov))

def _score(x):
    """Compute the C3' ratio: max|autoconv| / (sum(x))^2."""
    s = np.sum(x)
    if abs(s) < 1e-12:
        return float('inf')
    return _autoconv_max_abs(x) / (s * s)

def construct():
    start_time = time.time()
    deadline = start_time + 120.0  # 120s limit, use 120s wall clock
    
    # Start with all ones (the seed)
    x = np.ones(N)
    best_x = x.copy()
    best_val = _score(x)
    
    # Try a few random initializations
    rng = np.random.RandomState(42)
    candidates = [x.copy()]
    
    # Random ±1 initializations
    for trial in range(5):
        if time.time() > deadline - 30:
            break
        x_rand = rng.choice([-1.0, 1.0], size=N)
        candidates.append(x_rand.copy())
    
    # Also try block-based patterns
    for block_size in [2, 4, 8, 16, 32]:
        if time.time() > deadline - 30:
            break
        n_blocks = N // block_size
        signs = rng.choice([-1.0, 1.0], size=n_blocks)
        x_block = np.repeat(signs, block_size)
        if len(x_block) < N:
            x_block = np.concatenate([x_block, np.ones(N - len(x_block))])
        candidates.append(x_block[:N].copy())
    
    # Local search: single sign flips
    def local_search(x_init, max_iters=50000):
        x_curr = x_init.copy()
        curr_val = _score(x_curr)
        
        for it in range(max_iters):
            if time.time() > deadline - 5:
                break
            # Pick a random index to flip
            idx = rng.randint(N)
            x_trial = x_curr.copy()
            x_trial[idx] *= -1
            trial_val = _score(x_trial)
            
            if trial_val < curr_val:
                x_curr = x_trial
                curr_val = trial_val
        
        return x_curr, curr_val
    
    best_x = None
    best_val = float('inf')
    
    for cand in candidates:
        if time.time() > deadline - 10:
            break
        x_opt, val_opt = local_search(cand, max_iters=30000)
        if val_opt < best_val:
            best_val = val_opt
            best_x = x_opt
    
    # If we have time, do more iterations on the best
    if best_x is not None and time.time() < deadline - 10:
        x_opt, val_opt = local_search(best_x, max_iters=50000)
        if val_opt < best_val:
            best_val = val_opt
            best_x = x_opt
    
    return best_x

# EVOLVE-BLOCK-START
# The construct function is defined above
# EVOLVE-BLOCK-END
