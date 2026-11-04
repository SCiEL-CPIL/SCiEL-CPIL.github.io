"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time


def _autoconv_max_abs(f):
    """
    Compute the upper bound on C3: the maximum absolute value of the
    autoconvolution (f * f) over all non-zero lags.
    
    For a step function with N intervals of height f[i] on [-1/4, 1/4],
    the autoconvolution at lag k (in interval units) is:
        (f * f)[k] = sum_i f[i] * f[i-k]  (for k > 0, with appropriate bounds)
    
    We use FFT to compute the full linear convolution efficiently.
    The self-lag (k=0) is sum(f^2), which we exclude.
    """
    n = len(f)
    # Pad to avoid circular convolution artifacts
    # Linear convolution of two length-n sequences has length 2n-1
    # We need a size that is at least 2n-1 and is a power of 2 for FFT efficiency
    conv_len = 2 * n - 1
    fft_size = 1
    while fft_size < conv_len:
        fft_size *= 2
    
    F = np.fft.rfft(f, n=fft_size)
    conv = np.fft.irfft(F * F, n=fft_size)[:conv_len]
    
    # conv[k] for k=0 is the sum of squares (self-lag)
    # We want max |conv[k]| for k != 0
    # The convolution is symmetric: conv[k] = conv[conv_len-1-k]
    # Non-zero lags correspond to indices 1..conv_len-1
    if conv_len <= 1:
        return 0.0
    
    non_zero = np.concatenate([conv[1:], conv[:conv_len-1][::-1][:0]])  # just conv[1:] is enough since we take abs
    # Actually conv has length conv_len = 2n-1. Indices 0 to 2n-2.
    # conv[0] = sum f[i]^2 (lag 0)
    # conv[1] = sum f[i]f[i-1] (lag 1)
    # ...
    # conv[n-1] = sum f[i]f[i-(n-1)] (lag n-1)
    # conv[n] = sum f[i]f[i-n] ... but f has only n elements, so this is effectively 0 for the linear conv
    # Wait, let me think again.
    
    # Linear convolution of f (length n) with itself:
    # (f*f)[k] = sum_{i=max(0,k-n+1)}^{min(n-1,k)} f[i] * f[k-i]
    # for k = 0, 1, ..., 2n-2
    
    # So conv[0] is the sum of squares.
    # We want max |conv[k]| for k = 1, ..., 2n-2
    # But by symmetry, conv[k] = conv[2n-2-k], so we only need to check k=1 to n-1 (or n-1 to 2n-2)
    
    max_val = np.max(np.abs(conv[1:conv_len]))
    return max_val


def _construct_candidate(seed):
    """Construct a signed step function based on a seed."""
    rng = np.random.RandomState(seed)
    # Start with random signs
    f = rng.choice([-1.0, 1.0], size=N)
    return f


def _local_search(f, max_iter=5000, deadline=None):
    """
    Local search: try flipping individual intervals to improve the score.
    """
    current_score = _autoconv_max_abs(f)
    best_f = f.copy()
    best_score = current_score
    
    for i in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        
        # Try flipping a random position
        pos = np.random.randint(0, N)
        f[pos] = -f[pos]
        new_score = _autoconv_max_abs(f)
        
        if new_score < best_score:
            best_score = new_score
            best_f = f.copy()
        else:
            f[pos] = -f[pos]  # revert
    
    return best_f, best_score


def construct():
    """
    Construct a signed step function with N intervals to minimize the
    autoconvolution upper bound.
    """
    start_time = time.time()
    deadline = start_time + 100.0  # Safety margin before 120s limit
    
    # Start with a random signed step function
    rng = np.random.RandomState(42)
    best_f = rng.choice([-1.0, 1.0], size=N)
    best_score = _autoconv_max_abs(best_f)
    
    # Perform multiple rounds of local search with different starting points
    num_restarts = 10
    for restart in range(num_restarts):
        if time.time() > deadline:
            break
        
        # New random starting point
        f = rng.choice([-1.0, 1.0], size=N)
        
        # Local search
        f, score = _local_search(f, max_iter=2000, deadline=deadline)
        
        if score < best_score:
            best_score = score
            best_f = f.copy()
    
    # Return the best signed step function found
    return best_f
# EVOLVE-BLOCK-END
