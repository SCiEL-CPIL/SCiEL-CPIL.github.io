"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400
import time

# EVOLVE-BLOCK-START
import numpy as np
import itertools

def _autoconv_abs_max(x):
    """
    Compute the maximum absolute value of the autoconvolution of x.
    x is a 1D array of length N.
    Autoconvolution: c[k] = sum_i x[i] * x[i-k] for k in [-(N-1), N-1].
    We want max |c[k]|.
    """
    # Use FFT for speed
    n = len(x)
    size = 1
    while size < 2 * n - 1:
        size *= 2
    X = np.fft.rfft(x, n=size)
    C = np.fft.irfft(X * X, n=size)
    # The full convolution length is 2*n - 1
    # irfft returns size/2 + 1 points, but we need the first 2*n-1 points
    # Actually, np.fft.irfft returns size/2 + 1 points if size is even.
    # The linear convolution is in the first 2*n-1 samples.
    c = C[:2*n-1]
    return np.max(np.abs(c))

def _integral_sq(x):
    """
    Squared integral of the step function.
    Assuming unit width steps, integral = sum(x).
    """
    return np.sum(x) ** 2

def _score(x):
    """
    Score is the ratio: max_abs_autoconv / integral_sq.
    Lower is better.
    """
    num = _autoconv_abs_max(x)
    den = _integral_sq(x)
    if den == 0:
        return float('inf')
    return num / den

def _random_sign_vector(n, seed=None):
    if seed is not None:
        rng = np.random.default_rng(seed)
    else:
        rng = np.random.default_rng()
    return rng.choice([-1.0, 1.0], size=n)

def _optimize_block(x, block_start, block_end, max_iter=100, deadline=None):
    """
    Local search on a block of the vector.
    Tries flipping individual bits in the block and small subsets.
    """
    n = len(x)
    best_x = x.copy()
    best_score = _score(best_x)
    if deadline is not None and time.time() > deadline:
        return best_x, best_score
    
    block_size = block_end - block_start
    if block_size <= 0:
        return best_x, best_score
    
    # Try flipping individual elements in the block
    for i in range(block_start, block_end):
        if deadline is not None and time.time() > deadline:
            break
        new_x = best_x.copy()
        new_x[i] *= -1.0
        s = _score(new_x)
        if s < best_score:
            best_score = s
            best_x = new_x
    
    # Try flipping pairs in the block (if block is small)
    if block_size <= 10:
        for i in range(block_start, block_end):
            if deadline is not None and time.time() > deadline:
                break
            for j in range(i+1, block_end):
                if deadline is not None and time.time() > deadline:
                    break
                new_x = best_x.copy()
                new_x[i] *= -1.0
                new_x[j] *= -1.0
                s = _score(new_x)
                if s < best_score:
                    best_score = s
                    best_x = new_x
    
    return best_x, best_score

def _construct_random_with_local_search(seed, deadline):
    rng = np.random.default_rng(seed)
    best_x = None
    best_score = float('inf')
    
    # Start with a random vector
    x = _random_sign_vector(N, seed)
    s = _score(x)
    if s < best_score:
        best_score = s
        best_x = x.copy()
    
    # Do a few rounds of local search on random blocks
    num_rounds = 5
    for r in range(num_rounds):
        if deadline is not None and time.time() > deadline:
            break
        # Pick a random block to optimize
        block_len = rng.integers(5, 20)
        block_start = rng.integers(0, max(1, N - block_len))
        block_end = block_start + block_len
        x, s = _optimize_block(x, block_start, block_end, max_iter=50, deadline=deadline)
        if s < best_score:
            best_score = s
            best_x = x.copy()
    
    return best_x, best_score

def _construct_structural(deadline):
    """
    Try some structured patterns.
    """
    best_x = None
    best_score = float('inf')
    
    # Pattern 1: Alternating
    x = np.ones(N)
    x[1::2] = -1.0
    s = _score(x)
    if s < best_score:
        best_score = s
        best_x = x.copy()
    
    # Pattern 2: Blocks of +1 and -1
    block_sizes = [1, 2, 4, 8, 16, 32]
    for bs in block_sizes:
        if deadline is not None and time.time() > deadline:
            break
        x = np.zeros(N)
        for i in range(0, N, bs):
            sign = 1.0 if (i // bs) % 2 == 0 else -1.0
            x[i:i+bs] = sign
        s = _score(x)
        if s < best_score:
            best_score = s
            best_x = x.copy()
    
    # Pattern 3: Rademacher-like (random but deterministic seed)
    for seed in [42, 123, 999]:
        if deadline is not None and time.time() > deadline:
            break
        x = _random_sign_vector(N, seed)
        s = _score(x)
        if s < best_score:
            best_score = s
            best_x = x.copy()
    
    return best_x, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 10 second safety margin
    
    best_x = None
    best_score = float('inf')
    
    # 1. Try structural patterns
    x, s = _construct_structural(deadline)
    if s < best_score:
        best_score = s
        best_x = x.copy()
    
    # 2. Try random restarts with local search
    num_restarts = 20
    for i in range(num_restarts):
        if time.time() > deadline:
            break
        seed = 1000 + i
        x, s = _construct_random_with_local_search(seed, deadline)
        if s < best_score:
            best_score = s
            best_x = x.copy()
    
    if best_x is None:
        best_x = np.ones(N)
    
    return best_x
# EVOLVE-BLOCK-END
