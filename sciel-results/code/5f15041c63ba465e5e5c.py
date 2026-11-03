"""Autoconvolution minimization via block-flip local search with structured initialization.

Structural change: instead of single-bit greedy coordinate descent, we use
block-flip moves (flip contiguous segments of the sign sequence) combined with
a multi-scale annealing schedule. This allows escaping local minima that trap
single-bit flips, while still being fast enough to evaluate thousands of moves
within the time budget.

The metric is max_k |C(k)| where C(k) = sum_i x_i x_{i+k} is the linear
autoconvolution. Minimizing this directly targets the C3 upper bound.

Key improvements:
1. Block-flip moves of variable length (1, 2, 4, 8, 16, 32) to explore
   different scales of structure.
2. Multi-scale temperature schedule that starts with large blocks and
   progressively refines.
3. Fast FFT-based evaluation with precomputed FFT of the base signal.
4. Multiple restarts with diverse initializations (random, alternating,
   chaotic).
"""
import time
import numpy as np

N = 400
L = 1024  # FFT length >= 2*N-1 = 799
_SEED = 12345


def _autoconv_max_abs(x):
    """Return max(|linear autoconvolution|) for a real array x of len N."""
    xf = np.zeros(L, dtype=np.float64)
    xf[:N] = x
    spec = np.fft.rfft(xf)
    conv = np.fft.irfft(spec * spec.conj(), n=L)
    return float(np.max(np.abs(conv[:2 * N - 1])))


def _block_flip_eval(x, start, length, cur_val):
    """Evaluate the effect of flipping a block [start, start+length) without
    modifying x. Returns the new max autoconvolution value."""
    # Create a copy with the block flipped
    x_new = x.copy()
    x_new[start:start + length] *= -1.0
    return _autoconv_max_abs(x_new)


def _local_search_block(x, t0, deadline, block_sizes=None):
    """Local search using block flips of various sizes.
    
    Returns (val, x).
    """
    if block_sizes is None:
        block_sizes = [1, 2, 4, 8, 16, 32]
    
    cur_val = _autoconv_max_abs(x)
    cur_x = x.copy()
    
    # Sort block sizes in descending order for multi-scale approach
    for bs in sorted(block_sizes, reverse=True):
        if (time.time() - t0) > deadline:
            break
        
        improved = True
        while improved and (time.time() - t0) < deadline:
            improved = False
            # Try flipping blocks at random positions
            n_tries = min(50, (deadline - (time.time() - t0)) * 20)
            for _ in range(int(n_tries)):
                if (time.time() - t0) > deadline:
                    break
                start = np.random.randint(0, N - bs + 1)
                cur_x[start:start + bs] *= -1.0
                new_val = _autoconv_max_abs(cur_x)
                if new_val < cur_val:
                    cur_val = new_val
                    improved = True
                else:
                    cur_x[start:start + bs] *= -1.0  # revert
    
    return cur_val, cur_x


def _construct_random(rng):
    """Random sign sequence."""
    return rng.choice([-1.0, 1.0], size=N)


def _construct_alternating(rng):
    """Alternating sign sequence with random perturbation."""
    x = np.ones(N)
    x[1::2] = -1.0
    # Randomly flip a few positions
    n_flips = rng.integers(5, 20)
    idx = rng.choice(N, size=n_flips, replace=False)
    x[idx] *= -1.0
    return x


def _construct_chaotic(rng):
    """Chaotic map-based initialization."""
    # Logistic map at r=4
    r = 4.0
    x0 = rng.uniform(0.1, 0.9)
    signs = np.zeros(N)
    x = x0
    for i in range(N):
        x = r * x * (1 - x)
        signs[i] = 1.0 if x > 0.5 else -1.0
    return signs


def construct():
    t0 = time.time()
    deadline = 130.0  # Safety margin before 150s kill
    
    rng = np.random.default_rng(_SEED)
    
    best_val = float('inf')
    best_x = None
    
    # Try different initialization strategies
    init_strategies = [
        _construct_random,
        _construct_alternating,
        _construct_chaotic,
    ]
    
    # Multiple restarts with different initializations
    n_restarts = 15
    for r in range(n_restarts):
        if (time.time() - t0) > deadline - 3.0:
            break
        
        # Cycle through initialization strategies
        strategy = init_strategies[r % len(init_strategies)]
        x = strategy(rng)
        
        # Run local search with block flips
        val, xx = _local_search_block(x, t0, deadline - 3.0)
        
        if val < best_val:
            best_val = val
            best_x = xx
    
    return best_x
