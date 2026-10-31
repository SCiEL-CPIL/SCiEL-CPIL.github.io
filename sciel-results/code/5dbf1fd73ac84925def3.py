"""Signed autocorrelation maximum absolute-value variant - C3-prime task."""
import numpy as np
import time

N = 400

_START_TIME = time.time()
_DEADLINE = 140.0  # safety margin before 150s wall-clock kill


def _autoconv(x):
    """Compute full autoconvolution."""
    return np.correlate(x, x, mode='full')


def _compute_score(x):
    """Compute max|autoconv| / (sum x)^2."""
    s = np.sum(x)
    if abs(s) < 1e-12:
        return np.inf
    autoconv = np.correlate(x, x, mode='full')
    max_abs = np.max(np.abs(autoconv))
    return max_abs / (s * s)


def _compute_score_fast(x, prev_autoconv=None):
    """Compute score, optionally reusing previous autoconv for small changes."""
    s = np.sum(x)
    if abs(s) < 1e-12:
        return np.inf
    autoconv = np.correlate(x, x, mode='full')
    max_abs = np.max(np.abs(autoconv))
    return max_abs / (s * s)


def _sign_flip_search(x, max_iter=50000, seed=42):
    """Local search via sign flips on binary ±1 sequence."""
    rng = np.random.RandomState(seed)
    x = x.copy()
    best_score = _compute_score(x)
    best_x = x.copy()
    
    # Precompute current autoconv
    ac = np.correlate(x, x, mode='full')
    s = np.sum(x)
    
    # For a sign flip at index i: x[i] -> -x[i]
    # New sum: s' = s - 2*x[i]
    # New autoconv: ac' = ac - 2*x[i]*(x shifted) - 2*x[i]*(x shifted) + 4*x[i]*x[i]... 
    # Actually for binary ±1, flipping x[i] changes autoconv by:
    # ac'[k] = ac[k] - 2*x[i]*(x[i+k] + x[i-k]) + 4*x[i]*x[i] (only at k=0 for the last term)
    # More precisely: ac'[k] = sum_j x'[j]*x'[j+k] where x'[i]=-x[i]
    # = ac[k] - x[i]*x[i+k] - x[i]*x[i-k] + (-x[i])*(-x[i+k]) + ... 
    # Let me just recompute for simplicity but use a smarter approach
    
    # Actually let's use the incremental update:
    # When we flip x[i] to -x[i], the change in autoconv at lag k is:
    # delta_ac[k] = -2*x[i]*(x[i+k] if in range) - 2*x[i]*(x[i-k] if in range) 
    # Wait, let me think more carefully.
    
    # ac[k] = sum_j x[j]*x[j+k] for k >= 0 (and symmetric for k < 0)
    # If we flip x[i]: x'[i] = -x[i], all other x'[j] = x[j]
    # ac'[k] = sum_j x'[j]*x'[j+k]
    # = ac[k] - x[i]*x[i+k] - x[i]*x[i-k] + x[i]*x[i+k] + x[i]*x[i-k] ... no
    # Terms involving x[i]: 
    #   j=i: x'[i]*x'[i+k] = -x[i]*x[i+k] (was x[i]*x[i+k]), change: -2*x[i]*x[i+k]
    #   j=i-k: x'[i-k]*x'[i] = x[i-k]*(-x[i]) (was x[i-k]*x[i]), change: -2*x[i]*x[i-k]
    # So delta_ac[k] = -2*x[i]*x[i+k] - 2*x[i]*x[i-k] (with appropriate bounds)
    
    # For efficiency, let's just do the full recompute but limit iterations
    # N=400, autoconv is 799 elements, so it's fast
    
    elapsed = time.time() - _START_TIME
    for iteration in range(max_iter):
        if time.time() - _START_TIME > _DEADLINE:
            break
        
        # Try a random sign flip
        i = rng.randint(N)
        old_val = x[i]
        x[i] = -old_val
        sc = _compute_score(x)
        if sc < best_score:
            best_score = sc
            best_x = x.copy()
            ac = np.correlate(x, x, mode='full')
            s = np.sum(x)
        else:
            x[i] = old_val
    
    return best_x, best_score


def _multi_start_search():
    """Multiple diverse initializations with sign-flip local search."""
    best_x = None
    best_score = np.inf
    
    # Different initializations
    inits = []
    
    # 1. All ones
    inits.append(np.ones(N))
    
    # 2. Random ±1
    rng = np.random.RandomState(123)
    inits.append(rng.choice([-1, 1], size=N))
    
    # 3. Chirp pattern: x[i] = sign(cos(2*pi*i^2/(2*N)))
    i_arr = np.arange(N)
    chirp = np.sign(np.cos(np.pi * i_arr**2 / N))
    chirp[chirp == 0] = 1
    inits.append(chirp)
    
    # 4. Block pattern: alternating blocks of +1 and -1
    block_size = 7
    n_blocks = N // block_size + 1
    block = np.tile(np.array([1]*block_size + [-1]*block_size), n_blocks)[:N]
    inits.append(block)
    
    # 5. Sine wave
    sine = np.sign(np.sin(2 * np.pi * i_arr / N * 3))
    sine[sine == 0] = 1
    inits.append(sine)
    
    # 6. Random with bias
    inits.append(rng.choice([-1, 1], size=N, p=[0.6, 0.4]))
    
    # 7. Another chirp
    chirp2 = np.sign(np.cos(np.pi * i_arr**2 / (2*N) + np.pi/4))
    chirp2[chirp2 == 0] = 1
    inits.append(chirp2)
    
    # 8. Random
    rng2 = np.random.RandomState(456)
    inits.append(rng2.choice([-1, 1], size=N))
    
    # 9. Block with different size
    block_size2 = 13
    n_blocks2 = N // block_size2 + 1
    block2 = np.tile(np.array([1]*block_size2 + [-1]*block_size2), n_blocks2)[:N]
    inits.append(block2)
    
    # 10. Low-frequency sine
    sine_low = np.sign(np.sin(2 * np.pi * i_arr / N))
    sine_low[sine_low == 0] = 1
    inits.append(sine_low)
    
    for idx, x0 in enumerate(inits):
        if time.time() - _START_TIME > _DEADLINE - 10:
            break
        
        x_opt, sc = _sign_flip_search(x0, max_iter=20000, seed=1000+idx)
        if sc < best_score:
            best_score = sc
            best_x = x_opt
    
    return best_x, best_score


def construct():
    """Construct signed step function minimizing max|autoconv|/integral^2.
    
    Uses multi-start local search with sign flips on binary ±1 sequences.
    """
    best_x, best_score = _multi_start_search()
    
    # Final polish with more iterations on the best found
    if time.time() - _START_TIME < _DEADLINE - 5:
        x_polished, sc_polished = _sign_flip_search(best_x, max_iter=30000, seed=9999)
        if sc_polished < best_score:
            best_x = x_polished
            best_score = sc_polished
    
    return best_x
