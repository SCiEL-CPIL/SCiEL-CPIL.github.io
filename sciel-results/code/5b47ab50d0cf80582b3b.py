"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
from scipy.signal import fftconvolve
import time

def _autoconv_excl_lag0(x):
    """Full autoconvolution of x, excluding the zero-lag term."""
    ac = fftconvolve(x, x[::-1], mode='full')
    # Zero lag is at index N-1 in full mode
    ac[N-1] = 0.0
    return ac

def _compute_objective(x):
    """Compute max|autoconv_excl_lag0| / (sum)^2."""
    s = x.sum()
    if abs(s) < 1e-12:
        return float('inf')
    ac = _autoconv_excl_lag0(x)
    return float(np.max(np.abs(ac)) / (s * s))

def _local_search(x_init, max_iters=10000, deadline=None, rng=None):
    """Local sign-flip search with efficient scoring."""
    if rng is None:
        rng = np.random.default_rng(42)
    
    x = x_init.copy().astype(np.float64)
    cur_score = _compute_objective(x)
    
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
        
        # Try flipping a random element
        idx = rng.integers(0, N)
        x[idx] *= -1
        new_score = _compute_objective(x)
        if new_score < cur_score:
            cur_score = new_score
        else:
            x[idx] *= -1  # revert
    
    return x, cur_score

def _greedy_init():
    """Build a sequence greedily to minimize peak autoconv."""
    x = np.zeros(N)
    # Start with a structured pattern
    for i in range(N):
        # Try both signs, pick the one that minimizes the resulting objective
        x[i] = 1.0
        score1 = _compute_objective(x)
        x[i] = -1.0
        score2 = _compute_objective(x)
        if score1 < score2:
            x[i] = 1.0
        else:
            x[i] = -1.0
    return x

def construct():
    start = time.time()
    deadline = start + 140.0  # safety margin before 150s kill
    
    rng = np.random.default_rng(123)
    
    best_x = np.ones(N, dtype=np.int64)
    best_score = _compute_objective(best_x)
    
    # Generate diverse initializations
    inits = []
    
    # 1. All ones
    inits.append(np.ones(N, dtype=np.float64))
    
    # 2. Random ±1 with multiple seeds
    for seed in range(1, 11):
        r = np.random.default_rng(seed)
        inits.append(r.choice([-1.0, 1.0], size=N))
    
    # 3. Structured: alternating blocks of various sizes
    for block_size in [5, 10, 15, 20, 25, 30, 40, 50]:
        x = np.zeros(N)
        for i in range(0, N, block_size * 2):
            x[i:i+block_size] = 1
            x[i+block_size:i+2*block_size] = -1
        inits.append(x)
    
    # 4. Cosine-based with various frequencies
    for freq in [1, 2, 3, 4, 5, 7, 10, 15]:
        t = np.linspace(0, freq * 2 * np.pi, N)
        x = np.sign(np.cos(t))
        inits.append(x)
    
    # 5. Sine-based
    for freq in [1, 2, 3, 5, 8]:
        t = np.linspace(0, freq * 2 * np.pi, N)
        x = np.sign(np.sin(t))
        inits.append(x)
    
    # 6. Walsh-like patterns
    for level in [1, 2, 3, 4]:
        x = np.zeros(N)
        period = 2 ** (level + 1)
        for i in range(N):
            bit = (i // (period // 2)) % 2
            x[i] = 1.0 if bit == 0 else -1.0
        inits.append(x)
    
    # 7. Low-pass filtered random
    from scipy.ndimage import uniform_filter1d
    for seed in [100, 200, 300, 400, 500]:
        r = np.random.default_rng(seed)
        x = r.standard_normal(N)
        x = uniform_filter1d(x, size=20)
        x = np.sign(x)
        inits.append(x)
    
    # Run local search on each initialization
    for init_idx, x0 in enumerate(inits):
        if time.time() > deadline - 20.0:
            break
        
        # Local search with generous budget
        x_ls, score_ls = _local_search(x0, max_iters=5000, 
                                        deadline=deadline - 10.0, rng=rng)
        
        if score_ls < best_score:
            best_score = score_ls
            best_x = np.round(x_ls).astype(np.int64)
    
    # Final local search on best with more iterations
    if time.time() < deadline - 5.0:
        x_final, score_final = _local_search(best_x.astype(np.float64), 
                                              max_iters=10000, 
                                              deadline=deadline - 2.0, rng=rng)
        if score_final < best_score:
            best_score = score_final
            best_x = np.round(x_final).astype(np.int64)
    
    return best_x
# EVOLVE-BLOCK-END
