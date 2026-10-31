"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

N = 400

# EVOLVE-BLOCK-START

def _max_abs_autoconv(x):
    """Compute the maximum absolute value of the autoconvolution of x."""
    # Use FFT for efficiency
    n = len(x)
    size = 1
    while size < 2 * n - 1:
        size <<= 1
    X = np.fft.rfft(x, size)
    ac = np.fft.irfft(X * np.conj(X), size)[:2 * n - 1]
    return np.max(np.abs(ac))


def _score(x):
    """Score: max |autoconv| / (integral)^2. Lower is better."""
    integral = np.sum(x)
    if integral == 0:
        return float('inf')
    return _max_abs_autoconv(x) / (integral * integral)


def _barker_like(n, seed=0):
    """Generate a Barker-like binary sequence using known Barker sequences tiled/padded."""
    # Known Barker sequences (lengths 2,3,4,5,7,11,13)
    barker_seqs = {
        2: np.array([1, 1]),
        3: np.array([1, 1, -1]),
        4: np.array([1, 1, -1, 1]),
        5: np.array([1, 1, 1, -1, 1]),
        7: np.array([1, 1, 1, -1, 1, -1, -1]),
        11: np.array([1, 1, 1, 1, 1, -1, -1, 1, -1, 1, -1]),
        13: np.array([1, 1, 1, 1, 1, -1, -1, 1, -1, 1, -1, -1, 1]),
    }
    # Pick the longest Barker sequence that divides or fits
    best = None
    for length in sorted(barker_seqs.keys(), reverse=True):
        if n % length == 0:
            best = barker_seqs[length]
            break
    if best is None:
        # Use the longest one and tile/truncate
        best = barker_seqs[13]
    
    # Tile to fill n
    reps = int(np.ceil(n / len(best)))
    seq = np.tile(best, reps)[:n]
    return seq


def _golay_complementary(n, seed=0):
    """Generate a sequence based on Golay complementary pairs (simplified)."""
    # For simplicity, use a pseudo-random binary sequence with some structure
    # Golay complementary pairs have the property that sum of auto-correlations = 0 for non-zero shifts
    # We'll use a simple construction: start with random, then optimize
    rng = np.random.RandomState(seed)
    seq = rng.choice([1, -1], size=n)
    return seq


def _local_optimize(x, max_time=100.0, pair_flips=True):
    """Local optimization via sign flips (single and pair) to reduce max |autoconv|."""
    x = x.copy().astype(float)
    n = len(x)
    best_score = _score(x)
    start_time = time.time()
    
    # Precompute autoconvolution for fast updates
    # For efficiency, we'll recompute periodically and use incremental updates for small changes
    # Actually for N=400, full FFT is fast enough
    
    no_improve = 0
    max_no_improve = 500
    
    while time.time() - start_time < max_time:
        improved = False
        no_improve += 1
        
        if no_improve > max_no_improve:
            break
        
        # Try single flips
        if not pair_flips and no_improve > 200:
            break
            
        # Random subset of positions to try (for speed)
        if no_improve < 100:
            # Try all positions initially
            candidates = np.arange(n)
        else:
            # Try a random subset
            rng = np.random.RandomState(int(time.time() * 1000) % (2**31))
            candidates = rng.choice(n, size=min(n, 50), replace=False)
        
        best_flip = None
        best_new_score = best_score
        
        for i in candidates:
            old_val = x[i]
            x[i] = -old_val
            s = _score(x)
            if s < best_new_score:
                best_new_score = s
                best_flip = i
            x[i] = old_val
        
        if best_flip is not None:
            x[best_flip] = -x[best_flip]
            best_score = best_new_score
            improved = True
            no_improve = 0
            max_no_improve = 500
        else:
            # Try pair flips if single didn't work
            if pair_flips and no_improve > 50:
                rng = np.random.RandomState(int(time.time() * 1000) % (2**31))
                pairs = rng.choice(n, size=2, replace=False)
                i, j = int(pairs[0]), int(pairs[1])
                if i == j:
                    continue
                old_i, old_j = x[i], x[j]
                x[i] = -old_i
                x[j] = -old_j
                s = _score(x)
                if s < best_score:
                    best_score = s
                    improved = True
                    no_improve = 0
                else:
                    x[i] = old_i
                    x[j] = old_j
    
    return x, best_score


def construct():
    """Construct a signed step sequence minimizing max |autoconv| / (sum)^2."""
    start_time = time.time()
    deadline = 110.0  # Safety margin before 120s limit
    
    best_x = None
    best_score = float('inf')
    
    # Generate diverse initial patterns
    initial_patterns = []
    
    # 1. Barker-like sequence
    initial_patterns.append(_barker_like(N, seed=0))
    initial_patterns.append(_barker_like(N, seed=1))
    
    # 2. Random binary sequences
    for seed in range(10):
        rng = np.random.RandomState(seed)
        initial_patterns.append(rng.choice([1, -1], size=N).astype(float))
    
    # 3. Block patterns (alternating blocks)
    for block_size in [2, 3, 5, 7, 11]:
        pattern = np.tile([1, -1], size=(N // block_size + 1, block_size))
        pattern = pattern.flatten()[:N].astype(float)
        initial_patterns.append(pattern)
    
    # 4. Sparse patterns (mostly 1s with some -1s)
    for density in [0.1, 0.2, 0.3, 0.5]:
        rng = np.random.RandomState(42)
        x = np.ones(N)
        neg_idx = rng.choice(N, size=int(density * N), replace=False)
        x[neg_idx] = -1
        initial_patterns.append(x)
    
    # 5. Golay-inspired
    initial_patterns.append(_golay_complementary(N, seed=0))
    initial_patterns.append(_golay_complementary(N, seed=1))
    
    # Evaluate and optimize each pattern
    for pat in initial_patterns:
        if time.time() - start_time > deadline - 30:
            break
        
        x, s = _local_optimize(pat, max_time=min(20.0, deadline - (time.time() - start_time)))
        
        if s < best_score:
            best_score = s
            best_x = x.copy()
    
    if best_x is None:
        # Fallback
        best_x = np.ones(N)
    
    return best_x.astype(int)

# EVOLVE-BLOCK-END
