"""Optimized C3-prime: signed step functions minimizing max|autoconv|/S^2."""
import numpy as np
import time

N = 400
TIME_LIMIT = 110.0  # safety margin under 120s

def _autoconv_max_abs(h):
    """Compute max |autoconvolution| for a ±1 array h."""
    # Full autoconvolution via FFT
    n = len(h)
    size = 2 * n
    H = np.fft.rfft(h, size)
    ac = np.fft.irfft(H * np.conj(H), size)
    # ac[0..n-1] is the autoconvolution at lags 0..n-1
    # By symmetry, ac[k] = ac[size-k] for negative lags
    return np.max(np.abs(ac[:n]))

def _score(h):
    """Compute the objective: max|autoconv| / S^2."""
    S = np.sum(h)
    if S == 0:
        return np.inf
    max_ac = _autoconv_max_abs(h)
    return max_ac / (S * S)

def _flip(h, i):
    """Return h with element i flipped (±1 -> ∓1)."""
    h2 = h.copy()
    h2[i] = -h2[i]
    return h2

def _local_search(h, time_budget, rng):
    """Greedy local search: try flips, keep improvements."""
    start = time.time()
    best_h = h.copy()
    best_score = _score(best_h)
    n = len(best_h)
    
    # Precompute for speed: use the fact that flipping h[i] changes autoconv
    # in a structured way. But for simplicity, just recompute.
    
    # First pass: try all single flips
    improved = True
    while improved and (time.time() - start) < time_budget:
        improved = False
        for i in range(n):
            if (time.time() - start) > time_budget:
                break
            h2 = best_h.copy()
            h2[i] = -h2[i]
            s2 = _score(h2)
            if s2 < best_score - 1e-15:
                best_h = h2
                best_score = s2
                improved = True
    
    # Second phase: random restarts within budget
    while (time.time() - start) < time_budget:
        # Perturb current best
        h2 = best_h.copy()
        for _ in range(5):
            idx = rng.integers(0, n)
            h2[idx] = -h2[idx]
        s2 = _score(h2)
        if s2 < best_score:
            best_h = h2
            best_score = s2
            # Continue local search from here
            improved = True
            while improved and (time.time() - start) < time_budget:
                improved = False
                for i in range(n):
                    if (time.time() - start) > time_budget:
                        break
                    h3 = best_h.copy()
                    h3[i] = -h3[i]
                    s3 = _score(h3)
                    if s3 < best_score - 1e-15:
                        best_h = h3
                        best_score = s3
                        improved = True
    
    return best_h, best_score

def construct():
    """Construct a signed step function minimizing max|autoconv|/S^2."""
    start = time.time()
    rng = np.random.default_rng(42)
    
    # Initial candidates
    candidates = []
    
    # 1. All ones (baseline)
    candidates.append(np.ones(N, dtype=np.int8))
    
    # 2. Random ±1 with slight bias to ensure S != 0
    for _ in range(20):
        h = rng.choice([-1, 1], size=N)
        candidates.append(h.astype(np.int8))
    
    # 3. Structured: blocks of +1 and -1
    # Try various block patterns
    for block_size in [2, 4, 8, 16]:
        n_blocks = N // block_size
        h = np.ones(N, dtype=np.int8)
        for b in range(n_blocks):
            if b % 2 == 1:
                h[b*block_size:(b+1)*block_size] = -1
        candidates.append(h)
    
    # 4. Sinusoidal-like patterns
    x = np.linspace(0, 2*np.pi, N, endpoint=False)
    for freq in [1, 2, 3, 5, 7]:
        h = np.sign(np.sin(freq * x)).astype(np.int8)
        h[h == 0] = 1
        candidates.append(h)
    
    # 5. Mersenne-like or pseudorandom with fixed seed
    for seed in range(10):
        r = np.random.default_rng(seed)
        h = r.choice([-1, 1], size=N).astype(np.int8)
        candidates.append(h)
    
    # Evaluate all candidates
    best_h = None
    best_score = np.inf
    for h in candidates:
        s = _score(h)
        if s < best_score:
            best_score = s
            best_h = h.copy()
    
    # Check time
    elapsed = time.time() - start
    remaining = TIME_LIMIT - elapsed
    if remaining < 5:
        return best_h
    
    # Run local search on best candidate
    best_h, best_score = _local_search(best_h, remaining * 0.7, rng)
    
    # Check time
    elapsed = time.time() - start
    remaining = TIME_LIMIT - elapsed
    if remaining < 5:
        return best_h
    
    # Try local search on second-best candidates too
    # Re-evaluate candidates and sort
    scored = []
    for h in candidates:
        s = _score(h)
        scored.append((s, h))
    scored.sort(key=lambda x: x[0])
    
    for s, h in scored[1:5]:
        if (time.time() - start) > TIME_LIMIT - 10:
            break
        h2, s2 = _local_search(h, min(remaining * 0.15, 10), rng)
        if s2 < best_score:
            best_score = s2
            best_h = h2
    
    return best_h

# EVOLVE-BLOCK-END
