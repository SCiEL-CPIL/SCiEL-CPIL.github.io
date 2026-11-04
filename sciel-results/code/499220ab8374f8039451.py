"""
Signed step function minimizing max|autoconvolution| / (sum f)^2.

Structural change: Use simulated annealing with incremental autoconvolution
updates and adaptive temperature scheduling. This allows thorough exploration
of the ±1 hypercube with escape from local minima, combined with smart
initialization from known low-PSL sequences.
"""
import numpy as np
import time

N = 400
_DEADLINE = 145.0
_T0 = time.monotonic()


def _remaining():
    return _DEADLINE - (time.monotonic() - _T0)


def _autoconv_fft(f):
    """Compute linear autoconvolution via FFT."""
    n = len(f)
    size = 1
    while size < 2 * n - 1:
        size *= 2
    F = np.fft.rfft(f, size)
    ac = np.fft.irfft(F * F, size)
    return ac[:2 * n - 1]


def _score(f):
    """Compute max|autoconv| / (sum f)^2."""
    s = np.sum(f)
    if abs(s) < 1e-12:
        return np.inf
    ac = _autoconv_fft(f)
    return np.max(np.abs(ac)) / (s * s)


def _golay_pair(n):
    """Generate Golay complementary pair of length n (power of 2)."""
    if n == 1:
        return np.array([1.0]), np.array([1.0])
    half = n // 2
    a, b = _golay_pair(half)
    a2 = np.concatenate([a, b])
    b2 = np.concatenate([a, -b])
    return a2, b2


def _initial_candidates():
    """Generate diverse initial candidates."""
    rng = np.random.RandomState(42)
    candidates = []
    
    # Random ±1 sequences
    for _ in range(10):
        candidates.append(rng.choice([-1.0, 1.0], size=N))
    
    # Golay sequences
    golay_len = 1
    while golay_len < N:
        golay_len *= 2
    a, b = _golay_pair(golay_len)
    if golay_len >= N:
        candidates.append(a[:N].copy())
        candidates.append(b[:N].copy())
    else:
        reps = (N + golay_len - 1) // golay_len
        candidates.append(np.tile(a, reps)[:N])
        candidates.append(np.tile(b, reps)[:N])
        candidates.append(np.tile(np.concatenate([a, b]), reps)[:N])
    
    # Barker-like
    barker13 = np.array([1, 1, 1, -1, 1, -1, -1, 1, -1, 1, -1, -1, 1], dtype=float)
    reps = (N + 12) // 13
    candidates.append(np.tile(barker13, reps)[:N])
    
    # Alternating
    candidates.append(np.array([1.0 if i % 2 == 0 else -1.0 for i in range(N)]))
    
    # All ones
    candidates.append(np.ones(N))
    
    return candidates


def _simulated_annealing(f, max_time=30.0, initial_temp=1.0, cooling_rate=0.999):
    """
    Simulated annealing on ±1 sequences with incremental autoconv updates.
    
    Key optimization: when flipping bit i from old to new, the autoconvolution
    changes only at specific lags. We track the current autoconv and update
    incrementally.
    """
    f = f.copy()
    ac = _autoconv_fft(f)
    s = np.sum(f)
    if abs(s) < 1e-12:
        return f, np.inf
    cur_score = np.max(np.abs(ac)) / (s * s)
    best_f = f.copy()
    best_score = cur_score
    
    rng = np.random.RandomState(42)
    temp = initial_temp
    n_flips = 0
    start_time = time.monotonic()
    
    while (time.monotonic() - start_time) < max_time and _remaining() > 5.0:
        n_flips += 1
        
        # Choose a random index to flip
        idx = rng.randint(N)
        old_val = f[idx]
        new_val = -old_val
        
        # Update autoconvolution incrementally
        # Flipping f[idx] changes ac[k] for k where f[idx] * f[k-idx] or f[idx] * f[idx-k] appears
        # ac[k] = sum_i f[i] * f[k-i]
        # When f[idx] changes by delta = new_val - old_val = 2*new_val (since old=-new for ±1)
        # delta = 2 * new_val
        delta = new_val - old_val  # = ±2
        
        # The change in ac[k] is: delta * f[k-idx] + delta * f[idx-k] for valid k
        # But we need to be careful about bounds
        # For k in [0, 2N-2]:
        #   ac[k] += delta * f[k-idx] if 0 <= k-idx < N
        #   ac[k] += delta * f[idx-k] if 0 <= idx-k < N
        
        # Pre-compute the affected range
        # k-idx in [0, N-1] => k in [idx, idx+N-1]
        # idx-k in [0, N-1] => k in [idx-N+1, idx]
        
        k_low = max(0, idx - N + 1)
        k_high = min(2 * N - 2, idx + N - 1)
        
        # For k in [k_low, k_high]:
        #   contribution from first term: delta * f[k-idx] if k >= idx
        #   contribution from second term: delta * f[idx-k] if k <= idx
        
        # Vectorized update
        ks = np.arange(k_low, k_high + 1)
        
        # First term: f[k - idx] for k >= idx
        mask1 = ks >= idx
        if np.any(mask1):
            ks1 = ks[mask1]
            ac[ks1] += delta * f[ks1 - idx]
        
        # Second term: f[idx - k] for k <= idx
        mask2 = ks <= idx
        if np.any(mask2):
            ks2 = ks[mask2]
            ac[ks2] += delta * f[idx - ks2]
        
        # Update sum
        s_new = s + delta
        
        # Compute new score
        if abs(s_new) < 1e-12:
            # Revert
            ac[k_low:k_high+1] -= delta * (f[ks >= idx][mask1] if np.any(mask1) else 0)
            # This is getting complicated, just revert fully
            f[idx] = old_val
            s = s_new - delta
            ac = _autoconv_fft(f)
            continue
        
        new_score = np.max(np.abs(ac)) / (s_new * s_new)
        
        # Acceptance criterion
        if new_score < cur_score or rng.rand() < np.exp(-(new_score - cur_score) / temp):
            f[idx] = new_val
            s = s_new
            cur_score = new_score
            
            if new_score < best_score:
                best_score = new_score
                best_f = f.copy()
        else:
            # Revert
            f[idx] = old_val
            # Revert autoconv
            ks = np.arange(k_low, k_high + 1)
            mask1 = ks >= idx
            if np.any(mask1):
                ks1 = ks[mask1]
                ac[ks1] -= delta * f[ks1 - idx]
            mask2 = ks <= idx
            if np.any(mask2):
                ks2 = ks[mask2]
                ac[ks2] -= delta * f[idx - ks2]
        
        # Cool down
        if n_flips % 100 == 0:
            temp *= cooling_rate
            if temp < 0.01:
                temp = 0.01
    
    return best_f, best_score


def construct():
    """
    Returns a signed step height vector of length N minimizing
    max|linear_autoconvolution| / (sum f)^2.
    
    Uses simulated annealing with incremental autoconvolution updates.
    """
    # Generate initial candidates
    candidates = _initial_candidates()
    
    # Evaluate all candidates
    best_f = None
    best_score = np.inf
    for c in candidates:
        s = _score(c)
        if s < best_score:
            best_score = s
            best_f = c.copy()
    
    # Run simulated annealing from the best candidate
    if _remaining() > 10.0:
        f_annealed, score_annealed = _simulated_annealing(
            best_f, max_time=_remaining() - 10.0, initial_temp=1.0, cooling_rate=0.9995
        )
        if score_annealed < best_score:
            best_score = score_annealed
            best_f = f_annealed
    
    # Try SA from a few other good candidates
    candidates_sorted = sorted(candidates, key=lambda c: _score(c))
    for c in candidates_sorted[1:4]:
        if _remaining() < 10.0:
            break
        s = _score(c)
        if s > best_score * 1.5:
            continue
        f_annealed, score_annealed = _simulated_annealing(
            c, max_time=min(15.0, _remaining() - 10.0), initial_temp=0.5, cooling_rate=0.9995
        )
        if score_annealed < best_score:
            best_score = score_annealed
            best_f = f_annealed
    
    return best_f
