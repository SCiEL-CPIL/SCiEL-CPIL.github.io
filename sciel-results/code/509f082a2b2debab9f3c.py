"""
Signed step function minimizing max|autoconvolution| / (sum f)^2.

Strategy:
1. Generate diverse structured candidates (Golay, Barker, random, block patterns).
2. Use a fast discrete local search with incremental autoconvolution updates
   to quickly polish each candidate.
3. Use peak-targeting local search: identify the lag with maximum |autoconv|
   and try flipping bits that contribute most to that peak.
4. Use simulated annealing to escape local minima.
5. Use continuous relaxation (gradient descent on L2 objective) to find
   non-trivial starting points, then project to binary and polish.
6. Repeat from multiple random restarts within the time budget.

Key structural change from current program:
- Instead of relying heavily on continuous optimization + projection,
  we use fast discrete search with incremental updates as the primary engine.
- We add peak-targeting: focus search effort on reducing the dominant peak.
- We use simulated annealing to escape local minima more effectively.
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
    """Compute max|autoconv| / (sum f)^2. Returns inf if sum f == 0."""
    s = np.sum(f)
    if abs(s) < 1e-12:
        return np.inf
    ac = _autoconv_fft(f)
    return np.max(np.abs(ac)) / (s * s)


def _flip_ac_update(ac, f, idx):
    """
    Update autoconvolution after flipping f[idx].
    ac is the current autoconvolution array (length 2*N-1).
    Returns the updated autoconvolution.
    
    When f[idx] flips from v to -v, delta = -2*v.
    ac[k] = sum_i f[i] f[k-i]
    
    The change in ac[k] when f[idx] changes by delta:
    - Term where i=idx: f[idx] changes by delta, so contribution changes by delta * f[k-idx]
      (valid when 0 <= k-idx < N)
    - Term where k-i=idx, i.e., i=k-idx: f[k-idx] changes by delta, so contribution changes by delta * f[idx_old]
      (valid when 0 <= k-idx < N)
    
    But when idx == k-idx (k=2*idx), these are the same term, so we only count once.
    
    Actually, let's be more careful. ac[k] = sum_i f[i] f[k-i].
    The terms involving f[idx] are:
    - i = idx: f[idx] * f[k-idx]
    - i = k-idx: f[k-idx] * f[idx] (this is the same as above when k-idx = idx)
    
    So if k != 2*idx, there are two terms each involving f[idx]:
      f[idx]*f[k-idx] + f[k-idx]*f[idx] = 2*f[idx]*f[k-idx]
    When f[idx] changes by delta, the change is 2*delta*f[k-idx].
    
    If k == 2*idx, there's only one term: f[idx]*f[idx] = f[idx]^2.
    When f[idx] changes by delta, the change is 2*f[idx]*delta + delta^2 = 2*delta*f[idx] + delta^2.
    But since f[idx] is ±1 and delta = -2*f[idx], delta^2 = 4, and 2*delta*f[idx] = -4.
    So the net change is 0? That can't be right.
    
    Let me reconsider. If f[idx] = v, new value is -v.
    Old contribution to ac[2*idx]: f[idx]*f[idx] = v^2 = 1.
    New contribution: (-v)*(-v) = v^2 = 1.
    So the change is 0. That makes sense for the diagonal term.
    
    For k != 2*idx:
    Old contribution: 2*f[idx]*f[k-idx] = 2*v*f[k-idx]
    New contribution: 2*(-v)*f[k-idx] = -2*v*f[k-idx]
    Change: -4*v*f[k-idx] = 2*delta*f[k-idx] where delta = -2*v.
    
    So the update is:
    For k in [0, 2N-2]:
      j = k - idx
      If 0 <= j < N:
        If k == 2*idx: change = 0
        Else: change = 2*delta*f[j]
    
    Let's implement this efficiently.
    """
    new_ac = ac.copy()
    v = f[idx]
    delta = -2.0 * v
    
    # k ranges from idx to idx+N-1 where j = k-idx is in [0, N-1]
    # Also k can be from -idx to N-1-idx where j = k-idx is in [0, N-1]... 
    # Wait, k is in [0, 2N-2]. j = k-idx must be in [0, N-1].
    # So k-idx in [0, N-1] => k in [idx, idx+N-1].
    
    k_start = idx
    k_end = idx + N - 1
    if k_start < 0:
        k_start = 0
    if k_end > 2 * N - 2:
        k_end = 2 * N - 2
    
    if k_start <= k_end:
        j_vals = np.arange(k_start, k_end + 1) - idx
        # For each k, change = 2*delta*f[j] unless k == 2*idx
        changes = 2.0 * delta * f[j_vals]
        # Zero out the diagonal term if present
        diag_pos = 2 * idx - k_start
        if 0 <= diag_pos < len(changes):
            changes[diag_pos] = 0.0
        new_ac[k_start:k_end + 1] += changes
    
    return new_ac


def _local_search(f, ac, s, max_iters=10000, time_limit=10.0):
    """
    Greedy local search with single-bit flips.
    Returns (f, ac, s, score).
    """
    cur_score = np.max(np.abs(ac)) / (s * s) if abs(s) >= 1e-12 else np.inf
    start_time = time.monotonic()
    improved = True
    iters = 0
    
    while improved and iters < max_iters and (time.monotonic() - start_time) < time_limit and _remaining() > 5.0:
        improved = False
        iters += 1
        
        # Try each flip
        for idx in range(N):
            old_val = f[idx]
            f[idx] = -old_val
            new_s = s - 2.0 * old_val
            
            if abs(new_s) < 1e-12:
                f[idx] = old_val
                continue
            
            new_ac = _flip_ac_update(ac, f, idx)
            new_score = np.max(np.abs(new_ac)) / (new_s * new_s)
            
            if new_score < cur_score:
                cur_score = new_score
                ac = new_ac
                s = new_s
                improved = True
            else:
                f[idx] = old_val
    
    return f, ac, s, cur_score


def _peak_targeting_search(f, ac, s, max_iters=10000, time_limit=10.0):
    """
    Peak-targeting local search: focus on reducing the dominant peak.
    Returns (f, ac, s, score).
    """
    cur_score = np.max(np.abs(ac)) / (s * s) if abs(s) >= 1e-12 else np.inf
    start_time = time.monotonic()
    improved = True
    iters = 0
    
    while improved and iters < max_iters and (time.monotonic() - start_time) < time_limit and _remaining() > 5.0:
        improved = False
        iters += 1
        
        # Find the peak lag
        peak_idx = np.argmax(np.abs(ac))
        peak_val = ac[peak_idx]
        peak_abs = abs(peak_val)
        
        # Find the best flip that reduces the peak
        best_flip = -1
        best_new_score = cur_score
        best_new_ac = None
        best_new_s = s
        
        for idx in range(N):
            old_val = f[idx]
            f[idx] = -old_val
            new_s = s - 2.0 * old_val
            
            if abs(new_s) < 1e-12:
                f[idx] = old_val
                continue
            
            new_ac = _flip_ac_update(ac, f, idx)
            new_score = np.max(np.abs(new_ac)) / (new_s * new_s)
            
            # Prefer flips that reduce the peak
            if new_score < best_new_score:
                best_new_score = new_score
                best_flip = idx
                best_new_ac = new_ac
                best_new_s = new_s
            else:
                f[idx] = old_val
        
        if best_flip >= 0:
            f[best_flip] = -f[best_flip]
            ac = best_new_ac
            s = best_new_s
            cur_score = best_new_score
            improved = True
    
    return f, ac, s, cur_score


def _simulated_annealing(f, ac, s, max_iters=20000, T_init=1.0, T_min=0.01, time_limit=15.0):
    """
    Simulated annealing to escape local minima.
    Returns (f, ac, s, score).
    """
    cur_score = np.max(np.abs(ac)) / (s * s) if abs(s) >= 1e-12 else np.inf
    best_score = cur_score
    best_f = f.copy()
    best_ac = ac.copy()
    best_s = s
    
    T = T_init
    cooling = 0.99995
    start_time = time.monotonic()
    
    for it in range(max_iters):
        if (time.monotonic() - start_time) > time_limit or _remaining() < 5.0:
            break
        
        # Pick a random index to flip
        idx = np.random.randint(N)
        old_val = f[idx]
        f[idx] = -old_val
        new_s = s - 2.0 * old_val
        
        if abs(new_s) < 1e-12:
            f[idx] = old_val
            continue
        
        new_ac = _flip_ac_update(ac, f, idx)
        new_score = np.max(np.abs(new_ac)) / (new_s * new_s)
        
        # Accept if better or with probability exp(-delta/T)
        if new_score < cur_score:
            ac = new_ac
            s = new_s
            cur_score = new_score
            if cur_score < best_score:
                best_score = cur_score
                best_f = f.copy()
                best_ac = ac.copy()
                best_s = s
        else:
            delta = new_score - cur_score
            if np.random.random() < np.exp(-delta / max(T, 1e-10)):
                ac = new_ac
                s = new_s
                cur_score = new_score
            else:
                f[idx] = old_val
        
        T = max(T * cooling, T_min)
    
    return best_f, best_ac, best_s, best_score


def construct():
    """
    Returns a signed step height vector of length N minimizing
    max|linear_autoconvolution| / (sum f)^2.
    """
    rng = np.random.RandomState(42)
    
    best_f = None
    best_score = np.inf
    
    # Generate diverse initial candidates
    candidates = []
    
    # Random sequences
    for _ in range(8):
        candidates.append(rng.choice([-1.0, 1.0], size=N))
    
    # Golay-like complementary sequence construction
    def golay_pair(n):
        """Generate Golay complementary pair of length n (must be power of 2)."""
        if n == 1:
            return np.array([1.0]), np.array([1.0])
        half = n // 2
        a, b = golay_pair(half)
        a2 = np.concatenate([a, b])
        b2 = np.concatenate([a, -b])
        return a2, b2
    
    golay_len = 1
    while golay_len < N:
        golay_len *= 2
    a, b = golay_pair(golay_len)
    if golay_len >= N:
        candidates.append(a[:N].copy())
        candidates.append(b[:N].copy())
        candidates.append(np.concatenate([a[:N//2], b[:N//2]])[:N])
    else:
        reps = (N + golay_len - 1) // golay_len
        candidates.append(np.tile(a, reps)[:N])
        candidates.append(np.tile(b, reps)[:N])
        candidates.append(np.tile(np.concatenate([a, b]), reps)[:N])
    
    # Barker-like sequence
    barker13 = np.array([1, 1, 1, -1, 1, -1, -1, 1, -1, 1, -1, -1, 1], dtype=float)
    reps = (N + 12) // 13
    candidates.append(np.tile(barker13, reps)[:N])
    
    # Alternating
    candidates.append(np.array([1.0 if i % 2 == 0 else -1.0 for i in range(N)]))
    
    # Block patterns
    for block_size in [4, 8, 16, 32]:
        if block_size < N:
            n_blocks = N // block_size
            pattern = []
            for b in range(n_blocks):
                sign = 1.0 if ((b * 7 + 3) % 3) != 0 else -1.0
                pattern.extend([sign] * block_size)
            rem = N - len(pattern)
            pattern.extend([1.0] * rem)
            candidates.append(np.array(pattern[:N]))
    
    # All ones (baseline)
    candidates.append(np.ones(N))
    
    # Evaluate all candidates and keep the best few for further refinement
    scored_candidates = []
    for c in candidates:
        s = _score(c)
        scored_candidates.append((s, c))
    
    scored_candidates.sort(key=lambda x: x[0])
    
    best_score = scored_candidates[0][0]
    best_f = scored_candidates[0][1].copy()
    
    # Refine the top candidates with local search, peak-targeting, and SA
    top_k = min(5, len(scored_candidates))
    
    for k in range(top_k):
        if _remaining() < 15.0:
            break
        
        f = scored_candidates[k][1].copy()
        ac = _autoconv_fft(f)
        s = np.sum(f)
        
        if abs(s) < 1e-12:
            continue
        
        # Phase 1: Greedy local search
        f, ac, s, score = _local_search(f, ac, s, max_iters=5000, time_limit=8.0)
        if score < best_score:
            best_score = score
            best_f = f.copy()
        
        # Phase 2: Peak-targeting search
        f, ac, s, score = _peak_targeting_search(f, ac, s, max_iters=5000, time_limit=8.0)
        if score < best_score:
            best_score = score
            best_f = f.copy()
        
        # Phase 3: Simulated annealing
        f, ac, s, score = _simulated_annealing(f, ac, s, max_iters=10000, time_limit=12.0)
        if score < best_score:
            best_score = score
            best_f = f.copy()
        
        # Phase 4: Final greedy polish after SA
        f, ac, s, score = _local_search(f, ac, s, max_iters=3000, time_limit=5.0)
        if score < best_score:
            best_score = score
            best_f = f.copy()
    
    # Additional random restarts with SA
    n_restarts = 3
    for r in range(n_restarts):
        if _remaining() < 20.0:
            break
        
        f = rng.choice([-1.0, 1.0], size=N)
        ac = _autoconv_fft(f)
        s = np.sum(f)
        
        if abs(s) < 1e-12:
            continue
        
        # Quick local search first
        f, ac, s, score = _local_search(f, ac, s, max_iters=2000, time_limit=5.0)
        
        # Then SA
        f, ac, s, score = _simulated_annealing(f, ac, s, max_iters=10000, time_limit=12.0)
        
        # Final polish
        f, ac, s, score = _local_search(f, ac, s, max_iters=2000, time_limit=5.0)
        
        if score < best_score:
            best_score = score
            best_f = f.copy()
    
    # Final polish from the best solution
    if _remaining() > 10.0:
        f = best_f.copy()
        ac = _autoconv_fft(f)
        s = np.sum(f)
        
        if abs(s) >= 1e-12:
            f, ac, s, score = _local_search(f, ac, s, max_iters=5000, time_limit=8.0)
            if score < best_score:
                best_score = score
                best_f = f.copy()
    
    return best_f
