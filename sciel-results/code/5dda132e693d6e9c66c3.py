"""
Signed step function minimizing max|autoconvolution| / (sum f)^2.

The metric is: score = max_k |autoconv[k]| / (sum f)^2
where autoconv[k] = sum_i f[i] * f[k - i] (linear autoconvolution).

Structural approach: Use incremental autoconvolution updates and a
peak-targeting local search. When flipping bit i, the autoconvolution
changes only at specific lags. We track the peak lag and try to reduce
it by flipping bits that contribute most to that peak.
"""
import numpy as np
import time
import random

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
    return ac[:2*n-1]


def _score(f):
    """Compute max|autoconv| / (sum f)^2. Returns inf if sum f == 0."""
    s = np.sum(f)
    if abs(s) < 1e-12:
        return np.inf
    ac = _autoconv_fft(f)
    return np.max(np.abs(ac)) / (s * s)


def _update_autoconv(ac, f, idx, old_val):
    """
    Update autoconvolution after flipping f[idx] from old_val to -old_val.
    ac is the current autoconvolution array (length 2*N-1).
    Returns the updated autoconvolution.
    
    When f[idx] changes by delta = -2*old_val, each term f[idx]*f[j] in
    autoconv[k] changes where k = idx + j or k = j - idx (depending on convention).
    
    For linear autoconvolution: ac[k] = sum_i f[i] * f[k-i]
    Flipping f[idx] by delta affects ac[k] for all k where the pair (idx, k-idx)
    or (k-idx, idx) is valid, i.e., k-idx is in [0, N-1] and idx is in [0, N-1].
    So ac[k] changes by delta * f[k-idx] for k in [idx, idx+N-1] (if k-idx in [0,N-1])
    plus delta * f[k-idx] ... actually let's think more carefully.
    
    ac[k] = sum_{i=0}^{N-1} f[i] * f[k-i] where f[j]=0 for j<0 or j>=N.
    
    When we flip f[idx] by delta, the change in ac[k] is:
    delta * f[k-idx]  (from the term where i=idx)
    + delta * f[idx]  (from the term where k-i=idx, i.e., i=k-idx)
    
    Wait, that's double counting if idx == k-idx. Let me be more careful.
    
    ac[k] = sum_i f[i] f[k-i]
    
    The terms involving f[idx] are:
    - i = idx: f[idx] * f[k-idx]
    - k-i = idx, i.e., i = k-idx: f[k-idx] * f[idx]
    
    These are the same term when idx = k-idx, i.e., k = 2*idx.
    
    So the change in ac[k] when f[idx] changes by delta is:
    - If k != 2*idx: delta * f[k-idx] + delta * f[k-idx] = 2 * delta * f[k-idx]
      but only if both idx and k-idx are in [0, N-1]
    - If k == 2*idx: delta * f[idx] (only one term)
      but we need f[idx] to be the NEW value... this is getting complicated.
    
    Actually, let's just recompute. It's simpler and N=400 is small enough.
    """
    # For N=400, recomputing via FFT is fast enough.
    # But let's do a direct update for single bit flip.
    new_ac = ac.copy()
    delta = -2.0 * old_val  # change in f[idx]
    
    # For each k, the change in ac[k] is:
    # delta * f[k-idx] if 0 <= k-idx < N (term i=idx)
    # + delta * f[k-idx] if 0 <= k-idx < N (term i=k-idx)
    # But when k-idx == idx, these are the same term, so only one delta*f[idx]
    
    # More carefully:
    # ac[k] = sum_i f[i] f[k-i]
    # Change = delta * f[k-idx] (from i=idx term) + delta * f[k-idx] (from i=k-idx term)
    #        = 2*delta*f[k-idx] if k-idx != idx
    #        = delta*f[idx_new] if k-idx == idx (but f[idx] is changing...)
    
    # Let's just handle it by considering which k values are affected.
    # k must satisfy: 0 <= idx < N and 0 <= k-idx < N
    # So k ranges from idx to idx+N-1, and k-idx ranges from 0 to N-1.
    
    # For k in [idx, idx+N-1]:
    #   j = k - idx, j in [0, N-1]
    #   Change in ac[k]:
    #     From i=idx term: f[idx] changes by delta, so contribution changes by delta*f[j]
    #     From i=j term (k-i=j means i=k-j=k-(k-idx)=idx): wait, that's the same.
    
    # I think I'm overcomplicating. Let me just recompute via FFT.
    # For N=400, this is fast.
    return new_ac  # Placeholder, we'll use FFT recompute instead


def construct():
    """
    Returns a signed step height vector of length N minimizing
    max|linear_autoconvolution| / (sum f)^2.
    
    Uses peak-targeting local search: identify the lag with maximum
    |autoconv|, then try flipping bits that contribute most to that peak.
    """
    rng = np.random.RandomState(42)
    
    # Generate diverse initial candidates
    candidates = []
    
    # Random sequences
    for _ in range(5):
        candidates.append(rng.choice([-1.0, 1.0], size=N))
    
    # Golay-like complementary sequence construction
    # Golay sequences have optimal autocorrelation properties
    def golay_pair(n):
        """Generate Golay complementary pair of length n (must be power of 2)."""
        if n == 1:
            return np.array([1.0]), np.array([1.0])
        half = n // 2
        a, b = golay_pair(half)
        a2 = np.concatenate([a, b])
        b2 = np.concatenate([a, -b])
        return a2, b2
    
    # Use Golay sequences as building blocks
    golay_len = 1
    while golay_len < N:
        golay_len *= 2
    a, b = golay_pair(golay_len)
    # Pad or truncate to N
    if golay_len >= N:
        candidates.append(a[:N].copy())
        candidates.append(b[:N].copy())
        candidates.append(np.concatenate([a[:N//2], b[:N//2]])[:N])
    else:
        # Tile
        reps = (N + golay_len - 1) // golay_len
        candidates.append(np.tile(a, reps)[:N])
        candidates.append(np.tile(b, reps)[:N])
        candidates.append(np.tile(np.concatenate([a, b]), reps)[:N])
    
    # Barker-like sequences (known good autocorrelation)
    # Barker sequence of length 13: + + + - + - - + - + - - +
    barker13 = np.array([1, 1, 1, -1, 1, -1, -1, 1, -1, 1, -1, -1, 1], dtype=float)
    reps = (N + 12) // 13
    candidates.append(np.tile(barker13, reps)[:N])
    
    # Alternating patterns
    candidates.append(np.array([1.0 if i % 2 == 0 else -1.0 for i in range(N)]))
    
    # Block patterns with varying block sizes
    for block_size in [4, 8, 16, 32]:
        if block_size < N:
            n_blocks = N // block_size
            pattern = []
            for b in range(n_blocks):
                # Use a pseudo-random sign based on block index
                sign = 1.0 if ((b * 7 + 3) % 3) != 0 else -1.0
                pattern.extend([sign] * block_size)
            # Fill remainder
            rem = N - len(pattern)
            pattern.extend([1.0] * rem)
            candidates.append(np.array(pattern[:N]))
    
    # Evaluate all candidates
    best_f = None
    best_score = np.inf
    for c in candidates:
        s = _score(c)
        if s < best_score:
            best_score = s
            best_f = c.copy()
    
    f = best_f.copy()
    cur_score = best_score
    ac = _autoconv_fft(f)
    s = np.sum(f)
    
    # Peak-targeting local search
    # Identify the lag with maximum |autoconv| and try to reduce it
    T = 1.0
    T_min = 0.01
    cooling = 0.99995
    
    iterations = 0
    while _remaining() > 5.0 and T > T_min:
        # Find the peak lag
        peak_idx = np.argmax(np.abs(ac))
        peak_val = ac[peak_idx]
        
        # Try flipping bits that contribute to this peak
        # ac[peak_idx] = sum_i f[i] * f[peak_idx - i]
        # Flipping f[j] changes ac[peak_idx] by:
        # -2*f[j] * f[peak_idx - j] if j != peak_idx - j
        # -2*f[j]^2 = -2 if j == peak_idx - j (i.e., peak_idx even and j = peak_idx/2)
        
        # Find the bit that, when flipped, most reduces |ac[peak_idx]|
        best_flip = -1
        best_new_peak = abs(peak_val)
        
        # Sample a subset of positions for speed
        if _remaining() > 30:
            sample_size = min(N, 200)
            sample = rng.choice(N, size=sample_size, replace=False)
        else:
            sample = np.arange(N)
        
        for j in sample:
            k_j = peak_idx - j
            if 0 <= k_j < N:
                # Contribution of f[j]*f[k_j] to ac[peak_idx]
                if j == k_j:
                    # Single term: f[j]^2 = 1, flipping changes it to -1... 
                    # Actually f[j] is ±1, so f[j]^2 = 1 always.
                    # Flipping f[j] to -f[j] doesn't change f[j]^2.
                    # So this term is invariant!
                    continue
                else:
                    # Two terms: f[j]*f[k_j] and f[k_j]*f[j], both change sign
                    # Current contribution: 2 * f[j] * f[k_j]
                    # After flip: 2 * (-f[j]) * f[k_j] = -2 * f[j] * f[k_j]
                    # Change: -4 * f[j] * f[k_j]
                    current_contrib = 2.0 * f[j] * f[k_j]
                    new_contrib = -current_contrib
                    new_peak = peak_val - 2.0 * current_contrib
                    new_abs = abs(new_peak)
                    if new_abs < best_new_peak:
                        best_new_peak = new_abs
                        best_flip = j
        
        if best_flip >= 0:
            # Apply the flip
            old_val = f[best_flip]
            f[best_flip] = -f[best_flip]
            
            # Update autoconvolution (recompute via FFT for correctness)
            ac = _autoconv_fft(f)
            s = np.sum(f)
            if abs(s) < 1e-12:
                # Revert
                f[best_flip] = old_val
                ac = _autoconv_fft(f)
                s = np.sum(f)
                cur_score = np.inf
            else:
                cur_score = np.max(np.abs(ac)) / (s * s)
                if cur_score < best_score:
                    best_score = cur_score
                    best_f = f.copy()
        else:
            # No single flip reduces the peak; try a random flip
            idx = rng.randint(N)
            old_val = f[idx]
            f[idx] = -f[idx]
            ac = _autoconv_fft(f)
            s = np.sum(f)
            if abs(s) < 1e-12:
                f[idx] = old_val
                ac = _autoconv_fft(f)
                s = np.sum(f)
                cur_score = np.inf
            else:
                new_score = np.max(np.abs(ac)) / (s * s)
                if new_score < cur_score or rng.random() < np.exp(-(new_score - cur_score) / T):
                    cur_score = new_score
                    if cur_score < best_score:
                        best_score = cur_score
                        best_f = f.copy()
                else:
                    f[idx] = old_val
                    ac = _autoconv_fft(f)
                    s = np.sum(f)
        
        T *= cooling
        iterations += 1
        
        # Occasionally try multi-flip moves
        if iterations % 1000 == 0 and _remaining() > 10:
            for _ in range(3):
                idx1 = rng.randint(N)
                idx2 = rng.randint(N)
                if idx1 == idx2:
                    continue
                f[idx1] = -f[idx1]
                f[idx2] = -f[idx2]
                ac_new = _autoconv_fft(f)
                s_new = np.sum(f)
                if abs(s_new) < 1e-12:
                    f[idx1] = -f[idx1]
                    f[idx2] = -f[idx2]
                    continue
                new_score = np.max(np.abs(ac_new)) / (s_new * s_new)
                if new_score < cur_score:
                    cur_score = new_score
                    ac = ac_new
                    s = s_new
                    if cur_score < best_score:
                        best_score = cur_score
                        best_f = f.copy()
                else:
                    f[idx1] = -f[idx1]
                    f[idx2] = -f[idx2]
    
    # Final greedy pass: try flipping each bit and keep if it improves
    f = best_f.copy()
    ac = _autoconv_fft(f)
    s = np.sum(f)
    if abs(s) < 1e-12:
        return best_f
    cur_score = np.max(np.abs(ac)) / (s * s)
    
    improved = True
    while improved and _remaining() > 2.0:
        improved = False
        for idx in range(N):
            f[idx] = -f[idx]
            ac_new = _autoconv_fft(f)
            s_new = np.sum(f)
            if abs(s_new) < 1e-12:
                f[idx] = -f[idx]
                continue
            new_score = np.max(np.abs(ac_new)) / (s_new * s_new)
            if new_score < cur_score:
                cur_score = new_score
                ac = ac_new
                s = s_new
                improved = True
            else:
                f[idx] = -f[idx]
        if cur_score < best_score:
            best_score = cur_score
            best_f = f.copy()
    
    return best_f
