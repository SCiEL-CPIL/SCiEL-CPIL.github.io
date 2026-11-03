"""Peak-targeting local search with vectorized flip deltas and pair moves."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.signal import fftconvolve

def _compute_score(x):
    """Compute the C3' metric: max |autoconvolution| / (sum(x))^2"""
    s = np.sum(x)
    if abs(s) < 1e-12:
        return float('inf')
    c = fftconvolve(x, x, mode='full')
    max_abs = np.max(np.abs(c))
    return max_abs / (s * s)

def _compute_conv(x):
    """Compute full autoconvolution using FFT."""
    return fftconvolve(x, x, mode='full')

def _soft_max_abs(c, tau=1.0):
    """Smooth approximation of max|c| using log-sum-exp."""
    ac = np.abs(c)
    m = ac.max()
    if m < 1e-15:
        return m
    return m + tau * np.log(np.sum(np.exp((ac - m) / tau)))

def _single_flip_deltas(c, x):
    """
    For each position i, compute the delta in autoconvolution if x[i] is flipped.
    Returns array of shape (N, len(c)) where deltas[i] is the change in c.
    
    c has length 2N-1, indexed by lag k-(N-1) for k in [-(N-1), N-1].
    c[k_idx] = sum_j x[j]*x[j+k] where k = k_idx - (N-1).
    
    Flipping x[i] (value v -> -v, change dv = -2v):
    For lag k: delta_c[k] = dv * (x[i-k] + x[i+k]) - dv * (k==0 ? x[i] : 0)
    More precisely:
      delta_c[k] = -2v * (x[i+k] + x[i-k]) for k != 0
      delta_c[0] = 0 (since x[i]^2 is unchanged)
    """
    n_c = len(c)
    center = N - 1
    deltas = np.zeros((N, n_c))
    
    for i in range(N):
        v = x[i]
        # For positive lags k=1..N-1: index center+k
        #   delta = -2v*(x[i+k] + x[i-k]) where out-of-range = 0
        for k in range(1, N):
            idx_pos = center + k
            idx_neg = center - k
            val = 0.0
            if i + k < N:
                val += x[i + k]
            if i - k >= 0:
                val += x[i - k]
            if val != 0.0:
                d = -2.0 * v * val
                deltas[i, idx_pos] = d
                deltas[i, idx_neg] = d
        # Lag 0: no change
        # deltas[i, center] = 0
    
    return deltas

def _single_flip_deltas_fast(c, x):
    """
    Vectorized computation of single-flip deltas.
    For flip at position i, the delta at lag k is:
      -2*x[i] * (x[i+k] + x[i-k]) for k != 0, 0 for k=0.
    
    We can express this as: delta_i = -2*x[i] * (shifted_x_forward + shifted_x_backward)
    where shifted_x_forward[k] = x[i+k] and shifted_x_backward[k] = x[i-k].
    
    The autoconvolution c[k] = sum_j x[j]*x[j+k].
    Flipping x[i]: new_c[k] = c[k] - 2*x[i]*x[i+k] - 2*x[i]*x[i-k] + (correction for k=0)
    For k=0: the term x[i]*x[i] appears once, so change is -2*x[i]^2 = -2, but x[i]^2=1 so...
    Actually c[0] = sum x[j]^2 = N always, so delta[0] = 0.
    """
    n_c = len(c)
    center = N - 1
    # Build the "profile" for each flip position
    # For position i, delta at lag k (k from -(N-1) to N-1):
    #   delta[k] = -2*x[i] * (x[i+k] + x[i-k])  for k != 0
    #   delta[0] = 0
    
    # We can use the fact that this is a convolution-like structure.
    # Let's just do it with numpy operations per position but vectorized over lags.
    
    deltas = np.zeros((N, n_c))
    
    for i in range(N):
        v = x[i]
        # x[i+k] for k=1..N-1: this is x[i+1], x[i+2], ..., x[N-1] (length N-1-i)
        # x[i-k] for k=1..N-1: this is x[i-1], x[i-2], ..., x[0] (length i)
        
        # Positive lags: k=1..N-1, index center+k
        #   contribution from x[i+k]: valid for k <= N-1-i
        #   contribution from x[i-k]: valid for k <= i
        pos_lags = np.arange(1, N)
        # x[i+k]
        valid_fwd = pos_lags <= (N - 1 - i)
        fwd_vals = np.zeros(N - 1)
        fwd_vals[:N-1-i] = x[i+1:]
        # x[i-k]
        valid_bwd = pos_lags <= i
        bwd_vals = np.zeros(N - 1)
        bwd_vals[:i] = x[i-1::-1]
        
        val = fwd_vals + bwd_vals
        d = -2.0 * v * val
        
        deltas[i, center+1:] = d
        deltas[i, :center][::-1] = d  # negative lags, reversed
        # deltas[i, center] = 0
    
    return deltas

def _try_flip(x, i, c, current_score):
    """Try flipping position i, return new score or inf."""
    v = x[i]
    x[i] = -v
    s = np.sum(x)
    if abs(s) < 1e-12:
        x[i] = v
        return float('inf')
    new_c = c - 2.0 * v * _flip_profile(x, i)
    # Wait, this isn't quite right. Let me just recompute.
    # Actually the delta is: new_c[k] = c[k] - 2*v*(x_old[i+k] + x_old[i-k]) for k!=0
    # But x_old[i] = v, and after flip x_new[i] = -v.
    # The correct delta uses the OLD values of x[i+k] and x[i-k].
    # So we need to save the old profile before flipping.
    x[i] = v  # revert
    return float('inf')  # placeholder, use vectorized approach

def _flip_profile(x, i):
    """Return the profile p[k] = x[i+k] + x[i-k] for k=-(N-1)..N-1 (excluding k=0 which is 0)."""
    n_c = 2 * N - 1
    center = N - 1
    p = np.zeros(n_c)
    for k in range(1, N):
        val = 0.0
        if i + k < N:
            val += x[i + k]
        if i - k >= 0:
            val += x[i - k]
        p[center + k] = val
        p[center - k] = val
    return p

def _local_search(x, deadline, max_rounds=500):
    """
    Local search with single flips. Uses vectorized delta computation.
    """
    best_seq = x.copy()
    best_score = _compute_score(best_seq)
    current = x.copy()
    current_score = best_score
    c = _compute_conv(current)
    
    for round_idx in range(max_rounds):
        if time.time() > deadline:
            break
        
        best_improvement = 0
        best_flip_idx = -1
        best_new_c = None
        
        s = np.sum(current)
        s2 = s * s
        
        for i in range(N):
            if time.time() > deadline:
                break
            
            v = current[i]
            # Compute delta for flipping position i
            # delta[k] = -2*v*(x[i+k] + x[i-k]) for k != 0, delta[0] = 0
            new_c = c.copy()
            
            # Efficient: only modify non-zero entries
            for k in range(1, N):
                val = 0.0
                if i + k < N:
                    val += current[i + k]
                if i - k >= 0:
                    val += current[i - k]
                if val != 0.0:
                    d = -2.0 * v * val
                    new_c[N-1+k] += d
                    new_c[N-1-k] += d
            
            new_max_abs = np.max(np.abs(new_c))
            new_score = new_max_abs / s2
            
            if new_score < current_score - 1e-12:
                if new_score < best_score - best_improvement:
                    best_improvement = current_score - new_score
                    best_flip_idx = i
                    best_new_c = new_c
        
        if best_flip_idx >= 0:
            current[best_flip_idx] *= -1
            current_score = best_score - best_improvement
            c = best_new_c
            best_score = current_score
            best_seq = current.copy()
        else:
            break
    
    return best_seq, best_score

def _pair_flip_search(x, deadline, max_rounds=100):
    """
    Try pairs of flips to escape local minima.
    """
    best_seq = x.copy()
    best_score = _compute_score(best_seq)
    current = x.copy()
    current_score = best_score
    c = _compute_conv(current)
    s = np.sum(current)
    
    for round_idx in range(max_rounds):
        if time.time() > deadline:
            break
        
        best_improvement = 0
        best_pair = (-1, -1)
        
        s2 = s * s
        
        # Try a subset of pairs for efficiency
        # Randomly sample pairs
        rng = np.random.RandomState(round_idx)
        n_pairs = min(2000, N * (N-1) // 2)
        
        for _ in range(n_pairs):
            if time.time() > deadline:
                break
            i, j = rng.choice(N, 2, replace=False)
            
            vi = current[i]
            vj = current[j]
            
            new_c = c.copy()
            
            # Flip i: delta_i[k] = -2*vi*(x[i+k]+x[i-k]) for k!=0
            for k in range(1, N):
                val_i = 0.0
                if i + k < N:
                    val_i += current[i + k]
                if i - k >= 0:
                    val_i += current[i - k]
                if val_i != 0.0:
                    d = -2.0 * vi * val_i
                    new_c[N-1+k] += d
                    new_c[N-1-k] += d
            
            # Flip j: delta_j[k] = -2*vj*(x[j+k]+x[j-k]) for k!=0
            # But note: if |i-j| = k, then x[i+k] or x[i-k] might be x[j] which is also flipped.
            # For simplicity, use old values (approximation).
            for k in range(1, N):
                val_j = 0.0
                if j + k < N:
                    val_j += current[j + k]
                if j - k >= 0:
                    val_j += current[j - k]
                if val_j != 0.0:
                    d = -2.0 * vj * val_j
                    new_c[N-1+k] += d
                    new_c[N-1-k] += d
            
            # Correction for interaction: when k = |i-j|, the term involving both flipped positions
            # is handled approximately. This is a heuristic.
            
            new_max_abs = np.max(np.abs(new_c))
            new_score = new_max_abs / s2
            
            if new_score < current_score - 1e-12:
                if new_score < best_score - best_improvement:
                    best_improvement = current_score - new_score
                    best_pair = (i, j)
        
        if best_pair != (-1, -1):
            i, j = best_pair
            current[i] *= -1
            current[j] *= -1
            current_score = best_score - best_improvement
            c = _compute_conv(current)
            s = np.sum(current)
            best_score = current_score
            best_seq = current.copy()
        else:
            break
    
    return best_seq, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    # Start with all ones
    x = np.ones(N)
    best_seq = x.copy()
    best_score = _compute_score(x)
    
    # Diverse initializations
    rng = np.random.RandomState(123)
    
    # Random with various biases
    for frac in [0.0, 0.02, 0.05, 0.08, 0.1, 0.12, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5]:
        if time.time() > deadline:
            break
        n_neg = int(frac * N)
        if n_neg == 0:
            x_try = np.ones(N)
        else:
            idx = rng.choice(N, n_neg, replace=False)
            x_try = np.ones(N)
            x_try[idx] = -1
        s = _compute_score(x_try)
        if s < best_score:
            best_score = s
            best_seq = x_try.copy()
    
    # Block patterns
    for block_len in [2, 3, 4, 5, 6, 7, 8, 10, 12, 16, 20, 25, 32, 40, 50]:
        if time.time() > deadline:
            break
        for ratio in [0.5, 0.3, 0.7]:
            if time.time() > deadline:
                break
            n_neg_in_block = int(block_len * ratio)
            pattern = np.ones(block_len)
            pattern[:n_neg_in_block] = -1
            x_try = np.tile(pattern, N // block_len)
            if len(x_try) < N:
                x_try = np.concatenate([x_try, np.ones(N - len(x_try))])
            x_try = x_try[:N]
            s = _compute_score(x_try)
            if s < best_score:
                best_score = s
                best_seq = x_try.copy()
    
    # Run local search
    if time.time() < deadline:
        best_seq, best_score = _local_search(best_seq, deadline - 20)
    
    # Pair flip search
    if time.time() < deadline:
        best_seq, best_score = _pair_flip_search(best_seq, deadline - 10)
    
    # Final single-flip polish
    if time.time() < deadline:
        x = best_seq.copy()
        c = _compute_conv(x)
        s = np.sum(x)
        s2 = s * s
        for i in range(N):
            if time.time() > deadline:
                break
            v = x[i]
            new_c = c.copy()
            for k in range(1, N):
                val = 0.0
                if i + k < N:
                    val += x[i + k]
                if i - k >= 0:
                    val += x[i - k]
                if val != 0.0:
                    d = -2.0 * v * val
                    new_c[N-1+k] += d
                    new_c[N-1-k] += d
            new_max_abs = np.max(np.abs(new_c))
            new_score = new_max_abs / s2
            if new_score < best_score - 1e-12:
                x[i] *= -1
                best_score = new_score
                best_seq = x.copy()
                c = new_c
                s = np.sum(x)
                s2 = s * s
            # else: no change
    
    return best_seq
# EVOLVE-BLOCK-END
