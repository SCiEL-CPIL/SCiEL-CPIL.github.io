import numpy as np
import time

N = 400

def _score(h):
    """Compute the max absolute autocorrelation for lags 1 to N-1."""
    n = len(h)
    fft_size = 1
    while fft_size < 2 * n:
        fft_size *= 2
    H = np.fft.rfft(h, n=fft_size)
    acf = np.fft.irfft(np.abs(H) ** 2, n=fft_size)
    return float(np.max(np.abs(acf[1:n])))

def _compute_acf(h):
    """Compute full autocorrelation via FFT."""
    n = len(h)
    fft_size = 1
    while fft_size < 2 * n:
        fft_size *= 2
    H = np.fft.rfft(h, n=fft_size)
    acf = np.fft.irfft(np.abs(H) ** 2, n=fft_size)
    return acf

def _psl_value(acf, n):
    """Get PSL value from acf array (max abs at non-zero lags)."""
    if n <= 1:
        return 0.0
    return float(np.max(np.abs(acf[1:n])))

def _gradient_step_vectorized(h, acf):
    """
    Vectorized Gordon's PSL minimization step.
    For each position i, compute the change in PSL if we flip h[i].
    Returns (best_i, best_new_psl, current_psl).
    """
    n = len(h)
    current_psl = _psl_value(acf, n)
    
    # Pre-compute the current ACF lags
    acf_lags = acf[1:n]
    
    # For each position i, the delta in ACF lags when flipping h[i] is:
    # delta[k-1] = -2 * h[i] * (h[i-k] + h[i+k]) for valid indices
    # We need to compute this for all i efficiently.
    
    # Build neighbor sums: neighbor_sum[i, k] = h[i-k] + h[i+k] for k=1..n-1
    # Use padding to handle boundaries
    h_padded = np.zeros(2*n - 1)
    h_padded[n-1:n-1+n] = h
    
    # For each i in 0..n-1, and each k in 1..n-1:
    # h[i-k] corresponds to h_padded[n-1 + i - k]
    # h[i+k] corresponds to h_padded[n-1 + i + k]
    
    # Create index arrays
    i_idx = np.arange(n)[:, None]  # shape (n, 1)
    k_idx = np.arange(1, n)[None, :]  # shape (1, n-1)
    
    # Left neighbors: h[i-k] for valid i-k >= 0
    left_idx = n - 1 + i_idx - k_idx  # shape (n, n-1)
    left_valid = (left_idx >= 0) & (left_idx < 2*n - 1)
    left_vals = np.where(left_valid, h_padded[np.clip(left_idx, 0, 2*n-2)], 0.0)
    
    # Right neighbors: h[i+k] for valid i+k < n
    right_idx = n - 1 + i_idx + k_idx  # shape (n, n-1)
    right_valid = (right_idx >= 0) & (right_idx < 2*n - 1)
    right_vals = np.where(right_valid, h_padded[np.clip(right_idx, 0, 2*n-2)], 0.0)
    
    # Neighbor sum
    neighbor_sum = left_vals + right_vals  # shape (n, n-1)
    
    # Delta for each i: delta[i, k] = -2 * h[i] * neighbor_sum[i, k]
    delta = -2.0 * h[:, None] * neighbor_sum  # shape (n, n-1)
    
    # New ACF lags for each i
    new_acf_lags = acf_lags[None, :] + delta  # shape (n, n-1)
    
    # PSL for each i
    new_psls = np.max(np.abs(new_acf_lags), axis=1)  # shape (n,)
    
    # Find best
    best_i = int(np.argmin(new_psls))
    best_new_psl = float(new_psls[best_i])
    
    return best_i, best_new_psl, current_psl

def _update_acf_flip(h, acf, i):
    """
    Incrementally update ACF after flipping h[i].
    Correctly handles the double-counting at lag 0.
    """
    n = len(h)
    h_i = h[i]
    
    # For lag k (can be negative or positive in full ACF)
    # ACF[k] = sum_j h[j]*h[j+k]
    # When h[i] flips, terms involving h[i] change:
    # For k > 0: term h[i]*h[i+k] changes by -2*h[i]*h[i+k]
    #            term h[i-k]*h[i] changes by -2*h[i]*h[i-k]
    # For k = 0: term h[i]*h[i] changes by -2*h[i]*h[i] = -2 (since h[i]^2=1)
    # For k < 0: similar to k > 0 by symmetry
    
    delta = np.zeros(2*n - 1)
    
    # For positive lags k = 1 to n-1
    for k in range(1, n):
        # Term from h[i]*h[i+k]
        if i + k < n:
            delta[n - 1 + k] -= 2.0 * h_i * h[i+k]
        # Term from h[i-k]*h[i]  
        if i - k >= 0:
            delta[n - 1 + k] -= 2.0 * h_i * h[i-k]
    
    # For negative lags k = -(n-1) to -1
    for k in range(1, n):
        # Term from h[i]*h[i-k] (this is ACF[-k])
        if i - k >= 0:
            delta[n - 1 - k] -= 2.0 * h_i * h[i-k]
        # Term from h[i+k]*h[i] (this is ACF[-k])
        if i + k < n:
            delta[n - 1 - k] -= 2.0 * h_i * h[i+k]
    
    # For lag 0: ACF[0] = sum h[j]^2 = n, doesn't change when flipping
    # Actually, h[i]^2 = 1 before and after, so no change
    
    acf_new = acf + delta
    return acf_new

def _fast_psl_optimize(h, max_passes=200, time_budget=None, start_time=None):
    """
    Fast PSL optimization using Gordon's gradient method.
    Repeatedly find the single best flip and apply it.
    Uses incremental ACF updates.
    """
    h = h.copy().astype(np.float64)
    n = len(h)
    
    if start_time is None:
        start_time = time.time()
    
    best_h = h.copy()
    acf = _compute_acf(h)
    best_psl = _psl_value(acf, n)
    
    passes = 0
    no_improve_count = 0
    while passes < max_passes:
        if time_budget is not None and (time.time() - start_time) > time_budget:
            break
        
        i, new_psl, cur_psl = _gradient_step_vectorized(h, acf)
        
        if i == -1 or new_psl >= cur_psl - 1e-12:
            no_improve_count += 1
            if no_improve_count >= 5:
                break
            continue
        
        h[i] = -h[i]
        passes += 1
        no_improve_count = 0
        
        # Incremental ACF update
        acf = _update_acf_flip(h, acf, i)
        cur_psl = _psl_value(acf, n)
        
        if cur_psl < best_psl:
            best_psl = cur_psl
            best_h = h.copy()
        
        if best_psl < 1e-10:
            break
    
    return best_h, best_psl

def _pairwise_flip_search(h, acf, max_pairs=200, time_budget=None, start_time=None):
    """
    Try flipping pairs of positions to escape local minima.
    Returns (best_h, best_psl).
    """
    n = len(h)
    if start_time is None:
        start_time = time.time()
    
    best_h = h.copy()
    best_psl = _psl_value(acf, n)
    
    # Sample random pairs
    rng = np.random.default_rng(12345)
    attempts = 0
    while attempts < max_pairs:
        if time_budget is not None and (time.time() - start_time) > time_budget:
            break
        
        # Pick two random distinct positions
        i, j = rng.choice(n, size=2, replace=False)
        if i == j:
            attempts += 1
            continue
        
        # Try flipping both i and j
        h[i] = -h[i]
        h[j] = -h[j]
        
        # Compute new ACF (full recompute for simplicity, or incremental)
        acf_new = _compute_acf(h)
        psl_new = _psl_value(acf_new, n)
        
        if psl_new < best_psl:
            best_psl = psl_new
            best_h = h.copy()
            acf = acf_new
        else:
            # Revert
            h[i] = -h[i]
            h[j] = -h[j]
        
        attempts += 1
    
    return best_h, best_psl

def _logistic_init(r, n, seed):
    """
    Generate a sequence of +/- 1 based on the logistic map.
    x_{k+1} = r * x_k * (1 - x_k)
    sign is determined by whether x_k > 0.5
    """
    rng = np.random.default_rng(seed)
    x = rng.uniform(0.1, 0.9) # Initial condition
    signs = np.zeros(n)
    for i in range(n):
        signs[i] = 1.0 if x > 0.5 else -1.0
        x = r * x * (1.0 - x)
        if x <= 0 or x >= 1:
            # Reset to avoid escape or stagnation at 0/1
            x = rng.uniform(0.1, 0.9)
    return signs

def _m_sequence(n, seed):
    """
    Generate a maximal-length sequence (m-sequence) using LFSR.
    These have optimal autocorrelation properties.
    """
    rng = np.random.default_rng(seed)
    # Use a primitive polynomial of degree m where 2^m - 1 >= n
    # For n=400, we need m >= 9 (2^9-1=511)
    m = 9
    # Primitive polynomial for m=9: x^9 + x^4 + 1
    # Tap positions (0-indexed from LSB): 0, 4, 8
    state = int(rng.integers(1, 2**m))
    
    seq = []
    for _ in range(n):
        seq.append(1.0 if (state & 1) else -1.0)
        # Shift right and apply feedback
        feedback = (state & 1) ^ ((state >> 4) & 1)
        state = (state >> 1) | (feedback << (m-1))
    
    return np.array(seq)

def construct():
    start_time = time.time()
    deadline = 115.0  # Safety margin before 120s limit
    
    best_h = None
    best_score = float('inf')
    
    rng = np.random.default_rng(42)
    
    # Generate initial candidates
    candidates = []
    
    # 1. Random
    for i in range(15):
        candidates.append(rng.choice([-1.0, 1.0], size=N))
        
    # 2. Quasi-periodic (Irrational frequencies)
    for trial in range(8):
        phases = rng.uniform(0, 2*np.pi, size=8)
        amps = rng.uniform(0.3, 1.0, size=8)
        t = np.linspace(0, 1, N, endpoint=False)
        val = np.zeros(N)
        for k in range(1, 9):
            val += amps[k-1] * np.cos(2*np.pi*k*t + phases[k-1])
        candidates.append(np.sign(val))
        
    # 3. Simple periodic patterns
    for period in [2, 3, 4, 5, 7, 11, 13, 17, 19, 23]:
        pattern = np.sign(np.cos(2*np.pi*np.arange(N)/period))
        candidates.append(pattern)
        
    # 4. Block structure with various block sizes
    for block_size in [5, 8, 10, 16, 20]:
        for i in range(2):
            h = np.zeros(N)
            for b in range(N // block_size + 1):
                start = b*block_size
                end = min(start + block_size, N)
                h[start:end] = rng.choice([-1.0, 1.0])
            candidates.append(h)
        
    # 5. Logistic map initializations (Chaotic regime)
    for r in [3.5, 3.7, 3.9, 3.99, 4.0]:
        for seed in range(5):
            candidates.append(_logistic_init(r, N, seed))
            
    # 6. Thue-Morse sequence
    thue = np.array([1.0 if bin(i).count('1') % 2 == 0 else -1.0 for i in range(N)])
    candidates.append(thue)
    
    # 7. M-sequences (maximal length sequences)
    for seed in range(5):
        candidates.append(_m_sequence(N, seed))
    
    # 8. More diverse random with different densities
    for density in [0.25, 0.35, 0.5, 0.65, 0.75]:
        h = np.zeros(N)
        h = np.where(rng.random(N) < density, 1.0, -1.0)
        candidates.append(h)
    
    # 9. Walsh functions
    for k in range(1, 8):
        t = np.arange(N)
        candidates.append(np.sign(np.cos(2*np.pi*k*t/N)))
    
    # Evaluate all candidates and keep the best few for further search
    scored_candidates = []
    for h in candidates:
        s = _score(h)
        scored_candidates.append((s, h))
    
    scored_candidates.sort(key=lambda x: x[0])
    
    # Refine the top 8 candidates with fast PSL optimization (Gordon's method)
    for i, (s, h) in enumerate(scored_candidates[:8]):
        if time.time() > deadline - 20:
            break
        # Allocate time based on rank
        time_limit = deadline - 12
        h_refined, s_refined = _fast_psl_optimize(h, max_passes=500, time_budget=time_limit, start_time=start_time)
        if s_refined < best_score:
            best_score = s_refined
            best_h = h_refined.copy()
            
        # After single-flip search, try pairwise flips to escape local minima
        if time.time() < deadline - 12:
            acf_best = _compute_acf(best_h)
            h_pair, s_pair = _pairwise_flip_search(best_h, acf_best, max_pairs=150, 
                                                    time_budget=deadline - 12, start_time=start_time)
            if s_pair < best_score:
                best_score = s_pair
                best_h = h_pair.copy()
    
    # If time remains, do more random restarts with fast PSL optimization
    while time.time() < deadline - 8:
        # Alternate between various initialization strategies
        r_choice = rng.random()
        if r_choice < 0.3:
            h_init = rng.choice([-1.0, 1.0], size=N)
        elif r_choice < 0.5:
            r = rng.uniform(3.5, 4.0)
            h_init = _logistic_init(r, N, int(rng.integers(0, 10000)))
        elif r_choice < 0.7:
            h_init = _m_sequence(N, int(rng.integers(0, 10000)))
        elif r_choice < 0.85:
            # Quasi-periodic
            phases = rng.uniform(0, 2*np.pi, size=6)
            amps = rng.uniform(0.3, 1.0, size=6)
            t = np.linspace(0, 1, N, endpoint=False)
            val = np.zeros(N)
            for k in range(1, 7):
                val += amps[k-1] * np.cos(2*np.pi*k*t + phases[k-1])
            h_init = np.sign(val)
        else:
            # Block structure
            block_size = int(rng.integers(4, 25))
            h_init = np.zeros(N)
            for b in range(N // block_size + 1):
                start = b*block_size
                end = min(start + block_size, N)
                h_init[start:end] = rng.choice([-1.0, 1.0])
            
        # Quick optimization with limited time
        time_limit = min(deadline - 6, time.time() + 5)
        h_refined, s_refined = _fast_psl_optimize(h_init, max_passes=300, time_budget=time_limit, start_time=start_time)
        if s_refined < best_score:
            best_score = s_refined
            best_h = h_refined.copy()

    if best_h is None:
        # Fallback
        best_h = np.ones(N)
        
    return best_h
