import numpy as np
import time

N = 400

def _score(h):
    """Compute the max absolute autocorrelation for lags 1 to N-1."""
    fft_size = 1
    while fft_size < 2 * N:
        fft_size *= 2
    H = np.fft.rfft(h, n=fft_size)
    acf = np.fft.irfft(np.abs(H) ** 2, n=fft_size)
    return np.max(np.abs(acf[1:N]))

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

def _neighbor_sum(h, i):
    """Compute neighbor sum vector for position i."""
    n = len(h)
    neighbor_sum = np.zeros(n - 1)
    max_k1 = min(i, n - 1)
    if max_k1 > 0:
        neighbor_sum[:max_k1] += h[i - 1 : i - max_k1 - 1 : -1]
    max_k2 = min(n - 1 - i, n - 1)
    if max_k2 > 0:
        neighbor_sum[:max_k2] += h[i + 1 : i + max_k2 + 1]
    return neighbor_sum

def _fast_psl_optimize(h, max_passes=200, time_budget=None, start_time=None):
    """
    Fast PSL optimization using Gordon's gradient method.
    Repeatedly find the single best flip and apply it.
    """
    h = h.copy().astype(np.float64)
    n = len(h)
    
    if start_time is None:
        start_time = time.time()
    
    best_h = h.copy()
    acf = _compute_acf(h)
    best_psl = _psl_value(acf, n)
    
    passes = 0
    while passes < max_passes:
        if time_budget is not None and (time.time() - start_time) > time_budget:
            break
        
        current_psl = _psl_value(acf, n)
        acf_lags = acf[1:n]
        
        best_i = -1
        best_new_psl = float('inf')
        
        for i in range(n):
            neighbor_sum = _neighbor_sum(h, i)
            delta = -2.0 * h[i] * neighbor_sum
            new_acf_lags = acf_lags + delta
            new_psl = np.max(np.abs(new_acf_lags))
            
            if new_psl < best_new_psl:
                best_new_psl = new_psl
                best_i = i
        
        if best_i == -1 or best_new_psl >= current_psl:
            break
        
        h[best_i] = -h[best_i]
        passes += 1
        
        acf = _compute_acf(h)
        cur_psl = _psl_value(acf, n)
        
        if cur_psl < best_psl:
            best_psl = cur_psl
            best_h = h.copy()
        
        if best_psl < 1e-10:
            break
    
    return best_h, best_psl

def _van_der_corput_signs(n, base=2):
    """Generate van der Corput sequence mapped to signs."""
    vdc = np.zeros(n)
    for i in range(n):
        f = 1.0 / base
        val = 0.0
        ii = i
        while ii > 0:
            val += f * (ii % base)
            ii //= base
            f /= base
        vdc[i] = val
    
    threshold = np.median(vdc)
    signs = np.where(vdc >= threshold, 1.0, -1.0)
    return signs

def _chirp_signs(n, rng):
    """Generate signs from a chirp (frequency-modulated) signal."""
    t = np.arange(n)
    f0 = rng.uniform(0.1, 0.5)
    f1 = rng.uniform(0.5, 1.0)
    freq = f0 + (f1 - f0) * t / n
    phase = 2 * np.pi * np.cumsum(freq) / n
    signs = np.where(np.sin(phase) >= 0, 1.0, -1.0)
    return signs

def _morse_sequence_signs(n, rng):
    """Generate signs from a Morse sequence (binary sequence with good autocorrelation)."""
    lfsr = 1
    signs = np.zeros(n)
    for i in range(n):
        signs[i] = 1.0 if (lfsr & 1) else -1.0
        bit = (lfsr >> 4) ^ (lfsr >> 5) ^ (lfsr >> 6) ^ (lfsr >> 7)
        lfsr = (lfsr >> 1) | (bit << 7)
    return signs

def _random_block_flip(h, block_size, rng):
    """Flip a random contiguous block of signs."""
    h_new = h.copy()
    start = int(rng.integers(0, len(h) - block_size + 1))
    h_new[start:start + block_size] *= -1.0
    return h_new

def _block_flip_optimize(h, time_budget=30.0, start_time=None, rng=None):
    """Optimization using random block flips."""
    if rng is None:
        rng = np.random.default_rng()
    
    h = h.copy().astype(np.float64)
    n = len(h)
    
    if start_time is None:
        start_time = time.time()
    
    acf = _compute_acf(h)
    current_psl = _psl_value(acf, n)
    
    best_h = h.copy()
    best_psl = current_psl
    
    block_sizes = [2, 3, 4, 5, 8, 10, 16]
    
    while time.time() - start_time < time_budget:
        block_size = int(rng.choice(block_sizes))
        h_new = _random_block_flip(h, block_size, rng)
        
        acf_new = _compute_acf(h_new)
        psl_new = _psl_value(acf_new, n)
        
        if psl_new < current_psl:
            h = h_new
            acf = acf_new
            current_psl = psl_new
            
            if current_psl < best_psl:
                best_psl = current_psl
                best_h = h.copy()
    
    return best_h, best_psl

def _simulated_annealing_psl(h, time_budget=30.0, start_time=None, rng=None):
    """
    Simulated annealing on PSL using the fast gradient to evaluate moves.
    """
    if rng is None:
        rng = np.random.default_rng()
        
    h = h.copy().astype(np.float64)
    n = len(h)
    
    if start_time is None:
        start_time = time.time()
    
    acf = _compute_acf(h)
    current_psl = _psl_value(acf, n)
    
    best_h = h.copy()
    best_psl = current_psl
    
    T = 10.0
    T_min = 0.1
    cooling = 0.95
    
    while T > T_min:
        if time.time() - start_time > time_budget:
            break
        
        i = int(rng.integers(n))
        neighbor_sum = _neighbor_sum(h, i)
        delta = -2.0 * h[i] * neighbor_sum
        new_acf_lags = acf[1:n] + delta
        new_psl = float(np.max(np.abs(new_acf_lags)))
        
        delta_psl = new_psl - current_psl
        
        if delta_psl < 0 or np.random.random() < np.exp(-delta_psl / T):
            h[i] = -h[i]
            acf = _compute_acf(h)
            current_psl = _psl_value(acf, n)
            
            if current_psl < best_psl:
                best_psl = current_psl
                best_h = h.copy()
        
        T *= cooling
    
    return best_h, best_psl

def construct():
    start_time = time.time()
    deadline = 140.0  # Safety margin before 150s limit
    
    best_h = None
    best_score = float('inf')
    
    rng = np.random.default_rng(42)
    
    # Generate initial candidates
    candidates = []
    
    # 1. Van der Corput sequences
    for base in [2, 3, 4, 5, 7]:
        candidates.append(_van_der_corput_signs(N, base=base))
    
    # 2. Chirp sequences
    for _ in range(5):
        candidates.append(_chirp_signs(N, rng))
    
    # 3. Morse sequences
    for _ in range(3):
        candidates.append(_morse_sequence_signs(N, rng))
    
    # 4. Random candidates
    for i in range(10):
        candidates.append(rng.choice([-1.0, 1.0], size=N))
    
    # Evaluate all candidates
    scored_candidates = []
    for h in candidates:
        s = _score(h)
        scored_candidates.append((s, h))
    
    scored_candidates.sort(key=lambda x: x[0])
    
    # Refine the top candidates with fast PSL optimization
    top_k = 10
    for i, (s, h) in enumerate(scored_candidates[:top_k]):
        if time.time() > deadline - 30:
            break
        time_limit = deadline - 20
        h_refined, s_refined = _fast_psl_optimize(h, max_passes=500, time_budget=time_limit, start_time=start_time)
        if s_refined < best_score:
            best_score = s_refined
            best_h = h_refined.copy()
    
    # If time remains, do random restarts with block flips
    while time.time() < deadline - 10:
        h_init = rng.choice([-1.0, 1.0], size=N)
        time_limit = min(deadline - 5, time.time() + 5)
        h_block, s_block = _block_flip_optimize(h_init, time_budget=time_limit, start_time=start_time, rng=rng)
        if s_block < best_score:
            best_score = s_block
            best_h = h_block.copy()
            
        # Then refine with single-bit
        if time.time() < deadline - 5:
            h_refined, s_refined = _fast_psl_optimize(h_block, max_passes=200, time_budget=deadline - 5, start_time=start_time)
            if s_refined < best_score:
                best_score = s_refined
                best_h = h_refined.copy()
                
    # If we still have time, try SA on the best found so far
    if best_h is not None and time.time() < deadline - 5:
        sa_time = min(10.0, deadline - 5 - (time.time() - start_time))
        if sa_time > 2.0:
            h_sa, s_sa = _simulated_annealing_psl(best_h, time_budget=sa_time, start_time=start_time, rng=rng)
            if s_sa < best_score:
                best_score = s_sa
                best_h = h_sa.copy()
                
    if best_h is None:
        best_h = np.ones(N)
        
    return best_h
