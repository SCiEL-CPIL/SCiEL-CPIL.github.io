import numpy as np
import time

N = 400

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

def _find_best_flip(h, acf):
    """
    Find the index i that minimizes the max absolute autocorrelation
    after flipping h[i].
    Returns (best_i, best_new_psl, current_psl).
    """
    n = len(h)
    current_psl = _psl_value(acf, n)
    acf_lags = acf[1:n]
    
    best_i = -1
    best_new_psl = float('inf')
    
    # Pre-allocate neighbor_sum to avoid repeated allocation
    # We loop over i, compute delta, check max.
    for i in range(n):
        # Compute neighbor sums: h[i-k] + h[i+k] for k=1..n-1
        # We can do this with slicing
        # Left part: h[i-1], h[i-2], ...
        # Right part: h[i+1], h[i+2], ...
        
        # Construct neighbor_sum vector of length n-1
        # It is the sum of h shifted by -1 and +1, centered at i.
        # Actually, for lag k, the term is h[i-k] + h[i+k].
        
        # Efficient way:
        # neighbor_sum[k-1] = h[i-k] + h[i+k]
        
        # We can use np.roll or slicing.
        # Let's just do the loop with slicing as it's clear and fast enough for N=400.
        
        max_k1 = min(i, n-1)
        max_k2 = min(n-1-i, n-1)
        
        neighbor_sum = np.zeros(n-1)
        
        if max_k1 > 0:
            # h[i-1], h[i-2], ..., h[i-max_k1]
            # Indices: i-1 down to i-max_k1
            neighbor_sum[:max_k1] += h[i-1 : i-max_k1-1 : -1]
            
        if max_k2 > 0:
            # h[i+1], h[i+2], ..., h[i+max_k2]
            neighbor_sum[:max_k2] += h[i+1 : i+max_k2+1]
            
        delta = -2.0 * h[i] * neighbor_sum
        new_acf_lags = acf_lags + delta
        new_psl = np.max(np.abs(new_acf_lags))
        
        if new_psl < best_new_psl:
            best_new_psl = new_psl
            best_i = i
            
    return best_i, float(best_new_psl), current_psl

def _local_search(h, time_budget=10.0, start_time=None):
    """
    Greedy local search: repeatedly flip the bit that gives the best improvement.
    """
    h = h.copy().astype(np.float64)
    n = len(h)
    
    if start_time is None:
        start_time = time.time()
        
    acf = _compute_acf(h)
    current_psl = _psl_value(acf, n)
    
    best_h = h.copy()
    best_psl = current_psl
    
    steps = 0
    while time.time() - start_time < time_budget:
        i, new_psl, cur_psl = _find_best_flip(h, acf)
        
        if i == -1 or new_psl >= cur_psl:
            break
            
        h[i] = -h[i]
        steps += 1
        
        # Update ACF. Recomputing via FFT is O(N log N) ~ 4000 ops.
        # The flip search was O(N^2) ~ 160000 ops.
        # So recompute is cheap.
        acf = _compute_acf(h)
        current_psl = _psl_value(acf, n)
        
        if current_psl < best_psl:
            best_psl = current_psl
            best_h = h.copy()
            
        if best_psl < 1e-10:
            break
            
    return best_h, best_psl

def _thue_morse(n):
    """Generate Thue-Morse sequence of length n as +/-1."""
    t = np.zeros(n)
    for i in range(n):
        # Count set bits
        t[i] = 1.0 if bin(i).count('1') % 2 == 0 else -1.0
    return t

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    best_h = None
    best_score = float('inf')
    
    rng = np.random.default_rng(42)
    
    # Generate initial candidates
    candidates = []
    
    # 1. Thue-Morse
    candidates.append(_thue_morse(N))
    
    # 2. Random
    for i in range(20):
        candidates.append(rng.choice([-1.0, 1.0], size=N))
        
    # 3. Periodic patterns
    for period in [2, 3, 4, 5, 7, 11, 13, 17]:
        pattern = np.sign(np.cos(2*np.pi*np.arange(N)/period))
        candidates.append(pattern)
        
    # 4. Quasi-periodic
    for trial in range(5):
        phases = rng.uniform(0, 2*np.pi, size=5)
        t = np.linspace(0, 1, N, endpoint=False)
        val = np.zeros(N)
        for k in range(1, 6):
            val += np.cos(2*np.pi*k*t + phases[k-1])
        candidates.append(np.sign(val))
        
    # Evaluate all candidates
    scored_candidates = []
    for h in candidates:
        s = _psl_value(_compute_acf(h), N)
        scored_candidates.append((s, h))
        
    scored_candidates.sort(key=lambda x: x[0])
    
    # Refine top candidates
    # We have ~140s.
    # Local search step is ~1ms.
    # So we can do ~100 steps per second.
    # Let's allocate time to the top 5 candidates.
    
    for i, (s, h) in enumerate(scored_candidates[:5]):
        if time.time() > deadline - 10:
            break
            
        # Allocate more time to better initial candidates
        time_limit = min(20.0, deadline - 5 - (time.time() - start_time))
        if time_limit < 1.0:
            break
            
        h_refined, s_refined = _local_search(h, time_budget=time_limit, start_time=start_time)
        
        if s_refined < best_score:
            best_score = s_refined
            best_h = h_refined.copy()
            
    # If time remains, do more random restarts
    while time.time() < deadline - 10:
        h_init = rng.choice([-1.0, 1.0], size=N)
        # Short local search
        time_limit = min(5.0, deadline - 5 - (time.time() - start_time))
        if time_limit < 1.0:
            break
        h_refined, s_refined = _local_search(h_init, time_budget=time_limit, start_time=start_time)
        if s_refined < best_score:
            best_score = s_refined
            best_h = h_refined.copy()
            
    if best_h is None:
        best_h = np.ones(N)
        
    return best_h
