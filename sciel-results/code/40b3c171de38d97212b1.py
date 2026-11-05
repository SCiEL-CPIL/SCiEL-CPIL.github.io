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

def _l3_value(acf, n):
    """Get L3 norm (sum of cubes) from acf array (non-zero lags)."""
    if n <= 1:
        return 0.0
    return float(np.sum(np.abs(acf[1:n]) ** 3))

def _score(h):
    """Compute the L3 norm (sum of cubes) of autocorrelation for lags 1 to N-1."""
    acf = _compute_acf(h)
    return _l3_value(acf, len(h))

def _gradient_step_l3(h, acf):
    """
    Find the best single flip to minimize L3 norm (sum of cubes).
    """
    n = len(h)
    current_l3 = _l3_value(acf, n)
    
    best_i = -1
    best_new_l3 = float('inf')
    
    acf_lags = acf[1:n]
    abs_acf_lags = np.abs(acf_lags)
    
    for i in range(n):
        neighbor_sum = np.zeros(n-1)
        
        max_k1 = min(i, n-1)
        if max_k1 > 0:
            neighbor_sum[:max_k1] += h[i-1 : i-max_k1-1 : -1]
            
        max_k2 = min(n-1-i, n-1)
        if max_k2 > 0:
            neighbor_sum[:max_k2] += h[i+1 : i+max_k2+1]
            
        delta = -2.0 * h[i] * neighbor_sum
        new_acf_lags = acf_lags + delta
        new_l3 = float(np.sum(np.abs(new_acf_lags) ** 3))
        
        if new_l3 < best_new_l3:
            best_new_l3 = new_l3
            best_i = i
            
    return best_i, float(best_new_l3), current_l3

def _fast_l3_optimize(h, max_passes=200, time_budget=None, start_time=None):
    """
    Fast L3 norm optimization using gradient method.
    Repeatedly find the single best flip and apply it.
    """
    h = h.copy().astype(np.float64)
    n = len(h)
    
    if start_time is None:
        start_time = time.time()
    
    best_h = h.copy()
    acf = _compute_acf(h)
    best_l3 = _l3_value(acf, n)
    
    passes = 0
    while passes < max_passes:
        if time_budget is not None and (time.time() - start_time) > time_budget:
            break
        
        i, new_l3, cur_l3 = _gradient_step_l3(h, acf)
        
        if i == -1 or new_l3 >= cur_l3:
            break
        
        h[i] = -h[i]
        passes += 1
        
        acf = _compute_acf(h)
        cur_l3 = _l3_value(acf, n)
        
        if cur_l3 < best_l3:
            best_l3 = cur_l3
            best_h = h.copy()
        
        if best_l3 < 1e-10:
            break
    
    return best_h, best_l3

def _simulated_annealing_l3(h, time_budget=30.0, start_time=None, rng=None):
    """
    Simulated annealing on L3 norm.
    """
    if rng is None:
        rng = np.random.default_rng()
        
    h = h.copy().astype(np.float64)
    n = len(h)
    
    if start_time is None:
        start_time = time.time()
    
    acf = _compute_acf(h)
    current_l3 = _l3_value(acf, n)
    
    best_h = h.copy()
    best_l3 = current_l3
    
    T = 100.0
    T_min = 0.1
    cooling = 0.95
    
    while T > T_min:
        if time.time() - start_time > time_budget:
            break
        
        i = int(rng.integers(n))
        
        neighbor_sum = np.zeros(n-1)
        max_k1 = min(i, n-1)
        if max_k1 > 0:
            neighbor_sum[:max_k1] += h[i-1 : i-max_k1-1 : -1]
        max_k2 = min(n-1-i, n-1)
        if max_k2 > 0:
            neighbor_sum[:max_k2] += h[i+1 : i+max_k2+1]
            
        delta = -2.0 * h[i] * neighbor_sum
        new_acf_lags = acf[1:n] + delta
        new_l3 = float(np.sum(np.abs(new_acf_lags) ** 3))
        
        delta_l3 = new_l3 - current_l3
        
        if delta_l3 < 0 or np.random.random() < np.exp(-delta_l3 / T):
            h[i] = -h[i]
            acf = _compute_acf(h)
            current_l3 = _l3_value(acf, n)
            
            if current_l3 < best_l3:
                best_l3 = current_l3
                best_h = h.copy()
        
        T *= cooling
    
    return best_h, best_l3

def _zadoff_chu_chirp(n, m, a=1):
    """
    Generate a Zadoff-Chu-like quadratic phase chirp.
    h[k] = exp(i * pi * a * k * (k+1) / m) for k=0..n-1
    Returns the imaginary part (which will be in [-1,1]).
    """
    k = np.arange(n, dtype=np.float64)
    phase = np.pi * a * k * (k + 1) / m
    return np.sin(phase)

def _zadoff_chu_product_sequence(n, m1, m2, a1=1, a2=1, rng=None):
    """
    Create a signed sequence from the sign of the imaginary part of the product
    of two independent quadratic phase chirps with distinct moduli.
    
    h[k] = sign(Im(exp(i*pi*a1*k*(k+1)/m1) * exp(i*pi*a2*k*(k+1)/m2)))
         = sign(sin(pi*a1*k*(k+1)/m1 + pi*a2*k*(k+1)/m2))
    """
    k = np.arange(n, dtype=np.float64)
    phase1 = np.pi * a1 * k * (k + 1) / m1
    phase2 = np.pi * a2 * k * (k + 1) / m2
    combined_phase = phase1 + phase2
    val = np.sin(combined_phase)
    # Handle zeros by using a small perturbation or defaulting to +1
    val[val == 0] = 1.0
    return np.sign(val)

def _generate_zadoff_chu_candidates(n, num_candidates=20, rng=None):
    """
    Generate multiple Zadoff-Chu product sequence candidates with different
    prime moduli and coefficients.
    """
    if rng is None:
        rng = np.random.default_rng()
    
    candidates = []
    
    # Use distinct prime moduli to ensure independence
    primes = [7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83]
    
    for i in range(num_candidates):
        # Pick two distinct primes
        m1_idx = int(rng.integers(len(primes)))
        m2_idx = int(rng.integers(len(primes)))
        while m2_idx == m1_idx:
            m2_idx = int(rng.integers(len(primes)))
        
        m1 = primes[m1_idx]
        m2 = primes[m2_idx]
        
        # Vary the coefficients
        a1 = int(rng.integers(1, min(5, m1)))
        a2 = int(rng.integers(1, min(5, m2)))
        
        h = _zadoff_chu_product_sequence(n, m1, m2, a1, a2, rng)
        candidates.append(h)
    
    return candidates

def construct():
    start_time = time.time()
    deadline = 140.0  # Safety margin before 150s limit
    
    best_h = None
    best_score = float('inf')
    
    rng = np.random.default_rng(42)
    
    # Generate initial candidates
    candidates = []
    
    # 1. Zadoff-Chu chirp product sequences (primary strategy)
    zc_candidates = _generate_zadoff_chu_candidates(N, num_candidates=30, rng=rng)
    candidates.extend(zc_candidates)
    
    # 2. Random
    for i in range(10):
        candidates.append(rng.choice([-1.0, 1.0], size=N))
        
    # 3. Quasi-periodic (Irrational frequencies)
    for trial in range(5):
        phases = rng.uniform(0, 2*np.pi, size=5)
        amps = rng.uniform(0.5, 1.0, size=5)
        t = np.linspace(0, 1, N, endpoint=False)
        val = np.zeros(N)
        for k in range(1, 6):
            val += amps[k-1] * np.cos(2*np.pi*k*t + phases[k-1])
        candidates.append(np.sign(val))
        
    # 4. Simple periodic patterns
    for period in [2, 3, 4, 5, 7, 11, 13]:
        pattern = np.sign(np.cos(2*np.pi*np.arange(N)/period))
        candidates.append(pattern)
        
    # 5. Block structure
    block_size = 10
    for i in range(4):
        h = np.zeros(N)
        for b in range(N // block_size):
            h[b*block_size:(b+1)*block_size] = rng.choice([-1.0, 1.0])
        candidates.append(h)
    
    # Evaluate all candidates and keep the best few for further search
    scored_candidates = []
    for h in candidates:
        s = _score(h)
        scored_candidates.append((s, h))
    
    scored_candidates.sort(key=lambda x: x[0])
    
    # Refine the top 8 candidates with fast L3 optimization
    for i, (s, h) in enumerate(scored_candidates[:8]):
        if time.time() > deadline - 20:
            break
        remaining = deadline - 10 - (time.time() - start_time)
        if remaining < 5:
            break
        time_limit = min(remaining, 15.0)
        h_refined, s_refined = _fast_l3_optimize(h, max_passes=500, time_budget=time_limit, start_time=start_time)
        if s_refined < best_score:
            best_score = s_refined
            best_h = h_refined.copy()
            
    # If time remains, do a few random restarts with fast L3 optimization
    while time.time() < deadline - 10:
        h_init = rng.choice([-1.0, 1.0], size=N)
        remaining = deadline - 5 - (time.time() - start_time)
        if remaining < 3:
            break
        time_limit = min(remaining, 5.0)
        h_refined, s_refined = _fast_l3_optimize(h_init, max_passes=100, time_budget=time_limit, start_time=start_time)
        if s_refined < best_score:
            best_score = s_refined
            best_h = h_refined.copy()
            
    # If we still have time, try SA on the best found so far
    if best_h is not None and time.time() < deadline - 5:
        sa_time = min(10.0, deadline - 5 - (time.time() - start_time))
        if sa_time > 2.0:
            h_sa, s_sa = _simulated_annealing_l3(best_h, time_budget=sa_time, start_time=start_time, rng=rng)
            if s_sa < best_score:
                best_score = s_sa
                best_h = h_sa.copy()
                
    if best_h is None:
        # Fallback
        best_h = np.ones(N)
        
    return best_h
