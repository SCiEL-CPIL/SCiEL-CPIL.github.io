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

def _compute_neighbor_sums(h):
    """
    Compute for each position i the vector of neighbor sums:
    neighbor_sum[i][k-1] = h[i-k] + h[i+k] for valid k.
    Returns array of shape (n, n-1).
    """
    n = len(h)
    neighbor_sums = np.zeros((n, n-1))
    for k in range(1, n):
        neighbor_sums[k:, k-1] += h[:n-k]
        neighbor_sums[:n-k, k-1] += h[k:]
    return neighbor_sums

def _gradient_step_fast(h, acf, neighbor_sums):
    """
    Fast PSL minimization step using precomputed neighbor sums.
    For each position i, compute the change in PSL if we flip h[i].
    """
    n = len(h)
    current_psl = _psl_value(acf, n)
    acf_lags = acf[1:n]
    deltas = -2.0 * h[:, np.newaxis] * neighbor_sums
    new_acf_lags_all = acf_lags[np.newaxis, :] + deltas
    new_psls = np.max(np.abs(new_acf_lags_all), axis=1)
    best_i = int(np.argmin(new_psls))
    best_new_psl = float(new_psls[best_i])
    return best_i, best_new_psl, current_psl

def _fast_psl_optimize(h, max_passes=200, time_budget=None, start_time=None):
    """
    Fast PSL optimization with incremental ACF updates.
    Repeatedly find the single best flip and apply it.
    Uses incremental ACF update for O(N) per step instead of O(N log N).
    """
    h = h.copy().astype(np.float64)
    n = len(h)
    
    if start_time is None:
        start_time = time.time()
    
    best_h = h.copy()
    acf = _compute_acf(h)
    best_psl = _psl_value(acf, n)
    neighbor_sums = _compute_neighbor_sums(h)
    
    passes = 0
    while passes < max_passes:
        if time_budget is not None and (time.time() - start_time) > time_budget:
            break
        
        i, new_psl, cur_psl = _gradient_step_fast(h, acf, neighbor_sums)
        
        if i == -1 or new_psl >= cur_psl:
            break
        
        # Flip h[i]
        old_val = h[i]
        h[i] = -h[i]
        new_val = h[i]
        passes += 1
        
        # Incremental ACF update: flipping h[i] changes acf[k] by -2*old_val*(h[i-k] + h[i+k])
        # which equals -2*old_val * neighbor_sums[i, k-1]
        # But we need to be careful: neighbor_sums[i, k-1] = h[i-k] + h[i+k]
        # After the flip, h[i] is new_val, but the neighbors haven't changed
        # The change in acf at lag k is: -2 * old_val * (h[i-k] + h[i+k])
        # = -2 * old_val * neighbor_sums[i, k-1]
        
        # Update acf incrementally
        delta_acf = -2.0 * old_val * neighbor_sums[i, :]
        acf[1:n] += delta_acf
        
        # Update neighbor sums: flipping h[i] affects neighbor_sums[j] for j where i is a neighbor
        # When h[i] flips from old to new:
        # For each lag k:
        #   If i+k < n: neighbor_sums[i+k, k-1] includes h[i] as h[(i+k)-k] = h[i]
        #   If i-k >= 0: neighbor_sums[i-k, k-1] includes h[i] as h[(i-k)+k] = h[i]
        # Change in h[i] is (new - old) = -2*old = 2*new
        for k in range(1, n):
            if i + k < n:
                neighbor_sums[i+k, k-1] += 2.0 * new_val
            if i - k >= 0:
                neighbor_sums[i-k, k-1] += 2.0 * new_val
        
        cur_psl = _psl_value(acf, n)
        
        if cur_psl < best_psl:
            best_psl = cur_psl
            best_h = h.copy()
        
        if best_psl < 1e-10:
            break
    
    return best_h, best_psl

def _logistic_init(r, n, seed, init_x=None):
    """
    Generate a sequence of +/- 1 based on the logistic map.
    x_{k+1} = r * x_k * (1 - x_k)
    sign is determined by whether x_k > 0.5
    """
    rng = np.random.default_rng(seed)
    if init_x is None:
        x = rng.uniform(0.1, 0.9)
    else:
        x = init_x
    signs = np.zeros(n)
    for i in range(n):
        signs[i] = 1.0 if x > 0.5 else -1.0
        x = r * x * (1.0 - x)
        if x <= 0 or x >= 1:
            x = rng.uniform(0.1, 0.9)
    return signs

def _discrete_logistic_init(mu, n, seed, init_x=None):
    """
    Generate a sequence using discrete logistic map: x_{n+1} = floor(mu * x_n * (1 - x_n))
    Then map to signs based on parity or threshold.
    """
    rng = np.random.default_rng(seed)
    if init_x is None:
        x = rng.uniform(0.1, 0.9)
    else:
        x = init_x
    
    signs = np.zeros(n)
    for i in range(n):
        signs[i] = 1.0 if x > 0.5 else -1.0
        # Discrete logistic: x_{n+1} = floor(mu * x_n * (1 - x_n))
        # But this gives integers, so we need a different approach
        # Let's use: x_{n+1} = mu * x_n * (1 - x_n) but with rounding to keep it discrete
        # Actually, let's use a variant: x_{n+1} = (mu * x_n * (1 - x_n)) % 1.0
        # This keeps x in [0, 1) and creates chaotic dynamics
        x = (mu * x * (1.0 - x)) % 1.0
        if x < 1e-15:
            x = rng.uniform(0.1, 0.9)
    return signs

def _optimize_logistic_params(n, seed, max_trials=50):
    """
    Search over logistic map parameters (r, init_x) to find sequences with low PSL.
    Returns the best (h, psl) pair.
    """
    rng = np.random.default_rng(seed)
    best_h = None
    best_psl = float('inf')
    
    for trial in range(max_trials):
        # Sample r in chaotic regime
        r = rng.uniform(3.5, 4.0)
        init_x = rng.uniform(0.1, 0.9)
        
        h = _logistic_init(r, n, seed + trial, init_x)
        psl = _score(h)
        
        if psl < best_psl:
            best_psl = psl
            best_h = h.copy()
    
    return best_h, best_psl

def _optimize_discrete_logistic_params(n, seed, max_trials=50):
    """
    Search over discrete logistic map parameters.
    """
    rng = np.random.default_rng(seed)
    best_h = None
    best_psl = float('inf')
    
    for trial in range(max_trials):
        mu = rng.uniform(2.5, 4.0)
        init_x = rng.uniform(0.1, 0.9)
        
        h = _discrete_logistic_init(mu, n, seed + trial, init_x)
        psl = _score(h)
        
        if psl < best_psl:
            best_psl = psl
            best_h = h.copy()
    
    return best_h, best_psl

def construct():
    start_time = time.time()
    deadline = 145.0  # Safety margin before 150s limit
    
    best_h = None
    best_score = float('inf')
    
    rng = np.random.default_rng(42)
    
    # Generate initial candidates
    candidates = []
    
    # 1. Random
    for i in range(10):
        candidates.append(rng.choice([-1.0, 1.0], size=N))
        
    # 2. Quasi-periodic (Irrational frequencies)
    for trial in range(5):
        phases = rng.uniform(0, 2*np.pi, size=5)
        amps = rng.uniform(0.5, 1.0, size=5)
        t = np.linspace(0, 1, N, endpoint=False)
        val = np.zeros(N)
        for k in range(1, 6):
            val += amps[k-1] * np.cos(2*np.pi*k*t + phases[k-1])
        candidates.append(np.sign(val))
        
    # 3. Simple periodic patterns
    for period in [2, 3, 4, 5, 7, 11, 13]:
        pattern = np.sign(np.cos(2*np.pi*np.arange(N)/period))
        candidates.append(pattern)
        
    # 4. Block structure
    block_size = 10
    for i in range(4):
        h = np.zeros(N)
        for b in range(N // block_size):
            h[b*block_size:(b+1)*block_size] = rng.choice([-1.0, 1.0])
        candidates.append(h)
        
    # 5. Logistic map initializations with parameter search
    h_log, psl_log = _optimize_logistic_params(N, seed=42, max_trials=30)
    if h_log is not None:
        candidates.append(h_log)
    
    # 6. Discrete logistic map initializations
    h_dlog, psl_dlog = _optimize_discrete_logistic_params(N, seed=42, max_trials=30)
    if h_dlog is not None:
        candidates.append(h_dlog)
    
    # Additional logistic with different seeds
    for seed in [100, 200, 300]:
        h_log, psl_log = _optimize_logistic_params(N, seed=seed, max_trials=10)
        if h_log is not None:
            candidates.append(h_log)
        h_dlog, psl_dlog = _optimize_discrete_logistic_params(N, seed=seed, max_trials=10)
        if h_dlog is not None:
            candidates.append(h_dlog)
            
    # Evaluate all candidates and keep the best few for further search
    scored_candidates = []
    for h in candidates:
        s = _score(h)
        scored_candidates.append((s, h))
    
    scored_candidates.sort(key=lambda x: x[0])
    
    # Refine the top candidates with fast PSL optimization
    for i, (s, h) in enumerate(scored_candidates[:8]):
        if time.time() > deadline - 15:
            break
        time_limit = deadline - 10
        h_refined, s_refined = _fast_psl_optimize(h, max_passes=5000, time_budget=time_limit, start_time=start_time)
        if s_refined < best_score:
            best_score = s_refined
            best_h = h_refined.copy()
            
    # If time remains, do random restarts with fast PSL optimization
    restart_count = 0
    while time.time() < deadline - 8:
        restart_count += 1
        # Alternate between random, logistic, and discrete logistic initializations
        r_choice = rng.random()
        if r_choice < 0.3:
            h_init = rng.choice([-1.0, 1.0], size=N)
        elif r_choice < 0.6:
            r = rng.uniform(3.5, 4.0)
            init_x = rng.uniform(0.1, 0.9)
            h_init = _logistic_init(r, N, int(rng.integers(0, 100000)), init_x)
        else:
            mu = rng.uniform(2.5, 4.0)
            init_x = rng.uniform(0.1, 0.9)
            h_init = _discrete_logistic_init(mu, N, int(rng.integers(0, 100000)), init_x)
            
        # Quick optimization with limited time
        time_limit = min(deadline - 5, time.time() + 3)
        h_refined, s_refined = _fast_psl_optimize(h_init, max_passes=5000, time_budget=time_limit, start_time=start_time)
        if s_refined < best_score:
            best_score = s_refined
            best_h = h_refined.copy()
        
        if restart_count >= 20:
            break

    if best_h is None:
        best_h = np.ones(N)
        
    return best_h
