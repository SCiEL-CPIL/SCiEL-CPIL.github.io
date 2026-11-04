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

def _soft_max_score(acf_lags, beta=100.0):
    """
    Soft-max surrogate for the PSL using log-sum-exp.
    This provides a smooth approximation to max(|acf_lags|).
    """
    x = np.abs(acf_lags)
    max_x = np.max(x)
    if max_x == 0:
        return 0.0
    # log-sum-exp for numerical stability
    shifted = x - max_x
    log_sum = np.log(np.sum(np.exp(beta * shifted))) / beta
    return max_x + log_sum

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

def _gradient_step(h, acf, use_soft_max=False, beta=100.0):
    """
    Gordon's fast PSL minimization step.
    For each position i, compute the change in PSL if we flip h[i].
    Optionally uses soft-max surrogate for smoother optimization.
    """
    n = len(h)
    current_psl = _psl_value(acf, n)
    
    best_i = -1
    best_new_psl = float('inf')
    
    # Pre-compute the current ACF lags
    acf_lags = acf[1:n]
    
    for i in range(n):
        # Construct the change in ACF
        neighbor_sum = np.zeros(n-1)
        
        max_k1 = min(i, n-1)
        if max_k1 > 0:
            neighbor_sum[:max_k1] += h[i-1 : i-max_k1-1 : -1]
            
        max_k2 = min(n-1-i, n-1)
        if max_k2 > 0:
            neighbor_sum[:max_k2] += h[i+1 : i+max_k2+1]
            
        delta = -2.0 * h[i] * neighbor_sum
        new_acf_lags = acf_lags + delta
        
        if use_soft_max:
            new_psl = _soft_max_score(new_acf_lags, beta=beta)
        else:
            new_psl = np.max(np.abs(new_acf_lags))
        
        if new_psl < best_new_psl:
            best_new_psl = new_psl
            best_i = i
            
    return best_i, float(best_new_psl), current_psl

def _fast_psl_optimize(h, max_passes=200, time_budget=None, start_time=None, use_soft_max=False, beta=100.0):
    """
    Fast PSL optimization using Gordon's gradient method.
    Repeatedly find the single best flip and apply it.
    Optionally uses soft-max surrogate for smoother landscape.
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
        
        i, new_psl, cur_psl = _gradient_step(h, acf, use_soft_max=use_soft_max, beta=beta)
        
        if i == -1 or new_psl >= cur_psl:
            break
        
        h[i] = -h[i]
        passes += 1
        
        acf = _compute_acf(h)
        cur_psl = _psl_value(acf, n)
        
        if cur_psl < best_psl:
            best_psl = cur_psl
            best_h = h.copy()
        
        if best_psl < 1e-10:
            break
    
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
        
        # Compute delta for this specific i
        neighbor_sum = np.zeros(n-1)
        max_k1 = min(i, n-1)
        if max_k1 > 0:
            neighbor_sum[:max_k1] += h[i-1 : i-max_k1-1 : -1]
        max_k2 = min(n-1-i, n-1)
        if max_k2 > 0:
            neighbor_sum[:max_k2] += h[i+1 : i+max_k2+1]
            
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

def _polish_pass(h, time_budget=10.0, start_time=None):
    """
    Final high-precision polish pass with tighter tolerance.
    Uses multiple gradient steps with soft-max to escape shallow local minima.
    """
    if start_time is None:
        start_time = time.time()
    
    h = h.copy().astype(np.float64)
    n = len(h)
    
    acf = _compute_acf(h)
    best_psl = _psl_value(acf, n)
    best_h = h.copy()
    
    # Try multiple beta values for soft-max
    for beta in [50.0, 100.0, 200.0]:
        if time.time() - start_time > time_budget:
            break
        
        current_h = best_h.copy()
        current_acf = _compute_acf(current_h)
        current_psl = _psl_value(current_acf, n)
        
        for _ in range(100):
            if time.time() - start_time > time_budget:
                break
            
            i, new_psl_soft, _ = _gradient_step(current_h, current_acf, use_soft_max=True, beta=beta)
            
            if i == -1:
                break
            
            # Check if the actual PSL improves
            current_h[i] = -current_h[i]
            new_acf = _compute_acf(current_h)
            actual_psl = _psl_value(new_acf, n)
            
            if actual_psl < current_psl:
                current_psl = actual_psl
                current_acf = new_acf
                if current_psl < best_psl:
                    best_psl = current_psl
                    best_h = current_h.copy()
            else:
                # Revert
                current_h[i] = -current_h[i]
                break
    
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
        
    # 5. Rule 90 cellular automaton (Sierpinski gasket structure)
    # This creates a deterministic, non-periodic sequence with hierarchical spectral nulling
    for k in range(1, 10):
        h = np.zeros(N)
        h[0] = 1.0
        for step in range(k):
            new_h = np.zeros(N)
            for i in range(N):
                left = h[i-1] if i > 0 else 0.0
                right = h[i+1] if i < N-1 else 0.0
                # XOR for binary floats 0.0/1.0 is equivalent to (left + right) % 2
                new_h[i] = (left + right) % 2.0
            h = new_h
        # Convert to +/-1
        h = np.where(h != 0, 1.0, -1.0)
        candidates.append(h)
    
    # Evaluate all candidates and keep the best few for further search
    scored_candidates = []
    for h in candidates:
        s = _score(h)
        scored_candidates.append((s, h))
    
    scored_candidates.sort(key=lambda x: x[0])
    
    # Refine the top 5 candidates with fast PSL optimization (Gordon's method)
    for i, (s, h) in enumerate(scored_candidates[:5]):
        if time.time() > deadline - 25:
            break
        # Allocate time based on rank
        time_limit = deadline - 15
        h_refined, s_refined = _fast_psl_optimize(h, max_passes=500, time_budget=time_limit, start_time=start_time)
        if s_refined < best_score:
            best_score = s_refined
            best_h = h_refined.copy()
            
    # If time remains, do a few random restarts with fast PSL optimization
    while time.time() < deadline - 20:
        h_init = rng.choice([-1.0, 1.0], size=N)
        # Quick optimization with limited time
        time_limit = min(deadline - 10, time.time() + 5)
        h_refined, s_refined = _fast_psl_optimize(h_init, max_passes=100, time_budget=time_limit, start_time=start_time)
        if s_refined < best_score:
            best_score = s_refined
            best_h = h_refined.copy()
            
    # If we still have time, try SA on the best found so far
    if best_h is not None and time.time() < deadline - 15:
        sa_time = min(10.0, deadline - 10 - (time.time() - start_time))
        if sa_time > 2.0:
            h_sa, s_sa = _simulated_annealing_psl(best_h, time_budget=sa_time, start_time=start_time, rng=rng)
            if s_sa < best_score:
                best_score = s_sa
                best_h = h_sa.copy()
    
    # Final polish pass with soft-max and tighter tolerance
    if best_h is not None and time.time() < deadline - 5:
        polish_time = min(10.0, deadline - 5 - (time.time() - start_time))
        if polish_time > 2.0:
            h_polished, s_polished = _polish_pass(best_h, time_budget=polish_time, start_time=start_time)
            if s_polished < best_score:
                best_score = s_polished
                best_h = h_polished.copy()
                
    if best_h is None:
        # Fallback
        best_h = np.ones(N)
        
    return best_h
