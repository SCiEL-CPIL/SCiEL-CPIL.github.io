import numpy as np
import time

N = 400
_FFT_SIZE = 1
while _FFT_SIZE < 2 * N:
    _FFT_SIZE *= 2

def _thue_morse(n):
    """Generate the Thue-Morse sequence of length n as a numpy array of +/-1."""
    t = np.array([1.0])
    while len(t) < n:
        t = np.concatenate([t, -t])
    return t[:n]

def _delta_for_flip(h, i):
    """
    Compute the change in autocorrelation A[k] when flipping h[i].
    A[k] = sum_j h[j] h[j+k]
    Flipping h[i] changes A[k] by -2*h[i]*(h[i+k] + h[i-k]) (with boundary handling).
    Returns an array delta of length N such that A_new = A + delta.
    """
    hi = h[i]
    # Vectorized construction of the delta terms
    # Term 1: -2*hi*h[i+k] for k=0..N-1-i -> corresponds to h[i:]
    # Term 2: -2*hi*h[i-k] for k=0..i     -> corresponds to h[i::-1]
    
    # We need to align these into a single array of length N.
    # delta[k] = -2*hi*(h[i+k] if i+k<N else 0 + h[i-k] if i-k>=0 else 0)
    
    delta = np.zeros(N)
    
    # Part 1: k ranges from 0 to N-1-i
    # h[i+k] is h[i], h[i+1], ..., h[N-1]
    # This maps to delta[0], delta[1], ..., delta[N-1-i]
    if N - i > 0:
        delta[:N-i] -= 2.0 * hi * h[i:]
        
    # Part 2: k ranges from 0 to i
    # h[i-k] is h[i], h[i-1], ..., h[0]
    # This maps to delta[0], delta[1], ..., delta[i]
    if i >= 0:
        delta[:i+1] -= 2.0 * hi * h[i::-1]
        
    return delta

def _compute_acf(h):
    """Compute aperiodic autocorrelation ac[0..N-1] via FFT."""
    H = np.fft.fft(h, _FFT_SIZE)
    ac = np.fft.ifft(H * np.conj(H)).real[:N]
    return ac

def _peak_sidelobe(h):
    """max_{k>=1} |sum_i h[i] h[i+k]|."""
    H = np.fft.fft(h, _FFT_SIZE)
    ac = np.fft.ifft(H * np.conj(H)).real[:N]
    return float(np.max(np.abs(ac[1:])))

def _anneal_logsumexp(seed, time_budget, init_h):
    """
    Simulated annealing using a log-sum-exp surrogate for the peak objective.
    This smooths the landscape, helping to escape local minima.
    """
    rng = np.random.default_rng(seed)
    h = init_h.copy()
    ac = _compute_acf(h)
    a = np.abs(ac[1:])
    
    # Initial surrogate value
    m0 = float(np.max(a))
    beta = 5.0
    cur_surrogate = m0 + np.log(np.sum(np.exp(beta * a - beta * m0))) / beta
    cur_exact = m0
    
    best_h = h.copy()
    best_exact = cur_exact
    
    T = 1.0
    T_min = 1e-9
    start = time.time()
    
    # Estimate steps based on budget
    # Each step is O(N) vector ops + overhead. ~10k steps/sec is realistic.
    total_steps = max(1000, int(time_budget * 8000))
    
    for step in range(total_steps):
        if time.time() - start >= time_budget:
            break
            
        i = int(rng.integers(0, N))
        d = _delta_for_flip(h, i)
        
        # Update ACF
        ac_new = ac + d
        a_new = np.abs(ac_new[1:])
        
        # Compute new surrogate
        m_new = float(np.max(a_new))
        # Numerical stability for log-sum-exp
        # s = sum(exp(beta * (a_new - m_new)))
        # surrogate = m_new + log(s) / beta
        beta_curr = min(100.0, beta) # Cap beta to avoid overflow
        shifted = beta_curr * (a_new - m_new)
        s = float(np.sum(np.exp(shifted)))
        new_surrogate = m_new + np.log(s) / beta_curr
        
        delta_surrogate = new_surrogate - cur_surrogate
        
        # Acceptance criterion
        accept = False
        if delta_surrogate <= 0.0:
            accept = True
        else:
            # Probability of accepting worse move
            # Use current T
            prob = np.exp(-delta_surrogate / max(T, 1e-9))
            if rng.random() < prob:
                accept = True
        
        if accept:
            ac = ac_new
            h[i] *= -1.0
            cur_surrogate = new_surrogate
            
            # Track best exact value
            if m_new < best_exact:
                best_exact = m_new
                best_h = h.copy()
        else:
            # Revert (we didn't change h in place yet, so nothing to do)
            pass
        
        # Cooling
        T = max(T_min, T * (0.9999 if step < total_steps // 2 else 0.99998))
        # Increase beta to sharpen the surrogate
        beta = min(100.0, beta * 1.001)
        
    return best_h, best_exact

def _greedy_polish(h, time_budget):
    """
    Deterministic local search: repeatedly scan all indices and flip if it improves the peak.
    """
    h = h.copy()
    ac = _compute_acf(h)
    cur = float(np.max(np.abs(ac[1:])))
    best = cur
    best_h = h.copy()
    start = time.time()
    
    while time.time() - start < time_budget:
        improved = False
        # Random order to avoid bias in deterministic passes
        indices = np.random.permutation(N)
        
        for i in indices:
            if time.time() - start >= time_budget:
                break
            
            d = _delta_for_flip(h, i)
            ac_try = ac + d
            c_try = float(np.max(np.abs(ac_try[1:])))
            
            if c_try < cur - 1e-12:
                ac = ac_try
                cur = c_try
                h[i] *= -1.0
                improved = True
                if cur < best:
                    best = cur
                    best_h = h.copy()
        
        if not improved:
            break
            
    return best_h, best

def construct():
    try:
        # Hard deadline to stay within 120s limit with safety margin
        hard_deadline = 115.0
        start = time.time()
        
        # Initial best: Thue-Morse sequence (known good structure)
        global_best_h = _thue_morse(N)
        global_best = _peak_sidelobe(global_best_h)
        
        # Also try all-ones (though usually worse)
        ones_h = np.ones(N)
        ones_val = _peak_sidelobe(ones_h)
        if ones_val < global_best:
            global_best = ones_val
            global_best_h = ones_h
            
        # Phase 1: Log-Sum-Exp Annealing from diverse starts
        # We have ~100s for this phase.
        # Let's run a few restarts.
        
        # Start 1: Thue-Morse
        budget1 = min(40.0, hard_deadline - (time.time() - start) - 30.0)
        if budget1 > 1.0:
            h1, v1 = _anneal_logsumexp(12345, budget1, global_best_h)
            if v1 < global_best:
                global_best = v1
                global_best_h = h1
        
        # Start 2: Random
        rng_init = np.random.default_rng(42)
        h_rand = rng_init.choice([-1.0, 1.0], size=N)
        budget2 = min(30.0, hard_deadline - (time.time() - start) - 20.0)
        if budget2 > 1.0:
            h2, v2 = _anneal_logsumexp(54321, budget2, h_rand)
            if v2 < global_best:
                global_best = v2
                global_best_h = h2
                
        # Start 3: Another random
        h_rand2 = rng_init.choice([-1.0, 1.0], size=N)
        budget3 = min(20.0, hard_deadline - (time.time() - start) - 10.0)
        if budget3 > 1.0:
            h3, v3 = _anneal_logsumexp(99999, budget3, h_rand2)
            if v3 < global_best:
                global_best = v3
                global_best_h = h3

        # Phase 2: Greedy Polish
        remaining = hard_deadline - (time.time() - start)
        if remaining > 5.0:
            h_polished, v_polished = _greedy_polish(global_best_h, remaining - 2.0)
            if v_polished < global_best:
                global_best = v_polished
                global_best_h = h_polished
                
        return global_best_h
    except Exception:
        return np.ones(N)
