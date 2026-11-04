"""
Strategy: Fast simulated annealing with vectorized peak evaluation.

The previous approach used a greedy single-flip heuristic which can get stuck in
local minima. This version uses simulated annealing with:
1. Fast FFT-based autocorrelation computation
2. Vectorized evaluation of flip candidates
3. Geometric cooling schedule
4. Multiple restarts from different initializations
5. Time-bounded search that returns the best solution found

The key insight is that simulated annealing can escape local minima by
accepting worse moves with decreasing probability, while the FFT-based
autocorrelation computation allows evaluating many candidates quickly.
"""

import numpy as np
import time

N = 400

def _thue_morse(n):
    """Generate Thue-Morse sequence of length n, mapped to ±1."""
    seq = np.ones(n, dtype=np.float64)
    for i in range(n):
        count = bin(i).count('1')
        if count % 2 == 1:
            seq[i] = -1.0
    return seq

def _compute_autocorr(f):
    """Compute linear autocorrelation R[k] for k=0..N-1 using FFT."""
    n = len(f)
    size = 1
    while size < 2 * n:
        size *= 2
    F = np.fft.rfft(f, size)
    R = np.fft.irfft(F * np.conj(F), size)
    return R[:n]

def _peak_value(R):
    """Return max absolute autocorrelation for non-zero lags."""
    if len(R) <= 1:
        return 0.0
    return float(np.max(np.abs(R[1:])))

def _eval_single_flip(f, R, i):
    """
    Evaluate the peak if we flip position i.
    Returns (new_peak, new_R) if flip is accepted, or (None, None) if not computed.
    """
    n = len(f)
    # Delta[k] = -2 * f[i] * (f[i+k] + f[i-k]) for valid indices
    k = np.arange(n)
    i_plus_k = i + k
    i_minus_k = i - k
    
    f_i_plus_k = np.where((i_plus_k >= 0) & (i_plus_k < n), f[i_plus_k], 0.0)
    f_i_minus_k = np.where((i_minus_k >= 0) & (i_minus_k < n), f[i_minus_k], 0.0)
    
    delta = -2.0 * f[i] * (f_i_plus_k + f_i_minus_k)
    new_R = R + delta
    new_peak = float(np.max(np.abs(new_R[1:])))
    return new_peak, new_R

def _simulated_annealing(f_init, start_time, deadline, rng):
    """
    Perform simulated annealing starting from f_init.
    Returns the best f found.
    """
    f = f_init.copy()
    R = _compute_autocorr(f)
    current_peak = _peak_value(R)
    
    best_f = f.copy()
    best_peak = current_peak
    
    T = 2.0  # Initial temperature
    T_min = 1e-6
    cooling = 0.9995
    
    while T > T_min:
        if time.time() > deadline:
            break
        
        # Randomly select a position to flip
        i = rng.randint(N)
        
        new_peak, new_R = _eval_single_flip(f, R, i)
        
        delta = new_peak - current_peak
        
        if delta < 0 or rng.random() < np.exp(-delta / T):
            # Accept the move
            f[i] = -f[i]
            R = new_R
            current_peak = new_peak
            
            if current_peak < best_peak:
                best_peak = current_peak
                best_f = f.copy()
        
        T *= cooling
    
    return best_f, best_peak

def construct():
    start_time = time.time()
    deadline = start_time + 118.0  # Safety margin before 150s kill
    
    best_f = None
    best_peak = float('inf')
    
    # Try multiple initializations
    inits = []
    
    # 1. Thue-Morse
    tm = _thue_morse(N)
    inits.append(tm)
    inits.append(-tm)  # Flip polarity
    
    # 2. Random sequences
    rng = np.random.RandomState(42)
    for _ in range(6):
        r = rng.choice([-1.0, 1.0], size=N)
        inits.append(r)
    
    # 3. Legendre-like sequence
    def is_prime(num):
        if num < 2:
            return False
        for i in range(2, int(num**0.5) + 1):
            if num % i == 0:
                return False
        return True
    
    p = N
    while not is_prime(p):
        p += 1
    
    leg = np.ones(N, dtype=np.float64)
    for i in range(N):
        val = pow((i + 1) % p, (p - 1) // 2, p)
        if val == p - 1:
            leg[i] = -1.0
    inits.append(leg)
    inits.append(-leg)
    
    # 4. Chaotic logistic map sequence
    # Generate using logistic map with parameter near edge of chaos
    for mu in [3.99, 3.999, 4.0]:
        x = 0.5
        seq = np.zeros(N)
        for i in range(N):
            x = mu * x * (1 - x)
            seq[i] = 1.0 if x > 0.5 else -1.0
        inits.append(seq)
    
    for init_f in inits:
        if time.time() > deadline - 3.0:
            break
        
        rng_sa = np.random.RandomState(int(time.time() * 1000) % (2**31))
        f_result, peak_result = _simulated_annealing(
            init_f, start_time, deadline, rng_sa
        )
        
        if peak_result < best_peak:
            best_peak = peak_result
            best_f = f_result.copy()
    
    if best_f is None:
        # Fallback: return Thue-Morse
        best_f = _thue_morse(N)
    
    return best_f
