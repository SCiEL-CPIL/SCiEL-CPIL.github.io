"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time

def _autocorr_max_abs(s):
    """Compute max |autocorrelation| over non-zero lags for a length-N ±1 sequence."""
    n_fft = 1
    while n_fft < 2 * N:
        n_fft *= 2
    s_f = s.astype(np.float64)
    F = np.fft.rfft(s_f, n=n_fft)
    acf = np.fft.irfft(F * np.conj(F), n=n_fft)[:N]
    # acf[k] = sum_{i=0}^{N-1-k} s[i]*s[i+k] for k=0..N-1
    if N <= 1:
        return 0.0
    return float(np.max(np.abs(acf[1:])))

def _local_search(s, max_time=5.0):
    """Greedy local search: flip individual elements to reduce max |autocorr|."""
    s = s.copy()
    current_val = _autocorr_max_abs(s)
    start_time = time.time()
    
    while time.time() - start_time < max_time:
        improved = False
        # Try flipping each element
        for i in range(N):
            s_old = s[i]
            s[i] = -s_old
            new_val = _autocorr_max_abs(s)
            if new_val < current_val:
                current_val = new_val
                improved = True
            else:
                s[i] = s_old
        if not improved:
            break
    return s, current_val

def _simulated_annealing(s, max_time=10.0):
    """Simulated annealing to escape local minima."""
    s = s.copy()
    current_val = _autocorr_max_abs(s)
    best_s = s.copy()
    best_val = current_val
    start_time = time.time()
    T = 1.0
    cooling = 0.999
    
    while time.time() - start_time < max_time and T > 0.01:
        # Pick a random position to flip
        i = np.random.randint(N)
        s[i] = -s[i]
        new_val = _autocorr_max_abs(s)
        delta = new_val - current_val
        if delta < 0 or np.random.random() < np.exp(-delta / T):
            current_val = new_val
            if new_val < best_val:
                best_val = new_val
                best_s = s.copy()
        else:
            s[i] = -s[i]  # revert
        T *= cooling
    
    return best_s, best_val

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10s safety margin before 120s limit
    
    # Try multiple initial structures
    candidates = []
    
    # 1. All ones (baseline)
    candidates.append(np.ones(N))
    
    # 2. Alternating
    alt = np.ones(N)
    alt[1::2] = -1
    candidates.append(alt)
    
    # 3. Sine-based sign pattern (low-frequency)
    x = np.linspace(-0.25, 0.25, N, endpoint=False)
    for freq in [1, 2, 3, 4, 5, 10, 20]:
        sine_sign = np.sign(np.sin(2 * np.pi * freq * x))
        # Handle zeros (should be rare)
        sine_sign[sine_sign == 0] = 1
        candidates.append(sine_sign)
    
    # 4. Random initializations
    rng = np.random.default_rng(42)
    for _ in range(10):
        candidates.append(rng.choice([-1.0, 1.0], size=N))
    
    # Evaluate all candidates
    best_s = None
    best_val = float('inf')
    
    for cand in candidates:
        val = _autocorr_max_abs(cand)
        if val < best_val:
            best_val = val
            best_s = cand.copy()
    
    if best_s is None:
        return np.ones(N)
    
    # Check time
    if time.time() > deadline:
        return best_s
    
    # Local search from best candidate
    best_s, best_val = _local_search(best_s, max_time=min(20.0, deadline - time.time()))
    
    # Check time
    if time.time() > deadline:
        return best_s
    
    # Simulated annealing to escape local minima
    remaining_time = deadline - time.time()
    if remaining_time > 5.0:
        sa_s, sa_val = _simulated_annealing(best_s, max_time=min(remaining_time - 2.0, 30.0))
        if sa_val < best_val:
            best_s = sa_s
            best_val = sa_val
    
    # Final local search polish
    remaining_time = deadline - time.time()
    if remaining_time > 2.0:
        best_s, best_val = _local_search(best_s, max_time=min(remaining_time - 1.0, 10.0))
    
    return best_s
# EVOLVE-BLOCK-END
