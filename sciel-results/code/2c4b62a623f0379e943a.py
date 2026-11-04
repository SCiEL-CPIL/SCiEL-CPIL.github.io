"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

N = 400

# EVOLVE-BLOCK-START

def _autocorr_peak(f):
    """Compute the peak of the autocorrelation (excluding lag 0)."""
    # Use FFT-based autocorrelation
    n = len(f)
    f_ext = np.zeros(2 * n)
    f_ext[:n] = f
    F = np.fft.fft(f_ext, n=2 * n)
    acf = np.fft.ifft(F * np.conj(F)).real[:n]
    # Exclude lag 0
    return np.max(np.abs(acf[1:]))

def _score(f):
    """Score is the upper bound on C3, which we approximate as peak autocorrelation."""
    return _autocorr_peak(f)

def _thue_morse(n):
    """Generate Thue-Morse sequence of length n, mapped to {-1, 1}."""
    seq = np.zeros(n, dtype=np.int8)
    for i in range(1, n):
        seq[i] = seq[i >> 1] ^ (i & 1)
    return 1 - 2 * seq  # Map 0->1, 1->-1

def _local_search(f, max_iters=5000, time_limit=100.0):
    """Local search: flip individual signs to minimize peak autocorrelation."""
    f = f.copy()
    current_score = _score(f)
    best_score = current_score
    best_f = f.copy()
    
    start_time = time.time()
    
    for it in range(max_iters):
        if time.time() - start_time > time_limit:
            break
        
        # Try flipping each position
        best_gain = 0
        best_pos = -1
        
        for pos in range(N):
            f[pos] = -f[pos]
            new_score = _score(f)
            gain = current_score - new_score
            if gain > best_gain:
                best_gain = gain
                best_pos = pos
            f[pos] = -f[pos]
        
        if best_pos >= 0 and best_gain > 1e-12:
            f[best_pos] = -f[best_pos]
            current_score = _score(f)
            if current_score < best_score:
                best_score = current_score
                best_f = f.copy()
        else:
            break
    
    return best_f, best_score

def _anneal_search(f, time_limit=80.0):
    """Simulated annealing to escape local minima."""
    f = f.copy().astype(np.float64)
    current_score = _score(f)
    best_score = current_score
    best_f = f.copy()
    
    start_time = time.time()
    T = 1.0
    cooling = 0.999
    
    while time.time() - start_time < time_limit:
        T *= cooling
        if T < 1e-10:
            break
        
        pos = np.random.randint(N)
        f[pos] = -f[pos]
        new_score = _score(f)
        delta = new_score - current_score
        
        if delta < 0 or np.random.random() < np.exp(-delta / max(T, 1e-10)):
            current_score = new_score
            if current_score < best_score:
                best_score = current_score
                best_f = f.copy()
        else:
            f[pos] = -f[pos]
    
    return best_f, best_score

def construct():
    """Construct a signed step function with 400 intervals to minimize autocorrelation peak."""
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit
    
    np.random.seed(42)
    
    best_f = None
    best_score = np.inf
    
    # Start with Thue-Morse sequence (known to have flat autocorrelation)
    tm = _thue_morse(N)
    tm_f = tm.astype(np.float64)
    tm_score = _score(tm_f)
    best_f = tm_f.copy()
    best_score = tm_score
    
    # Local search from Thue-Morse
    if time.time() < deadline - 30:
        f_ls, score_ls = _local_search(tm_f, max_iters=3000, time_limit=25.0)
        if score_ls < best_score:
            best_score = score_ls
            best_f = f_ls
    
    # Try random restarts
    n_restarts = 5
    time_per_restart = (deadline - time.time()) / (n_restarts + 1)
    
    for r in range(n_restarts):
        if time.time() > deadline - 15:
            break
        
        # Random initialization
        f_rand = np.random.choice([-1.0, 1.0], size=N)
        rand_score = _score(f_rand)
        
        # Local search from random start
        f_ls, score_ls = _local_search(f_rand, max_iters=2000, time_limit=max(10.0, time_per_restart - 5))
        
        if score_ls < best_score:
            best_score = score_ls
            best_f = f_ls
    
    # Final annealing pass if time remains
    if time.time() < deadline - 20 and best_f is not None:
        f_anneal, score_anneal = _anneal_search(best_f, time_limit=15.0)
        if score_anneal < best_score:
            best_score = score_anneal
            best_f = f_anneal
    
    # Ensure we return a valid result
    if best_f is None:
        best_f = tm_f
    
    return best_f

# EVOLVE-BLOCK-END
