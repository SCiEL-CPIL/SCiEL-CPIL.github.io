import numpy as np
import time

N = 400
# Precompute FFT size for full autocorrelation
n_fft = 1
while n_fft < 2 * N:
    n_fft *= 2

def _calc_acf(s):
    """Compute full autocorrelation array A[0..N-1]"""
    s_f = s.astype(np.float64)
    F = np.fft.rfft(s_f, n=n_fft)
    P = F * np.conj(F)
    acf = np.fft.irfft(P, n=n_fft)
    return np.real(acf[:N])

def _max_abs_acf_from_array(A):
    return float(np.max(np.abs(A[1:])))

def _acf_delta_flip(s, i, A):
    """
    Compute the delta in the autocorrelation array A when flipping s[i] -> -s[i].
    Returns the new A array.
    A_new[k] = A[k] - 2*s[i]*(s[i+k] + s[i-k])
    """
    si = s[i]
    if si == 0:
        return A
    
    k = np.arange(N)
    p1 = i + k
    p2 = i - k
    
    valid1 = p1 < N
    valid2 = p2 >= 0
    
    contrib = np.zeros(N, dtype=np.float64)
    contrib[valid1] += s[p1[valid1]]
    contrib[valid2] += s[p2[valid2]]
    
    D = -2.0 * si * contrib
    return A + D

def _fast_local_search(s, current_val, time_limit, deadline=None):
    """
    Fast local search using O(N) delta updates.
    """
    start = time.time()
    A = _calc_acf(s)
    current_val = float(np.max(np.abs(A[1:])))
    
    improved = True
    passes = 0
    max_passes = 10000 # High limit, time is the constraint
    
    while improved and passes < max_passes:
        if time.time() - start > time_limit:
            break
        if deadline is not None and time.time() > deadline:
            break
            
        improved = False
        passes += 1
        indices = np.random.permutation(N)
        
        for i in indices:
            if deadline is not None and time.time() > deadline:
                break
                
            # Calculate new A
            A_new = _acf_delta_flip(s, i, A)
            val_new = float(np.max(np.abs(A_new[1:])))
            
            if val_new < current_val - 1e-12:
                A = A_new
                s[i] = -s[i]
                current_val = val_new
                improved = True
    
    return s, current_val

def _ising_cooling_init(n_iter=200, deadline=None):
    """
    Spin-glass cooling schedule for initialization.
    """
    x = np.random.uniform(-1, 1, size=N)
    
    T_start = 0.1
    T_end = 0.001
    cooling_rate = (T_end / T_start) ** (1.0 / n_iter)
    T = T_start
    
    for it in range(n_iter):
        if deadline is not None and time.time() > deadline:
            break
            
        A = _calc_acf(x)
        A_for_grad = np.zeros(N)
        A_for_grad[1:] = A[1:]
        
        Fx = np.fft.rfft(x, n=n_fft)
        FA = np.fft.rfft(A_for_grad, n=n_fft)
        grad = np.fft.irfft(4 * Fx * FA, n=n_fft)[:N]
        
        noise = np.random.normal(0, T * 0.1, size=N)
        x_new = x - 0.01 * grad + noise
        x_new = np.clip(x_new, -1, 1)
        x = x_new
        T *= cooling_rate
        
    s = np.where(x >= 0, 1.0, -1.0)
    return s

def construct():
    start_time = time.time()
    # Evaluator kills at 150s. We should aim to finish well before.
    # Let's set a hard deadline at 145s to be safe.
    deadline = start_time + 145.0
    
    best_val = float('inf')
    best_seq = np.ones(N)
    
    # Strategy 1: Multiple initializations with intensive local search
    # The cell findings suggest greedy coordinate descent is key.
    # We will use Ising cooling to get a decent start, then hammer it with local search.
    
    num_restarts = 20
    
    for r in range(num_restarts):
        if time.time() > deadline - 10.0:
            break
            
        # Initialize
        s = _ising_cooling_init(n_iter=100, deadline=deadline)
        
        # Initial local search to settle
        s, current_val = _fast_local_search(s, float('inf'), time_limit=2.0, deadline=deadline)
        
        if current_val < best_val:
            best_val = current_val
            best_seq = s.copy()
            
        # If we have found a very good solution, we might spend more time polishing it
        # But for now, let's keep restarting to find better basins.
        
    # Strategy 2: Final Polish on the best found sequence
    remaining = deadline - time.time()
    if remaining > 5.0:
        s = best_seq.copy()
        # Use a longer time budget for the final polish
        s, current_val = _fast_local_search(s, best_val, time_limit=remaining - 2.0, deadline=deadline)
        if current_val < best_val:
            best_val = current_val
            best_seq = s
    
    return best_seq
