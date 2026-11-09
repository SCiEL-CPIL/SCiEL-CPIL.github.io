import numpy as np
import time

N = 400

def compute_value(f):
    """Score value: peak aperiodic autocorrelation sidelobe / (N-1). Lower is better."""
    R = np.correlate(f, f, mode='full')
    # R is of length 2N-1, centered at N-1
    # Aperiodic autocorrelation: R_aper(k) for k=0..N-1
    # np.correlate(f, f, 'full')[N-1:] gives R(0), R(1), ..., R(N-1)
    R_aper = R[N-1:]
    max_side = np.max(np.abs(R_aper[1:]))
    return max_side / (N - 1)

def legendre_sequence(p):
    """Generate Legendre sequence of prime length p."""
    f = np.empty(p, dtype=np.int8)
    f[0] = -1
    for k in range(1, p):
        f[k] = 1 if pow(k, (p-1)//2, p) == 1 else -1
    return f

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    p = 401
    seq_full = legendre_sequence(p)
    f = seq_full[:N].copy()
    
    current_value = compute_value(f)
    best_f = f.copy()
    best_value = current_value
    
    T = 0.5
    cooling = 0.999
    min_T = 0.005
    
    while time.time() < deadline and T > min_T:
        idx = np.random.randint(N)
        f[idx] *= -1
        new_value = compute_value(f)
        delta = new_value - current_value
        
        if delta < 0 or np.random.random() < np.exp(-delta / T):
            current_value = new_value
            if new_value < best_value:
                best_value = new_value
                best_f = f.copy()
        else:
            f[idx] *= -1
        
        T *= cooling
    
    return best_f
