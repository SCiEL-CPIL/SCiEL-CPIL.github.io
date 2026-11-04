"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time

def _objective(x):
    # Compute circular autocorrelation
    f = np.fft.fft(x, n=2*N)
    r = np.fft.ifft(f * np.conj(f))
    # Sidelobes are indices 1..N-1 and N+1..2N-1
    # Due to symmetry, we check 1..N-1
    return np.max(np.abs(r[1:N]))

def _local_search(x, deadline, max_passes=20):
    x = x.copy()
    current_obj = _objective(x)
    improved = True
    passes = 0
    while improved and passes < max_passes:
        if time.time() > deadline:
            break
        improved = False
        passes += 1
        for i in range(N):
            if time.time() > deadline:
                break
            old_val = x[i]
            x[i] = -old_val
            new_obj = _objective(x)
            if new_obj < current_obj:
                current_obj = new_obj
                improved = True
            else:
                x[i] = old_val
    return x, current_obj

def construct():
    start_time = time.time()
    deadline = start_time + 115.0
    
    best_x = None
    best_obj = float('inf')
    
    # Quasi-periodic initializations
    alpha_candidates = [
        (np.sqrt(5) - 1) / 2, # Golden ratio
        np.pi / 2,
        np.e,
        np.sqrt(2),
        np.sqrt(3),
        np.log(2)
    ]
    
    for alpha in alpha_candidates:
        if time.time() > deadline:
            break
        for phase in range(5):
            if time.time() > deadline:
                break
            phi = np.random.uniform(0, 2*np.pi)
            t = np.arange(N)
            vals = np.cos(2 * np.pi * alpha * t + phi)
            x = np.sign(vals)
            x[x == 0] = 1
            
            x_opt, obj = _local_search(x, deadline)
            
            if obj < best_obj:
                best_obj = obj
                best_x = x_opt
                
    # Random initializations if time permits
    if time.time() < deadline - 10.0:
        for _ in range(5):
            if time.time() > deadline:
                break
            x = np.random.choice([-1, 1], N)
            x_opt, obj = _local_search(x, deadline)
            if obj < best_obj:
                best_obj = obj
                best_x = x_opt

    if best_x is None:
        return np.ones(N)
        
    return best_x

# EVOLVE-BLOCK-END
