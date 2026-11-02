"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def _full_autoconv(x):
    n = len(x)
    size = 1
    while size < 2 * n - 1:
        size *= 2
    X = np.fft.rfft(x, size)
    ac = np.fft.irfft(X * X.conj(), size)
    return ac[:2 * n - 1]

def _score(x):
    S = x.sum()
    if S == 0:
        return float('inf')
    ac = _full_autoconv(x)
    return float(np.max(np.abs(ac)) / (S * S))

def construct():
    start = time.time()
    deadline = start + 140.0
    
    best_x = np.ones(N, dtype=np.float64)
    best_val = _score(best_x)
    
    rng = random.Random(99999)
    
    # Simulated annealing
    T = 0.003
    T_min = 0.00003
    alpha = 0.9995
    
    x = best_x.copy()
    current_val = best_val
    
    steps = 0
    while time.time() < deadline:
        steps += 1
        
        i = rng.randint(0, N - 1)
        x[i] = -x[i]
        
        new_val = _score(x)
        
        if new_val < current_val:
            current_val = new_val
            if new_val < best_val:
                best_val = new_val
                best_x = x.copy()
        else:
            delta = new_val - current_val
            if delta > 0:
                if rng.random() < np.exp(-delta / max(T, 1e-15)):
                    current_val = new_val
                else:
                    x[i] = -x[i]
            else:
                current_val = new_val
        
        T = max(T * alpha, T_min)
        
        if steps % 5000 == 0 and current_val > best_val * 1.01:
            x = best_x.copy()
            current_val = best_val
            T = 0.003
    
    return best_x
# EVOLVE-BLOCK-END
