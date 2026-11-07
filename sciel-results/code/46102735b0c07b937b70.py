import numpy as np
import time

N = 100

# EVOLVE-BLOCK-START
import numpy as np

def _value(h):
    g = 1.0 - h
    corr = np.correlate(h, g, mode='full')
    return np.max(corr) / N * 2.0

def construct():
    deadline = time.time() + 110
    
    best_h = np.full(N, 0.5)
    best_val = _value(best_h)
    
    # Random binary search
    rng = np.random.RandomState(42)
    for _ in range(50000):
        if time.time() > deadline:
            break
        h = np.zeros(N)
        h[rng.choice(N, size=50, replace=False)] = 1.0
        val = _value(h)
        if val < best_val:
            best_val = val
            best_h = h.copy()
    
    # Local search on best
    h = best_h.copy()
    for iteration in range(200):
        if time.time() > deadline:
            break
        improved = False
        ones_idx = np.where(h == 1.0)[0]
        zeros_idx = np.where(h == 0.0)[0]
        for oi in ones_idx:
            if time.time() > deadline:
                break
            for zi in zeros_idx:
                h[oi] = 0.0
                h[zi] = 1.0
                val = _value(h)
                if val < best_val:
                    best_val = val
                    best_h = h.copy()
                    improved = True
                else:
                    h[oi] = 1.0
                    h[zi] = 0.0
        if not improved:
            break
    
    return best_h
# EVOLVE-BLOCK-END
