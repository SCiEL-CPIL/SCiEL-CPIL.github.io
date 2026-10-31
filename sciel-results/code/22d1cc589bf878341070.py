import numpy as np
import time

N = 50
START_TIME = time.time()
TIME_LIMIT = 145.0

def compute_ratio(h):
    """Compute ||h||_2^2 / (||h||_1 * ||h*h||_inf) for step function."""
    h = np.asarray(h, dtype=float)
    if np.any(h < -1e-15):
        return 0.0
    h = np.maximum(0, h)
    dx = 1.0 / N
    l1 = dx * np.sum(h)
    if l1 < 1e-15:
        return 0.0
    l2_sq = dx * np.sum(h**2)
    conv = np.convolve(h, h, mode='full')
    conv_scaled = conv * dx
    conv_inf = np.max(conv_scaled)
    if conv_inf < 1e-15:
        return 0.0
    return l2_sq / (l1 * conv_inf)

def construct():
    np.random.seed(42)
    best_val = 0.0
    best_h = np.ones(N)  # default
    
    # Try single spike
    for i in range(N):
        h = np.zeros(N)
        h[i] = 1.0
        val = compute_ratio(h)
        if val > best_val:
            best_val = val
            best_h = h.copy()
    
    elapsed = time.time() - START_TIME
    if elapsed > TIME_LIMIT - 60:
        return [float(x) for x in best_h]
    
    # Try two-point solutions with various separations
    for i in range(N):
        for j in range(i + 1, N):
            if time.time() - START_TIME > TIME_LIMIT - 30:
                break
            # Optimize the ratio a/b for two spikes
            best_local = 0.0
            best_a_frac = 0.5
            for a_frac in np.linspace(0.01, 0.99, 100):
                h = np.zeros(N)
                h[i] = a_frac
                h[j] = 1.0 - a_frac
                val = compute_ratio(h)
                if val > best_local:
                    best_local = val
                    best_a_frac = a_frac
            if best_local > best_val:
                best_val = best_local
                best_h = np.zeros(N)
                best_h[i] = best_a_frac
                best_h[j] = 1.0 - best_a_frac
    
    elapsed = time.time() - START_TIME
    if elapsed > TIME_LIMIT - 15:
        return [float(x) for x in best_h]
    
    # Try three-point solutions for nearby triples
    for i in range(N):
        for d1 in range(1, 8):
            for d2 in range(1, 8):
                j = i + d1
                k = j + d2
                if k >= N:
                    continue
                if time.time() - START_TIME > TIME_LIMIT - 5:
                    break
                best_local = 0.0
                best_abc = (1/3, 1/3, 1/3)
                for a in np.linspace(0.02, 0.96, 30):
                    for b in np.linspace(0.02, 0.96 - a, 30):
                        c = 1.0 - a - b
                        if c < 0.02:
                            continue
                        h = np.zeros(N)
                        h[i] = a
                        h[j] = b
                        h[k] = c
                        val = compute_ratio(h)
                        if val > best_local:
                            best_local = val
                            best_abc = (a, b, c)
                if best_local > best_val:
                    best_val = best_local
                    best_h = np.zeros(N)
                    best_h[i] = best_abc[0]
                    best_h[j] = best_abc[1]
                    best_h[k] = best_abc[2]
    
    return [float(x) for x in best_h]
