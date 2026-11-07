import numpy as np
import time

N = 100

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    best_h = None
    best_val = np.inf
    
    def evaluate(h):
        if not np.all(np.isfinite(h)):
            return np.inf
        if not (np.all(h >= 0) and np.all(h <= 1)):
            return np.inf
        if abs(np.sum(h) - N / 2.0) > 1e-9 * N:
            return np.inf
        corr = np.correlate(h, 1 - h, mode="full")
        return np.max(corr) / N * 2.0

    def optimize(h_init, max_iters=5000):
        h = h_init.copy()
        eta = 0.05
        for _ in range(max_iters):
            if time.time() > deadline:
                break
            
            corr = np.correlate(h, 1 - h, mode="full")
            max_idx = np.argmax(corr)
            k = max_idx - (N - 1)
            
            # Gradient
            j_idx = np.arange(N)
            m1 = j_idx - k
            valid1 = (m1 >= 0) & (m1 < N)
            g = np.zeros(N)
            g[valid1] += (1 - h[m1[valid1]])
            
            m2 = j_idx + k
            valid2 = (m2 >= 0) & (m2 < N)
            g[valid2] -= h[m2[valid2]]
            
            g = g - np.mean(g)
            
            h_new = h - eta * g
            h_new = np.clip(h_new, 0, 1)
            
            # Normalize
            s = np.sum(h_new)
            if s > 1e-9:
                h_new = h_new * (N / 2.0 / s)
                # Re-clip
                h_new = np.clip(h_new, 0, 1)
                # If re-clip changed sum significantly, we might need to fix it
                # But for now, accept it. The next iteration's gradient will correct it.
            
            h = h_new
            
            # Check value
            val = evaluate(h)
            if val < best_val:
                best_val = val
                best_h = h.copy()
                
            # Decay eta
            eta *= 0.999
            
        return best_h, best_val

    # Restart 1: Uniform
    h1 = np.full(N, 0.5)
    h_best, val_best = optimize(h1)
    
    # Restart 2: Random
    if time.time() < deadline:
        rng = np.random.default_rng(42)
        h2 = rng.random(N)
        h2 = h2 / np.sum(h2) * (N / 2.0)
        h2 = np.clip(h2, 0, 1)
        # Fix sum
        h2 = h2 * (N / 2.0 / np.sum(h2))
        
        h_best2, val_best2 = optimize(h2)
        if val_best2 < val_best:
            val_best = val_best2
            h_best = h_best2

    # Restart 3: Binary-ish
    if time.time() < deadline:
        h3 = np.zeros(N)
        h3[:N//2] = 1.0
        # Shuffle
        rng = np.random.default_rng(123)
        h3 = rng.permutation(h3)
        # This sums to 50.
        h_best3, val_best3 = optimize(h3)
        if val_best3 < val_best:
            val_best = val_best3
            h_best = h_best3

    if best_h is None:
        return np.full(N, 0.5)
        
    # Final check
    if evaluate(best_h) > 1.0: # Fallback to seed if bad
        return np.full(N, 0.5)
        
    return best_h
