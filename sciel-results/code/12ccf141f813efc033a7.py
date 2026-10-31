import numpy as np
import time

N = 50

def _compute_ratio(f):
    """
    Computes the ratio ||f||_2^2 / (||f||_1 * ||f*f||_inf).
    Assumes f is non-negative.
    """
    f = np.asarray(f, dtype=float)
    if np.any(f < -1e-12):
        return -1e9
    f = np.maximum(f, 0.0)
    l1 = np.sum(f)
    if l1 < 1e-12:
        return -1e9
    l2_sq = np.sum(f**2)
    conv = np.convolve(f, f, mode='full')
    conv_max = np.max(conv)
    if conv_max < 1e-12:
        return -1e9
    return l2_sq / (l1 * conv_max)

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    # Initialize with several diverse candidates
    candidates = []
    
    # 1. Uniform
    candidates.append(np.ones(N) / N)
    
    # 2. Delta (Sparse)
    for i in range(N):
        d = np.zeros(N)
        d[i] = 1.0
        candidates.append(d)
        
    # 3. Random Sparse
    rng = np.random.default_rng(42)
    for _ in range(20):
        mask = np.zeros(N, dtype=bool)
        k = rng.integers(1, N//2 + 1)
        indices = rng.choice(N, k, replace=False)
        mask[indices] = True
        vals = rng.random(k)
        f = np.zeros(N)
        f[indices] = vals
        f = f / np.sum(f)
        candidates.append(f)
        
    # 4. Gaussian
    center = N / 2 - 0.5
    for sigma in [1, 2, 5, 10, 20]:
        g = np.exp(-0.5 * ((np.arange(N) - center) / sigma)**2)
        g = g / np.sum(g)
        candidates.append(g)
        
    # 5. Sidon-like (low autocorrelation)
    # Try to place 1s at positions with minimal overlap
    # Simple heuristic: random permutation of a subset
    for _ in range(5):
        k = 10
        indices = rng.choice(N, k, replace=False)
        f = np.zeros(N)
        f[indices] = 1.0
        f = f / np.sum(f)
        candidates.append(f)

    best_f = None
    best_ratio = -1e9
    
    for c in candidates:
        c = np.maximum(c, 0)
        s = np.sum(c)
        if s < 1e-12: continue
        c = c / s
        r = _compute_ratio(c)
        if r > best_ratio:
            best_ratio = r
            best_f = c.copy()
            
    if best_f is None:
        return [1.0/N] * N

    # Local Optimization: Coordinate Descent + Random Perturbation
    current_f = best_f.copy()
    current_ratio = best_ratio
    
    max_iters = 20000
    no_improve = 0
    max_no_improve = 1000
    
    # Adaptive step size
    step_size = 0.1
    
    for i in range(max_iters):
        if time.time() > deadline - 10:
            break
            
        # Randomly pick two indices to swap mass
        idx = rng.integers(0, N)
        jdx = rng.integers(0, N)
        if idx == jdx:
            continue
            
        # Try moving mass from jdx to idx
        # Also try different fractions
        improved = False
        for frac in [0.1, 0.05, 0.2, 0.01, 0.5]:
            transfer = current_f[jdx] * frac
            if transfer < 1e-10:
                continue
                
            new_f = current_f.copy()
            new_f[idx] += transfer
            new_f[jdx] -= transfer
            
            if new_f[jdx] < 0:
                new_f[idx] += -new_f[jdx]
                new_f[jdx] = 0
                
            s = np.sum(new_f)
            if s < 1e-12:
                continue
            new_f = new_f / s
            
            r = _compute_ratio(new_f)
            if r > current_ratio + 1e-9:
                current_f = new_f
                current_ratio = r
                improved = True
                if r > best_ratio:
                    best_ratio = r
                    best_f = new_f.copy()
                break
        if not improved:
            # Try reverse
            transfer = current_f[idx] * 0.1
            new_f = current_f.copy()
            new_f[jdx] += transfer
            new_f[idx] -= transfer
            if new_f[idx] < 0:
                new_f[jdx] += -new_f[idx]
                new_f[idx] = 0
            s = np.sum(new_f)
            if s > 1e-12:
                new_f = new_f / s
                r = _compute_ratio(new_f)
                if r > current_ratio + 1e-9:
                    current_f = new_f
                    current_ratio = r
                    improved = True
                    if r > best_ratio:
                        best_ratio = r
                        best_f = new_f.copy()
                        
        if improved:
            no_improve = 0
        else:
            no_improve += 1
            
        # If stuck, perturb
        if no_improve > max_no_improve:
            perturbation = rng.normal(0, 0.1, N)
            new_f = np.maximum(current_f + perturbation, 0)
            s = np.sum(new_f)
            if s > 1e-12:
                new_f = new_f / s
                r = _compute_ratio(new_f)
                if r > current_ratio:
                    current_f = new_f
                    current_ratio = r
                    if r > best_ratio:
                        best_ratio = r
                        best_f = new_f.copy()
            no_improve = 0

    # Final Polish
    if time.time() < deadline - 5:
        polish_f = best_f.copy()
        polish_ratio = _compute_ratio(polish_f)
        for _ in range(10000):
            if time.time() > deadline - 2:
                break
            idx = rng.integers(0, N)
            jdx = rng.integers(0, N)
            if idx == jdx: continue
            for frac in [0.01, 0.001, 0.05]:
                transfer = polish_f[jdx] * frac
                new_f = polish_f.copy()
                new_f[idx] += transfer
                new_f[jdx] -= transfer
                if new_f[jdx] < 0:
                    new_f[idx] += -new_f[jdx]
                    new_f[jdx] = 0
                s = np.sum(new_f)
                if s < 1e-12: continue
                new_f = new_f / s
                r = _compute_ratio(new_f)
                if r > polish_ratio + 1e-9:
                    polish_f = new_f
                    polish_ratio = r
                    if r > best_ratio:
                        best_ratio = r
                        best_f = new_f.copy()
                    break

    best_f = np.maximum(best_f, 0.0)
    s = np.sum(best_f)
    if s > 0:
        best_f = best_f / s
        
    return [float(x) for x in best_f]
