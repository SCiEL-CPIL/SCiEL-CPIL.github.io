import numpy as np
import time
from scipy.optimize import minimize

N = 50

def compute_ratio(h):
    """
    Compute the ratio ||h||_2^2 / (||h||_1 * ||h*h||_inf) for h >= 0.
    """
    h = np.asarray(h, dtype=np.float64)
    # Ensure non-negativity
    h = np.maximum(h, 0.0)
    
    l1 = np.sum(h)
    if l1 < 1e-300:
        return 0.0
    
    l2_sq = np.sum(h * h)
    
    # Efficient autoconvolution using FFT
    # h*h has length 2N-1
    # Zero pad to power of 2 for FFT efficiency
    n_fft = 1 << (2 * N - 1).bit_length()
    H = np.fft.rfft(h, n=n_fft)
    hh = np.fft.irfft(H * H, n=n_fft)
    hh_max = np.max(hh)
    
    if hh_max < 1e-300:
        return 0.0
        
    return l2_sq / (l1 * hh_max)

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    rng = np.random.default_rng(42)
    
    best_h = None
    best_score = -np.inf
    
    # Strategy:
    # 1. Use scale invariance: we can fix ||h||_1 = 1 to simplify the search space.
    #    The ratio is invariant to scaling.
    # 2. Instead of discrete support search, we optimize the full 50-dimensional vector
    #    using L-BFGS-B, which handles non-negativity constraints natively.
    # 3. We use multiple diverse initializations to avoid local optima:
    #    - Uniform
    #    - Geometric (decaying)
    #    - Sinusoidal
    #    - Sparse Sidon-like (randomly selected indices with optimized weights)
    # 4. For each initialization, we run L-BFGS-B on the log-sum-exp transformed variables
    #    or simply on the raw variables with bounds [0, inf], fixing the L1 norm via
    #    normalization in the objective (or just maximizing the ratio directly).
    
    # Objective function for L-BFGS-B
    # We maximize R(h) = ||h||_2^2 / (||h||_1 * ||h*h||_inf)
    # We minimize -R(h)
    
    def neg_ratio(h):
        val = compute_ratio(h)
        return -val
    
    # We will run multiple L-BFGS-B optimizations from different starting points.
    # Each run has a time budget.
    
    # Generate diverse initializations
    inits = []
    
    # 1. Uniform
    inits.append(np.ones(N))
    
    # 2. Geometric
    for ratio in [0.9, 0.8, 1.1, 1.2]:
        g = np.array([ratio**i for i in range(N)])
        inits.append(g)
        inits.append(g[::-1].copy())
        
    # 3. Sinusoidal
    for freq in [1, 2, 3]:
        for phase in [0, np.pi/2, np.pi]:
            s = np.abs(np.sin(np.pi * freq * np.arange(N) / N + phase))
            inits.append(s)
            
    # 4. Sparse Sidon-like: randomly pick K indices, set them to 1, others 0
    for K in [5, 10, 15, 20]:
        for _ in range(5):
            idx = rng.choice(N, size=K, replace=False)
            s = np.zeros(N)
            s[idx] = 1.0
            inits.append(s)
            
    # 5. Random Gaussian (clipped)
    for _ in range(5):
        r = np.abs(rng.normal(0, 1, N))
        inits.append(r)
        
    # Shuffle inits
    rng.shuffle(inits)
    
    # Run L-BFGS-B on each init with a time limit
    # We need to break out of L-BFGS-B if time runs out.
    # scipy.optimize.minimize does not have a direct time limit, but we can use a callback.
    
    def time_callback(xk):
        if time.time() > deadline - 5.0:
            return True
        return False
    
    for init in inits:
        if time.time() > deadline - 10.0:
            break
            
        # Normalize init to L1=1 for consistency, though not strictly required
        s = np.sum(init)
        if s > 0:
            init_norm = init / s
        else:
            init_norm = init
            
        # Run L-BFGS-B
        try:
            res = minimize(
                neg_ratio, 
                init_norm, 
                method='L-BFGS-B', 
                bounds=[(0, None)] * N,
                options={'maxiter': 1000, 'maxfun': 10000, 'callback': time_callback, 'disp': False}
            )
            
            if res.success or res.fun < best_score:
                # L-BFGS-B minimizes, so res.fun is -ratio
                # We need to recompute the ratio to be safe
                candidate_h = np.maximum(res.x, 0)
                score = compute_ratio(candidate_h)
                if score > best_score:
                    best_score = score
                    best_h = candidate_h.copy()
        except Exception:
            continue
            
        # Also try a local polish with more iterations if time permits
        if best_h is not None and time.time() < deadline - 20.0:
            # Use the best found so far as a new init for a longer run?
            # Or just let the loop continue.
            pass
            
    # If no valid result, return uniform
    if best_h is None:
        best_h = np.ones(N) / N
        
    return list(best_h)

if __name__ == "__main__":
    h = construct()
    print(compute_ratio(h))
