import numpy as np
import time
from scipy.optimize import minimize, differential_evolution

N = 50
w = 1.0 / 100.0  # spacing on [-0.5, 0.5] with m+2=102 points

def compute_value_fast(h):
    h = np.asarray(h, dtype=float)
    g = np.convolve(h, h)
    m = len(g)  # 99
    # l2 = (w/3) * (2*sum(g^2) + sum(g[j]*g[j+1] for j=0..m-2))
    g2 = np.sum(g * g)
    if m > 1:
        cross = np.sum(g[:-1] * g[1:])
    else:
        cross = 0.0
    l2 = (w / 3.0) * (2.0 * g2 + cross)
    norm_1 = np.sum(np.abs(g)) / (m + 1)
    norm_inf = np.max(np.abs(g))
    if norm_inf == 0 or norm_1 == 0:
        return 0.0
    return l2 / (norm_1 * norm_inf)

def neg_value_direct(h):
    h = np.maximum(h, 0)
    return -compute_value_fast(h)

def construct():
    deadline = time.time() + 145.0  # 5s safety margin before 150s kill
    best_h = np.ones(N)
    best_val = compute_value_fast(best_h)  # should be 1.0

    rng = np.random.RandomState(12345)

    def try_optimize_log(start_log_h, maxiter, ftol=1e-10, gtol=1e-8):
        nonlocal best_h, best_val
        t0 = time.time()
        try:
            def neg_value_log(log_h):
                h = np.exp(log_h)
                return -compute_value_fast(h)
            
            res = minimize(
                neg_value_log,
                start_log_h,
                method='L-BFGS-B',
                options={'maxiter': maxiter, 'maxfun': maxiter * 10, 'ftol': ftol, 'gtol': gtol}
            )
            h_cand = np.exp(res.x)
            v_cand = compute_value_fast(h_cand)
            if v_cand > best_val:
                best_val = v_cand
                best_h = h_cand.copy()
        except Exception:
            pass
        return time.time() - t0

    def try_optimize_direct(start_h, maxiter, ftol=1e-10, gtol=1e-8):
        nonlocal best_h, best_val
        t0 = time.time()
        try:
            res = minimize(
                neg_value_direct,
                start_h,
                method='L-BFGS-B',
                bounds=[(0, None)] * N,
                options={'maxiter': maxiter, 'maxfun': maxiter * 10, 'ftol': ftol, 'gtol': gtol}
            )
            h_cand = np.maximum(res.x, 0)
            v_cand = compute_value_fast(h_cand)
            if v_cand > best_val:
                best_val = v_cand
                best_h = h_cand.copy()
        except Exception:
            pass
        return time.time() - t0

    # Phase 1: Start from uniform and optimize
    try_optimize_log(np.zeros(N), maxiter=5000)
    
    # Phase 2: Try diverse starting points
    starting_points = []
    
    # Uniform
    starting_points.append(np.zeros(N))
    
    # Sine shapes
    for freq in [1, 2, 3]:
        x = np.sin(2 * np.pi * freq * np.arange(N) / N)
        x = x - x.mean()
        starting_points.append(x)
    
    # Cosine shapes
    for freq in [1, 2, 3]:
        x = np.cos(2 * np.pi * freq * np.arange(N) / N)
        x = x - x.mean()
        starting_points.append(x)
    
    # Gaussian bumps
    for center in [N//4, N//2, 3*N//4]:
        for width in [10, 20, 30]:
            x = np.exp(-((np.arange(N) - center)**2) / (2 * width**2))
            x = x - x.mean()
            starting_points.append(x)
    
    # Plateau
    x = np.zeros(N)
    x[N//4:3*N//4] = 1.0
    x = x - x.mean()
    starting_points.append(x)
    
    # Triangle
    x = 1 - 2 * np.abs(np.arange(N) / N - 0.5)
    x = x - x.mean()
    starting_points.append(x)
    
    # Random smooth
    for i in range(8):
        x = rng.randn(N)
        kernel = np.ones(5) / 5.0
        x = np.convolve(x, kernel, mode='same')
        x = x - x.mean()
        starting_points.append(x)
    
    # Two bumps
    x = np.exp(-((np.arange(N) - N//4)**2) / 20.0) + np.exp(-((np.arange(N) - 3*N//4)**2) / 20.0)
    x = x - x.mean()
    starting_points.append(x)
    
    # Alternating
    x = np.array([1.0 if j % 2 == 0 else -0.5 for j in range(N)])
    x = x - x.mean()
    starting_points.append(x)
    
    # Quadratic
    x = (np.arange(N) / N - 0.5)**2 * 4 - 1
    x = x - x.mean()
    starting_points.append(x)
    
    # Linear ramp
    x = np.linspace(-1, 1, N)
    x = x - x.mean()
    starting_points.append(x)
    
    # Process starting points
    for i, x in enumerate(starting_points):
        if time.time() > deadline - 20.0:
            break
        
        x = np.maximum(x, -5.0)  # avoid underflow
        try_optimize_log(x, maxiter=3000)
        
        # Also try direct optimization
        if time.time() < deadline - 25.0:
            h_start = np.exp(x)
            try_optimize_direct(h_start, maxiter=2000)
        
        # Periodic high-precision polish
        if time.time() < deadline - 30.0:
            try_optimize_log(np.log(best_h + 1e-10), maxiter=10000, ftol=1e-12, gtol=1e-10)
    
    # Phase 3: Final polish with tighter tolerances
    if time.time() < deadline - 10.0:
        try_optimize_log(np.log(best_h + 1e-10), maxiter=20000, ftol=1e-14, gtol=1e-12)
        try_optimize_direct(best_h, maxiter=20000, ftol=1e-14, gtol=1e-12)
    
    # Phase 4: If time remains, try differential evolution for global exploration
    if time.time() < deadline - 60.0:
        try:
            def neg_value_de(h):
                h = np.maximum(h, 0)
                return -compute_value_fast(h)
            
            bounds = [(0, 5)] * N
            res = differential_evolution(
                neg_value_de,
                bounds,
                seed=42,
                maxiter=200,
                tol=1e-10,
                polish=True,
                init='sobol'
            )
            h_cand = np.maximum(res.x, 0)
            v_cand = compute_value_fast(h_cand)
            if v_cand > best_val:
                best_val = v_cand
                best_h = h_cand.copy()
        except Exception:
            pass
    
    # Final polish
    if time.time() < deadline - 5.0:
        try_optimize_log(np.log(best_h + 1e-10), maxiter=5000, ftol=1e-14, gtol=1e-12)
    
    return list(np.asarray(best_h, dtype=float))
