import numpy as np
import time
from scipy.optimize import minimize

N = 50

def compute_ratio(h):
    """Compute ||h||_2^2 / (||h||_1 * ||h*h||_inf) for a given height array."""
    h = np.array(h, dtype=np.float64)
    if np.sum(h) == 0:
        return 0.0
    l2_sq = np.sum(h**2)
    l1 = np.sum(h)
    # Linear autoconvolution
    conv = np.convolve(h, h, mode='full')
    linf = np.max(conv)
    if l1 * linf == 0:
        return 0.0
    return l2_sq / (l1 * linf)

def neg_ratio(h):
    """Negative of ratio for minimization."""
    return -compute_ratio(h)

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin (5s buffer before 150s kill)
    
    best_h = np.ones(N, dtype=np.float64)
    best_score = compute_ratio(best_h)
    
    # Generate diverse initial conditions
    def make_initial_conditions():
        inits = []
        rng = np.random.RandomState(42)
        
        # Uniform
        inits.append(np.ones(N) / N)
        
        # Single spike at various positions
        for pos in [0, N//4, N//2, 3*N//4, N-1]:
            h = np.zeros(N)
            h[pos] = 1.0
            inits.append(h)
        
        # Two-point clusters
        for pos in range(0, N-1, 5):
            for ratio in [0.5, 1.0, 2.0]:
                h = np.zeros(N)
                h[pos] = 1.0
                h[pos+1] = ratio
                inits.append(h)
        
        # Gaussian profiles with different widths
        center = N // 2
        for sigma in [1.0, 2.0, 3.0, 5.0, 8.0, 12.0]:
            x = np.arange(N)
            h = np.exp(-0.5 * ((x - center) / sigma) ** 2)
            h = h / np.sum(h)
            inits.append(h)
        
        # Triangular profiles
        for width in [3, 5, 7, 11, 15]:
            h = np.zeros(N)
            center = N // 2
            for i in range(-width//2, width//2 + 1):
                pos = center + i
                if 0 <= pos < N:
                    h[pos] = 1.0 - abs(i) / (width/2.0 + 0.001)
                    if h[pos] < 0:
                        h[pos] = 0.0
            if np.sum(h) > 0:
                h = h / np.sum(h)
                inits.append(h)
        
        # Random nonnegative vectors
        for _ in range(10):
            h = rng.random(N)
            h = h / np.sum(h)
            inits.append(h)
        
        # Sparse random
        for _ in range(5):
            h = np.zeros(N)
            k = rng.randint(1, 8)
            idx = rng.choice(N, k, replace=False)
            h[idx] = rng.random(k)
            h = h / np.sum(h)
            inits.append(h)
        
        return inits
    
    initial_conditions = make_initial_conditions()
    
    # Optimize from each initial condition using Nelder-Mead
    for i, h0 in enumerate(initial_conditions):
        if time.time() > deadline - 10:
            break
        
        try:
            result = minimize(
                neg_ratio,
                h0,
                method='Nelder-Mead',
                options={
                    'maxiter': 500,
                    'xatol': 1e-8,
                    'fatol': 1e-10,
                    'adaptive': True
                }
            )
            
            h_opt = np.maximum(result.x, 0.0)  # Ensure nonnegativity
            if np.sum(h_opt) > 0:
                h_opt = h_opt / np.sum(h_opt)  # Normalize
                score = compute_ratio(h_opt)
                if score > best_score:
                    best_score = score
                    best_h = h_opt.copy()
        except Exception:
            continue
    
    # Try a few more iterations from the best found so far
    if time.time() < deadline - 15:
        for _ in range(3):
            if time.time() > deadline - 5:
                break
            try:
                result = minimize(
                    neg_ratio,
                    best_h + 0.01 * np.random.RandomState(int(time.time()*1000) % 10000).random(N),
                    method='Nelder-Mead',
                    options={
                        'maxiter': 300,
                        'xatol': 1e-9,
                        'fatol': 1e-11,
                        'adaptive': True
                    }
                )
                h_opt = np.maximum(result.x, 0.0)
                if np.sum(h_opt) > 0:
                    h_opt = h_opt / np.sum(h_opt)
                    score = compute_ratio(h_opt)
                    if score > best_score:
                        best_score = score
                        best_h = h_opt.copy()
            except Exception:
                continue
    
    return list(best_h)
