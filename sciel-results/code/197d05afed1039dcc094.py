import numpy as np
from scipy.optimize import minimize
import time

N = 50
# Heights are on equal intervals of [-1/4, 1/4]
# This means 50 intervals, so 51 grid points? Or 50 heights?
# "Choose 50 nonnegative heights on equal intervals of [-1/4,1/4]"
# Usually this implies 50 bins or 50 values.
# Let's assume 50 values $h_i$ for $i=0..49$.
# The domain is $[-1/4, 1/4]$.
# If there are 50 intervals, the width of each is $(1/4 - (-1/4))/50 = 1/20 = 0.05$.
# The autoconvolution $f*f$ is defined on $[-1/2, 1/2]$.
# The grid for the convolution will have $2N-1 = 99$ points.
# The spacing is $dx = 0.05$.

# Objective: Maximize $\frac{\|h\|_2^2}{\|h\|_1 \cdot \|h*h\|_\infty}$
# Note: $\|h\|_2^2 = \sum h_i^2$.
# $\|h\|_1 = \sum h_i$.
# $\|h*h\|_\infty = \max_k |(h*h)_k|$.
# Since $h_i \ge 0$, $(h*h)_k \ge 0$.
# The convolution sum is discrete: $(h*h)_k = \sum_i h_i h_{k-i}$.
# We need to be careful with the grid alignment.
# Let's define the grid for $h$ as $x_i = -1/4 + i \cdot dx$ for $i=0..N-1$.
# The convolution grid will be $y_k = -1/2 + k \cdot dx$ for $k=0..2N-2$.
# $(h*h)_k = \sum_{i} h_i h_{k-i}$ where the indices are valid.
# Actually, standard discrete convolution of two length-N sequences has length $2N-1$.
# Let $h$ be indexed $0..N-1$.
# $(h*h)_k = \sum_{i=\max(0, k-(N-1))}^{\min(N-1, k)} h_i h_{k-i}$.

def compute_convolution(h):
    # h is a numpy array of length N
    # Returns the autoconvolution array of length 2N-1
    # Using np.convolve is efficient
    return np.convolve(h, h, mode='full')

def score(h):
    h = np.asarray(h)
    l1 = np.sum(h)
    if l1 == 0:
        return 0.0
    l2_sq = np.sum(h**2)
    conv = compute_convolution(h)
    l_inf = np.max(conv)
    if l_inf == 0:
        return 0.0
    return l2_sq / (l1 * l_inf)

def neg_score(h):
    return -score(h)

def construct():
    start_time = time.time()
    deadline = 145.0  # 5s margin
    
    best_h = np.ones(N)
    best_score = score(best_h)
    
    # Initial diverse set of candidates
    candidates = [best_h]
    
    # 1. Uniform
    candidates.append(np.ones(N))
    
    # 2. Centered peak (Gaussian-like)
    x = np.linspace(-1, 1, N)
    for sigma in [0.2, 0.5, 1.0, 2.0]:
        candidates.append(np.exp(-x**2 / (2*sigma**2)))
    
    # 3. Triangular
    candidates.append(1 - np.abs(x))
    
    # 4. Sparse (few non-zeros)
    for k in [5, 10, 25]:
        h_sparse = np.zeros(N)
        h_sparse[:k] = 1.0
        candidates.append(h_sparse)
        
    # 5. Random
    np.random.seed(42)
    for _ in range(20):
        candidates.append(np.random.rand(N))
        
    # 6. Alternating
    h_alt = np.zeros(N)
    h_alt[::2] = 1.0
    candidates.append(h_alt)
    
    # Evaluate initial candidates
    for h in candidates:
        h = np.clip(h, 0, None)
        if np.sum(h) == 0: continue
        s = score(h)
        if s > best_score:
            best_score = s
            best_h = h.copy()
            
    # Local optimization using L-BFGS-B
    # We optimize in log-space or just directly with bounds?
    # Directly with bounds [0, 1] might be better to avoid scaling issues, 
    # but the objective is scale-invariant (homogeneous of degree 0).
    # So we can fix the scale, e.g., sum(h) = 1 or max(h) = 1.
    # Let's fix sum(h) = 1 to reduce the dimensionality of the search space effectively.
    # However, L-BFGS-B doesn't handle equality constraints well.
    # Instead, we can just bound h in [0, 1] and let the optimizer find the shape.
    # Since it's scale invariant, the absolute magnitude doesn't matter, only the shape.
    
    bounds = [(0, 1)] * N
    
    # Multi-start local optimization
    starts = []
    # Add best found so far
    starts.append(best_h / np.sum(best_h) if np.sum(best_h) > 0 else np.ones(N)/N)
    
    # Add some random normalized starts
    for _ in range(10):
        h_rand = np.random.rand(N)
        h_rand /= np.sum(h_rand)
        starts.append(h_rand)
        
    # Add specific shapes normalized
    for h in candidates:
        h_norm = h / np.sum(h) if np.sum(h) > 0 else np.ones(N)/N
        starts.append(h_norm)

    # Deduplicate starts roughly
    unique_starts = []
    for s in starts:
        is_unique = True
        for u in unique_starts:
            if np.allclose(s, u, atol=1e-3):
                is_unique = False
                break
        if is_unique:
            unique_starts.append(s)
    
    for h0 in unique_starts:
        if time.time() > deadline:
            break
            
        try:
            res = minimize(neg_score, h0, method='L-BFGS-B', bounds=bounds, 
                           options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-8})
            if res.success or res.fun < neg_score(best_h):
                h_opt = res.x
                h_opt = np.clip(h_opt, 0, None)
                s_opt = score(h_opt)
                if s_opt > best_score:
                    best_score = s_opt
                    best_h = h_opt.copy()
        except Exception:
            pass
            
    # Second phase: Polish with SLSQP or another L-BFGS-B run from best
    if time.time() < deadline - 10:
        try:
            # Normalize for stability
            h_start = best_h / np.sum(best_h)
            res = minimize(neg_score, h_start, method='L-BFGS-B', bounds=bounds,
                           options={'maxiter': 2000, 'ftol': 1e-14, 'gtol': 1e-10})
            if res.fun < neg_score(best_h):
                h_opt = np.clip(res.x, 0, None)
                s_opt = score(h_opt)
                if s_opt > best_score:
                    best_score = s_opt
                    best_h = h_opt.copy()
        except Exception:
            pass
            
    # Return list of floats
    return [float(x) for x in best_h]

# EVOLVE-BLOCK-START
# The construct function is defined above.
# We need to ensure the interface is correct.
# The task says: Interface: construct() -> sequence of nonnegative step heights
# The seed program had:
# def construct():
#     return [1.0] * N
# My construct returns a list of floats. This matches.
# EVOLVE-BLOCK-END
