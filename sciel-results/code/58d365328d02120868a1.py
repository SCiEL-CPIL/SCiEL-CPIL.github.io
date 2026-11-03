"""Erdos minimum overlap: minimize max overlap of h with shifted 1-h.
Approach: Smoothed minimization using log-sum-exp approximation of the max overlap,
optimized via Projected Gradient Descent with adaptive step size and multiple restarts.
"""
import time
import numpy as np

N = 100
_DEADLINE = time.time() + 115


def _compute_overlaps(h):
    """Compute C_k = (2/N)*(N/2 - r_k) for k=0..N-1."""
    n = len(h)
    r = np.zeros(n)
    for k in range(n):
        r[k] = np.dot(h[k:], h[:n-k])
    return (2.0 / n) * (n / 2.0 - r)


def _score(h):
    """Compute the maximum linear overlap."""
    h = np.asarray(h, dtype=np.float64)
    cs = _compute_overlaps(h)
    return float(np.max(cs))


def _project_to_feasible(y):
    """
    Project vector y onto the set {h | 0 <= h_i <= 1, sum(h) = N/2}.
    Uses bisection on the threshold tau.
    """
    n = len(y)
    target_sum = n / 2.0
    
    lo = np.min(y) - 2.0
    hi = np.max(y) + 2.0
    
    for _ in range(60):
        mid = (lo + hi) / 2.0
        h = np.clip(y - mid, 0.0, 1.0)
        s = np.sum(h)
        if s > target_sum:
            lo = mid
        else:
            hi = mid
            
    h = np.clip(y - lo, 0.0, 1.0)
    # Correct any numerical drift in the sum
    diff = target_sum - np.sum(h)
    if abs(diff) > 1e-12:
        mask = (h > 1e-8) & (h < 1 - 1e-8)
        if np.any(mask):
            h[mask] += diff / np.sum(mask)
            h = np.clip(h, 0.0, 1.0)
    return h


def _soft_max_obj(h, alpha):
    """
    Compute log-sum-exp of overlaps and its gradient.
    """
    cs = _compute_overlaps(h)
    
    c_max = np.max(cs)
    exp_cs = np.exp(alpha * (cs - c_max))
    sum_exp = np.sum(exp_cs)
    
    obj = (1.0 / alpha) * (np.log(sum_exp) + alpha * c_max)
    
    w = exp_cs / sum_exp
    
    grad_raw = np.zeros(N)
    for j in range(N):
        part1 = np.dot(w[:j+1], h[j::-1])
        part2 = np.dot(w[:N-j], h[j:])
        grad_raw[j] = -(2.0 / N) * (part1 + part2)
        
    return obj, grad_raw


def _optimize_from_start(h0, alphas, max_steps_per_alpha, base_eta):
    """Run projected gradient descent from a given start point."""
    h = h0.copy()
    best_h = h.copy()
    best_score = _score(best_h)
    
    for alpha in alphas:
        if time.time() > _DEADLINE - 5:
            break
            
        eta = base_eta
        no_improve_count = 0
        
        for step in range(max_steps_per_alpha):
            if time.time() > _DEADLINE - 5:
                break
                
            obj, grad = _soft_max_obj(h, alpha)
            
            h_new = h - eta * grad
            h_proj = _project_to_feasible(h_new)
            
            s = _score(h_proj)
            if s < best_score:
                best_score = s
                best_h = h_proj.copy()
                no_improve_count = 0
            else:
                no_improve_count += 1
                
            h = h_proj
            eta *= 0.997
            
            if no_improve_count > 50:
                break
    
    return best_h, best_score


def construct():
    """
    Construct step heights minimizing maximum overlap with shifted complement.
    Uses Projected Gradient Descent on a log-sum-exp smoothed max overlap
    with multiple restarts from different initializations.
    """
    t0 = time.time()
    
    # Multiple initializations to explore different basins
    initializations = []
    
    # 1. Cosine wave
    i = np.arange(N)
    h1 = 0.5 + 0.3 * np.cos(2.0 * np.pi * i / N)
    h1 = _project_to_feasible(np.clip(h1, 0.0, 1.0))
    initializations.append(h1)
    
    # 2. Different frequency cosine
    h2 = 0.5 + 0.25 * np.cos(4.0 * np.pi * i / N)
    h2 = _project_to_feasible(np.clip(h2, 0.0, 1.0))
    initializations.append(h2)
    
    # 3. Sine wave
    h3 = 0.5 + 0.3 * np.sin(2.0 * np.pi * i / N)
    h3 = _project_to_feasible(np.clip(h3, 0.0, 1.0))
    initializations.append(h3)
    
    # 4. Random with fixed seed
    np.random.seed(42)
    h4 = np.random.uniform(0.0, 1.0, N)
    h4 = _project_to_feasible(h4)
    initializations.append(h4)
    
    # 5. Alternating pattern
    h5 = np.where(i % 2 == 0, 0.7, 0.3)
    h5 = _project_to_feasible(h5)
    initializations.append(h5)
    
    # 6. Block pattern
    h6 = np.zeros(N)
    h6[:N//4] = 0.8
    h6[N//4:3*N//4] = 0.4
    h6[3*N//4:] = 0.8
    h6 = _project_to_feasible(h6)
    initializations.append(h6)
    
    alphas = [1.0, 5.0, 10.0, 20.0, 50.0, 100.0, 200.0, 500.0]
    
    best_h = None
    best_score = float('inf')
    
    for h0 in initializations:
        if time.time() > _DEADLINE - 10:
            break
        
        h_opt, score = _optimize_from_start(h0, alphas, 300, 0.01)
        
        if score < best_score:
            best_score = score
            best_h = h_opt.copy()
    
    # Final cleanup
    if best_h is None:
        best_h = initializations[0]
    
    best_h = np.clip(best_h, 0.0, 1.0)
    diff = N / 2.0 - np.sum(best_h)
    if abs(diff) > 1e-12:
        mask = (best_h > 1e-10) & (best_h < 1 - 1e-10)
        if np.any(mask):
            best_h[mask] += diff / np.sum(mask)
            best_h = np.clip(best_h, 0.0, 1.0)
            
    return best_h
