"""Rewritten program: model-and-optimize for Erdos minimum overlap."""
import numpy as np
from scipy.optimize import minimize
import time

N = 200

# EVOLVE-BLOCK-START

def _all_overlaps(h):
    """Compute all circular cross-correlation overlaps h[i]*(1-h[i+k])."""
    N = len(h)
    overlaps = np.zeros(N)
    for k in range(N):
        shifted = np.roll(h, -k)
        overlaps[k] = np.sum(h * (1.0 - shifted))
    return overlaps

def _overlap_max(h):
    """Compute max over shifts k of sum_i h[i]*(1-h[i+k]), circular."""
    return np.max(_all_overlaps(h))

def _overlap_max_grad(h):
    """Compute max overlap and its gradient w.r.t. h."""
    N = len(h)
    overlaps = np.zeros(N)
    for k in range(N):
        shifted = np.roll(h, -k)
        overlaps[k] = np.sum(h * (1.0 - shifted))
    max_idx = np.argmax(overlaps)
    max_val = overlaps[max_idx]
    
    # Gradient of overlap_k = sum_i h[i]*(1-h[i+k])
    # d/dh[j] overlap_k = (1-h[j+k]) - h[j-k]
    k = max_idx
    shifted_plus = np.roll(h, -k)   # h[i+k]
    shifted_minus = np.roll(h, k)    # h[i-k]
    grad = (1.0 - shifted_plus) - shifted_minus
    return max_val, grad

def _project_to_simplex(h, target_sum):
    """Project h onto [0,1]^N with sum = target_sum using water-filling."""
    n = len(h)
    h = np.clip(h, 0.0, 1.0)
    # Iterative projection
    for _ in range(50):
        diff = target_sum - np.sum(h)
        if abs(diff) < 1e-12:
            break
        # Distribute evenly among non-saturated coordinates
        if diff > 0:
            free = h < 1.0 - 1e-10
        else:
            free = h > 1e-10
        if not np.any(free):
            break
        h[free] += diff / np.sum(free)
        h = np.clip(h, 0.0, 1.0)
    return h

def construct():
    deadline = time.time() + 140.0
    target_sum = N / 2.0
    best = np.full(N, 0.5)
    best_score = _overlap_max(best)
    
    rng = np.random.RandomState(42)
    
    # Strategy 1: L-BFGS-B on h directly with penalty for sum constraint
    # Use augmented Lagrangian: minimize max_overlap + mu*(sum(h) - N/2)^2
    # But simpler: project after each step
    
    def objective_and_grad(h):
        h_proj = _project_to_simplex(h, target_sum)
        val, grad = _overlap_max_grad(h_proj)
        return val, grad
    
    # Try several random starts
    for trial in range(8):
        if time.time() > deadline - 15.0:
            break
        
        # Random initialization with sum constraint
        h0 = rng.uniform(0.2, 0.8, size=N)
        h0 = _project_to_simplex(h0, target_sum)
        
        try:
            res = minimize(
                lambda x: _overlap_max(_project_to_simplex(x, target_sum)),
                h0,
                jac=lambda x: _overlap_max_grad(_project_to_simplex(x, target_sum))[1],
                method='L-BFGS-B',
                bounds=[(0.0, 1.0)] * N,
                options={'maxiter': 100, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            if time.time() > deadline - 10.0:
                break
            h_opt = _project_to_simplex(res.x, target_sum)
            score = _overlap_max(h_opt)
            if score < best_score - 1e-10:
                best = h_opt
                best_score = score
        except Exception:
            pass
    
    # Strategy 2: Local refinement with Nelder-Mead on a low-dim harmonic parameterization
    num_harmonics = 5
    def params_to_h(params):
        h = np.full(N, 0.5)
        for m in range(1, num_harmonics + 1):
            a = params[2*(m-1)]
            b = params[2*(m-1)+1]
            h += a * np.sin(2*np.pi*m*np.arange(N)/N) + b * np.cos(2*np.pi*m*np.arange(N)/N)
        h = _project_to_simplex(h, target_sum)
        return h
    
    n_params = 2 * num_harmonics
    for trial in range(5):
        if time.time() > deadline - 10.0:
            break
        x0 = rng.randn(n_params) * 0.03
        def obj(x):
            return _overlap_max(params_to_h(x))
        try:
            res = minimize(obj, x0, method='Nelder-Mead',
                           options={'maxiter': 300, 'xatol': 1e-7, 'fatol': 1e-9})
            if time.time() > deadline - 5.0:
                break
            h_opt = params_to_h(res.x)
            score = _overlap_max(h_opt)
            if score < best_score - 1e-10:
                best = h_opt
                best_score = score
        except Exception:
            pass
    
    # Final projection to ensure exact sum
    best = _project_to_simplex(best, target_sum)
    return best

# EVOLVE-BLOCK-END
