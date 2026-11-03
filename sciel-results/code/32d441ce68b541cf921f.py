"""Erdos minimum overlap: minimize max overlap of h with shifted 1-h.
Approach: Global optimization using Differential Evolution with projection onto feasible set.
"""
import time
import numpy as np
from scipy.optimize import differential_evolution

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
    
    for _ in range(60): # 60 bisection steps
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


def construct():
    """
    Construct step heights minimizing maximum overlap with shifted complement.
    Uses Differential Evolution on an unconstrained parameterization projected to feasible set.
    """
    t0 = time.time()
    
    # Define objective function for DE
    # x is unconstrained, we project it to h
    def objective(x):
        if time.time() > _DEADLINE - 1:
            return 1e10 # Stop early
        h = _project_to_feasible(x)
        return _score(h)
    
    # Bounds for x. Since projection clips to [0,1], x values far from [0,1] 
    # just saturate. A range of [-2, 2] is usually sufficient to explore the space 
    # because the gradient of the projection is 1 in the interior.
    # However, to be safe, let's use [-5, 5].
    bounds = [(-5.0, 5.0) for _ in range(N)]
    
    # Run Differential Evolution
    # popsize: number of points in population. Default is 15.
    # maxiter: maximum number of iterations.
    # tol: tolerance.
    # seed: for reproducibility.
    # strategy: 'best1bin' is default.
    
    # We want to use most of the time.
    # Let's estimate evals.
    # If 1 eval takes ~0.5 ms, 10000 evals take 5s.
    # We have 115s.
    # Let's try popsize=20, maxiter=2000.
    # 40,000 evals. ~20s.
    
    result = differential_evolution(
        objective, 
        bounds=bounds, 
        popsize=20, 
        maxiter=2000, 
        seed=42, 
        tol=1e-9,
        polish=False # We will polish manually or skip if time is tight
    )
    
    best_x = result.x
    best_h = _project_to_feasible(best_x)
    best_score = _score(best_h)
    
    # Optional: Local polish with Nelder-Mead on the projected space?
    # Nelder-Mead needs a differentiable or at least continuous objective.
    # We can define a function f(h) that projects h (identity if feasible, else project) 
    # but Nelder-Mead works on unconstrained variables too.
    # Let's try a few steps of Nelder-Mead on x.
    
    if time.time() < _DEADLINE - 10:
        from scipy.optimize import minimize
        # Use the DE result as initial guess
        # Nelder-Mead is good for local refinement
        res_nm = minimize(
            objective, 
            best_x, 
            method='Nelder-Mead', 
            options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-10}
        )
        if res_nm.success or res_nm.fun < best_score:
            best_x = res_nm.x
            best_h = _project_to_feasible(best_x)
            best_score = _score(best_h)
            
    best_h = np.clip(best_h, 0.0, 1.0)
    diff = N / 2.0 - np.sum(best_h)
    if abs(diff) > 1e-12:
        mask = (best_h > 1e-10) & (best_h < 1 - 1e-10)
        if np.any(mask):
            best_h[mask] += diff / np.sum(mask)
            best_h = np.clip(best_h, 0.0, 1.0)
            
    return best_h
