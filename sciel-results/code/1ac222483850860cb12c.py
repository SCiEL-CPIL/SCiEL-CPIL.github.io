"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import minimize, differential_evolution
import time

N = 50

def _score(heights):
    """Compute the raw value: ||h||_2^2 / (||h||_1 * ||h*h||_inf)"""
    h = np.asarray(heights, dtype=float)
    l1 = np.sum(h)
    if l1 == 0:
        return 0.0
    l2_sq = np.sum(h ** 2)
    # Autoconvolution (full)
    conv = np.convolve(h, h, mode='full')
    sup = np.max(np.abs(conv))
    if sup == 0:
        return 0.0
    return l2_sq / (l1 * sup)

def _gaussian_basis_params_to_heights(params):
    """
    Convert Gaussian basis parameters to a height vector.
    params: [c1, w1, a1, c2, w2, a2, ...] for K Gaussians
    Returns: array of N nonnegative heights
    """
    K = len(params) // 3
    heights = np.zeros(N)
    x = np.linspace(-0.25, 0.25, N)
    for k in range(K):
        c = params[3*k]
        w = max(params[3*k+1], 0.001)
        a = max(params[3*k+2], 0.0)
        heights += a * np.exp(-0.5 * ((x - c) / w) ** 2)
    return heights

def _evaluate_gaussian_params(params):
    """Score for Gaussian parameter vector"""
    heights = _gaussian_basis_params_to_heights(params)
    return _score(heights)

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 145s to be safe within 150s
    
    best_heights = np.ones(N)
    best_score = _score(best_heights)
    
    # Try different numbers of Gaussian components
    for K in [1, 2, 3]:
        if time.time() > deadline - 20:
            break
        
        # Differential evolution over Gaussian parameters
        # Each Gaussian: center in [-0.25, 0.25], width in [0.01, 0.25], amplitude in [0, 5]
        bounds = []
        for k in range(K):
            bounds.append((-0.25, 0.25))  # center
            bounds.append((0.01, 0.25))   # width
            bounds.append((0.0, 5.0))     # amplitude
        
        def objective(params):
            return -_evaluate_gaussian_params(params)
        
        try:
            result = differential_evolution(
                objective, bounds,
                maxiter=200, popsize=15,
                tol=1e-8, mutation=(0.3, 1.0),
                recombination=0.8, seed=42,
                polish=True, init='latinhypercube'
            )
            if time.time() > deadline - 10:
                break
            
            if result.fun < 0:  # Valid result
                score = -result.fun
                heights = _gaussian_basis_params_to_heights(result.x)
                if score > best_score:
                    best_score = score
                    best_heights = heights.copy()
        except Exception:
            pass
    
    # Local refinement using SLSQP on the height vector directly
    if time.time() < deadline - 15:
        # Normalize and optimize
        def objective_h(h):
            return -_score(h)
        
        # Constraint: nonnegativity handled by bounds
        bounds_h = [(0, None)] * N
        
        # Try from current best and a few other starts
        starts = [best_heights.copy()]
        
        # Add a few structured starts
        x = np.linspace(-0.25, 0.25, N)
        starts.append(np.exp(-10 * x**2))  # Single peak
        starts.append(np.exp(-20 * (x - 0.1)**2) + np.exp(-20 * (x + 0.1)**2))  # Two peaks
        starts.append(np.ones(N))  # Flat
        
        for start in starts:
            if time.time() > deadline - 5:
                break
            start = np.maximum(start, 0)
            if np.sum(start) == 0:
                continue
            try:
                res = minimize(
                    objective_h, start,
                    method='SLSQP',
                    bounds=bounds_h,
                    options={'maxiter': 300, 'ftol': 1e-14}
                )
                if time.time() > deadline - 5:
                    break
                if res.fun < 0:
                    s = -res.fun
                    if s > best_score:
                        best_score = s
                        best_heights = np.maximum(res.x, 0).copy()
            except Exception:
                pass
    
    # Final polish
    if time.time() < deadline - 2:
        try:
            def objective_h2(h):
                return -_score(h)
            res = minimize(
                objective_h2, best_heights,
                method='SLSQP',
                bounds=[(0, None)] * N,
                options={'maxiter': 500, 'ftol': 1e-15}
            )
            if res.fun < 0 and -res.fun > best_score:
                best_heights = np.maximum(res.x, 0).copy()
        except Exception:
            pass
    
    return list(np.maximum(best_heights, 0))
