"""Packing 50 equal circles in a unit square: maximise the common radius."""
import numpy as np
from scipy.optimize import minimize
import time

N = 50
DEADLINE = 150.0
MARGIN = 5.0  # seconds safety margin


def _score(centres):
    """Compute the packing radius for a set of centres."""
    n = len(centres)
    # Wall distances
    wall = np.min(np.minimum(0.5 - np.abs(centres), 0.5), axis=1)
    # Pairwise distances
    min_pair = np.inf
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(centres[i] - centres[j])
            if d < min_pair:
                min_pair = d
    return min(min_pair / 2.0, np.min(wall))


def _neg_score(centres):
    """Negative score for minimization."""
    return -_score(centres)


def _gradient(centres):
    """Finite-difference gradient of the negative score."""
    n = len(centres)
    grad = np.zeros_like(centres)
    h = 1e-8
    f0 = _neg_score(centres)
    for i in range(n):
        for k in range(2):
            orig = centres[i, k]
            centres[i, k] = orig + h
            fp = _neg_score(centres)
            centres[i, k] = orig - h
            fm = _neg_score(centres)
            centres[i, k] = orig
            grad[i, k] = (fp - fm) / (2 * h)
    return grad


def _random_init(rng):
    """Generate a random initial configuration."""
    return (rng.random((N, 2)) - 0.5) * 0.9


def _grid_init():
    """Generate a grid initial configuration."""
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    return grid[:N].astype(np.float64)


def _refine(centres, max_iter=200, tol=1e-12):
    """Refine a configuration using Nelder-Mead style local search."""
    centres = centres.copy()
    best = _score(centres)
    
    # Use scipy's minimize with a simple objective
    # We'll use a multi-stage approach
    stages = [
        {'method': 'Nelder-Mead', 'options': {'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-12}},
        {'method': 'Nelder-Mead', 'options': {'maxiter': 1000, 'xatol': 1e-12, 'fatol': 1e-14}},
    ]
    
    for stage in stages:
        result = minimize(
            _neg_score,
            centres,
            method=stage['method'],
            options=stage['options']
        )
        if result.success and _score(result.x) > best:
            centres = result.x
            best = _score(centres)
    
    return centres, best


def _anneal_refine(centres, beta_schedule, max_iter_per_stage=500):
    """Multi-stage annealing refinement."""
    centres = centres.copy()
    best = _score(centres)
    
    for beta in beta_schedule:
        def obj(x, b=beta):
            # Smooth approximation using log-sum-exp for the min
            n = len(x)
            # Wall constraints
            wall = 0.5 - np.abs(x)
            wall_min = np.min(wall, axis=1)
            
            # Pairwise distances
            min_pair = np.inf
            for i in range(n):
                for j in range(i + 1, n):
                    d = np.linalg.norm(x[i] - x[j]) / 2.0
                    if d < min_pair:
                        min_pair = d
            
            # Soft min
            vals = np.concatenate([wall_min, [min_pair]])
            # Use log-sum-exp for soft min
            m = np.min(vals)
            soft_min = m + np.log(np.sum(np.exp(vals - m))) / len(vals)
            return -soft_min
        
        result = minimize(
            obj,
            centres,
            method='Nelder-Mead',
            options={'maxiter': max_iter_per_stage, 'xatol': 1e-10, 'fatol': 1e-12}
        )
        if result.success:
            sc = _score(result.x)
            if sc > best:
                centres = result.x
                best = sc
    
    return centres, best


def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE - MARGIN
    
    rng = np.random.default_rng(42)
    
    # Start with grid initialization
    best_centres = _grid_init()
    best_score = _score(best_centres)
    
    # Try several random initializations
    n_restarts = 8
    for i in range(n_restarts):
        if time.time() > deadline:
            break
        
        centres = _random_init(rng)
        
        # Quick refinement
        centres, sc = _refine(centres, max_iter=100)
        if sc > best_score:
            best_centres = centres
            best_score = sc
    
    # If we have time, do more thorough optimization
    if time.time() < deadline - 10:
        # More restarts with better refinement
        for i in range(10):
            if time.time() > deadline:
                break
            
            centres = _random_init(rng)
            # Perturb the best solution too
            if i % 2 == 1:
                centres = best_centres + rng.normal(0, 0.01, size=(N, 2))
            
            centres, sc = _refine(centres, max_iter=200)
            if sc > best_score:
                best_centres = centres
                best_score = sc
    
    # Final high-precision polish
    if time.time() < deadline - 5:
        # Use annealing with increasing beta
        beta_schedule = [1.0, 5.0, 20.0, 100.0, 500.0, 2000.0, 5000.0]
        centres, sc = _anneal_refine(best_centres, beta_schedule, max_iter_per_stage=300)
        if sc > best_score:
            best_centres = centres
            best_score = sc
        
        # One more Nelder-Mead pass with very tight tolerance
        if time.time() < deadline - 2:
            result = minimize(
                _neg_score,
                best_centres,
                method='Nelder-Mead',
                options={'maxiter': 5000, 'xatol': 1e-14, 'fatol': 1e-16}
            )
            if result.success:
                sc = _score(result.x)
                if sc > best_score:
                    best_centres = result.x
                    best_score = sc
    
    # Ensure all centres are strictly inside the square
    best_centres = np.clip(best_centres, -0.499999999, 0.499999999)
    
    return best_centres
