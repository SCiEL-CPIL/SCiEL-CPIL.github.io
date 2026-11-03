"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution


def _compute_radius(cents):
    """Compute the largest common radius for given centres."""
    # Boundary constraint
    min_boundary = np.min(np.minimum(
        np.minimum(0.5 - cents[:, 0], 0.5 - cents[:, 1]),
        np.minimum(cents[:, 0] + 0.5, cents[:, 1] + 0.5)
    ))
    
    # Pairwise distances
    n = len(cents)
    dists = np.zeros((n, n))
    for i in range(n):
        diff = cents[i+1:] - cents[i]
        d = np.sqrt(np.sum(diff**2, axis=1))
        dists[i, i+1:] = d
        dists[i+1:, i] = d
    
    # Find minimum pairwise distance (excluding self)
    if n > 1:
        # Use a trick to find min off-diagonal
        np.fill_diagonal(dists, np.inf)
        min_pair = np.min(dists) / 2.0
    else:
        min_pair = min_boundary
    
    return min(min_pair, min_boundary)


def _objective(cents_flat):
    """Objective to minimize (negative of radius)."""
    cents = cents_flat.reshape(-1, 2)
    r = _compute_radius(cents)
    return -r


def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    # Initial guess: grid arrangement
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    best_cents = grid[:N].copy()
    best_r = _compute_radius(best_cents)
    
    # Try SLSQP optimization from multiple starting points
    # Starting point 1: grid
    # Starting point 2: slightly perturbed grid
    np.random.seed(42)
    
    for trial in range(5):
        if time.time() > deadline - 20:
            break
        
        if trial == 0:
            x0 = best_cents.flatten()
        else:
            # Perturbed grid
            perturbation = 0.02 * np.random.randn(N, 2)
            x0 = (best_cents + perturbation).flatten()
        
        # Bounds: each coordinate in [-0.5, 0.5]
        bounds = [(-0.5, 0.5)] * (2 * N)
        
        try:
            result = minimize(
                _objective,
                x0,
                method='SLSQP',
                bounds=bounds,
                options={'maxiter': 500, 'ftol': 1e-12}
            )
            
            if result.success or result.fun < _objective(x0):
                new_cents = result.x.reshape(-1, 2)
                new_r = _compute_radius(new_cents)
                if new_r > best_r:
                    best_r = new_r
                    best_cents = new_cents.copy()
        except Exception:
            pass
    
    # Try differential evolution for global search if time permits
    if time.time() < deadline - 30:
        try:
            bounds_de = [(-0.5, 0.5)] * (2 * N)
            result_de = differential_evolution(
                _objective,
                bounds_de,
                seed=42,
                maxiter=100,
                tol=1e-8,
                polish=False,
                init='latinhypercube'
            )
            
            if result_de.fun < _objective(best_cents.flatten()):
                new_cents = result_de.x.reshape(-1, 2)
                new_r = _compute_radius(new_cents)
                if new_r > best_r:
                    best_r = new_r
                    best_cents = new_cents.copy()
        except Exception:
            pass
    
    # Final polish with SLSQP if time permits
    if time.time() < deadline - 5:
        try:
            x0 = best_cents.flatten()
            bounds = [(-0.5, 0.5)] * (2 * N)
            result = minimize(
                _objective,
                x0,
                method='SLSQP',
                bounds=bounds,
                options={'maxiter': 1000, 'ftol': 1e-14}
            )
            
            if result.success:
                new_cents = result.x.reshape(-1, 2)
                new_r = _compute_radius(new_cents)
                if new_r > best_r:
                    best_r = new_r
                    best_cents = new_cents.copy()
        except Exception:
            pass
    
    return best_cents
# EVOLVE-BLOCK-END
