"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.optimize import minimize

N = 14
D = 3

# EVOLVE-BLOCK-START
import itertools


def _pairwise_distances(X):
    """Compute all pairwise squared distances."""
    # X shape (N, D)
    diff = X[:, np.newaxis, :] - X[np.newaxis, :, :]
    sq_dists = np.sum(diff ** 2, axis=-1)
    # Get upper triangle indices (i < j)
    iu = np.triu_indices(N, k=1)
    return sq_dists[iu]


def _score(X):
    """Compute the squared ratio of max to min pairwise distance."""
    sq_dists = _pairwise_distances(X)
    # Filter out any zero distances (shouldn't happen with distinct points)
    sq_dists = sq_dists[sq_dists > 1e-12]
    if len(sq_dists) == 0:
        return float('inf')
    min_d = np.min(sq_dists)
    max_d = np.max(sq_dists)
    if min_d < 1e-12:
        return float('inf')
    return (max_d / min_d) ** 2


def _objective(params):
    """Objective function for optimization: minimize squared ratio."""
    X = params.reshape(N, D)
    return _score(X)


def construct():
    """Place 14 distinct points in 3D to minimize squared max/min distance ratio."""
    deadline = time.time() + 110  # Safety margin before 120s limit
    
    # Start with a regular 14-point configuration
    # Use a known good starting point: points on a sphere or grid
    # Let's start with a regular grid perturbed slightly
    
    best_score = float('inf')
    best_X = None
    
    # Initial configuration: try a few different starting points
    # 1. Regular tetrahedron-based expansion
    # 2. Grid-based
    # 3. Random
    
    # Use a structured starting point: 14 points approximating a sphere
    # First, let's try a simple grid
    side = int(np.ceil(N ** (1 / D)))
    grid = list(itertools.product(range(side), repeat=D))[:N]
    X0 = np.array(grid, dtype=float)
    # Scale to unit sphere-ish
    X0 = X0 - X0.mean(axis=0)
    norm = np.linalg.norm(X0, axis=1)
    norm[norm < 1e-10] = 1.0
    X0 = X0 / norm[:, np.newaxis]
    
    initial_score = _score(X0)
    if initial_score < best_score:
        best_score = initial_score
        best_X = X0.copy()
    
    # Try random starting points
    np.random.seed(42)
    for trial in range(20):
        X_rand = np.random.randn(N, D)
        X_rand -= X_rand.mean(axis=0)
        score = _score(X_rand)
        if score < best_score:
            best_score = score
            best_X = X_rand.copy()
    
    # Now do local optimization from the best starting point
    # Use BFGS or L-BFGS-B
    
    if best_X is not None:
        result = minimize(
            _objective,
            best_X.flatten(),
            method='L-BFGS-B',
            options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-8}
        )
        
        if result.fun < best_score:
            best_score = result.fun
            best_X = result.x.reshape(N, D)
    
    # Try multiple random restarts with local optimization
    for trial in range(50):
        if time.time() > deadline:
            break
        
        X_rand = np.random.randn(N, D)
        X_rand -= X_rand.mean(axis=0)
        
        result = minimize(
            _objective,
            X_rand.flatten(),
            method='L-BFGS-B',
            options={'maxiter': 300, 'ftol': 1e-10, 'gtol': 1e-8}
        )
        
        if result.fun < best_score:
            best_score = result.fun
            best_X = result.x.reshape(N, D)
    
    # Final polish with tighter tolerance
    if best_X is not None and time.time() < deadline - 10:
        result = minimize(
            _objective,
            best_X.flatten(),
            method='L-BFGS-B',
            options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-10}
        )
        
        if result.fun < best_score:
            best_score = result.fun
            best_X = result.x.reshape(N, D)
    
    if best_X is None:
        # Fallback to grid
        best_X = X0
    
    return best_X
# EVOLVE-BLOCK-END
