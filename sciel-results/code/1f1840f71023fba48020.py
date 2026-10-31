"""Rewritten program: 4x4 grid with separable low-rank Y = A B^T, optimized via DE."""
import time
import numpy as np
from scipy.optimize import differential_evolution

N = 16
D = 2

# EVOLVE-BLOCK-START

def _pairwise_distances(points):
    """Compute all pairwise distances for n points in 2D."""
    n = len(points)
    # Use broadcasting for efficiency
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    # Set diagonal to inf to ignore self-distances
    np.fill_diagonal(dists, np.inf)
    return dists

def _score_from_points(points):
    """Compute the squared ratio of max to min pairwise distance."""
    dists = _pairwise_distances(points)
    min_d = np.min(dists)
    max_d = np.max(dists)
    if min_d < 1e-12:
        return np.inf
    return (max_d / min_d) ** 2

def _reconstruct_from_params(params):
    """
    Reconstruct 16 points from parameter vector.
    Points are arranged in a 4x4 grid.
    x-coordinates: fixed integers [0, 1, 2, 3] for each column
    y-coordinates: from low-rank matrix Y = A @ B.T where A, B are 4xk
    """
    k = 2  # rank
    # params layout: A (4*k), B (4*k)
    A_flat = params[:4 * k]
    B_flat = params[4 * k:8 * k]
    
    A = A_flat.reshape(4, k)
    B = B_flat.reshape(4, k)
    
    Y = A @ B.T  # 4x4 matrix of y-coordinates
    
    # Build points: for row i (0..3) and col j (0..3):
    # x = j, y = Y[i, j]
    points = np.zeros((16, 2))
    for i in range(4):
        for j in range(4):
            idx = i * 4 + j
            points[idx, 0] = float(j)
            points[idx, 1] = float(Y[i, j])
    
    return points

def _objective(params):
    """Objective function for optimization."""
    points = _reconstruct_from_params(params)
    return _score_from_points(points)

def _initial_guess():
    """Generate initial parameter vector."""
    k = 2
    # Start with a reasonable grid-like configuration
    # Y should be approximately identity-like or linear
    A = np.array([[0, 0], [1, 0], [2, 0], [3, 0]], dtype=float)
    B = np.eye(4)[:, :k]  # first 2 columns of identity
    # Actually let's use a simpler init
    A = np.random.randn(4, k) * 0.5
    B = np.random.randn(4, k) * 0.5
    return np.concatenate([A.flatten(), B.flatten()])

def construct():
    """
    Construct 16 points in 2D to minimize (max/min pairwise distance)^2.
    Uses a 4x4 grid with separable low-rank y-coordinates optimized by DE.
    """
    deadline = time.time() + 110  # Safety margin before 120s limit
    k = 2
    
    # Initial solution: simple grid
    init_params = np.concatenate([
        np.array([0, 0, 1, 0, 2, 0, 3, 0], dtype=float),  # A
        np.array([0, 1, 0, 1, 0, 1, 0, 1], dtype=float),  # B (makes Y[i,j] = i if j even, ... hmm)
    ])
    # Let me just use a clean init
    A_init = np.array([[0, 0], [1, 0], [2, 0], [3, 0]], dtype=float)
    B_init = np.array([[1, 0], [1, 0], [1, 0], [1, 0]], dtype=float)
    init_params = np.concatenate([A_init.flatten(), B_init.flatten()])
    
    best_params = init_params.copy()
    best_score = _objective(best_params)
    
    # Define bounds for DE
    bounds = [(-5, 5)] * (8 * k)
    
    # Run differential evolution with time limit
    try:
        result = differential_evolution(
            _objective,
            bounds,
            seed=42,
            maxiter=200,
            popsize=15,
            tol=1e-6,
            mutation=(0.3, 1.0),
            recombination=0.8,
            polish=True,
            init='sobol',
            x0=init_params,
            callback=_make_callback(deadline, best_params, best_score),
            workers=1
        )
        if result.success or result.fun < best_score:
            best_params = result.x
            best_score = result.fun
    except Exception:
        pass
    
    # Check if we're still within time budget for a polish step
    if time.time() < deadline - 5:
        # Try a few more local refinements with different seeds
        for seed in [1, 7, 13, 42, 99]:
            if time.time() > deadline - 5:
                break
            try:
                res = differential_evolution(
                    _objective,
                    bounds,
                    seed=seed,
                    maxiter=50,
                    popsize=10,
                    tol=1e-7,
                    polish=True,
                    init='sobol',
                    workers=1
                )
                if res.fun < best_score:
                    best_params = res.x
                    best_score = res.fun
            except Exception:
                pass
    
    points = _reconstruct_from_params(best_params)
    return points

def _make_callback(deadline, best_params, best_score):
    """Create a callback that checks time and updates best."""
    state = {'best_params': best_params.copy(), 'best_score': best_score}
    
    def callback(xk, convergence):
        if time.time() > deadline:
            raise TimeoutError("Time limit reached")
        return False
    
    return callback

# EVOLVE-BLOCK-END
