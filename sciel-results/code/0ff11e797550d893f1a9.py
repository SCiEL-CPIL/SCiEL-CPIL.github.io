"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.optimize import differential_evolution, minimize

N = 16
D = 2

# EVOLVE-BLOCK-START

def _score_from_points(pts):
    """Compute (max pairwise dist / min pairwise dist)^2 for a set of points."""
    n = len(pts)
    # Compute all pairwise squared distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    sq_dists = np.sum(diff ** 2, axis=-1)
    # Exclude diagonal
    sq_dists = sq_dists[np.triu_indices(n, k=1)]
    if len(sq_dists) == 0:
        return np.inf
    max_sq = np.max(sq_dists)
    min_sq = np.min(sq_dists)
    if min_sq <= 0:
        return np.inf
    ratio_sq = max_sq / min_sq
    return ratio_sq


def _parametric_grid(params):
    """
    Map a low-dimensional parameter vector to 16 points in 2D.
    We use a 4x4 grid deformed by sinusoidal modes.
    params: array of shape (6,)
      [amp_x, amp_y, freq_x, freq_y, phase_x, phase_y]
    """
    amp_x, amp_y, freq_x, freq_y, phase_x, phase_y = params
    
    # Base 4x4 grid centered at origin
    i = np.arange(4)
    j = np.arange(4)
    I, J = np.meshgrid(i, j, indexing='ij')
    I = I.ravel()  # shape (16,)
    J = J.ravel()
    
    # Normalize to [-1, 1]
    I_norm = (I - 1.5) / 1.5
    J_norm = (J - 1.5) / 1.5
    
    # Apply sinusoidal deformation
    x = I_norm + amp_x * np.sin(freq_x * J_norm + phase_x)
    y = J_norm + amp_y * np.sin(freq_y * I_norm + phase_y)
    
    pts = np.column_stack([x, y])
    return pts


def _objective(params):
    """Objective: minimize the squared ratio of max to min pairwise distance."""
    pts = _parametric_grid(params)
    return _score_from_points(pts)


def _refine_direct(pts_init, time_budget):
    """
    Refine points directly in 32D using Nelder-Mead with a smooth surrogate.
    Uses log-sum-exp for smooth max/min.
    """
    start = time.time()
    
    def smooth_objective(pts_flat):
        if time.time() - start > time_budget:
            return 1e10
        pts = pts_flat.reshape(16, 2)
        n = 16
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        sq_dists = np.sum(diff ** 2, axis=-1)
        sq_dists = sq_dists[np.triu_indices(n, k=1)]
        
        # Soft max and min using log-sum-exp
        beta = 50.0  # sharpness
        max_logsumexp = np.log(np.sum(np.exp(beta * sq_dists))) / beta
        min_logsumexp = -np.log(np.sum(np.exp(-beta * sq_dists))) / beta
        
        if min_logsumexp <= 0:
            return 1e10
        return (max_logsumexp / min_logsumexp) ** 2
    
    # Initial point
    x0 = pts_init.ravel()
    
    try:
        res = minimize(
            smooth_objective,
            x0,
            method='Nelder-Mead',
            options={'maxiter': 5000, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True}
        )
        if res.success or res.fun < _score_from_points(pts_init):
            return res.x.reshape(16, 2)
    except Exception:
        pass
    return pts_init


def construct():
    deadline = time.time() + 110  # Safety margin before 120s limit
    best_pts = None
    best_score = np.inf
    
    # Strategy 1: Differential Evolution on parametric grid
    bounds = [
        (0.0, 0.5),   # amp_x
        (0.0, 0.5),   # amp_y
        (0.5, 4.0),   # freq_x
        (0.5, 4.0),   # freq_y
        (0.0, 2*np.pi),  # phase_x
        (0.0, 2*np.pi),  # phase_y
    ]
    
    try:
        if time.time() < deadline - 60:
            de_res = differential_evolution(
                _objective,
                bounds,
                seed=42,
                maxiter=200,
                popsize=20,
                tol=1e-6,
                polish=True,
                init='sobol'
            )
            if de_res.fun < best_score:
                best_score = de_res.fun
                best_pts = _parametric_grid(de_res.x)
    except Exception:
        pass
    
    # Strategy 2: Try a few random parametric initializations and refine
    rng = np.random.RandomState(123)
    for attempt in range(5):
        if time.time() > deadline - 40:
            break
        params = np.array([
            rng.uniform(0.0, 0.5),
            rng.uniform(0.0, 0.5),
            rng.uniform(0.5, 4.0),
            rng.uniform(0.5, 4.0),
            rng.uniform(0.0, 2*np.pi),
            rng.uniform(0.0, 2*np.pi),
        ])
        pts = _parametric_grid(params)
        s = _score_from_points(pts)
        if s < best_score:
            best_score = s
            best_pts = pts
    
    # Strategy 3: Direct refinement of best points found
    if best_pts is not None and time.time() < deadline - 15:
        refined = _refine_direct(best_pts, time_budget=25)
        s_ref = _score_from_points(refined)
        if s_ref < best_score:
            best_score = s_ref
            best_pts = refined
    
    # Final sanity check
    if best_pts is None:
        # Fallback to grid
        side = int(np.ceil(N ** (1 / D)))
        grid = list(__import__('itertools').product(range(side), repeat=D))[:N]
        best_pts = np.array(grid, dtype=float)
        best_score = _score_from_points(best_pts)
    
    return best_pts

# EVOLVE-BLOCK-END
