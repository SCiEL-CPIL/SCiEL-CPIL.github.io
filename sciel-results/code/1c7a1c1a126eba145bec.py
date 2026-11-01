"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 14
D = 3

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize

_DEADLINE = time.time() + 140

def _pairwise_dist_sq(pts):
    """Compute squared pairwise distances for N x D points. Returns (N, N) matrix."""
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    return np.sum(diff * diff, axis=2)

def _objective_repulsive(pts_flat, repulsion_strength=100.0, tau=0.05):
    """
    Objective: Smooth ratio + Repulsion term to prevent collapse.
    """
    pts = pts_flat.reshape(N, D)
    dsq = _pairwise_dist_sq(pts)
    iu = np.triu_indices(N, k=1)
    vals = dsq[iu]
    
    # Repulsion term: Sum of 1/(d^2 + eps)
    # This creates a strong force pushing points apart when they are close.
    repulsion = np.sum(1.0 / (vals + 1e-6))
    
    if np.any(vals < 1e-14):
        return 1e10 + repulsion * repulsion_strength
        
    # Soft max and soft min for ratio
    v_max = np.max(vals)
    log_sum_exp_max = v_max + tau * np.log(np.sum(np.exp((vals - v_max) / tau)))
    
    v_min = np.min(vals)
    log_sum_exp_min = -v_min - tau * np.log(np.sum(np.exp(-(vals - v_min) / tau)))
    
    if log_sum_exp_min < 1e-10:
        return 1e10 + repulsion * repulsion_strength
        
    ratio = log_sum_exp_max / log_sum_exp_min
    return ratio + repulsion * repulsion_strength

def _true_score(pts):
    """Compute the actual squared ratio: (max_d / min_d)^2 = max_dsq / min_dsq."""
    dsq = _pairwise_dist_sq(pts)
    iu = np.triu_indices(N, k=1)
    vals = dsq[iu]
    min_dsq = np.min(vals)
    max_dsq = np.max(vals)
    if min_dsq < 1e-14:
        return 1e10
    return max_dsq / min_dsq

def _random_init(rng):
    """Random initialization in unit hypercube, then center and scale."""
    pts = rng.uniform(-1, 1, size=(N, D))
    pts -= pts.mean(axis=0)
    dsq = _pairwise_dist_sq(pts)
    iu = np.triu_indices(N, k=1)
    max_dsq = np.max(dsq[iu])
    if max_dsq > 0:
        pts *= 1.0 / np.sqrt(max_dsq)
    return pts

def _local_optimize(pts, tau=0.05, max_iter=500, tol=1e-10, repulsion_strength=100.0):
    """Local optimization using L-BFGS-B on the repulsive objective."""
    x0 = pts.flatten()
    res = minimize(
        _objective_repulsive,
        x0,
        args=(repulsion_strength, tau),
        method='L-BFGS-B',
        options={'maxiter': max_iter, 'ftol': tol, 'gtol': 1e-8}
    )
    return res.x.reshape(N, D)

def _polish(pts, max_iter=2000, tol=1e-12):
    """High-precision polish with decreasing tau and increasing repulsion."""
    current = pts.copy()
    # Start with strong repulsion to ensure separation, then relax
    for tau, rep in [(0.01, 100.0), (0.005, 50.0), (0.001, 10.0), (0.0005, 5.0), (0.0001, 1.0)]:
        if time.time() > _DEADLINE:
            break
        current = _local_optimize(current, tau=tau, max_iter=max_iter, tol=tol, repulsion_strength=rep)
    return current

def construct():
    rng = np.random.default_rng(42)
    
    candidates = []
    
    # 1. Random restarts
    n_restarts = 50
    for i in range(n_restarts):
        if time.time() > _DEADLINE:
            break
        pts = _random_init(rng)
        pts = _local_optimize(pts, tau=0.05, max_iter=200, tol=1e-8, repulsion_strength=100.0)
        candidates.append(pts)
        
    # 2. Structured starts
    for trial in range(5):
        if time.time() > _DEADLINE:
            break
        tetra = np.array([
            [1, 1, 1],
            [1, -1, -1],
            [-1, 1, -1],
            [-1, -1, 1]
        ], dtype=float)
        tetra /= np.sqrt(3)
        extra = rng.uniform(-0.5, 0.5, size=(N - 4, D))
        pts = np.vstack([tetra, extra])
        pts -= pts.mean(axis=0)
        pts = _local_optimize(pts, tau=0.05, max_iter=200, tol=1e-8, repulsion_strength=100.0)
        candidates.append(pts)
        
    if not candidates:
        side = int(np.ceil(N ** (1 / D)))
        grid = np.array([[i, j, k] for i in range(side) for j in range(side) for k in range(side)][:N], dtype=float)
        return grid
        
    best_pts = None
    best_score = np.inf
    for pts in candidates:
        if time.time() > _DEADLINE:
            break
        s = _true_score(pts)
        if s < best_score:
            best_score = s
            best_pts = pts.copy()
            
    if best_pts is not None:
        best_pts = _polish(best_pts)
        best_score = _true_score(best_pts)
        
    # Final safety check for distinctness
    # If points are too close, perturb them
    for _ in range(10):
        dsq = _pairwise_dist_sq(best_pts)
        iu = np.triu_indices(N, k=1)
        min_dsq = np.min(dsq[iu])
        if min_dsq > 1e-8:
            break
        best_pts += rng.normal(0, 1e-4, size=(N, D))
        
    return best_pts
# EVOLVE-BLOCK-END
