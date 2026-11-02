"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 14
D = 3

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize, differential_evolution

DEADLINE = 150.0
START_TIME = time.time()

def _time_left():
    return DEADLINE - (time.time() - START_TIME)

def _score(pts):
    """Compute (max_dist / min_dist)^2 for a set of points."""
    pts = np.asarray(pts, dtype=float)
    n = pts.shape[0]
    # Compute all pairwise squared distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    sq_dists = np.sum(diff * diff, axis=-1)
    np.fill_diagonal(sq_dists, np.inf)
    min_sq = np.min(sq_dists)
    max_sq = np.max(sq_dists)
    if min_sq <= 0:
        return 1e12
    return (max_sq / min_sq) ** 2

def _optimize_from_start(pts0, time_budget):
    """Run Powell's method from a given start, with re-centering and scaling."""
    pts0 = np.asarray(pts0, dtype=float)
    n, d = pts0.shape
    
    def objective(x):
        if _time_left() < 5.0:
            return 1e12
        pts = x.reshape(n, d)
        # Center
        pts = pts - np.mean(pts, axis=0)
        # Scale to unit norm (or avoid zero)
        norm = np.linalg.norm(pts)
        if norm < 1e-12:
            return 1e12
        pts = pts / norm
        return _score(pts)
    
    x0 = pts0.flatten()
    # Center and scale initial point
    x0 = x0.reshape(n, d)
    x0 = x0 - np.mean(x0, axis=0)
    norm = np.linalg.norm(x0)
    if norm > 1e-12:
        x0 = x0 / norm
    x0 = x0.flatten()
    
    result = minimize(
        objective,
        x0,
        method='Powell',
        options={'maxiter': 200, 'maxfev': 5000, 'xatol': 1e-10, 'fatol': 1e-10}
    )
    
    best_pts = result.x.reshape(n, d)
    best_pts = best_pts - np.mean(best_pts, axis=0)
    norm = np.linalg.norm(best_pts)
    if norm > 1e-12:
        best_pts = best_pts / norm
    
    return best_pts, _score(best_pts)

def _generate_initial_points():
    """Generate diverse initial points for optimization."""
    inits = []
    
    # 1. Spherical Fibonacci-like distribution
    n = N
    golden_angle = np.pi * (3 - np.sqrt(5))
    pts = np.zeros((n, 3))
    for i in range(n):
        z = 1 - 2 * i / (n - 1)
        radius = np.sqrt(1 - z**2)
        theta = golden_angle * i
        pts[i] = [radius * np.cos(theta), radius * np.sin(theta), z]
    inits.append(pts)
    
    # 2. Random points on sphere (multiple seeds)
    for seed in range(10):
        rng = np.random.RandomState(seed)
        pts = rng.randn(n, 3)
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        pts = pts / norms
        inits.append(pts)
    
    # 3. Tesseract-like projections
    # Take 14 vertices of a 4D tesseract and project to 3D
    tesseract = np.array(list(np.ndindex(2, 2, 2, 2)))[:n]
    tesseract = (tesseract - 0.5) * 2
    # Project by dropping one coordinate with different weights
    for k in range(4):
        proj = np.delete(tesseract, k, axis=1)
        norms = np.linalg.norm(proj, axis=1, keepdims=True)
        proj = proj / (norms + 1e-10)
        inits.append(proj)
    
    # 4. Grid-based perturbations
    rng = np.random.RandomState(42)
    for _ in range(5):
        pts = rng.randn(n, 3)
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        pts = pts / norms
        inits.append(pts)
    
    return inits

def construct():
    global START_TIME
    START_TIME = time.time()
    
    best_pts = None
    best_score = np.inf
    
    # Generate initial points
    inits = _generate_initial_points()
    
    # Optimize from each starting point with time budget
    n_inits = len(inits)
    for i, pts0 in enumerate(inits):
        if _time_left() < 10.0:
            break
        
        # Allocate time per optimization run
        time_per_run = max(2.0, (_time_left() - 10.0) / max(1, n_inits - i))
        
        pts_opt, score = _optimize_from_start(pts0, time_per_run)
        
        if score < best_score:
            best_score = score
            best_pts = pts_opt
    
    # If we have time, do a final polish on the best point
    if best_pts is not None and _time_left() > 15.0:
        # Try a more precise optimization from the best point
        def objective_precise(x):
            if _time_left() < 5.0:
                return 1e12
            pts = x.reshape(N, D)
            pts = pts - np.mean(pts, axis=0)
            norm = np.linalg.norm(pts)
            if norm < 1e-12:
                return 1e12
            pts = pts / norm
            return _score(pts)
        
        x0 = best_pts.flatten()
        x0 = x0.reshape(N, D)
        x0 = x0 - np.mean(x0, axis=0)
        norm = np.linalg.norm(x0)
        if norm > 1e-12:
            x0 = x0 / norm
        x0 = x0.flatten()
        
        result = minimize(
            objective_precise,
            x0,
            method='Powell',
            options={'maxiter': 500, 'maxfev': 10000, 'xatol': 1e-12, 'fatol': 1e-12}
        )
        
        pts_polished = result.x.reshape(N, D)
        pts_polished = pts_polished - np.mean(pts_polished, axis=0)
        norm = np.linalg.norm(pts_polished)
        if norm > 1e-12:
            pts_polished = pts_polished / norm
        
        score_polished = _score(pts_polished)
        if score_polished < best_score:
            best_score = score_polished
            best_pts = pts_polished
    
    # Safety check: ensure we have valid distinct points
    if best_pts is None:
        # Fallback to simple grid
        side = int(np.ceil(N ** (1 / D)))
        import itertools
        grid = list(itertools.product(range(side), repeat=D))[:N]
        best_pts = np.array(grid, dtype=float)
    
    # Ensure points are distinct
    # If any two points are too close, perturb slightly
    diff = best_pts[:, np.newaxis, :] - best_pts[np.newaxis, :, :]
    sq_dists = np.sum(diff * diff, axis=-1)
    np.fill_diagonal(sq_dists, 1.0)
    if np.min(sq_dists) < 1e-10:
        # Add small random perturbation
        rng = np.random.RandomState(0)
        best_pts = best_pts + rng.randn(*best_pts.shape) * 1e-5
    
    return best_pts
# EVOLVE-BLOCK-END
