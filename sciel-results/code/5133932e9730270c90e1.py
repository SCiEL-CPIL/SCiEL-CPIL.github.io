"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize
from math import cos, sin, pi

def _score(points):
    """Compute Delta / n^n for 8 points, where Delta is product of ordered pair distances
    after scaling to diameter 2."""
    n = len(points)
    pts = np.array(points, dtype=float)
    # Compute all pairwise distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    # Diameter = max distance
    diam = np.max(dists)
    if diam < 1e-15:
        return 0.0
    # Scale to diameter 2
    scaled_dists = dists / diam * 2.0
    # Product of ordered pair distances (exclude self-distances which are 0)
    mask = ~np.eye(n, dtype=bool)
    ordered_dists = scaled_dists[mask]
    # Delta = product of all ordered pair distances
    # Use log to avoid underflow
    log_delta = np.sum(np.log(ordered_dists))
    log_score = log_delta - n * np.log(n)
    return np.exp(log_score)

def _neg_score(points):
    return -_score(points)

def _neg_score_and_grad(points):
    """Compute negative score and its gradient using numerical differentiation."""
    pts = np.array(points, dtype=float)
    n = len(pts)
    score = _score(pts)
    
    # Numerical gradient
    eps = 1e-7
    grad = np.zeros_like(pts)
    for i in range(n):
        for j in range(2):
            pts_plus = pts.copy()
            pts_plus[i, j] += eps
            pts_minus = pts.copy()
            pts_minus[i, j] -= eps
            grad[i, j] = (-_score(pts_plus) + _score(pts_minus)) / (2 * eps)
    
    return -score, grad

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # safety margin before 150s kill
    
    # Start with regular octagon
    octagon = [[cos(2 * pi * k / 8), sin(2 * pi * k / 8)] for k in range(8)]
    best_points = np.array(octagon, dtype=float)
    best_score = _score(best_points)
    
    rng = np.random.RandomState(42)
    
    # Generate diverse starting points
    def generate_starts():
        starts = []
        # Regular octagon
        starts.append(np.array(octagon, dtype=float))
        
        # Perturbed octagons with various patterns
        for trial in range(20):
            angles = 2 * pi * np.arange(8) / 8
            # Different perturbation strategies
            strategy = trial % 5
            
            if strategy == 0:
                # Random radius perturbations
                radii = 1.0 + 0.2 * rng.randn(8)
                radii = np.clip(radii, 0.5, 1.5)
                angles_pert = angles + 0.1 * rng.randn(8)
            elif strategy == 1:
                # Alternating radii (two concentric circles)
                base_r = 1.0 + 0.1 * rng.randn()
                alt_r = 1.0 - 0.1 * rng.randn()
                radii = np.where(np.arange(8) % 2 == 0, base_r, alt_r)
                angles_pert = angles + 0.05 * rng.randn(8)
            elif strategy == 2:
                # Grouped perturbations
                radii = 1.0 + 0.15 * rng.randn(4)
                radii = np.tile(radii, 2)
                angles_pert = angles + 0.08 * rng.randn(8)
            elif strategy == 3:
                # Sine wave perturbation
                phase = rng.randn()
                radii = 1.0 + 0.15 * np.sin(2 * angles + phase)
                angles_pert = angles + 0.05 * rng.randn(8)
            else:
                # Larger random perturbations
                radii = 1.0 + 0.3 * rng.randn(8)
                radii = np.clip(radii, 0.3, 1.7)
                angles_pert = angles + 0.15 * rng.randn(8)
            
            pts = np.column_stack([radii * np.cos(angles_pert), radii * np.sin(angles_pert)])
            starts.append(pts)
        
        # Square with vertices at corners and midpoints
        # Try configurations with 4-fold symmetry
        for r_inner in [0.5, 0.7, 0.8, 0.9]:
            angles_outer = np.array([0, pi/2, pi, 3*pi/2])
            angles_inner = angles_outer + pi/4
            pts_outer = np.column_stack([np.cos(angles_outer), np.sin(angles_outer)])
            pts_inner = r_inner * np.column_stack([np.cos(angles_inner), np.sin(angles_inner)])
            pts = np.vstack([pts_outer, pts_inner])
            starts.append(pts)
        
        # Try configurations with 2-fold symmetry
        for dx in [0.1, 0.2, 0.3]:
            base = np.array(octagon, dtype=float)
            # Shift alternating points
            for i in range(8):
                if i % 2 == 0:
                    base[i, 0] += dx
                else:
                    base[i, 0] -= dx
            starts.append(base.copy())
        
        return starts
    
    starts = generate_starts()
    
    # Optimize each starting point
    for idx, x0 in enumerate(starts):
        if time.time() > deadline - 5.0:
            break
        
        x0_flat = x0.flatten()
        
        # Try L-BFGS-B with numerical gradient
        try:
            if time.time() > deadline - 3.0:
                break
            result = minimize(_neg_score, x0_flat, method='L-BFGS-B',
                            options={'maxiter': 2000, 'ftol': 1e-15, 'gtol': 1e-12})
            
            if time.time() > deadline - 3.0:
                break
            
            candidate = result.x.reshape(8, 2)
            # Check distinctness
            diff = candidate[:, np.newaxis, :] - candidate[np.newaxis, :, :]
            dists = np.linalg.norm(diff, axis=2)
            np.fill_diagonal(dists, np.inf)
            if np.min(dists) < 1e-8:
                continue
            
            cand_score = _score(candidate)
            if cand_score > best_score:
                best_score = cand_score
                best_points = candidate
                # Refine with Nelder-Mead
                if time.time() < deadline - 10.0:
                    result2 = minimize(_neg_score, best_points.flatten(), method='Nelder-Mead',
                                      options={'maxiter': 3000, 'xatol': 1e-12, 'fatol': 1e-14})
                    candidate2 = result2.x.reshape(8, 2)
                    diff2 = candidate2[:, np.newaxis, :] - candidate2[np.newaxis, :, :]
                    dists2 = np.linalg.norm(diff2, axis=2)
                    np.fill_diagonal(dists2, np.inf)
                    if np.min(dists2) >= 1e-8:
                        cand_score2 = _score(candidate2)
                        if cand_score2 > best_score:
                            best_score = cand_score2
                            best_points = candidate2
                            
        except Exception:
            pass
    
    # Final local refinement with multiple small perturbations
    if time.time() < deadline - 5.0:
        for pert_trial in range(10):
            if time.time() > deadline - 3.0:
                break
            x0 = best_points.flatten() + 0.01 * rng.randn(16)
            try:
                result = minimize(_neg_score, x0, method='Nelder-Mead',
                                options={'maxiter': 2000, 'xatol': 1e-13, 'fatol': 1e-15})
                candidate = result.x.reshape(8, 2)
                diff = candidate[:, np.newaxis, :] - candidate[np.newaxis, :, :]
                dists = np.linalg.norm(diff, axis=2)
                np.fill_diagonal(dists, np.inf)
                if np.min(dists) < 1e-8:
                    continue
                cand_score = _score(candidate)
                if cand_score > best_score:
                    best_score = cand_score
                    best_points = candidate
            except Exception:
                pass
    
    # Return the best points found
    return [[float(p[0]), float(p[1])] for p in best_points]
# EVOLVE-BLOCK-END
