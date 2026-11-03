"""Structural change: L-BFGS-B continuous optimization with jittered hexagonal initialization."""
import numpy as np
import time
from scipy.optimize import minimize
import math

N = 200
_DEADLINE = 140.0  # safety margin below 150s kill


def _score(centers):
    """Return the largest common radius for given centers in [-0.5,0.5]^2."""
    if centers.shape != (N, 2):
        return 0.0
    if not np.all(np.isfinite(centers)):
        return 0.0
    # Boundary constraint
    boundary = np.min(np.minimum(
        0.5 - np.abs(centers[:, 0]),
        0.5 - np.abs(centers[:, 1])
    ))
    if boundary < 0:
        return 0.0
    # Pairwise distances
    diff = centers[:, None, :] - centers[None, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists)
    return min(boundary, min_pair / 2.0)


def _negative_score(flat):
    """Negative of the score for minimization, with penalty for boundary violations."""
    centers = flat.reshape(N, 2)
    if not np.all(np.isfinite(centers)):
        return 1e6
    
    # Check boundary violations with penalty
    dist_to_boundary = np.minimum(
        0.5 - np.abs(centers[:, 0]),
        0.5 - np.abs(centers[:, 1])
    )
    
    # If any point is outside the square, heavy penalty
    min_boundary = np.min(dist_to_boundary)
    if min_boundary < -0.01:
        return 1e6 + (abs(min_boundary) + 0.01) * 1000
    
    # Pairwise distances
    diff = centers[:, None, :] - centers[None, :, :]
    dists_sq = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(dists_sq, np.inf)
    min_dist_sq = np.min(dists_sq)
    min_pair = np.sqrt(min_dist_sq)
    
    # Compute effective radius
    boundary_eff = max(min_boundary, 0)
    r = min(boundary_eff, min_pair / 2.0)
    
    # Penalty if boundary is violated (negative)
    penalty = 0.0
    if min_boundary < 0:
        penalty = -min_boundary * 100  # strong penalty
    
    return -(r - penalty)


def _jittered_hex_lattice(n_points, seed=42):
    """Generate a jittered hexagonal lattice of approximately n_points within [-0.5,0.5]^2."""
    rng = np.random.RandomState(seed)
    
    # Estimate spacing for ~n_points in unit square
    # Hexagonal packing density is ~0.9069
    area_per_point = 1.0 / n_points
    # For hexagonal lattice, area per point = s^2 * sqrt(3)/2
    s = np.sqrt(2 * area_per_point / np.sqrt(3))
    
    # Generate hex lattice points
    points = []
    dx = s
    dy = s * np.sqrt(3) / 2
    
    margin = 0.02
    x_min, x_max = -0.5 + margin, 0.5 - margin
    y_min, y_max = -0.5 + margin, 0.5 - margin
    
    row = 0
    y = y_min
    while y <= y_max:
        x_offset = (dx / 2) if (row % 2 == 1) else 0.0
        x = x_min + x_offset
        while x <= x_max:
            points.append([x, y])
            x += dx
        y += dy
        row += 1
    
    points = np.array(points, dtype=np.float64)
    
    if len(points) < n_points:
        # Reduce spacing and regenerate
        s *= 0.9
        return _jittered_hex_lattice(n_points, seed)
    
    # Add small jitter to break symmetry
    jitter = rng.normal(0, 0.003, size=points.shape)
    points += jitter
    
    # Clip to stay within square
    points[:, 0] = np.clip(points[:, 0], -0.48, 0.48)
    points[:, 1] = np.clip(points[:, 1], -0.48, 0.48)
    
    # Select n_points with best initial spacing
    if len(points) > n_points:
        # Compute pairwise distances for each candidate
        # Use a greedy selection: keep points that maximize minimum distance
        selected = [points[0]]
        for p in points[1:]:
            if len(selected) >= n_points:
                break
            # Check if p is far enough from all selected
            dists = np.sqrt(np.sum((np.array(selected) - p) ** 2, axis=1))
            if len(dists) > 0 and np.min(dists) > s * 0.7:
                selected.append(p)
        
        # If we didn't get enough, take the rest
        while len(selected) < n_points:
            remaining = [p for p in points if p not in selected]
            if not remaining:
                break
            selected.append(remaining[0])
        
        points = np.array(selected[:n_points])
    
    return points


def construct():
    start_time = time.time()
    
    # Generate initial hexagonal lattice with jitter
    centers = _jittered_hex_lattice(N, seed=42)
    flat = centers.flatten()
    
    best_score = _score(centers)
    best_centers = centers.copy()
    
    # Bounds for L-BFGS-B
    bounds = [(-0.49, 0.49)] * (N * 2)
    
    # Run L-BFGS-B optimization
    max_time = _DEADLINE - start_time
    max_time = min(max_time, 100)  # cap to leave margin
    
    if max_time > 5:
        # First optimization run
        res = minimize(
            _negative_score,
            flat,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 500, 'maxfun': 2000, 'disp': False}
        )
        
        elapsed = time.time() - start_time
        if res.success or res.status == 0:
            new_centers = res.x.reshape(N, 2)
            new_score = _score(new_centers)
            if new_score > best_score:
                best_score = new_score
                best_centers = new_centers
        
        # Second run from perturbed start if time permits
        if time.time() - start_time < _DEADLINE - 10:
            rng = np.random.RandomState(123)
            perturbed = best_centers + rng.normal(0, 0.002, size=best_centers.shape)
            perturbed[:, 0] = np.clip(perturbed[:, 0], -0.48, 0.48)
            perturbed[:, 1] = np.clip(perturbed[:, 1], -0.48, 0.48)
            
            res2 = minimize(
                _negative_score,
                perturbed.flatten(),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 300, 'maxfun': 1500, 'disp': False}
            )
            
            new_centers2 = res2.x.reshape(N, 2)
            new_score2 = _score(new_centers2)
            if new_score2 > best_score:
                best_score = new_score2
                best_centers = new_centers2
    
    # Final validation
    final_score = _score(best_centers)
    if final_score <= 0:
        # Fallback to simple grid
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        best_centers = grid[:N]
    
    return best_centers
