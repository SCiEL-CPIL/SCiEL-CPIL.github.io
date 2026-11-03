"""
Packing of 50 equal spheres in a unit cube using grid-based candidate selection
followed by local optimization.

Strategy:
1. Generate candidate positions on a fine 3D grid inside the cube.
2. Use a greedy independent-set heuristic to select 50 non-conflicting candidates.
3. Refine the selected positions via local optimization (soft-min gradient ascent)
   to maximize the minimum clearance.
4. Multiple restarts with different grid offsets and random tie-breaking.
"""

import numpy as np
import time
from scipy.optimize import minimize

N = 50
DEADLINE_SECONDS = 115  # Safety margin before 120s limit


def _score(points):
    """Compute the maximum valid radius for the given sphere centers."""
    if points.shape != (N, 3):
        return 0.0
    if not np.all(np.isfinite(points)):
        return 0.0
    # Distance between all pairs
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists) / 2.0

    # Distance to walls: cube is [-0.5, 0.5]^3
    wall_dists = 0.5 - np.abs(points)
    min_wall = np.min(wall_dists)

    return min(min_pair, min_wall)


def _soft_min_energy(points, beta=10.0):
    """
    Soft-min energy: a smooth approximation of the negative of the minimum clearance.
    Lower energy = better packing.
    """
    # Pairwise distances
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dists, 1e10)
    
    # Wall distances
    wall_dists = 0.5 - np.abs(points)
    
    # Combine all constraints into a single vector
    # For pairs: use dists/2 (radius)
    # For walls: use wall_dists
    pair_vals = dists / 2.0
    # Only upper triangle to avoid double counting
    mask = np.triu(np.ones((N, N), dtype=bool), k=1)
    pair_vals = pair_vals[mask]
    
    all_vals = np.concatenate([pair_vals, wall_dists.flatten()])
    
    # Soft min: -log(sum(exp(-beta * x))) / beta
    # This approximates min(x) as beta -> infinity
    max_val = np.max(all_vals)
    shifted = all_vals - max_val
    log_sum = np.log(np.sum(np.exp(-beta * shifted)))
    soft_min = max_val - log_sum / beta
    
    return -soft_min  # Negative because we want to minimize energy = maximize min clearance


def _soft_min_grad(points, beta=10.0):
    """Gradient of the soft-min energy with respect to point positions."""
    # Pairwise distances
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dists, 1e10)
    
    # Wall distances
    wall_dists = 0.5 - np.abs(points)
    
    pair_vals = dists / 2.0
    mask = np.triu(np.ones((N, N), dtype=bool), k=1)
    pair_vals = pair_vals[mask]
    
    all_vals = np.concatenate([pair_vals, wall_dists.flatten()])
    
    max_val = np.max(all_vals)
    shifted = all_vals - max_val
    weights = np.exp(-beta * shifted)
    weights /= np.sum(weights)
    
    # Gradient with respect to pair distances
    grad = np.zeros_like(points)
    
    # Pair contributions
    pair_idx = np.argwhere(mask)
    for idx, (i, j) in enumerate(pair_idx):
        d = points[i] - points[j]
        dist = dists[i, j]
        if dist < 1e-10:
            continue
        # d(soft_min)/d(dist_ij) = -weights[idx] / 2 (since pair_val = dist/2)
        # d(dist_ij)/d(points[i]) = d / dist
        # d(dist_ij)/d(points[j]) = -d / dist
        w = weights[idx]
        grad[i] += w * d / dist
        grad[j] -= w * d / dist
    
    # Wall contributions
    wall_idx_offset = len(pair_idx)
    for i in range(N):
        for k in range(3):
            idx = wall_idx_offset + i * 3 + k
            w = weights[idx]
            # wall_dist = 0.5 - |point_ik|
            # d(wall_dist)/d(point_ik) = -sign(point_ik)
            val = points[i, k]
            if abs(val) > 1e-10:
                grad[i, k] -= w * np.sign(val)
    
    return grad


def _grid_candidates(grid_size, offset=None):
    """
    Generate candidate positions on a grid inside the cube.
    grid_size: number of divisions per axis
    offset: optional offset to shift the grid
    """
    if offset is None:
        offset = np.array([0.0, 0.0, 0.0])
    
    # Grid points from -0.5 to 0.5
    coords = np.linspace(-0.5 + 1.0 / grid_size / 2, 0.5 - 1.0 / grid_size / 2, grid_size)
    coords = coords + offset
    
    # Create 3D grid
    xs, ys, zs = np.meshgrid(coords, coords, coords, indexing='ij')
    candidates = np.column_stack([xs.ravel(), ys.ravel(), zs.ravel()])
    
    # Filter to only keep points inside the cube
    inside = np.all(np.abs(candidates) <= 0.5, axis=1)
    candidates = candidates[inside]
    
    return candidates


def _greedy_select(candidates, n_select, r_est, rng):
    """
    Greedy independent-set heuristic to select n_select non-conflicting candidates.
    r_est: estimated radius for conflict detection
    rng: numpy random generator for tie-breaking
    """
    selected = []
    remaining = np.arange(len(candidates))
    
    while len(selected) < n_select and len(remaining) > 0:
        # For each remaining candidate, compute its "score" based on distance to already selected
        if len(selected) == 0:
            # Pick randomly from all candidates
            idx = rng.integers(len(remaining))
            chosen = remaining[idx]
        else:
            # Compute minimum distance to already selected points
            sel_pts = candidates[selected]
            rem_pts = candidates[remaining]
            
            # Distance matrix
            diff = rem_pts[:, np.newaxis, :] - sel_pts[np.newaxis, :, :]
            dists = np.sqrt(np.sum(diff ** 2, axis=2))
            min_dists = np.min(dists, axis=1)
            
            # Prefer points that are far from already selected
            # Add some randomness
            scores = min_dists + rng.random(len(remaining)) * 0.1
            idx = np.argmax(scores)
            chosen = remaining[idx]
        
        selected.append(chosen)
        remaining = np.delete(remaining, idx)
        
        # Remove candidates that are too close to the newly selected one
        if len(remaining) > 0:
            new_pt = candidates[chosen]
            rem_pts = candidates[remaining]
            diff = rem_pts - new_pt
            dists = np.sqrt(np.sum(diff ** 2, axis=1))
            keep_mask = dists >= 2 * r_est
            remaining = remaining[keep_mask]
    
    return np.array(selected)


def _local_optimize(points, max_time, beta_start=5.0, beta_end=50.0):
    """
    Local optimization using gradient-based soft-min minimization.
    """
    start_time = time.time()
    best_points = points.copy()
    best_score = _score(best_points)
    
    # Use L-BFGS-B with the soft-min energy
    def energy(x):
        pts = x.reshape(N, 3)
        # Ensure points stay in the cube
        pts = np.clip(pts, -0.499, 0.499)
        return _soft_min_energy(pts, beta=beta_start)
    
    def gradient(x):
        pts = x.reshape(N, 3)
        pts = np.clip(pts, -0.499, 0.499)
        g = _soft_min_grad(pts, beta=beta_start)
        return g.flatten()
    
    # Bounds: keep points strictly inside the cube
    bounds = [(-0.499, 0.499)] * (N * 3)
    
    x0 = best_points.flatten()
    
    # Run L-BFGS-B with time limit
    result = None
    try:
        result = minimize(
            energy, x0, jac=gradient,
            method='L-BFGS-B', bounds=bounds,
            options={'maxiter': 1000, 'ftol': 1e-10, 'gtol': 1e-8}
        )
    except Exception:
        pass
    
    if result is not None and result.success:
        new_points = np.clip(result.x.reshape(N, 3), -0.499, 0.499)
        new_score = _score(new_points)
        if new_score > best_score:
            best_points = new_points
            best_score = new_score
    
    # Second pass with higher beta for sharper approximation
    beta2 = beta_end
    x0 = best_points.flatten()
    
    def energy2(x):
        pts = x.reshape(N, 3)
        pts = np.clip(pts, -0.499, 0.499)
        return _soft_min_energy(pts, beta=beta2)
    
    def gradient2(x):
        pts = x.reshape(N, 3)
        pts = np.clip(pts, -0.499, 0.499)
        g = _soft_min_grad(pts, beta=beta2)
        return g.flatten()
    
    try:
        result2 = minimize(
            energy2, x0, jac=gradient2,
            method='L-BFGS-B', bounds=bounds,
            options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-8}
        )
    except Exception:
        pass
    
    if result2 is not None and result2.success:
        new_points = np.clip(result2.x.reshape(N, 3), -0.499, 0.499)
        new_score = _score(new_points)
        if new_score > best_score:
            best_points = new_points
            best_score = new_score
    
    return best_points, best_score


def construct():
    """Construct the sphere centers."""
    start_time = time.time()
    rng = np.random.default_rng(42)
    
    best_points = None
    best_score = 0.0
    
    # Different grid sizes to try
    grid_sizes = [7, 8, 9, 10]
    
    for grid_size in grid_sizes:
        if time.time() - start_time > DEADLINE_SECONDS - 5:
            break
        
        # Multiple restarts with different offsets
        n_restarts = 3
        for restart in range(n_restarts):
            if time.time() - start_time > DEADLINE_SECONDS - 5:
                break
            
            # Random offset for the grid
            offset = rng.random(3) * (1.0 / grid_size) * 0.5
            
            # Generate candidates
            candidates = _grid_candidates(grid_size, offset)
            
            if len(candidates) < N:
                continue
            
            # Estimate radius for greedy selection
            # For a grid of size g, spacing is ~1/g, so radius ~1/(2g)
            r_est = 1.0 / (2.0 * grid_size) * 0.9
            
            # Greedy selection
            selected_idx = _greedy_select(candidates, N, r_est, rng)
            
            if len(selected_idx) < N:
                continue
            
            points = candidates[selected_idx].copy()
            
            # Add small perturbation
            points += rng.random(N, 3) * 0.005
            points = np.clip(points, -0.499, 0.499)
            
            # Local optimization
            remaining_time = DEADLINE_SECONDS - (time.time() - start_time)
            if remaining_time < 5:
                break
            
            opt_points, opt_score = _local_optimize(points, remaining_time)
            
            if opt_score > best_score:
                best_score = opt_score
                best_points = opt_points.copy()
    
    if best_points is None:
        # Fallback: simple grid
        m = 4
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
        idx = np.random.permutation(len(grid))
        best_points = grid[idx[:N]].copy()
        best_points = np.clip(best_points, -0.499, 0.499)
    
    return best_points


# For quick testing
if __name__ == "__main__":
    centers = construct()
    print(f"Score: {_score(centers)}")
    print(f"Shape: {centers.shape}")
