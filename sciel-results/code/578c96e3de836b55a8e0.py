"""Improved packing for 100 equal spheres in a unit cube."""
import numpy as np
from scipy.optimize import minimize
import itertools

N = 100

# EVOLVE-BLOCK-START
def _fcc_lattice_points(n_per_axis=5):
    """Generate FCC lattice points. FCC has points at corners and face centers of each cube."""
    points = []
    for i in range(n_per_axis):
        for j in range(n_per_axis):
            for k in range(n_per_axis):
                points.append([i, j, k])
                points.append([i + 0.5, j + 0.5, k])
                points.append([i + 0.5, j, k + 0.5])
                points.append([i, j + 0.5, k + 0.5])
    return np.array(points, dtype=float)

def _compute_radius(centers):
    """Compute the maximum radius for given centers."""
    diffs = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diffs ** 2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists) / 2.0
    wall_dists = np.min(np.abs(centers - 0.5), axis=1)
    min_wall = np.min(wall_dists)
    return min(min_pair, min_wall)

def _negative_radius(x):
    """Objective: negative of the radius (to minimize)."""
    x = x.reshape(-1, 3)
    x = np.clip(x, -0.5 + 1e-12, 0.5 - 1e-12)
    return -_compute_radius(x)

def _local_optimize(x0, max_iter=300):
    """Local optimization using L-BFGS-B."""
    bounds = [(-0.5 + 1e-10, 0.5 - 1e-10)] * (N * 3)
    result = minimize(
        _negative_radius,
        x0,
        method='L-BFGS-B',
        bounds=bounds,
        options={'maxiter': max_iter, 'ftol': 1e-14, 'gtol': 1e-10}
    )
    return result.x, -result.fun

def _repulsion_step(x, step_size=0.005):
    """One step of repulsion-based gradient descent (vectorized)."""
    x = x.reshape(-1, 3).copy()
    diffs = x[:, np.newaxis, :] - x[np.newaxis, :, :]
    dists_sq = np.sum(diffs ** 2, axis=2)
    np.fill_diagonal(dists_sq, np.inf)
    
    with np.errstate(divide='ignore', invalid='ignore'):
        inv_d3 = 1.0 / (dists_sq ** 1.5)
    np.fill_diagonal(inv_d3, 0.0)
    
    forces = np.sum(diffs * inv_d3[:, :, np.newaxis], axis=1)
    
    wall_dists = np.min(np.abs(x - 0.5), axis=1, keepdims=True)
    with np.errstate(divide='ignore', invalid='ignore'):
        wall_force_mag = 1.0 / (np.maximum(wall_dists, 1e-6) ** 2)
    wall_dir = np.sign(x - 0.5)
    wall_forces = wall_dir * wall_force_mag
    
    x_new = x + step_size * (forces + wall_forces)
    x_new = np.clip(x_new, -0.49, 0.49)
    return x_new.flatten()

def construct():
    """Construct 100 sphere centers in [-0.5, 0.5]^3 maximizing the radius."""
    best_x = None
    best_r = 0.0
    
    # Strategy 1: FCC lattice with various parameters
    for n_per_axis in [4, 5]:
        pts = _fcc_lattice_points(n_per_axis)
        
        min_pt = pts.min(axis=0)
        max_pt = pts.max(axis=0)
        span = max_pt - min_pt
        max_span = np.max(span)
        scale = 1.0 / max_span
        center = (min_pt + max_pt) / 2.0
        pts = (pts - center) * scale
        pts = np.clip(pts, -0.49, 0.49)
        
        if len(pts) >= N:
            selected = [np.random.randint(len(pts))]
            while len(selected) < N:
                sel_pts = pts[selected]
                min_dists = np.full(len(pts), np.inf)
                for sp in sel_pts:
                    d = np.sqrt(np.sum((pts - sp) ** 2, axis=1))
                    min_dists = np.minimum(min_dists, d)
                for s in selected:
                    min_dists[s] = 0
                next_idx = np.argmax(min_dists)
                selected.append(next_idx)
            
            initial = pts[selected].flatten()
        else:
            initial = pts.flatten()
            while len(initial) < N * 3:
                initial = np.concatenate([initial, np.random.uniform(-0.4, 0.4, 3)])
            initial = initial[:N * 3]
        
        x_opt, r = _local_optimize(initial)
        if r > best_r:
            best_r = r
            best_x = x_opt
    
    # Strategy 2: Random restarts with repulsion-based initialization
    for restart in range(10):
        x0 = np.random.uniform(-0.45, 0.45, N * 3)
        
        for _ in range(30):
            x0 = _repulsion_step(x0, step_size=0.005)
        
        x_opt, r = _local_optimize(x0)
        if r > best_r:
            best_r = r
            best_x = x_opt
    
    # Final polish
    if best_x is not None:
        x_opt, r = _local_optimize(best_x, max_iter=500)
        if r > best_r:
            best_r = r
            best_x = x_opt
    
    result = best_x.reshape(N, 3)
    result = np.clip(result, -0.5 + 1e-10, 0.5 - 1e-10)
    
    return result
# EVOLVE-BLOCK-END
