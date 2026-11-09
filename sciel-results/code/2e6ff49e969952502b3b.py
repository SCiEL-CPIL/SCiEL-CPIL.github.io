"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


def _score(points):
    """Compute the packing score: min of pairwise distances/2 and (1 - |c_i|)."""
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=-1))
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists) / 2.0
    
    norms = np.sqrt(np.sum(points * points, axis=-1))
    min_boundary = np.min(1.0 - norms)
    
    return min(min_pair, min_boundary)


def _compute_score(points):
    """Alias for _score."""
    return _score(points)


def _hex_lattice_init(n=50):
    """Generate a hexagonal lattice initial configuration with better spacing."""
    # Try multiple spacings and pick the best one
    best_points = None
    best_score = -np.inf
    
    for spacing in [0.22, 0.23, 0.24, 0.25, 0.26]:
        points = []
        for j in range(-4, 5):
            for i in range(-4, 5):
                x = (i + 0.5 * (j % 2)) * spacing
                y = j * spacing * np.sqrt(3.0) / 2.0
                if x*x + y*y <= 0.9**2:
                    points.append([x, y])
        
        if len(points) < n:
            continue
            
        points = np.array(points)
        radii = np.hypot(points[:, 0], points[:, 1])
        idx = np.argsort(radii)[:n]
        points = points[idx]
        
        s = _score(points)
        if s > best_score:
            best_score = s
            best_points = points.copy()
    
    if best_points is None:
        # Fallback: random
        best_points = np.random.uniform(-0.8, 0.8, (n, 2))
    
    return best_points


def _slsqp_optimize(points, deadline=None):
    """Use SLSQP to maximize the minimum radius with explicit constraints."""
    N = points.shape[0]
    x0 = points.flatten().copy()
    
    def neg_score(x):
        pts = x.reshape(N, 2)
        return -_score(pts)
    
    # Constraints: all points must be inside unit circle
    constraints = []
    for i in range(N):
        constraints.append({
            'type': 'ineq',
            'fun': lambda x, i=i: 1.0 - np.sqrt(x[2*i]**2 + x[2*i+1]**2) - 0.001
        })
    
    # Bounds: keep points within [-1, 1]
    bounds = [(-1.0, 1.0)] * (N * 2)
    
    result = minimize(
        neg_score, x0, method='SLSQP',
        bounds=bounds, constraints=constraints,
        options={'maxiter': 200, 'ftol': 1e-12}
    )
    
    if result.success or result.fun < neg_score(x0):
        return result.x.reshape(N, 2)
    return points


def _local_polish(points, n_iter=1000, deadline=None):
    """Gradient-based local optimization with adaptive step size."""
    N = points.shape[0]
    points = points.copy()
    best_score = _score(points)
    step_size = 0.01
    
    for it in range(n_iter):
        if deadline is not None and time.time() > deadline - 2.0:
            break
        
        # Compute forces
        forces = np.zeros_like(points)
        
        # Pairwise repulsion - vectorized
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff * diff, axis=-1))
        np.fill_diagonal(dists, 1.0)
        
        # Only consider close pairs
        close_mask = dists < 0.35
        for i in range(N):
            for j in range(i+1, N):
                if close_mask[i, j]:
                    d = dists[i, j]
                    force_mag = 0.005 / max(d, 0.001)
                    direction = (points[i] - points[j]) / max(d, 0.001)
                    forces[i] += force_mag * direction
                    forces[j] -= force_mag * direction
        
        # Boundary attraction
        norms = np.hypot(points[:, 0], points[:, 1])
        for i in range(N):
            if norms[i] > 0.85:
                force_mag = 0.005 * (norms[i] - 0.85)
                direction = -points[i] / max(norms[i], 0.001)
                forces[i] += force_mag * direction
        
        # Update positions
        new_points = points + forces * step_size
        
        # Project into unit disk
        norms = np.hypot(new_points[:, 0], new_points[:, 1])
        mask = norms > 0.99
        if np.any(mask):
            scale = 0.99 / norms[mask]
            new_points[mask] *= scale[:, np.newaxis]
        
        new_score = _score(new_points)
        if new_score > best_score:
            points = new_points
            best_score = new_score
            step_size = min(step_size * 1.1, 0.05)
        else:
            step_size *= 0.9
    
    return points


def _final_polish(points, deadline=None):
    """High-precision final polish with very small steps."""
    N = points.shape[0]
    points = points.copy()
    best_score = _score(points)
    step_size = 0.001
    
    for it in range(2000):
        if deadline is not None and time.time() > deadline - 1.0:
            break
        
        # Very fine-grained perturbation
        new_points = points.copy()
        
        # Compute forces
        forces = np.zeros_like(points)
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff * diff, axis=-1))
        np.fill_diagonal(dists, 1.0)
        
        for i in range(N):
            for j in range(i+1, N):
                d = dists[i, j]
                if d < 0.28:
                    force_mag = 0.001 / max(d, 0.001)
                    direction = (points[i] - points[j]) / max(d, 0.001)
                    forces[i] += force_mag * direction
                    forces[j] -= force_mag * direction
        
        norms = np.hypot(points[:, 0], points[:, 1])
        for i in range(N):
            if norms[i] > 0.95:
                force_mag = 0.001 * (norms[i] - 0.95)
                direction = -points[i] / max(norms[i], 0.001)
                forces[i] += force_mag * direction
        
        new_points = points + forces * step_size
        
        # Project
        norms = np.hypot(new_points[:, 0], new_points[:, 1])
        mask = norms > 0.999
        if np.any(mask):
            scale = 0.999 / norms[mask]
            new_points[mask] *= scale[:, np.newaxis]
        
        new_score = _score(new_points)
        if new_score > best_score:
            points = new_points
            best_score = new_score
        else:
            step_size *= 0.95
    
    return points


def construct():
    """Construct optimal packing of 50 circles in unit circle."""
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin
    
    # Initial configuration from hexagonal lattice
    init_points = _hex_lattice_init(N)
    best_points = init_points.copy()
    best_score = _score(best_points)
    
    # SLSQP optimization
    if time.time() < deadline - 60.0:
        slsqp_points = _slsqp_optimize(init_points, deadline=deadline)
        slsqp_score = _score(slsqp_points)
        if slsqp_score > best_score:
            best_points = slsqp_points
            best_score = slsqp_score
    
    # Local polish
    if time.time() < deadline - 20.0:
        polished_points = _local_polish(best_points, n_iter=500, deadline=deadline)
        polished_score = _score(polished_points)
        if polished_score > best_score:
            best_points = polished_points
            best_score = polished_score
    
    # Final high-precision polish
    if time.time() < deadline - 5.0:
        final_points = _final_polish(best_points, deadline=deadline)
        final_score = _score(final_points)
        if final_score > best_score:
            best_points = final_points
            best_score = final_score
    
    # Final check: ensure all points are strictly inside
    norms = np.hypot(best_points[:, 0], best_points[:, 1])
    mask = norms >= 0.999
    if np.any(mask):
        scale = 0.999 / norms[mask]
        best_points[mask] *= scale[:, np.newaxis]
    
    return best_points
# EVOLVE-BLOCK-END
