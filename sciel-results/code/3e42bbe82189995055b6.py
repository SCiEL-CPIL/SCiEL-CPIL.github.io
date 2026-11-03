"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 300

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
import random

def _score(positions):
    """Compute the packing radius for a set of positions."""
    if len(positions) < 2:
        return 0.0
    # Minimum distance to boundary
    radii = np.linalg.norm(positions, axis=1)
    boundary_dist = 1.0 - radii
    min_boundary = np.min(boundary_dist)
    
    # Minimum pairwise distance / 2
    # Use a more efficient approach for pairwise distances
    # But for 300 points, O(N^2) is fine
    n = len(positions)
    min_pair = float('inf')
    for i in range(n):
        for j in range(i+1, n):
            d = np.linalg.norm(positions[i] - positions[j])
            if d < min_pair:
                min_pair = d
    return min(min_boundary, min_pair / 2.0)


def _hexagonal_init(n_points, seed=42):
    """Generate initial hexagonal packing."""
    rng = np.random.RandomState(seed)
    
    # Start with a hexagonal grid
    spacing = 2.0 / (np.sqrt(n_points) + 1.0)
    while True:
        k = int(1.0 / spacing) + 2
        rows = [((i + 0.5 * (j % 2)) * spacing, j * spacing * np.sqrt(3.0) / 2.0)
                for i in range(-k, k + 1) for j in range(-k, k + 1)]
        points = np.array(rows)
        radius = np.hypot(points[:, 0], points[:, 1])
        points, radius = points[radius <= 1.0 - spacing / 2.0], radius[radius <= 1.0 - spacing / 2.0]
        if len(points) >= n_points:
            # Add small random perturbation to break symmetry
            points = points + rng.randn(n_points, 2) * spacing * 0.05
            # Keep within bounds
            norms = np.linalg.norm(points, axis=1)
            points[norms > 1.0] = points[norms > 1.0] * (0.99 / norms[norms > 1.0, None])
            return points[:n_points]
        spacing *= 0.98


def _soft_min_penalty(positions, r, beta=50.0):
    """
    Soft minimum of pairwise distances and boundary distances.
    Returns a differentiable objective to maximize.
    """
    n = len(positions)
    # Boundary terms
    radii = np.linalg.norm(positions, axis=1)
    boundary = 1.0 - radii
    
    # Pairwise terms
    # Compute all pairwise distances
    diffs = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    dists = np.linalg.norm(diffs, axis=2)
    np.fill_diagonal(dists, float('inf'))
    
    # Soft min: -log(sum(exp(-beta * x))) / beta
    # For boundary
    boundary_soft = -np.log(np.sum(np.exp(-beta * boundary))) / beta
    # For pairwise (only upper triangle to avoid double counting)
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    pair_dists = dists[mask]
    pair_soft = -np.log(np.sum(np.exp(-beta * pair_dists))) / beta
    
    # The actual constraint is min(boundary, pair_dists/2) >= r
    # So we want to maximize the soft min of (boundary - r) and (pair_dists/2 - r)
    # But for optimization, let's maximize the soft min of the actual values
    # and check if it exceeds r
    
    # Actually, let's define the objective as the soft minimum of all constraints
    # constraint_i = boundary_i - r for i in range(n)
    # constraint_j = pair_dists_j/2 - r for j in pairs
    
    # Simpler: maximize the soft min of (boundary) and (pair_dists/2)
    # and see if it's > r
    
    # Combine all terms
    all_terms = np.concatenate([boundary, pair_dists / 2.0])
    soft_min_val = -np.log(np.sum(np.exp(-beta * all_terms))) / beta
    return soft_min_val


def _gradient_optimize(positions, max_iter=2000, time_limit=100.0):
    """
    Use gradient-based optimization to maximize the packing radius.
    """
    start_time = time.time()
    n = len(positions)
    
    def objective(x):
        """Negative of soft min packing radius (to minimize)."""
        elapsed = time.time() - start_time
        if elapsed > time_limit:
            return 0.0  # Signal timeout
        pos = x.reshape(-1, 2)
        # Keep within unit circle
        norms = np.linalg.norm(pos, axis=1)
        pos[norms > 1.0] = pos[norms > 1.0] * (0.999 / norms[norms > 1.0, None])
        return -_soft_min_penalty(pos, r=0.0, beta=30.0)
    
    def grad(x):
        """Gradient of the objective via finite differences (fallback)."""
        elapsed = time.time() - start_time
        if elapsed > time_limit:
            return np.zeros_like(x)
        # Use numerical gradient
        eps = 1e-6
        g = np.zeros_like(x)
        for i in range(len(x)):
            x_plus = x.copy()
            x_plus[i] += eps
            x_minus = x.copy()
            x_minus[i] -= eps
            g[i] = (objective(x_plus) - objective(x_minus)) / (2 * eps)
        return g
    
    # Use L-BFGS-B for box-constrained optimization
    x0 = positions.flatten()
    bounds = [(-1.0, 1.0)] * (2 * n)
    
    best_x = x0.copy()
    best_val = objective(x0)
    
    for restart in range(3):
        if time.time() - start_time > time_limit - 10:
            break
        if restart > 0:
            # Add perturbation for restart
            x0 = x0 + np.random.randn(2 * n) * 0.01
            norms = np.linalg.norm(x0.reshape(-1, 2), axis=1)
            x0[norms > 1.0] = x0[norms > 1.0] * (0.99 / norms[norms > 1.0, None])
        
        result = minimize(objective, x0, method='L-BFGS-B', bounds=bounds,
                          options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-8})
        
        if result.fun < best_val:
            best_val = result.fun
            best_x = result.x
        
        x0 = result.x
    
    pos = best_x.reshape(-1, 2)
    # Final cleanup: ensure inside unit circle
    norms = np.linalg.norm(pos, axis=1)
    pos[norms > 1.0] = pos[norms > 1.0] * (0.999 / norms[norms > 1.0, None])
    return pos


def _local_refine(positions, max_iter=500, time_limit=50.0):
    """
    Local refinement using direct coordinate optimization.
    """
    start_time = time.time()
    n = len(positions)
    pos = positions.copy()
    
    for it in range(max_iter):
        if time.time() - start_time > time_limit:
            break
        
        # Compute current packing radius
        r = _score(pos)
        
        # Find the most constrained pair or boundary
        radii = np.linalg.norm(pos, axis=1)
        boundary_dist = 1.0 - radii
        min_boundary_idx = np.argmin(boundary_dist)
        
        # Find closest pair
        min_pair_dist = float('inf')
        min_pair = (0, 1)
        for i in range(n):
            for j in range(i+1, n):
                d = np.linalg.norm(pos[i] - pos[j])
                if d < min_pair_dist:
                    min_pair_dist = d
                    min_pair = (i, j)
        
        # If boundary is the constraint, push the point inward
        if boundary_dist[min_boundary_idx] < min_pair_dist / 2.0:
            # Move the point toward the center
            direction = -pos[min_boundary_idx] / (np.linalg.norm(pos[min_boundary_idx]) + 1e-10)
            pos[min_boundary_idx] += direction * 0.001
        else:
            # Move the pair apart
            i, j = min_pair
            diff = pos[j] - pos[i]
            norm = np.linalg.norm(diff)
            if norm < 1e-10:
                continue
            direction = diff / norm
            pos[i] -= direction * 0.0005
            pos[j] += direction * 0.0005
        
        # Ensure all points stay within the unit circle
        norms = np.linalg.norm(pos, axis=1)
        pos[norms > 1.0] = pos[norms > 1.0] * (0.999 / norms[norms > 1.0, None])
    
    return pos


def construct():
    """Construct 300 circle centers in the unit disk to maximize packing radius."""
    start_time = time.time()
    deadline = start_time + 140.0  # Leave 10s margin
    
    # Step 1: Generate initial hexagonal packing
    positions = _hexagonal_init(N, seed=42)
    current_score = _score(positions)
    
    # Step 2: Gradient-based optimization
    if time.time() < deadline - 30:
        positions = _gradient_optimize(positions, max_iter=1000, time_limit=deadline - time.time() - 20)
        new_score = _score(positions)
        if new_score > current_score:
            current_score = new_score
    
    # Step 3: Local refinement
    if time.time() < deadline - 10:
        positions = _local_refine(positions, max_iter=300, time_limit=deadline - time.time() - 5)
        new_score = _score(positions)
        if new_score > current_score:
            current_score = new_score
    
    # Final check: ensure validity
    norms = np.linalg.norm(positions, axis=1)
    positions[norms > 1.0] = positions[norms > 1.0] * (0.999 / norms[norms > 1.0, None])
    
    return positions
# EVOLVE-BLOCK-END
