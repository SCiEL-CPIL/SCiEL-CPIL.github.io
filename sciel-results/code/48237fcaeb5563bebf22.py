"""Rewritten program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import minimize
import time

N = 300


# EVOLVE-BLOCK-START
def _hex_lattice_init(n):
    """Generate a hexagonal lattice of points inside the unit disk, return n closest to origin."""
    # Estimate spacing
    # Area of hexagon with side s is (3*sqrt(3)/2)*s^2, but for packing circles of radius r in hex lattice,
    # the spacing between centers is 2r.
    # For n circles of radius r in a hex lattice packed in a circle of radius 1:
    # Approximate: n * (2r)^2 * (2/sqrt(3)) ≈ pi * 1^2
    # So r ≈ sqrt(pi * n / (4 * n * 2/sqrt(3))) ... let's just use a simple estimate.
    # Actually, the area per circle in hex lattice is (2r)^2 * sqrt(3)/2 = 2*sqrt(3)*r^2
    # Total area ≈ n * 2*sqrt(3)*r^2 ≈ pi
    # r ≈ sqrt(pi / (2*sqrt(3)*n))
    r_est = np.sqrt(np.pi / (2 * np.sqrt(3) * n))
    spacing = 2 * r_est * 0.95  # slight shrink to ensure we get enough points
    
    while True:
        k = int(1.0 / spacing) + 3
        rows = [((i + 0.5 * (j % 2)) * spacing, j * spacing * np.sqrt(3.0) / 2.0)
                for i in range(-k, k + 1) for j in range(-k, k + 1)]
        points = np.array(rows)
        radius = np.hypot(points[:, 0], points[:, 1])
        mask = radius <= 1.0 - spacing / 2.0
        points = points[mask]
        if len(points) >= n:
            # Sort by distance to origin, take closest n
            idx = np.argsort(radius, kind="stable")[:n]
            return points[idx]
        spacing *= 0.98


def _compute_score(centers):
    """Compute the score: min(min pairwise distance / 2, min (1 - |c_i|))."""
    if len(centers) < 2:
        return 0.0
    # Pairwise distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists)
    # Distance to boundary
    radii = np.hypot(centers[:, 0], centers[:, 1])
    min_boundary = np.min(1.0 - radii)
    return min(min_pair_dist / 2.0, min_boundary)


def construct():
    start_time = time.time()
    deadline = start_time + 120.0  # 120s wall-clock limit, keep safety margin
    
    # Step 1: Generate initial hexagonal lattice
    centers = _hex_lattice_init(N).astype(np.float64)
    current_score = _compute_score(centers)
    
    if time.time() > deadline - 5.0:
        return centers
    
    # Step 2: Use scipy minimize to maximize the minimum distance
    # We minimize the negative of the score, i.e., minimize -min(min_pair_dist/2, min_boundary)
    # This is equivalent to maximizing the minimum of (pair_dist/2, boundary_dist)
    
    def objective(x):
        """Objective to minimize: -min(min_pair_dist/2, min_boundary)"""
        c = x.reshape(-1, 2)
        if np.any(np.isnan(c)) or np.any(np.isinf(c)):
            return 1e6
        diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff * diff, axis=2))
        np.fill_diagonal(dists, np.inf)
        min_pair_dist = np.min(dists)
        radii = np.hypot(c[:, 0], c[:, 1])
        min_boundary = np.min(1.0 - radii)
        # We want to maximize min(min_pair_dist/2, min_boundary)
        # So minimize -min(min_pair_dist/2, min_boundary)
        return -min(min_pair_dist / 2.0, min_boundary)
    
    # Use Nelder-Mead or Powell for derivative-free optimization
    # Since the objective is non-smooth, Nelder-Mead is a good choice
    # But it can be slow for 600 dimensions. Let's try a more efficient approach.
    
    # Alternative: Use a local search approach with random restarts
    # Or use scipy.optimize.minimize with method='Nelder-Mead' but with limited iterations
    
    # Let's try a simpler approach: iterative local improvement
    # For each point, try to move it slightly to improve the minimum distance
    
    best_centers = centers.copy()
    best_score = current_score
    
    # Iterative local optimization
    max_iter = 50
    step_size = 0.01
    np.random.seed(42)
    
    for iteration in range(max_iter):
        if time.time() > deadline - 10.0:
            break
        
        # Try to improve each point
        for i in range(N):
            if time.time() > deadline - 10.0:
                break
            
            # Calculate current minimum distance for point i
            diff_i = centers[i] - centers
            dists_i = np.sqrt(np.sum(diff_i * diff_i, axis=1))
            dists_i[i] = np.inf
            min_dist_i = np.min(dists_i)
            
            # Distance to boundary for point i
            radius_i = np.hypot(centers[i, 0], centers[i, 1])
            boundary_i = 1.0 - radius_i
            
            # The limiting factor for point i
            limit = min(min_dist_i / 2.0, boundary_i)
            
            # Try small random perturbations
            improved = False
            for _ in range(20):
                if time.time() > deadline - 10.0:
                    break
                
                # Random direction
                direction = np.random.randn(2)
                direction /= np.linalg.norm(direction)
                
                # Try moving in this direction
                new_center = centers[i] + direction * step_size
                new_radius = np.hypot(new_center[0], new_center[1])
                if new_radius > 1.0:
                    continue
                
                # Check if this improves the global minimum
                # Calculate new distances from new_center to all other points
                diff_new = new_center - centers
                diff_new[i] = 0  # ignore self
                dists_new = np.sqrt(np.sum(diff_new * diff_new, axis=1))
                min_dist_new = np.min(dists_new)
                boundary_new = 1.0 - new_radius
                
                new_limit = min(min_dist_new / 2.0, boundary_new)
                
                # Also need to check if this affects other points' minimum distances
                # This is expensive, so let's just check if the global score improves
                temp_centers = centers.copy()
                temp_centers[i] = new_center
                temp_score = _compute_score(temp_centers)
                
                if temp_score > best_score:
                    centers = temp_centers
                    best_centers = centers.copy()
                    best_score = temp_score
                    improved = True
                    break
        
        if not improved:
            # No improvement in this iteration, reduce step size
            step_size *= 0.8
            if step_size < 1e-6:
                break
    
    return best_centers
# EVOLVE-BLOCK-END
