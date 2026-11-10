"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 300

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


def _compute_score(points):
    """Compute the packing radius for given centres."""
    if len(points) != N:
        return 0.0
    if not np.all(np.isfinite(points)):
        return 0.0
    # Distance to boundary
    r_boundary = 1.0 - np.linalg.norm(points, axis=1)
    min_boundary = np.min(r_boundary)
    if min_boundary < 0:
        return 0.0
    # Pairwise distances
    # Use vectorized computation
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dist = np.linalg.norm(diff, axis=2)
    # Set diagonal to inf
    np.fill_diagonal(dist, np.inf)
    min_pair = np.min(dist) / 2.0
    return min(min_boundary, min_pair)


def _initial_hex_lattice():
    """Generate initial points using hexagonal lattice."""
    spacing = 2.0 / (np.sqrt(N) + 1.0)
    while True:
        k = int(1.0 / spacing) + 2
        rows = [((i + 0.5 * (j % 2)) * spacing, j * spacing * np.sqrt(3.0) / 2.0)
                for i in range(-k, k + 1) for j in range(-k, k + 1)]
        points = np.array(rows)
        radius = np.hypot(points[:, 0], points[:, 1])
        mask = radius <= 1.0 - spacing / 2.0
        points, radius = points[mask], radius[mask]
        if len(points) >= N:
            return points[np.argsort(radius, kind="stable")[:N]]
        spacing *= 0.98


def _local_optimize(points, time_limit=110.0):
    """Local optimization using gradient-free methods on the packing radius."""
    start_time = time.time()
    best_points = points.copy()
    best_score = _compute_score(best_points)
    
    # Use Nelder-Mead or similar? Too slow for 300*2=600 dims.
    # Instead, use a simple iterative repulsion + boundary attraction method.
    
    # We'll use a simple gradient ascent on a smooth surrogate.
    # Objective: maximize r = min(min_pair_dist/2, min_boundary_dist)
    # Surrogate: soft-min using log-sum-exp with temperature beta.
    
    beta = 50.0  # temperature for soft-min
    lr = 0.01
    iterations = 5000
    
    for it in range(iterations):
        if time.time() - start_time > time_limit:
            break
        
        # Compute distances
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dist = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dist, np.inf)
        
        # Boundary distance
        r_boundary = 1.0 - np.linalg.norm(points, axis=1)
        
        # Compute gradient of soft-min
        # soft_min(x) = -1/beta * log(sum(exp(-beta*x_i)))
        # d(soft_min)/dx_i = -exp(-beta*x_i) / sum(exp(-beta*x_j))
        
        # For pair distances: each pair (i,j) contributes to both i and j
        # We need to handle this carefully.
        
        # Let's use a simpler approach: just do repulsion + boundary constraint
        # Force: each point repels others, attracted to center if too far, repelled from boundary
        
        forces = np.zeros_like(points)
        
        # Pairwise repulsion
        for i in range(N):
            for j in range(i+1, N):
                d = points[i] - points[j]
                dist_ij = np.linalg.norm(d)
                if dist_ij < 1e-10:
                    d = np.random.randn(2) * 1e-5
                    dist_ij = np.linalg.norm(d)
                # Repulsion force
                force_mag = 1.0 / (dist_ij ** 2)
                forces[i] += force_mag * d / dist_ij
                forces[j] -= force_mag * d / dist_ij
        
        # Boundary force: keep points inside
        r = np.linalg.norm(points, axis=1)
        for i in range(N):
            if r[i] > 0.95:  # near boundary
                # Push inward
                forces[i] -= (points[i] / r[i]) * 10.0 * (r[i] - 0.95)
        
        # Update
        points = points + lr * forces
        
        # Project back to valid region if needed
        r = np.linalg.norm(points, axis=1)
        mask = r > 1.0
        if np.any(mask):
            points[mask] = points[mask] * (1.0 / r[mask])
        
        # Check score
        score = _compute_score(points)
        if score > best_score:
            best_score = score
            best_points = points.copy()
        
        # Decay learning rate
        lr *= 0.999
    
    return best_points, best_score


def construct():
    start_time = time.time()
    deadline = 110.0  # safety margin before 120s limit
    
    # Initial hexagonal lattice
    points = _initial_hex_lattice()
    score = _compute_score(points)
    
    # Local optimization
    points, score = _local_optimize(points, time_limit=deadline - (time.time() - start_time))
    
    # Final check
    if time.time() - start_time < deadline:
        # Try a few more iterations with smaller step
        for _ in range(100):
            if time.time() - start_time > deadline:
                break
            diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
            dist = np.linalg.norm(diff, axis=2)
            np.fill_diagonal(dist, np.inf)
            min_dist = np.min(dist)
            
            # Find closest pair and separate them slightly
            idx = np.unravel_index(np.argmin(dist), dist.shape)
            i, j = idx
            d = points[i] - points[j]
            dist_ij = np.linalg.norm(d)
            if dist_ij < min_dist + 1e-10:
                d = d / dist_ij if dist_ij > 1e-10 else np.array([1.0, 0.0])
                points[i] += 0.001 * d
                points[j] -= 0.001 * d
                # Keep inside boundary
                r = np.linalg.norm(points, axis=1)
                mask = r > 1.0
                if np.any(mask):
                    points[mask] = points[mask] * (1.0 / r[mask])
    
    return points
# EVOLVE-BLOCK-END
