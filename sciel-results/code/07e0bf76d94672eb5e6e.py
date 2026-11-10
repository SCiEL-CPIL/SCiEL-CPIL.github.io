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
    """Vectorized local optimization using repulsion and boundary forces."""
    start_time = time.time()
    best_points = points.copy()
    best_score = _compute_score(best_points)
    
    lr = 0.02
    decay = 0.999
    
    # Pre-allocate for speed
    # We will use a vectorized force calculation
    
    while time.time() - start_time < time_limit:
        # Compute pairwise differences and distances
        # diff[i, j] = points[i] - points[j]
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dist = np.linalg.norm(diff, axis=2)
        
        # Avoid division by zero
        dist[dist < 1e-10] = 1e-10
        
        # Repulsion force: F_ij = (1 / dist^2) * (p_i - p_j) / dist
        # We want the force on i due to j.
        # force_ij_vec = diff[i, j] / dist[i, j]
        # force_mag = 1 / dist[i, j]^2
        # F_on_i = sum_j force_mag * force_ij_vec
        
        # Vectorized:
        # inv_dist2 = 1.0 / (dist ** 2)
        # inv_dist = 1.0 / dist
        # Force on i from j is: diff[i, j] * inv_dist[i, j] * inv_dist2[i, j]
        # Sum over j for each i.
        
        # Mask diagonal to avoid self-interaction
        np.fill_diagonal(dist, 1e10) # Large distance -> small force, avoids div by 0 issues in logs if any, but we do direct div
        
        inv_dist = 1.0 / dist
        inv_dist2 = inv_dist * inv_dist
        
        # Force tensor: shape (N, N, 2)
        # F[i, j] = diff[i, j] * inv_dist[i, j, None] * inv_dist2[i, j, None]
        force_tensor = diff * (inv_dist * inv_dist2)[:, :, np.newaxis]
        
        # Sum over j (axis 1) to get net force on each i
        forces = np.sum(force_tensor, axis=1)
        
        # Boundary force
        r = np.linalg.norm(points, axis=1)
        # Push inward if r > 0.9
        # Force direction is -points / r
        # Magnitude proportional to (r - 0.9)
        boundary_force = np.zeros_like(points)
        mask_boundary = r > 0.9
        if np.any(mask_boundary):
            # Normalized radial vector
            radial = points[mask_boundary] / r[mask_boundary, np.newaxis]
            # Push inward
            boundary_force[mask_boundary] = -radial * (r[mask_boundary, np.newaxis] - 0.9) * 5.0
            
        forces += boundary_force
        
        # Update points
        points = points + lr * forces
        
        # Project to unit circle
        r = np.linalg.norm(points, axis=1)
        mask = r > 1.0
        if np.any(mask):
            # Fix broadcasting: r[mask] is (K,), points[mask] is (K, 2)
            # Need (K, 1) for r
            points[mask] = points[mask] * (1.0 / r[mask])[:, np.newaxis]
        
        # Check score
        score = _compute_score(points)
        if score > best_score:
            best_score = score
            best_points = points.copy()
        
        lr *= decay

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
                    points[mask] = points[mask] * (1.0 / r[mask])[:, np.newaxis]
    
    return points
# EVOLVE-BLOCK-END
