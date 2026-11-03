"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time


def _compute_radius(centers):
    """Compute the maximum common radius for given centers."""
    n = len(centers)
    # Distance to walls
    wall_dists = 0.5 - np.abs(centers)  # shape (n, 3)
    min_wall = np.min(wall_dists)  # min over all coords and spheres
    
    # Pairwise distances
    # Use broadcasting for pairwise distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    # Set diagonal to inf
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists) / 2.0
    
    return min(min_wall, min_pair)


def _bcc_initialization(n):
    """Initialize n points on a BCC lattice scaled to fit unit cube."""
    # BCC lattice: points at corners and body centers of sub-cubes
    # We need to find a grid size that gives us approximately n points
    # BCC has 2 points per unit cell (1 corner + 1 body center)
    # So we need roughly n/2 unit cells, i.e., m^3 ≈ n/2
    
    # Try different grid sizes
    best_points = None
    best_radius = -1
    
    for m in range(2, 10):
        # Create BCC lattice points in [0, 1]^3
        points = []
        for i in range(m):
            for j in range(m):
                for k in range(m):
                    # Corner point
                    points.append([i/m, j/m, k/m])
                    # Body center
                    points.append([(i+0.5)/m, (j+0.5)/m, (k+0.5)/m])
        
        points = np.array(points)
        # Scale to fit in [-0.5, 0.5]^3
        points = points - 0.5
        
        # If we have more than n points, take the first n
        if len(points) >= n:
            pts = points[:n]
            r = _compute_radius(pts)
            if r > best_radius:
                best_radius = r
                best_points = pts
        
        # If we have fewer, pad with random points near center
        if len(points) < n:
            pts = points.copy()
            remaining = n - len(pts)
            # Add random points
            rng = np.random.RandomState(42)
            for _ in range(remaining):
                pts = np.vstack([pts, rng.uniform(-0.3, 0.3, 3)])
            r = _compute_radius(pts)
            if r > best_radius:
                best_radius = r
                best_points = pts
    
    return best_points


def _gradient_ascent(centers, max_iter=200, step_size=0.01, time_limit=None):
    """Perform gradient ascent on the minimum radius objective."""
    n = len(centers)
    centers = centers.copy()
    
    # Track best
    best_centers = centers.copy()
    best_radius = _compute_radius(centers)
    
    start_time = time.time()
    
    for it in range(max_iter):
        if time_limit is not None and (time.time() - start_time) > time_limit:
            break
        
        # Compute current radius and identify constraining pairs/walls
        wall_dists = 0.5 - np.abs(centers)  # (n, 3)
        min_wall_val = np.min(wall_dists)
        
        # Pairwise distances
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dists, np.inf)
        min_pair_val = np.min(dists) / 2.0
        
        current_r = min(min_wall_val, min_pair_val)
        
        # Compute gradient: direction to increase min radius
        grad = np.zeros_like(centers)
        
        # Wall constraints
        if min_wall_val <= min_pair_val:
            # Gradient from walls
            for i in range(n):
                for k in range(3):
                    d = wall_dists[i, k]
                    if d <= min_wall_val + 1e-10:
                        # Push away from wall
                        if centers[i, k] > 0:
                            grad[i, k] -= 1.0  # push toward negative
                        else:
                            grad[i, k] += 1.0  # push toward positive
        
        # Pair constraints
        if min_pair_val <= min_wall_val:
            for i in range(n):
                for j in range(i+1, n):
                    d = dists[i, j]
                    if d <= 2*min_pair_val + 1e-10:
                        # Push apart
                        direction = (centers[i] - centers[j]) / (d + 1e-15)
                        grad[i] += direction
                        grad[j] -= direction
        
        # Normalize gradient
        grad_norm = np.linalg.norm(grad)
        if grad_norm > 1e-15:
            grad = grad / grad_norm
        
        # Take step
        new_centers = centers + step_size * grad
        
        # Clamp to cube
        new_centers = np.clip(new_centers, -0.4999, 0.4999)
        
        new_radius = _compute_radius(new_centers)
        
        if new_radius > current_r - 1e-12:
            centers = new_centers
            if new_radius > best_radius:
                best_radius = new_radius
                best_centers = centers.copy()
        
        # Adapt step size
        if new_radius > current_r + 1e-10:
            step_size = min(step_size * 1.1, 0.05)
        else:
            step_size = max(step_size * 0.8, 0.001)
    
    return best_centers, best_radius


def construct():
    start_time = time.time()
    time_budget = 140.0  # Leave 10s margin
    
    # Initial BCC lattice
    centers = _bcc_initialization(N)
    if centers is None:
        # Fallback to grid
        m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        centers = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])[:N]
    
    best_centers = centers.copy()
    best_radius = _compute_radius(centers)
    
    # Multiple rounds of optimization with decreasing step sizes
    step_sizes = [0.03, 0.01, 0.005, 0.002, 0.001]
    max_iters = [100, 150, 200, 250, 300]
    
    for i, (ss, mi) in enumerate(zip(step_sizes, max_iters)):
        elapsed = time.time() - start_time
        if elapsed > time_budget - 5:
            break
        
        remaining_time = time_budget - elapsed - 2
        centers, radius = _gradient_ascent(
            centers, max_iter=mi, step_size=ss, time_limit=remaining_time
        )
        
        if radius > best_radius:
            best_radius = radius
            best_centers = centers.copy()
        
        elapsed = time.time() - start_time
        if elapsed > time_budget - 3:
            break
    
    # Final polish with very small steps
    elapsed = time.time() - start_time
    if elapsed < time_budget - 5:
        centers, radius = _gradient_ascent(
            best_centers, max_iter=100, step_size=0.0005, 
            time_limit=time_budget - elapsed - 2
        )
        if radius > best_radius:
            best_radius = radius
            best_centers = centers.copy()
    
    # Ensure validity: clip to cube
    best_centers = np.clip(best_centers, -0.5 + 1e-10, 0.5 - 1e-10)
    
    return best_centers
# EVOLVE-BLOCK-END
