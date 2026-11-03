"""Improved packing for 200 equal circles in a unit square."""
import numpy as np
import time

N = 200

# EVOLVE-BLOCK-START

def _compute_score(centers):
    """Compute the minimum radius for given centers."""
    # Boundary constraint: min distance to boundary
    # Square is [-0.5, 0.5]^2
    dist_to_boundary = np.minimum(
        np.minimum(0.5 - centers[:, 0], 0.5 + centers[:, 0]),
        np.minimum(0.5 - centers[:, 1], 0.5 + centers[:, 1])
    )
    boundary_min = np.min(dist_to_boundary)
    
    # Pairwise distances
    # Compute all pairwise distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=-1))
    # Set diagonal to inf
    np.fill_diagonal(dists, np.inf)
    pair_min = np.min(dists)
    
    return min(boundary_min, pair_min / 2.0)


def _hex_grid_init(seed=42):
    """Generate a hexagonal grid initialization with random offset."""
    rng = np.random.default_rng(seed)
    
    # Hexagonal grid spacing
    # For 200 circles in unit square, approximate spacing
    # Hex grid: rows offset by half spacing
    # Try a few different spacings
    
    best_centers = None
    best_score = -1
    
    for attempt in range(20):
        # Random offset for the hex grid
        offset_x = rng.uniform(-0.05, 0.05)
        offset_y = rng.uniform(-0.05, 0.05)
        
        # Hex grid parameters
        # Spacing in x direction
        dx = 1.0 / 15.0  # roughly 15 columns
        dy = dx * np.sqrt(3) / 2
        
        # Generate hex grid points
        points = []
        row = 0
        y = -0.5 + offset_y
        while y <= 0.5:
            x_offset = (dx / 2) if (row % 2 == 1) else 0
            x = -0.5 + offset_x + x_offset
            while x <= 0.5:
                points.append([x, y])
                x += dx
            y += dy
            row += 1
        
        points = np.array(points)
        
        if len(points) < N:
            continue
        
        # Select N points greedily: maximize min distance
        # Start with a random point, then add points that maximize min distance to existing
        rng_local = np.random.default_rng(seed + attempt)
        
        selected = [points[rng_local.integers(len(points))]]
        remaining = set(range(len(points)))
        remaining.remove(selected[0])
        
        for _ in range(N - 1):
            best_idx = -1
            best_min_dist = -1
            for idx in remaining:
                dists_to_selected = np.min(np.linalg.norm(points[idx] - np.array(selected), axis=1))
                if dists_to_selected > best_min_dist:
                    best_min_dist = dists_to_selected
                    best_idx = idx
            selected.append(points[best_idx])
            remaining.remove(best_idx)
        
        centers = np.array(selected)
        score = _compute_score(centers)
        
        if score > best_score:
            best_score = score
            best_centers = centers.copy()
    
    return best_centers


def _push_apart(centers, iterations=1000, learning_rate=0.01):
    """Push apart overlapping circles using gradient-like updates."""
    centers = centers.copy()
    n = len(centers)
    
    for it in range(iterations):
        # Compute forces
        forces = np.zeros_like(centers)
        
        for i in range(n):
            for j in range(i + 1, n):
                diff = centers[i] - centers[j]
                dist = np.linalg.norm(diff)
                if dist < 0.001:
                    # Random small push
                    forces[i] += np.random.randn(2) * 0.001
                    forces[j] -= np.random.randn(2) * 0.001
                else:
                    # Repulsive force if too close
                    if dist < 0.08:
                        force_mag = (0.08 - dist) / dist
                        forces[i] += force_mag * diff
                        forces[j] -= force_mag * diff
        
        # Boundary repulsion
        for i in range(n):
            for k in range(2):
                if centers[i, k] > 0.5 - 0.04:
                    forces[i, k] -= (centers[i, k] - (0.5 - 0.04)) * 0.5
                elif centers[i, k] < -0.5 + 0.04:
                    forces[i, k] += ((-0.5 + 0.04) - centers[i, k]) * 0.5
        
        # Update
        centers += learning_rate * forces
        
        # Project back into square
        centers[:, 0] = np.clip(centers[:, 0], -0.5, 0.5)
        centers[:, 1] = np.clip(centers[:, 1], -0.5, 0.5)
        
        # Decay learning rate
        learning_rate *= 0.999
    
    return centers


def _coordinate_ascent_polish(centers, max_iters=5000, tol=1e-12):
    """Polish using coordinate ascent: move each point to maximize its min distance."""
    centers = centers.copy()
    n = len(centers)
    
    for it in range(max_iters):
        improved = False
        for i in range(n):
            # Try to move point i to maximize min distance to others and boundary
            # Use a local search: try small perturbations
            best_pos = centers[i].copy()
            best_min_dist = _point_min_dist(centers, i)
            
            # Try moving in each direction
            for dx in [-0.002, -0.001, 0.001, 0.002]:
                for dy in [-0.002, -0.001, 0.001, 0.002]:
                    new_pos = centers[i] + np.array([dx, dy])
                    # Check boundary
                    if new_pos[0] < -0.5 or new_pos[0] > 0.5 or new_pos[1] < -0.5 or new_pos[1] > 0.5:
                        continue
                    
                    old_pos = centers[i]
                    centers[i] = new_pos
                    min_dist = _point_min_dist(centers, i)
                    
                    if min_dist > best_min_dist + tol:
                        best_min_dist = min_dist
                        best_pos = new_pos.copy()
                        improved = True
                    else:
                        centers[i] = old_pos
            
            centers[i] = best_pos
        
        if not improved:
            break
    
    return centers


def _point_min_dist(centers, i):
    """Compute the minimum distance from point i to all other points and boundary."""
    # Boundary
    dist_boundary = np.minimum(
        np.minimum(0.5 - centers[i, 0], 0.5 + centers[i, 0]),
        np.minimum(0.5 - centers[i, 1], 0.5 + centers[i, 1])
    )
    
    # Other points
    diff = centers - centers[i]
    dists = np.linalg.norm(diff, axis=1)
    dists[i] = np.inf
    dist_point = np.min(dists)
    
    return min(dist_boundary, dist_point)


def construct():
    start_time = time.time()
    deadline = start_time + 120  # 120 second limit
    
    # Step 1: Hex grid initialization
    centers = _hex_grid_init(seed=42)
    if centers is None:
        # Fallback to square grid
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        centers = grid[:N]
    
    best_centers = centers.copy()
    best_score = _compute_score(centers)
    
    # Step 2: Push apart to resolve overlaps
    if time.time() < deadline - 60:
        centers = _push_apart(centers, iterations=500, learning_rate=0.01)
        score = _compute_score(centers)
        if score > best_score:
            best_score = score
            best_centers = centers.copy()
    
    # Step 3: Coordinate ascent polish
    if time.time() < deadline - 30:
        centers = _coordinate_ascent_polish(best_centers, max_iters=2000, tol=1e-10)
        score = _compute_score(centers)
        if score > best_score:
            best_score = score
            best_centers = centers.copy()
    
    # Ensure all points are strictly inside the square
    best_centers[:, 0] = np.clip(best_centers[:, 0], -0.5 + 1e-10, 0.5 - 1e-10)
    best_centers[:, 1] = np.clip(best_centers[:, 1], -0.5 + 1e-10, 0.5 - 1e-10)
    
    return best_centers

# EVOLVE-BLOCK-END
