"""Rewritten program: force-directed relaxation starting from a hexagonal lattice."""
import time
import numpy as np

N = 200

def _score(points):
    """Compute the packing radius for a set of 2D points in [-0.5, 0.5]^2."""
    # Pairwise distances
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    # Fill diagonal with inf to ignore self-distances
    np.fill_diagonal(dists, np.inf)
    min_dist = np.min(dists)
    
    # Distance to boundary
    # Points must be in [-0.5, 0.5]^2
    # Distance to each wall:
    # Left wall: x - (-0.5) = x + 0.5
    # Right wall: 0.5 - x
    # Bottom wall: y - (-0.5) = y + 0.5
    # Top wall: 0.5 - y
    d_left = points[:, 0] + 0.5
    d_right = 0.5 - points[:, 0]
    d_bottom = points[:, 1] + 0.5
    d_top = 0.5 - points[:, 1]
    min_boundary = np.min(np.min(np.stack([d_left, d_right, d_bottom, d_top], axis=1), axis=1))
    
    # The radius is half the minimum of (min pairwise distance, min boundary distance)
    r = min(min_dist / 2.0, min_boundary)
    return r

def _hexagonal_lattice_init(n):
    """
    Generate an initial configuration of n points based on a hexagonal lattice
    within the unit square [-0.5, 0.5]^2.
    """
    # Hexagonal lattice parameters
    # For a hexagonal lattice, the distance between nearest neighbors is 'a'.
    # We want to fit as many points as possible.
    # Let's try to fit roughly sqrt(N) rows.
    # The optimal packing fraction for hexagonal is pi / (2 * sqrt(3)) * a^2 / area_per_point?
    # Area per point in hex lattice = (sqrt(3)/2) * a^2.
    # Total area = 1.
    # N * (sqrt(3)/2) * a^2 <= 1  =>  a <= sqrt(2 / (N * sqrt(3)))
    
    a = np.sqrt(2.0 / (N * np.sqrt(3.0)))
    
    # Generate points on hex lattice
    # Row spacing: a * sqrt(3) / 2
    # Column spacing: a
    # Offset for odd rows: a / 2
    
    points = []
    # We need to cover [-0.5, 0.5] x [-0.5, 0.5]
    # Let's generate a grid of points and then select the best N
    
    # Estimate number of rows and columns
    row_spacing = a * np.sqrt(3.0) / 2.0
    col_spacing = a
    
    # Generate a range of x and y
    # To be safe, generate a slightly larger grid and clip
    x_min, x_max = -0.5 - a, 0.5 + a
    y_min, y_max = -0.5 - a, 0.5 + a
    
    y = y_min
    row_idx = 0
    while y <= y_max:
        x_offset = 0.0 if row_idx % 2 == 0 else col_spacing / 2.0
        x = x_min + x_offset
        while x <= x_max:
            points.append([x, y])
            x += col_spacing
        y += row_spacing
        row_idx += 1
    
    points = np.array(points)
    
    # Keep only points strictly inside the square (with some margin? No, just inside)
    # Actually, the scoring function handles boundaries, but for initialization,
    # we want points that are not too close to the boundary to avoid immediate conflicts?
    # No, we just want a valid starting point. Points on the boundary are okay, but
    # force-directed will push them in.
    # Let's filter points that are within [-0.5, 0.5]
    mask = (points[:, 0] >= -0.5) & (points[:, 0] <= 0.5) & (points[:, 1] >= -0.5) & (points[:, 1] <= 0.5)
    points = points[mask]
    
    # If we have more than N points, we need to select N.
    # A simple strategy: sort by distance to center? Or just take the first N?
    # Taking the first N might bias towards one corner.
    # Better: compute the score for all subsets? Too expensive.
    # Heuristic: select points that are well-distributed.
    # We can use a greedy farthest-point sampling on the lattice points to pick N.
    
    if len(points) > N:
        # Farthest point sampling
        selected = []
        # Start with the point closest to the center (0,0)
        dists_to_center = np.sum(points ** 2, axis=1)
        start_idx = np.argmin(dists_to_center)
        selected.append(points[start_idx])
        
        remaining = np.delete(points, start_idx, axis=0)
        
        for _ in range(N - 1):
            if len(remaining) == 0:
                break
            # For each remaining point, find its minimum distance to any selected point
            min_dists = []
            for p in remaining:
                d = np.min(np.sum((p - np.array(selected)) ** 2, axis=1))
                min_dists.append(d)
            min_dists = np.array(min_dists)
            farthest_idx = np.argmax(min_dists)
            selected.append(remaining[farthest_idx])
            remaining = np.delete(remaining, farthest_idx, axis=0)
        
        points = np.array(selected)
    elif len(points) < N:
        # If we don't have enough points, add random points inside the square
        # This shouldn't happen with the grid generation, but just in case
        while len(points) < N:
            p = np.random.uniform(-0.5, 0.5, 2)
            points = np.vstack([points, p])
    
    return points[:N]

def _force_directed_relaxation(points, max_iterations=5000, deadline=None):
    """
    Apply force-directed relaxation to maximize the minimum distance.
    """
    n = len(points)
    points = points.copy()
    
    # Parameters
    k = 1.0  # Spring constant
    damping = 0.5
    step_size = 0.01
    
    # Precompute indices for efficiency? No, N=200 is small enough.
    
    for i in range(max_iterations):
        if deadline is not None and time.time() > deadline:
            break
            
        # Compute forces
        forces = np.zeros_like(points)
        
        for j in range(n):
            for k_idx in range(j + 1, n):
                diff = points[j] - points[k_idx]
                dist = np.linalg.norm(diff)
                if dist < 1e-10:
                    # Add small random perturbation
                    diff = np.random.uniform(-1e-6, 1e-6, 2)
                    dist = np.linalg.norm(diff)
                
                # Repulsive force
                # We want to maximize min distance.
                # A common approach is to use a potential energy function like 1/d^2
                # or simply push apart if dist < target.
                # Let's use a simple repulsive force: F = k / dist^2
                # This will push all points apart, but we need to keep them in the box.
                
                force_mag = k / (dist ** 2)
                force_dir = diff / dist
                
                forces[j] += force_mag * force_dir
                forces[k_idx] -= force_mag * force_dir
            
            # Boundary forces
            # Push points away from boundaries if they are too close
            # We want the distance to boundary to be at least r.
            # But we don't know r yet.
            # Let's use a penalty if distance to boundary < 0.1 (arbitrary)
            # Or better, just constrain the points to be inside the box.
            # If a point is outside, push it in.
            # If a point is close to the boundary, we might want to push it in more?
            # Actually, for maximizing min distance, we want points to be as far from boundary as possible.
            # So, let's add an attractive force towards the center? No, that would reduce distances.
            # Let's just clamp the points to the box after updating.
            
        # Update positions
        # Use a simple gradient ascent on the minimum distance?
        # Or just use the forces calculated above.
        # The forces above push points apart.
        # We also need to keep them in the box.
        
        # Simple update:
        points += damping * forces * step_size
        
        # Clamp to box
        points[:, 0] = np.clip(points[:, 0], -0.5, 0.5)
        points[:, 1] = np.clip(points[:, 1], -0.5, 0.5)
        
        # Optional: annealing? Reduce step size over time?
        # step_size *= 0.999
        
    return points

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    # Initialize with hexagonal lattice
    points = _hexagonal_lattice_init(N)
    
    # Relax
    points = _force_directed_relaxation(points, max_iterations=10000, deadline=deadline)
    
    # Check time
    if time.time() > deadline - 5.0:
        return points
    
    # Try a few more iterations with different parameters?
    # Or just return.
    
    return points
