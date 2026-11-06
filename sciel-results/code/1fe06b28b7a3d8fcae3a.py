"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 200

# EVOLVE-BLOCK-START
import numpy as np
import time
import math

def _score(pts):
    """Compute the radius (score) for a given set of points."""
    if N == 1:
        return min(0.5 - np.abs(pts[0, 0]), 0.5 - np.abs(pts[0, 1]))
    
    # Pairwise distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists)
    
    # Distance to walls
    wall_dists = np.minimum(0.5 - np.abs(pts), 0.5)
    min_wall_dist = np.min(wall_dists)
    
    return min(min_pair_dist / 2.0, min_wall_dist)

def _better_init():
    """
    Generate a better initial configuration using a hexagonal grid 
    that is adjusted to fit the square, then randomly subsample or add points.
    """
    # Target N=200
    # Hex grid parameters
    # We want roughly 200 points.
    # Area of square = 1.
    # Approx area per point = 1/200 = 0.005.
    # Hex cell area = s^2 * sqrt(3)/2.
    # s^2 * 0.866 = 0.005 => s^2 = 0.00576 => s = 0.0759.
    
    s = 0.075
    dy = s * math.sqrt(3) / 2
    
    points = []
    # Iterate over rows
    # y from -0.5 + s/2 to 0.5 - s/2
    y = -0.5 + s/2
    row_idx = 0
    while y <= 0.5 - s/2:
        # x offset for this row
        offset = 0 if row_idx % 2 == 0 else s/2
        x = -0.5 + s/2 + offset
        while x <= 0.5 - s/2:
            points.append([x, y])
            x += s
        y += dy
        row_idx += 1
        
    points = np.array(points)
    
    if len(points) > N:
        # Randomly select N points
        idx = np.random.choice(len(points), N, replace=False)
        points = points[idx]
    elif len(points) < N:
        # Add random points in empty spots
        # This is a bit crude, but SA will fix it
        while len(points) < N:
            # Pick random point in square
            p = np.random.uniform(-0.45, 0.45, 2)
            # Check if it's far enough from existing points (roughly s/2)
            # To speed up, just add it, SA will resolve overlaps
            points = np.vstack([points, p])
            
    return points

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    # Initialize
    np.random.seed(42)
    pts = _better_init()
    
    # Ensure valid shape
    if len(pts) != N:
        # Fallback to random if init failed badly
        pts = np.random.uniform(-0.45, 0.45, (N, 2))
        
    current_radius = _score(pts)
    best_radius = current_radius
    best_pts = pts.copy()
    
    # Simulated Annealing
    T = 0.01  # Initial temperature
    T_min = 1e-6
    alpha = 0.9999  # Cooling rate
    sigma = 0.005  # Initial step size
    
    steps = 0
    while time.time() < deadline:
        # Pick a random point to move
        i = np.random.randint(N)
        
        # Generate new position
        # Gaussian noise
        delta = np.random.normal(0, sigma, 2)
        new_pts = pts.copy()
        new_pts[i] += delta
        
        # Check bounds
        if np.any(new_pts[i] < -0.5) or np.any(new_pts[i] > 0.5):
            # Reject or reflect? Reject for now, or clip?
            # Clipping might cause overlaps. Reject is safer.
            # But if we reject too much, we waste time.
            # Let's try to keep it inside by reflecting or just rejecting.
            # Given sigma is small, this should be rare.
            continue
            
        # Calculate new radius
        new_radius = _score(new_pts)
        
        # Acceptance
        if new_radius > current_radius:
            pts = new_pts
            current_radius = new_radius
            if current_radius > best_radius:
                best_radius = current_radius
                best_pts = pts.copy()
        else:
            # Boltzmann probability
            delta_r = new_radius - current_radius
            if delta_r < 0:
                prob = math.exp(delta_r / T)
                if np.random.rand() < prob:
                    pts = new_pts
                    current_radius = new_radius
            
        # Cooling
        T *= alpha
        if T < T_min:
            T = T_min
            # If T is very low, we can also reduce sigma
            sigma *= 0.99
            
        steps += 1
        
    return best_pts
# EVOLVE-BLOCK-END
