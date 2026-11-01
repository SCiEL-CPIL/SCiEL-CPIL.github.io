"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 130

# EVOLVE-BLOCK-START
import numpy as np
import time

def _normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm

def construct():
    # 1. Initialization: Fibonacci Spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # Pre-compute a grid on the sphere for evaluating covering radius
    # Using a lat-long grid with sufficient resolution
    # 100x200 grid = 20000 points, which is fast enough
    n_lat = 100
    n_lon = 200
    lats = np.linspace(-np.pi/2, np.pi/2, n_lat, endpoint=False) + np.pi/n_lat
    lons = np.linspace(0, 2*np.pi, n_lon, endpoint=False)
    
    # Create meshgrid
    LATS, LONS = np.meshgrid(lats, lons, indexing='ij')
    
    # Convert to Cartesian
    grid_x = np.cos(LATS) * np.cos(LONS)
    grid_y = np.cos(LATS) * np.sin(LONS)
    grid_z = np.sin(LATS)
    grid_points = np.column_stack([grid_x.ravel(), grid_y.ravel(), grid_z.ravel()])
    # Normalize just in case (they should be unit)
    grid_points = grid_points / np.linalg.norm(grid_points, axis=1, keepdims=True)
    
    start_time = time.time()
    time_limit = 115.0
    
    step_size = 0.01
    repulsive_strength = 0.005
    patience = 0
    max_patience = 200
    prev_radius = np.pi
    
    while time.time() - start_time < time_limit:
        # 1. Evaluate covering radius on the grid
        # Calculate distance from each grid point to each configuration point
        # Use dot product for speed: cos(d) = dot(g, p)
        dots = np.dot(grid_points, points.T) # Shape (G, N)
        dots = np.clip(dots, -1.0, 1.0)
        # We want the MAX distance, which corresponds to the MIN dot product
        # Actually, we want the distance to the NEAREST point.
        # Nearest point has MAX dot product (closest to 1).
        # So min angular distance = arccos(max dot product).
        max_dots = np.max(dots, axis=1) # Shape (G,)
        min_dists = np.arccos(max_dots) # Angular distance to nearest point
        
        current_radius = np.max(min_dists)
        
        if current_radius < prev_radius:
            prev_radius = current_radius
            patience = 0
        else:
            patience += 1
            if patience > max_patience:
                break
        
        # Find the grid point with the largest distance (the "hole")
        worst_idx = np.argmax(min_dists)
        worst_point = grid_points[worst_idx]
        
        # Find the 3 nearest points to the worst_point
        dots_to_worst = np.dot(points, worst_point)
        # Sort indices by dot product descending (closest first)
        nearest_indices = np.argsort(-dots_to_worst)[:3]
        
        # Move these 3 points away from the worst_point
        for idx in nearest_indices:
            p = points[idx]
            # Tangent vector at p pointing away from worst_point
            # Project worst_point onto tangent space of p
            # t = worst_point - (worst_point . p) p
            # We want to move AWAY from worst_point, so we move in direction -t?
            # If we move p away from worst_point, the distance increases.
            # The vector from p to worst_point is roughly worst_point - p.
            # Tangential component: w - (w.p)p.
            # If we add this to p, we move TOWARDS w.
            # So we subtract it to move AWAY.
            proj = worst_point - np.dot(worst_point, p) * p
            t = _normalize(proj)
            points[idx] = _normalize(p - step_size * t)
            
        # Apply global repulsive force to maintain uniformity
        # This prevents points from clumping together due to the local moves
        # Force on i from j is along tangent, magnitude 1/d^2
        for i in range(N):
            force = np.zeros(3)
            pi = points[i]
            for j in range(N):
                if i == j: continue
                pj = points[j]
                d = np.linalg.norm(pi - pj)
                if d < 1e-6: continue
                # Repulsive force direction: from pj to pi
                # Tangential component of (pi - pj) at pi
                # vec = pi - pj
                # tang = vec - (vec . pi) pi
                vec = pi - pj
                tang = vec - np.dot(vec, pi) * pi
                norm_tang = np.linalg.norm(tang)
                if norm_tang > 1e-6:
                    # Magnitude decays with distance
                    mag = repulsive_strength / (d * d)
                    force += (tang / norm_tang) * mag
            if np.linalg.norm(force) > 1e-6:
                points[i] = _normalize(pi + force)
                
        # Re-normalize all points to ensure unit sphere
        points = points / np.linalg.norm(points, axis=1, keepdims=True)

    return points
# EVOLVE-BLOCK-END
