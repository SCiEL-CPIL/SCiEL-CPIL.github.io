"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

N = 300

# EVOLVE-BLOCK-START
def construct():
    # 1. Initialization: Hexagonal Lattice
    # Generate a hexagonal grid and select points inside the unit circle.
    # We aim for a density that fits N points.
    # Approximate radius of each circle r ~ 0.053.
    # Hexagonal spacing d = 2 * r * 1.1 (slightly larger to avoid initial overlaps)
    
    # Estimate optimal spacing
    area_unit = np.pi
    area_per_point = area_unit / N
    # Hexagon area = sqrt(3) * d^2 / 2 ? No, area of hexagon with side s is 3*sqrt(3)*s^2/2.
    # Distance between centers d = 2*s.
    # Area per point in hex packing = d^2 * sqrt(3) / 2.
    d_est = np.sqrt(2 * area_per_point / np.sqrt(3))
    
    # Generate grid points
    # We generate a square grid of hex points covering [-1.5, 1.5]
    points_list = []
    # Step in x
    dx = d_est
    # Step in y
    dy = d_est * np.sqrt(3) / 2
    
    # Grid range
    x_min, x_max = -1.2, 1.2
    y_min, y_max = -1.2, 1.2
    
    x = x_min
    row = 0
    while x <= x_max:
        y = y_min
        offset = (row % 2) * (dx / 2)
        while y <= y_max:
            px = x + offset
            py = y
            if px**2 + py**2 <= 1.0:
                points_list.append([px, py])
            y += dy
        x += dx
        row += 1
        
    points = np.array(points_list)
    
    # We might have more or fewer than N points.
    if len(points) > N:
        # Keep the N points closest to the center? Or random?
        # Keeping closest to center creates a solid disk, which is good for packing.
        dists = np.linalg.norm(points, axis=1)
        indices = np.argsort(dists)
        points = points[indices[:N]]
    elif len(points) < N:
        # If too few, add random points?
        # This shouldn't happen with the estimate, but just in case.
        # Add random points inside circle
        while len(points) < N:
            r = np.sqrt(np.random.rand())
            theta = np.random.rand() * 2 * np.pi
            points = np.vstack([points, [r * np.cos(theta), r * np.sin(theta)]])
            
    # Ensure we have exactly N points
    points = points[:N]
    
    # 2. Simulated Annealing Optimization
    # Objective: Maximize min( min_pair_dist / 2, min_boundary_dist )
    
    def calc_score(pts):
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dists, np.inf)
        min_pair_dist = np.min(dists)
        norms = np.linalg.norm(pts, axis=1)
        min_boundary_dist = np.min(1.0 - norms)
        return min(min_pair_dist / 2.0, min_boundary_dist)
    
    current_points = points.copy()
    current_score = calc_score(current_points)
    best_points = current_points.copy()
    best_score = current_score
    
    # SA Parameters
    T_start = 0.05
    T_end = 0.001
    alpha = 0.999
    steps = 50000
    
    start_time = time.time()
    time_limit = 110.0
    
    # Pre-allocate for speed? No, we update one point at a time.
    # We need to be careful with the score calculation.
    # Full recalculation is O(N^2).
    
    for step in range(steps):
        if time.time() - start_time > time_limit:
            break
            
        # Pick a random point
        i = np.random.randint(N)
        
        # Generate a new position
        # Step size should be related to the current scale
        # We use a Gaussian step
        step_size = 0.05 * (T_start / (T_start + (T_end - T_start) * (step / steps))) # Adaptive?
        # Simpler: fixed step size scaled by T
        # Let's use a fixed exploration radius that shrinks
        radius = 0.05 * (1.0 - 0.5 * (step / steps))
        
        dx = np.random.normal(0, radius)
        dy = np.random.normal(0, radius)
        
        old_pos = current_points[i].copy()
        new_pos = old_pos + np.array([dx, dy])
        
        # Check if new position is valid (inside unit circle roughly)
        if new_pos[0]**2 + new_pos[1]**2 > 1.0:
            # Reject immediately or reflect?
            # Reflecting might be better to stay in bounds
            # But SA can handle it with penalty.
            # Let's just reject if it goes too far out, or calculate score with penalty.
            # For simplicity, if it's outside, we calculate the score which will be low.
            pass
            
        current_points[i] = new_pos
        
        new_score = calc_score(current_points)
        
        # Delta
        delta = new_score - current_score
        
        if delta > 0:
            # Accept
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                best_points = current_points.copy()
        else:
            # Accept with probability exp(delta / T)
            T = T_start * (T_end / T_start) ** (step / steps)
            if np.random.rand() < np.exp(delta / T):
                current_score = new_score
            else:
                # Reject, revert
                current_points[i] = old_pos
    
    return best_points
# EVOLVE-BLOCK-END
