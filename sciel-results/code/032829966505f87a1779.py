"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import minimize
import time

N = 12

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill
    
    # Define a compact honeycomb cluster of 12 unit hexagons
    # Using axial coordinates in a hex grid, then converting to Cartesian
    # A good compact shape: rows of sizes 2,3,4,3 (total 12)
    # Or 3,4,3,2 - let's try several topologies
    
    def hex_vertices(cx, cy, side, angle_deg=0.0):
        """Return 6 vertices of a regular hexagon centered at (cx,cy) with given side and rotation."""
        angle = np.radians(angle_deg)
        angles = angle + np.pi/6 + np.arange(6) * np.pi/3  # flat-top orientation
        return np.column_stack([cx + side * np.cos(angles), cy + side * np.sin(angles)])
    
    def hex_grid_positions(rows):
        """Generate positions for hexagons in a honeycomb pattern.
        rows: list of (count, offset) for each row
        """
        positions = []
        # For flat-top hexagons, horizontal spacing = sqrt(3)*side, vertical = 1.5*side
        # For pointy-top, swap
        # Let's use pointy-top: vertices at angles 0, 60, 120, 180, 240, 300
        # Spacing: dx = sqrt(3), dy = 1.5
        dx = np.sqrt(3)
        dy = 1.5
        y = 0
        for i, count in enumerate(rows):
            x_offset = (max(rows) - count) * dx / 2
            for j in range(count):
                x = x_offset + j * dx
                positions.append((x, y))
            y += dy
        return np.array(positions)
    
    def min_enclosing_hex_side(points, outer_angle_deg=0.0):
        """Compute minimum side length of a regular hexagon (with given orientation) 
        that contains all given points. The hexagon center is optimized."""
        angle = np.radians(outer_angle_deg)
        # A regular hexagon with center (cx,cy), side s, rotation angle
        # can be described by 6 half-plane constraints:
        # For each of 6 directions (at angles angle + k*pi/3 + pi/6 for k=0..5),
        # the projection of any point onto that direction must be within [-s*sqrt(3)/2, s*sqrt(3)/2]
        # Actually, for a regular hexagon with circumradius R = s, the apothem is s*sqrt(3)/2
        # The hexagon is the intersection of 6 half-planes at distance apothem from center
        
        # Directions for the 6 sides (normals)
        normals = []
        for k in range(6):
            theta = angle + k * np.pi/3 + np.pi/6
            normals.append(np.array([np.cos(theta), np.sin(theta)]))
        normals = np.array(normals)
        
        # For a hexagon centered at (cx, cy) with side s:
        # For each normal n_k: n_k . (p - c) <= s * sqrt(3) / 2
        # So: n_k . p - n_k . c <= s * sqrt(3) / 2
        # And: -n_k . p + n_k . c <= s * sqrt(3) / 2
        # i.e., |n_k . (p - c)| <= s * sqrt(3) / 2
        
        # To minimize s, for a fixed center c, we need:
        # s >= max over all points p and normals n_k of |n_k . (p - c)| * 2 / sqrt(3)
        
        # For a given center c, the required s is:
        # s(c) = max_k max_p |n_k . (p - c)| * 2 / sqrt(3)
        
        # We want to minimize s(c) over c.
        # This is a convex optimization problem.
        
        # Let's use a simple approach: for a given angle, find optimal center by
        # minimizing the maximum projection range.
        
        # The optimal center is the one that minimizes the width in each of the 6 directions.
        # For direction n_k, the width is max_p(n_k.p) - min_p(n_k.p)
        # The center should be at the midpoint: c_k = (max_p(n_k.p) + min_p(n_k.p)) / 2 * n_k ... 
        # Actually, for each direction, the center component along n_k should be at the midpoint.
        # But these constraints may not be simultaneously satisfiable, so we need to find
        # the center that minimizes the maximum over all directions.
        
        # Simple approach: use scipy minimize to find optimal center
        def objective(c):
            cx, cy = c
            max_proj = 0
            for n in normals:
                projections = points @ n
                max_p = np.max(projections)
                min_p = np.min(projections)
                # The hexagon center projection onto n should be at (max_p + min_p)/2
                # The half-width is (max_p - min_p)/2
                # Required s: (max_p - min_p)/2 * 2/sqrt(3) = (max_p - min_p)/sqrt(3)
                half_width = (max_p - min_p) / 2
                max_proj = max(max_proj, half_width)
            return max_proj * 2 / np.sqrt(3)
        
        # Initial guess: centroid of points
        c0 = np.mean(points, axis=0)
        
        result = minimize(objective, c0, method='Nelder-Mead', 
                         options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-10})
        
        return result.fun, result.x
    
    # Try different honeycomb topologies
    topologies = [
        [2, 3, 4, 3],
        [3, 4, 3, 2],
        [1, 3, 4, 4, 1],
        [2, 4, 4, 2],
        [3, 3, 3, 3],
        [4, 4, 4],
        [2, 3, 3, 4],
        [4, 3, 3, 2],
        [1, 4, 4, 3],
        [3, 4, 4, 1],
        [2, 4, 3, 3],
        [3, 3, 4, 2],
    ]
    
    best_side = float('inf')
    best_config = None
    
    for rows in topologies:
        if sum(rows) != N:
            continue
        
        positions = hex_grid_positions(rows)
        
        # Center the cluster at origin
        centroid = np.mean(positions, axis=0)
        positions = positions - centroid
        
        # Try different outer hexagon orientations
        for outer_angle in range(0, 30, 3):  # 0 to 27 degrees, step 3
            if time.time() > deadline:
                break
            
            side, center = min_enclosing_hex_side(positions, outer_angle)
            
            if side < best_side:
                best_side = side
                best_config = (rows, positions.copy(), outer_angle, center.copy())
        
        if time.time() > deadline:
            break
    
    if best_config is None:
        # Fallback: trivial arrangement
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0
    
    rows, positions, outer_angle, center = best_config
    
    # Build inner hexagon list
    inner = []
    for i in range(N):
        x, y = positions[i]
        # All inner hexagons have the same orientation (0 degrees) for simplicity
        # Actually, in a honeycomb, adjacent hexagons have the same orientation
        inner.append([x, y, 0.0])
    
    # The outer hexagon center and orientation
    outer_center = (float(center[0]), float(center[1]))
    outer_side = float(best_side)
    outer_rotation = float(outer_angle)
    
    return inner, outer_center, outer_side, outer_rotation
# EVOLVE-BLOCK-END
