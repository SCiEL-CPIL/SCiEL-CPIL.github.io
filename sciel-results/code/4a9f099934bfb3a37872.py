"""Packing 12 unit hexagons inside a regular hexagon, minimizing outer side length."""
import numpy as np
from scipy.optimize import minimize
import time

N = 12

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # safety margin
    
    # Unit regular hexagon vertices (flat-top orientation, side=1)
    # Vertices at angles 0, 60, 120, 180, 240, 300 degrees
    base_hex = np.array([[np.cos(np.radians(a)), np.sin(np.radians(a))] for a in range(0, 360, 60)])
    
    def hexagon_vertices(center, angle_deg, side=1.0):
        """Get vertices of a regular hexagon."""
        angle_rad = np.radians(angle_deg)
        rot = np.array([[np.cos(angle_rad), -np.sin(angle_rad)],
                        [np.sin(angle_rad), np.cos(angle_rad)]])
        return center + side * (base_hex @ rot.T)
    
    def point_in_hexagon(pt, center, side, angle_deg):
        """Check if point is inside a regular hexagon (with tolerance)."""
        angle_rad = np.radians(angle_deg)
        rot_inv = np.array([[np.cos(-angle_rad), -np.sin(-angle_rad)],
                            [np.sin(-angle_rad), np.cos(-angle_rad)]])
        local_pt = rot_inv @ (pt - center)
        # For a unit hexagon centered at origin, check |x| <= 1, |y| <= 1, |x+y| <= 1.5... 
        # Actually for regular hexagon with vertices at distance 1 from center:
        # The hexagon is intersection of 6 half-planes
        # For flat-top: vertices at angles 0,60,120,180,240,300
        # Normal directions: 30, 90, 150, 210, 270, 330 degrees
        # Apothem = cos(30) = sqrt(3)/2
        apothem = side * np.sqrt(3) / 2
        for i in range(6):
            normal_angle = np.radians(30 + 60 * i)
            normal = np.array([np.cos(normal_angle), np.sin(normal_angle)])
            if np.dot(local_pt, normal) > apothem + 1e-9:
                return False
        return True
    
    def hex_in_hexagon(inner_center, inner_angle, outer_center, outer_side, outer_angle):
        """Check if inner hexagon is fully inside outer hexagon."""
        inner_verts = hexagon_vertices(inner_center, inner_angle, 1.0)
        for v in inner_verts:
            if not point_in_hexagon(v, outer_center, outer_side, outer_angle):
                return False
        return True
    
    def objective(params):
        """Minimize outer hexagon side length with penalty for violations."""
        # params: [outer_side, outer_angle, then for each inner: x, y, angle]
        outer_side = params[0]
        outer_angle = params[1]
        outer_center = np.array([0.0, 0.0])  # Fix outer center at origin for symmetry
        
        penalty = 0.0
        for i in range(N):
            idx = 2 + 3 * i
            ix = params[idx]
            iy = params[idx + 1]
            iangle = params[idx + 2]
            
            inner_verts = hexagon_vertices(np.array([ix, iy]), iangle, 1.0)
            for v in inner_verts:
                if not point_in_hexagon(v, outer_center, outer_side, outer_angle):
                    # Compute how far outside
                    angle_rad = np.radians(outer_angle)
                    rot_inv = np.array([[np.cos(-angle_rad), -np.sin(-angle_rad)],
                                        [np.sin(-angle_rad), np.cos(-angle_rad)]])
                    local_pt = rot_inv @ (v - outer_center)
                    apothem = outer_side * np.sqrt(3) / 2
                    for j in range(6):
                        normal_angle = np.radians(30 + 60 * j)
                        normal = np.array([np.cos(normal_angle), np.sin(normal_angle)])
                        dist = np.dot(local_pt, normal) - apothem
                        if dist > 0:
                            penalty += dist * dist
        
        return outer_side + 100.0 * penalty
    
    # Initial guess: hexagonal lattice arrangement
    # Place hexagons in a pattern that's roughly 4 wide, 3 tall
    # Hexagonal lattice spacing: sqrt(3) for flat-top hexagons touching
    spacing = np.sqrt(3)
    
    # Try a 3-4-3 arrangement (rows of 3, 4, 3)
    # Row y positions: 0, spacing/2*sqrt(3)... actually for hex lattice:
    # For flat-top hexagons, horizontal spacing = 1.5 (vertex to vertex horizontally is 2, but center to center when touching on side is sqrt(3))
    # Let me use: centers at triangular lattice points
    # For flat-top hex: adjacent centers distance = sqrt(3)
    # Directions: 0, 60, 120, 180, 240, 300
    
    # Let me just use a grid-like initial placement
    init_params = np.zeros(2 + 3 * N)
    init_params[0] = 7.0  # initial outer side
    init_params[1] = 0.0  # initial outer angle
    
    # Place in a hexagonal pattern
    # Rows: 2, 3, 4, 3 or similar
    # Let's do 3 rows: 4, 4, 4 or 3, 5, 4... 
    # Actually for 12: let's do 4 columns x 3 rows
    col_spacing = 1.5  # horizontal distance between hex centers in same row (for flat-top, this is 3/2 * side = 1.5)
    row_spacing = np.sqrt(3) / 2 * 2  # = sqrt(3)
    
    # Hmm, let me think more carefully.
    # Flat-top regular hexagon, side s=1:
    # Width (horizontal) = 2 (vertex to vertex)
    # Height (vertical) = sqrt(3) (flat to flat)
    # When placing in a row, center-to-center horizontal distance for touching = 1.5 (since each hex extends 0.5 beyond center on each side horizontally... no)
    # Actually: vertex at (1,0), next at (0.5, sqrt(3)/2). The horizontal distance between centers of two adjacent hexagons in the same row (touching on the slanted side) is 1.5.
    # Wait, let me reconsider. For flat-top hex with vertices at angles 0,60,120,180,240,300:
    # Vertices: (1,0), (0.5, 0.866), (-0.5, 0.866), (-1, 0), (-0.5, -0.866), (0.5, -0.866)
    # Two hexagons side by side touching: the right vertex of left hex at (1,0) relative to its center, and left vertex of right hex at (-1,0) relative to its center. For them to touch, centers are 2 apart? No, that's vertex-to-vertex.
    # For flat-to-flat (touching on the flat top/bottom), the vertical distance is sqrt(3).
    # For touching on the slanted sides, the distance is... 
    # The normal to the slanted side (between vertices at 0 and 60 deg) is at 30 deg. The apothem is sqrt(3)/2.
    # So center-to-center distance when touching on that side = 2 * apothem = sqrt(3).
    
    # So in a hexagonal lattice, nearest neighbor distance = sqrt(3)
    # Let me place them in a triangular lattice with spacing sqrt(3)
    
    positions = []
    # 3 rows of 4, or 4 rows of 3, etc.
    # Let's try 4 rows: 3, 4, 4, 1? No.
    # 12 = 4+4+4 (3 rows of 4) or 3+4+5 or 2+3+4+3+2...
    # Let's do 3 rows of 4
    for row in range(3):
        for col in range(4):
            x = col * 1.5  # horizontal spacing
            y = row * np.sqrt(3)
            if row % 2 == 1:
                x += 0.75  # offset every other row
            positions.append([x, y])
    
    # Center the arrangement
    xs = [p[0] for p in positions]
    ys = [p[1] for p in positions]
    cx = (max(xs) - min(xs)) / 2
    cy = (max(ys) - min(ys)) / 2
    for i, p in enumerate(positions):
        init_params[2 + 3*i] = p[0] - cx
        init_params[2 + 3*i + 1] = p[1] - cy
        init_params[2 + 3*i + 2] = 0.0
    
    best_result = None
    best_score = np.inf
    
    # Try multiple initial configurations and outer angles
    for outer_angle_init in [0.0, 30.0, 15.0]:
        for _ in range(3):  # multiple restarts
            if time.time() > deadline:
                break
                
            params = init_params.copy()
            params[1] = outer_angle_init
            
            # Add small random perturbation
            if _ > 0:
                params[2:] += np.random.normal(0, 0.3, size=3*N)
            
            result = minimize(objective, params, method='Nelder-Mead', 
                             options={'maxiter': 5000, 'xatol': 1e-6, 'fatol': 1e-8})
            
            if result.fun < best_score:
                best_score = result.fun
                best_result = result.x
    
    if best_result is None:
        # Fallback
        return [[p[0], p[1], 0.0] for p in positions], (0.0, 0.0), 7.0, 0.0
    
    # Extract solution
    outer_side = best_result[0]
    outer_angle = best_result[1]
    
    inner = []
    for i in range(N):
        idx = 2 + 3 * i
        ix = best_result[idx]
        iy = best_result[idx + 1]
        iangle = best_result[idx + 2]
        inner.append([ix, iy, iangle])
    
    return inner, (0.0, 0.0), outer_side, outer_angle
# EVOLVE-BLOCK-END
