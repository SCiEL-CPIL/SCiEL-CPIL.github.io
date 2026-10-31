"""Packing 11 unit hexagons in minimum outer hexagon."""
import math
import time
import random
import numpy as np

N = 11

def hex_vertices(cx, cy, side, angle_deg):
    """Return vertices of a regular hexagon centered at (cx,cy) with given side and rotation."""
    a = math.radians(angle_deg)
    verts = []
    for i in range(6):
        theta = a + i * math.pi / 3
        verts.append((cx + side * math.cos(theta), cy + side * math.sin(theta)))
    return verts

def point_in_hex(px, py, cx, cy, side, angle_deg):
    """Check if point (px,py) is inside hexagon centered at (cx,cy) with given side and rotation."""
    # Transform to hexagon's local frame
    dx = px - cx
    dy = py - cy
    a = math.radians(angle_deg)
    ca, sa = math.cos(-a), math.sin(-a)
    lx = dx * ca - dy * sa
    ly = dx * sa + dy * ca
    # For a regular hexagon with flat top (angle=0), vertices at angles 0,60,120,180,240,300
    # The hexagon in local coords: |lx| <= side, |ly| <= side*sqrt(3)/2, and the corners
    # Actually for point-in-hex test, use the six half-planes
    for i in range(6):
        theta = i * math.pi / 3
        nx = math.cos(theta)
        ny = math.sin(theta)
        # The normal points outward, the supporting line is at distance side from center
        if lx * nx + ly * ny > side + 1e-9:
            return False
    return True

def hexagon_area(side):
    return (3 * math.sqrt(3) / 2) * side * side

def min_enclosing_hexagon(points):
    """Find minimum side length of a regular hexagon (any rotation) enclosing all points.
    Returns (cx, cy, side, angle_deg)."""
    best_side = float('inf')
    best_cx, best_cy, best_angle = 0.0, 0.0, 0.0
    
    # Try different rotations
    for angle_deg in range(0, 60, 5):  # 12 angles, 5-degree steps
        a = math.radians(angle_deg)
        ca, sa = math.cos(a), math.sin(a)
        
        # Project points onto the 6 axis directions of the hexagon
        # For a hexagon with rotation a, the 6 normal directions are at angles a + i*60
        # The hexagon is the intersection of 6 half-planes
        # For each direction, find the max and min projection
        projections = []
        for i in range(3):
            theta = a + i * math.pi / 3
            nx = math.cos(theta)
            ny = math.sin(theta)
            proj = [px * nx + py * ny for px, py in points]
            projections.append((min(proj), max(proj)))
        
        # The hexagon must contain all points
        # For each of the 3 axis pairs, the center must be positioned so that
        # the half-plane constraints are satisfied
        # The side length is determined by the maximum extent
        # For a regular hexagon, the distance from center to each side is 'side'
        # So for each direction i, we need: center_proj - side <= min_proj and center_proj + side >= max_proj
        # This gives: side >= (max_proj - min_proj) / 2 for each direction
        # But also the center must be consistent across all directions
        
        # The minimum side for this rotation is max over all 6 directions of (max_proj - min_proj)/2
        # Wait, that's not quite right. Let me think again.
        
        # For a regular hexagon with side s and rotation a, centered at (cx, cy):
        # A point (px, py) is inside iff for all i in 0..5:
        #   (px-cx)*cos(a+i*pi/3) + (py-cy)*sin(a+i*pi/3) <= s
        # So: (px-cx)*nx_i + (py-cy)*ny_i <= s for all i
        # => cx*nx_i + cy*ny_i >= px*nx_i + py*ny_i - s for all i
        # => cx*nx_i + cy*ny_i >= max_j (px_j*nx_i + py_j*ny_i) - s
        # Similarly for the opposite direction (i+3):
        #   (px-cx)*nx_{i+3} + (py-cy)*ny_{i+3} <= s
        #   nx_{i+3} = -nx_i, ny_{i+3} = -ny_i
        #   => -(px-cx)*nx_i - (py-cy)*ny_i <= s
        #   => (px-cx)*nx_i + (py-cy)*ny_i >= -s
        #   => cx*nx_i + cy*ny_i <= min_j (px_j*nx_i + py_j*ny_i) + s
        
        # So for each direction i (0,1,2):
        #   max_proj_i - s <= cx*nx_i + cy*ny_i <= min_proj_i + s  (wait, min should be max)
        #   Actually: max_j(proj_j) - s <= cx*nx_i + cy*ny_i <= min_j(proj_j) + s
        #   This requires: max_j(proj_j) - s <= min_j(proj_j) + s
        #   => s >= (max_j(proj_j) - min_j(proj_j)) / 2
        
        # So the minimum s for a given rotation is max over i of (max_proj_i - min_proj_i)/2
        # And the center (cx,cy) must satisfy the constraints for all 3 directions simultaneously
        
        max_extent = 0
        for i in range(3):
            theta = a + i * math.pi / 3
            nx = math.cos(theta)
            ny = math.sin(theta)
            proj = [px * nx + py * ny for px, py in points]
            mn, mx = min(proj), max(proj)
            extent = (mx - mn) / 2
            max_extent = max(max_extent, extent)
        
        # For this rotation, minimum side is max_extent
        # The center must be at the intersection of the constraint regions
        # For simplicity, let's try the center as the average of midpoints
        # or solve for a feasible center
        
        if max_extent < best_side:
            # Find a feasible center: the center must satisfy
            # for each direction i: mid_i - s <= c_dot_n_i <= mid_i + s
            # where mid_i = (max_proj_i + min_proj_i)/2
            # Let's try the centroid of the points as a starting point
            cx0 = sum(px for px, py in points) / len(points)
            cy0 = sum(py for px, py in points) / len(points)
            
            # Check if this center works
            valid = True
            s = max_extent
            for i in range(6):
                theta = a + i * math.pi / 3
                nx = math.cos(theta)
                ny = math.sin(theta)
                proj = [px * nx + py * ny for px, py in points]
                if any(p - (cx0 * nx + cy0 * ny) > s + 1e-9 for p in proj):
                    valid = False
                    break
            
            if valid:
                best_side = max_extent
                best_cx, best_cy, best_angle = cx0, cy0, angle_deg
            else:
                # Try to find a better center by optimization
                # For now, skip this rotation or use a simple search
                pass
    
    return best_cx, best_cy, best_side, best_angle

def hex_centers_hexagonal(arrangement):
    """Generate hexagon centers in a hexagonal grid pattern."""
    centers = []
    # Hexagonal grid: neighbors at distance 2 (for unit hexagons touching)
    # Actually for unit hexagons (side 1), the distance between centers of touching hexagons is 2*sqrt(3)/2 = sqrt(3)
    # Wait: for regular hexagons of side 1, the distance between centers of adjacent hexagons is 2 * (sqrt(3)/2) = sqrt(3)
    # No wait. The apothem of a unit hexagon is sqrt(3)/2. Two touching hexagons have centers 2*apothem = sqrt(3) apart.
    # Hmm, actually for hexagons with side length 1, the distance between centers of two edge-adjacent hexagons is 2 * (sqrt(3)/2) = sqrt(3).
    # Let me verify: a regular hexagon with side 1 has vertices at distance 1 from center. The apothem (distance from center to side) is cos(30°)*1 = sqrt(3)/2.
    # Two hexagons sharing an edge: the edge is at distance apothem from each center, so centers are 2*apothem = sqrt(3) apart. Yes.
    
    # Hexagonal grid coordinates: use axial coordinates
    # Directions: (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1) in axial coords
    # Convert to Cartesian: x = sqrt(3)*(q + r/2), y = 3*r/2  (for pointy-top)
    # Or for flat-top: x = 3/2 * q, y = sqrt(3)*(r + q/2)
    
    # Let's use a simple approach: generate candidate positions
    return centers

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    # Initial arrangement: hexagonal pattern
    # For 11 hexagons: center + 6 in first ring + 4 in second ring
    # Hex grid spacing: sqrt(3) between centers
    sp = math.sqrt(3)
    
    # Axial coordinates for a compact 11-hex arrangement
    # Ring 0: (0,0)
    # Ring 1: (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)
    # Ring 2: pick 4 from the 12 positions
    axial_coords = [(0,0)]
    # First ring
    for d in [(1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)]:
        axial_coords.append(d)
    # Second ring: pick 4 adjacent ones for compactness
    # Second ring positions: (2,0), (1,1), (0,2), (-1,2), (-2,2), (-2,1), (-2,0), (-1,-1), (0,-2), (1,-2), (2,-1), (2,-1)
    # Let's pick (2,0), (1,1), (-2,1), (-1,-1) for a roughly symmetric shape
    # Actually let me pick 4 that make a more compact shape
    second_ring = [(2,0), (1,1), (-2,1), (-1,-1), (0,2), (-2,2), (2,-1), (1,-2)]
    # Pick 4: let's try (2,0), (1,1), (-1,-1), (0,-2) - this makes a diagonal
    # Or (2,0), (1,1), (-2,2), (-1,-1) - more spread
    # Let's try a few options
    
    best_result = None
    best_side = float('inf')
    
    # Try several arrangements
    arrangements = [
        # Option 1: linear
        [(k, 0) for k in range(-5, 6)],
        # Option 2: hexagonal 1-6-4
        [(0,0), (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1), (2,0), (1,1), (-1,-1), (0,-2)],
        # Option 3: 3-5-3
        [(-1,0), (0,0), (1,0), (-1,1), (0,1), (1,1), (0,2), (-1,-1), (0,-1), (1,-1), (0,-2)],
        # Option 4: 2-3-3-2-1
        [(0,0), (1,0), (-1,1), (0,1), (1,1), (-1,0), (0,-1), (1,-1), (-1,-1), (0,-2), (1,-2)],
    ]
    
    for arr in arrangements:
        if time.time() > deadline:
            break
        
        # Convert axial to Cartesian (flat-top hexagons)
        points = []
        for q, r in arr:
            x = 1.5 * sp * q
            y = sp * (r + 0.5 * q)
            points.append((x, y))
        
        # Get minimum enclosing hexagon
        cx, cy, side, angle = min_enclosing_hexagon(points)
        
        if side < best_side:
            best_side = side
            best_result = (points, cx, cy, side, angle)
    
    # Local optimization: try to improve by moving hexagons
    if best_result is not None:
        points, cx, cy, side, angle = best_result
        best_side = side
        
        # Try small perturbations
        np.random.seed(42)
        for iteration in range(2000):
            if time.time() > deadline:
                break
            
            # Pick a random hexagon and nudge it
            idx = np.random.randint(0, N)
            dx = np.random.uniform(-0.3, 0.3)
            dy = np.random.uniform(-0.3, 0.3)
            new_points = list(points)
            new_points[idx] = (points[idx][0] + dx, points[idx][1] + dy)
            
            new_cx, new_cy, new_side, new_angle = min_enclosing_hexagon(new_points)
            
            if new_side < best_side - 1e-6:
                best_side = new_side
                points = new_points
                cx, cy, angle = new_cx, new_cy, new_angle
                best_result = (points, cx, cy, side, angle)
    
    if best_result is None:
        # Fallback: linear arrangement
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0
    
    points, cx, cy, side, angle = best_result
    
    # Construct inner hexagon positions
    inner = []
    for px, py in points:
        inner.append([px, py, 0.0])
    
    return inner, (cx, cy), side, angle

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    sp = math.sqrt(3)
    
    best_result = None
    best_side = float('inf')
    
    arrangements = [
        [(k, 0) for k in range(-5, 6)],
        [(0,0), (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1), (2,0), (1,1), (-1,-1), (0,-2)],
        [(-1,0), (0,0), (1,0), (-1,1), (0,1), (1,1), (0,2), (-1,-1), (0,-1), (1,-1), (0,-2)],
        [(0,0), (1,0), (-1,1), (0,1), (1,1), (-1,0), (0,-1), (1,-1), (-1,-1), (0,-2), (1,-2)],
        [(0,0), (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1), (2,0), (-2,1), (-2,2), (2,-1)],
    ]
    
    for arr in arrangements:
        if time.time() > deadline:
            break
        points = []
        for q, r in arr:
            x = 1.5 * sp * q
            y = sp * (r + 0.5 * q)
            points.append((x, y))
        cx, cy, side, angle = min_enclosing_hexagon(points)
        if side < best_side:
            best_side = side
            best_result = (points, cx, cy, side, angle)
    
    if best_result is None:
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0
    
    points, cx, cy, side, angle = best_result
    best_side = side
    
    np.random.seed(42)
    for iteration in range(5000):
        if time.time() > deadline:
            break
        idx = np.random.randint(0, N)
        dx = np.random.uniform(-0.2, 0.2)
        dy = np.random.uniform(-0.2, 0.2)
        new_points = list(points)
        new_points[idx] = (points[idx][0] + dx, points[idx][1] + dy)
        new_cx, new_cy, new_side, new_angle = min_enclosing_hexagon(new_points)
        if new_side < best_side - 1e-9:
            best_side = new_side
            points = new_points
            cx, cy, angle = new_cx, new_cy, new_angle
            best_result = (points, cx, cy, side, angle)
    
    points, cx, cy, side, angle = best_result
    inner = []
    for px, py in points:
        inner.append([px, py, 0.0])
    return inner, (cx, cy), side, angle
# EVOLVE-BLOCK-END
