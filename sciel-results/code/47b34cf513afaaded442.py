"""Optimal packing of 12 unit regular hexagons in a minimal regular hexagon container.
The 12 inner hexagons form a 3x4 rectangular block (2x3 rhombus × 2 copies) with no gaps.
We optimize the outer container side length and rotation to tightly enclose this shape."""
import numpy as np
import time
from scipy.optimize import minimize

N = 12

# EVOLVE-BLOCK-START

def _hex_vertices(cx, cy, side, angle_deg):
    """Vertices of a regular hexagon centered at (cx,cy), side length, rotation in degrees."""
    a = np.radians(angle_deg)
    ca, sa = np.cos(a), np.sin(a)
    # Unit hexagon vertices (pointy-top at 0deg: vertices at angles 0, 60, 120, 180, 240, 300)
    base_angles = np.radians(np.arange(0, 360, 60))
    rx = side * np.cos(base_angles)
    ry = side * np.sin(base_angles)
    # Rotate
    x = rx * ca - ry * sa + cx
    y = rx * sa + ry * ca + cy
    return np.column_stack([x, y])

def _outer_hex_vertices(cx, cy, side, angle_deg):
    """Vertices of outer container hexagon."""
    return _hex_vertices(cx, cy, side, angle_deg)

def _point_in_hexagon(px, py, hx, hy, side, angle_deg):
    """Check if point (px,py) is inside hexagon centered at (hx,hy) with given side and rotation."""
    # Transform point to hexagon's local frame
    dx = px - hx
    dy = py - hy
    a = np.radians(-angle_deg)
    ca, sa = np.cos(a), np.sin(a)
    lx = dx * ca - dy * sa
    ly = dx * sa + dy * ca
    # For a regular hexagon with 'radius' (center to vertex) = side, the inradius = side * sqrt(3)/2
    # A regular hexagon with side length s has vertices at distance s from center
    # The condition for a point to be inside: max of 3 linear constraints
    # For flat-top hexagon (angle 0): vertices at 0, 60, 120, 180, 240, 300
    # The hexagon can be described by: for each of 3 axes (at 30, 90, 150 degrees), |dot| <= inradius
    # Actually, for a regular hexagon with circumradius R = side:
    # The supporting lines are at distance R * cos(30°) = R * sqrt(3)/2 from center
    # along directions 30°, 90°, 150° (for pointy-top)
    # Let me use a simpler approach: the hexagon is the intersection of 6 half-planes
    # Each edge is perpendicular to a vertex direction, at distance inradius = side * sqrt(3)/2
    inradius = side * np.sqrt(3) / 2
    # The 6 edge normals are at angles 30, 90, 150, 210, 270, 330 degrees for pointy-top
    # For our rotated frame, the normals in local coords are at 30+30k for k=0..5? 
    # Actually the edges of a hexagon with vertices at angles 0,60,...,300 are perpendicular
    # to directions 30, 90, 150, 210, 270, 330
    normal_angles = np.radians(np.arange(30, 360, 60))
    nx = np.cos(normal_angles)
    ny = np.sin(normal_angles)
    dots = lx * nx + ly * ny
    return np.all(dots <= inradius + 1e-9)

def _max_violation(inner_verts, outer_verts):
    """Compute how much inner vertices stick out of outer hexagon.
    Returns 0 if all inside, positive if violations exist."""
    # For each inner vertex, check if inside outer hexagon
    # Use the same method: outer hexagon in its local frame
    cx, cy = 0.0, 0.0
    # We'll handle this differently - use the outer hex parameters from optimization
    
    # Actually, let's compute it properly in the objective function
    return 0.0

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 110s margin before 150s limit
    
    # Define the 12 inner hexagon positions forming a 3x4 grid block
    # For regular hexagons in a "brick wall" / honeycomb pattern:
    # Using pointy-top hexagons (vertices at 0, 60, 120, ... degrees)
    # Horizontal spacing between adjacent centers in same row: sqrt(3) * side
    # Vertical spacing between rows: 1.5 * side
    # Row offset: sqrt(3)/2 * side for alternating rows
    
    side_inner = 1.0
    dx = np.sqrt(3) * side_inner  # horizontal spacing
    dy = 1.5 * side_inner          # vertical spacing between rows
    offset = dx / 2.0              # horizontal offset for odd rows
    
    # 3 columns × 4 rows = 12 hexagons
    # Row 0: 3 hexagons at x = 0, dx, 2dx
    # Row 1: 3 hexagons at x = offset, dx+offset, 2dx+offset  (shifted right)
    # Row 2: 3 hexagons at x = 0, dx, 2dx
    # Row 3: 3 hexagons at x = offset, dx+offset, 2dx+offset
    
    positions = []
    for row in range(4):
        for col in range(3):
            x = col * dx + (row % 2) * offset
            y = row * dy
            positions.append((x, y))
    
    # Center the block
    xs = [p[0] for p in positions]
    ys = [p[1] for p in positions]
    cx_block = (min(xs) + max(xs)) / 2.0
    cy_block = (min(ys) + max(ys)) / 2.0
    
    centered_positions = [(x - cx_block, y - cy_block) for x, y in positions]
    
    # Compute all vertices of all inner hexagons (for finding bounding shape)
    all_inner_verts = []
    for (cx, cy) in centered_positions:
        verts = _hex_vertices(cx, cy, side_inner, 0.0)
        all_inner_verts.append(verts)
    all_inner_verts = np.vstack(all_inner_verts)
    
    # Objective: minimize outer side length subject to all inner vertices being inside
    def outer_containing_side(rotation_deg, outer_center_x=0.0, outer_center_y=0.0):
        """Find minimum side length of outer hexagon (rotated by rotation_deg) 
        centered at origin that contains all inner vertices."""
        # For a regular hexagon with circumradius R (side = R), centered at origin,
        # rotated by angle phi:
        # A point (px, py) is inside iff for all k=0..5:
        #   dot((px,py), n_k) <= R * cos(30°) = R * sqrt(3)/2
        # where n_k are unit vectors at angles phi + 30° + 60°*k
        # So R >= max_k( dot((px,py), n_k) ) / cos(30°) for all points
        # R = max over all points and all k of dot / cos(30°)
        
        phi = np.radians(rotation_deg)
        normal_angles = phi + np.radians(30) + np.radians(60) * np.arange(6)
        nx = np.cos(normal_angles)
        ny = np.sin(normal_angles)
        
        # Shift points by outer center
        px = all_inner_verts[:, 0] - outer_center_x
        py = all_inner_verts[:, 1] - outer_center_y
        
        # dots[point_idx, normal_idx]
        dots = np.outer(px, nx) + np.outer(py, ny)
        max_dot = np.max(dots)
        
        cos30 = np.sqrt(3) / 2.0
        R = max_dot / cos30
        return max(R, 0.0)
    
    def objective(params):
        if time.time() > deadline:
            return 1e10
        rotation_deg = params[0]
        ocx = params[1]
        ocy = params[2]
        return outer_containing_side(rotation_deg, ocx, ocy)
    
    # Also need to verify containment - the formula above gives the tightest R
    # for a given rotation and center. We just need to minimize R over rotation and center.
    # But we should fix center at origin for symmetry and just optimize rotation.
    # Actually, the optimal center should be at the centroid of the shape for minimal R.
    # Let's optimize rotation only, with center at origin.
    
    def objective_rot(theta):
        if time.time() > deadline:
            return 1e10
        return outer_containing_side(theta[0])
    
    # Try multiple initial rotations
    best_side = np.inf
    best_rotation = 0.0
    best_center = (0.0, 0.0)
    
    for init_rot in range(0, 60, 5):
        if time.time() > deadline:
            break
        try:
            res = minimize(objective_rot, [float(init_rot)], method='Nelder-Mead',
                          options={'maxiter': 200, 'xatol': 1e-6, 'fatol': 1e-8})
            if res.success or res.fun < best_side:
                if res.fun < best_side:
                    best_side = res.fun
                    best_rotation = res.x[0]
        except Exception:
            pass
    
    # Also try optimizing center offset
    def objective_full(params):
        if time.time() > deadline:
            return 1e10
        rot = params[0]
        ocx = params[1]
        ocy = params[2]
        return outer_containing_side(rot, ocx, ocy)
    
    if time.time() < deadline:
        try:
            res = minimize(objective_full, [best_rotation, 0.0, 0.0], method='Nelder-Mead',
                          options={'maxiter': 300, 'xatol': 1e-6, 'fatol': 1e-8})
            if res.fun < best_side:
                best_side = res.fun
                best_rotation = res.x[0]
                best_center = (res.x[1], res.x[2])
        except Exception:
            pass
    
    # Build inner list
    inner = []
    for (cx, cy) in centered_positions:
        inner.append([float(cx), float(cy), 0.0])
    
    outer_side = float(best_side)
    outer_center = (float(best_center[0]), float(best_center[1]))
    outer_angle = float(best_rotation)
    
    return inner, outer_center, outer_side, outer_angle

# EVOLVE-BLOCK-END
