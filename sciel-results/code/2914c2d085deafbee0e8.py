import numpy as np
from scipy.optimize import minimize
import time

N = 11

def _hex_vertices(cx, cy, side, angle_deg):
    """Return 6 vertices of a regular hexagon with center (cx,cy), side length, rotation in degrees."""
    theta = np.radians(angle_deg)
    angles = theta + np.pi/3 * np.arange(6)
    return np.column_stack([cx + side * np.cos(angles), cy + side * np.sin(angles)])

def _point_in_hexagon(px, py, cx, cy, side, angle_deg):
    """Check if point (px,py) is inside hexagon with center (cx,cy), side, angle_deg."""
    # Transform to local frame
    theta = np.radians(angle_deg)
    dx = px - cx
    dy = py - cy
    # Rotate by -theta
    x_local = dx * np.cos(-theta) - dy * np.sin(-theta)
    y_local = dx * np.sin(-theta) + dy * np.cos(-theta)
    # In local frame, hexagon has vertices at angles 0, 60, 120, 180, 240, 300 degrees
    # The hexagon is defined by |x| <= side, |y| <= side*sqrt(3)/2, |x + y/sqrt(3)| <= side ... 
    # Actually, for a hexagon with vertices at (side*cos(k*pi/3), side*sin(k*pi/3)) for k=0..5:
    # The constraints are:
    # |x| <= side
    # |y| <= side * sqrt(3)/2  
    # |x*sqrt(3) + y| <= side * sqrt(3) ... hmm let me think again
    
    # A regular hexagon with circumradius R (= side length) has:
    # Vertices at (R*cos(k*60°), R*sin(k*60°)) for k=0..5
    # The edges are at distance R*cos(30°) = R*sqrt(3)/2 from center
    # The half-width in x is R, half-height in y is R*sqrt(3)/2
    # But the hexagon is not an axis-aligned rectangle.
    
    # The six edges are perpendicular to directions 30°, 90°, 150°, 210°, 270°, 330°
    # Each edge is at distance R*sqrt(3)/2 from center
    # So the constraint is: for each of 6 directions at 30°+k*60°, the projection of (x,y) onto that direction <= R*sqrt(3)/2
    
    # Direction vectors for the 6 edge normals:
    normal_angles = np.radians(30 + 60 * np.arange(6))
    normals = np.column_stack([np.cos(normal_angles), np.sin(normal_angles)])
    projections = np.array([x_local * normals[:, 0] + y_local * normals[:, 1]])
    inradius = side * np.sqrt(3) / 2
    return np.all(projections <= inradius + 1e-10)

def _max_projection(vertices, cx, cy, angle_deg):
    """Max projection of vertices onto hexagon edge normals, relative to hexagon center."""
    theta = np.radians(angle_deg)
    dx = vertices[:, 0] - cx
    dy = vertices[:, 1] - cy
    # Rotate
    x_local = dx * np.cos(-theta) - dy * np.sin(-theta)
    y_local = dx * np.sin(-theta) + dy * np.cos(-theta)
    
    normal_angles = np.radians(30 + 60 * np.arange(6))
    normals = np.column_stack([np.cos(normal_angles), np.sin(normal_angles)])
    projections = x_local * normals[:, 0] + y_local * normals[:, 1]
    return np.max(projections)

def _required_side(vertices_all, cx, cy, angle_deg):
    """Minimum side length of hexagon at (cx,cy) with rotation angle_deg to contain all vertices."""
    theta = np.radians(angle_deg)
    dx = vertices_all[:, 0] - cx
    dy = vertices_all[:, 1] - cy
    x_local = dx * np.cos(-theta) - dy * np.sin(-theta)
    y_local = dx * np.sin(-theta) + dy * np.cos(-theta)
    
    normal_angles = np.radians(30 + 60 * np.arange(6))
    normals = np.column_stack([np.cos(normal_angles), np.sin(normal_angles)])
    projections = x_local * normals[:, 0] + y_local * normals[:, 1]
    max_proj = np.max(projections)
    # inradius = side * sqrt(3)/2, so side = max_proj / (sqrt(3)/2) = 2*max_proj/sqrt(3)
    return 2.0 * max_proj / np.sqrt(3)

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    # Initial configuration: hexagonal close packing
    # Flat-top hexagons, side=1
    # Center hexagon at origin
    # 6 neighbors at distance sqrt(3) in directions 0, 60, 120, 180, 240, 300 degrees
    # Wait, for flat-top hexagons, the neighbors are at specific positions
    
    # Let me use a different approach: place hexagons in a triangular lattice
    # For flat-top hexagons with side 1:
    # Horizontal spacing: sqrt(3)
    # Vertical spacing: 3/2 (for adjacent rows), with horizontal offset sqrt(3)/2
    
    # Actually, let me just use circumradius-based placement
    # Two flat-top hexagons sharing an edge: centers separated by 2*inradius = sqrt(3)
    # The 6 neighbor directions for flat-top are at 0°, 60°, 120°, 180°, 240°, 300° from center
    
    # Let's create a compact 11-hexagon pattern
    # Pattern: row-based
    # Row 0 (y=0): 3 hexagons
    # Row 1 (y=3/2): 3 hexagons (shifted by sqrt(3)/2)
    # Row 2 (y=3): 3 hexagons
    # Row 3 (y=9/2): 2 hexagons (shifted by sqrt(3)/2)
    # That's 11
    
    # For flat-top hexagons:
    # Horizontal neighbor distance: sqrt(3)
    # Vertical neighbor distance: 3/2 (with horizontal shift sqrt(3)/2)
    
    sqrt3 = np.sqrt(3)
    positions = []
    # Row 0: y=0, x = 0, sqrt3, 2*sqrt3
    for i in range(3):
        positions.append((i * sqrt3, 0.0))
    # Row 1: y=1.5, x = sqrt3/2, 3*sqrt3/2, 5*sqrt3/2
    for i in range(3):
        positions.append((sqrt3/2 + i * sqrt3, 1.5))
    # Row 2: y=3.0, x = 0, sqrt3, 2*sqrt3
    for i in range(3):
        positions.append((i * sqrt3, 3.0))
    # Row 3: y=4.5, x = sqrt3/2, 3*sqrt3/2
    for i in range(2):
        positions.append((sqrt3/2 + i * sqrt3, 4.5))
    
    positions = np.array(positions)
    
    # All inner hexagons have side=1, angle=0 (flat-top)
    # Get all vertices of all inner hexagons
    all_vertices = []
    for (px, py) in positions:
        verts = _hex_vertices(px, py, 1.0, 0.0)
        all_vertices.append(verts)
    all_vertices = np.vstack(all_vertices)
    
    # Optimize outer hexagon: center (cx, cy), angle, minimize side
    # Parameters: [cx, cy, angle_rad]
    # Side is computed from the max projection
    
    def objective(params):
        cx, cy, angle_rad = params
        angle_deg = np.degrees(angle_rad)
        return _required_side(all_vertices, cx, cy, angle_deg)
    
    # Initial guess: center of all vertices
    cx0 = np.mean(all_vertices[:, 0])
    cy0 = np.mean(all_vertices[:, 1])
    angle0 = 0.0
    
    x0 = [cx0, cy0, angle0]
    
    best_result = None
    best_side = _required_side(all_vertices, cx0, cy0, 0.0)
    
    # Try optimization
    try:
        result = minimize(objective, x0, method='Nelder-Mead', 
                         options={'maxiter': 2000, 'xatol': 1e-6, 'fatol': 1e-8})
        if time.time() < deadline:
            if result.fun < best_side:
                best_side = result.fun
                best_result = result.x
    except Exception:
        pass
    
    if time.time() >= deadline:
        # Return best known
        if best_result is not None:
            cx, cy, angle_rad = best_result
            angle_deg = np.degrees(angle_rad)
        else:
            cx, cy, angle_deg = cx0, cy0, 0.0
        inner = [[px, py, 0.0] for px, py in positions]
        return inner, (float(cx), float(cy)), float(best_side), float(angle_deg)
    
    # Try multiple angle initializations
    for angle_init in [0, 30, 60, 90, 120, 150]:
        if time.time() >= deadline:
            break
        x0_angle = [cx0, cy0, np.radians(angle_init)]
        try:
            result = minimize(objective, x0_angle, method='Nelder-Mead',
                             options={'maxiter': 1000, 'xatol': 1e-6, 'fatol': 1e-8})
            if time.time() >= deadline:
                break
            if result.fun < best_side:
                best_side = result.fun
                best_result = result.x
        except Exception:
            pass
    
    if best_result is not None:
        cx, cy, angle_rad = best_result
        angle_deg = np.degrees(angle_rad)
    else:
        cx, cy, angle_deg = cx0, cy0, 0.0
    
    inner = [[px, py, 0.0] for px, py in positions]
    return inner, (float(cx), float(cy)), float(best_side), float(angle_deg)
