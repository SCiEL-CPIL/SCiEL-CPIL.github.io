"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import minimize
import time

N = 11

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 110s limit to be safe under 120s/150s

    # 1. Generate the optimal honeycomb packing of 11 unit hexagons
    # A regular hexagon of side length 1 has:
    # - Width (flat-to-flat) = sqrt(3)
    # - Height (point-to-point) = 2
    # In a honeycomb grid:
    # - Horizontal spacing between centers in same row: sqrt(3)
    # - Vertical spacing between rows: 1.5
    # - Horizontal offset for alternating rows: sqrt(3)/2
    
    # The 11-hexagon cluster is a hexagon of side length 2 (in hexagon units).
    # Rows:
    # Row 0 (y=0): 3 hexes
    # Row 1 (y=1.5): 4 hexes
    # Row 2 (y=3.0): 3 hexes
    # Row 3 (y=4.5): 1 hex  <- Wait, standard hexagon packing of 11 is:
    # Let's visualize the "hexagonal number" 11.
    # Actually, 11 is not a hexagonal number (1, 7, 19...).
    # The optimal packing of 11 hexagons is usually a 2x2 block plus extras, or a distorted hexagon.
    # Let's use the standard "flower" shape with center + 6 around + 4 more?
    # No, the densest packing of 11 hexagons is a hexagon of side 2 (which is 7) + 4 attached?
    # Actually, the "hexagon of side 2" has 7 hexagons.
    # Adding 4 more to make 11: attach them to the sides.
    # A common optimal configuration for 11 is a 3-4-3-1 or similar?
    # Let's use a simple, valid, dense packing.
    # A 2x5 rectangle of hexagons? No, that's 10.
    # Let's use the "staircase" or just a tight cluster.
    # The inspiration mentioned "asymmetric 11-hexagon cluster".
    # Let's construct a valid cluster:
    # Center (0,0)
    # 6 neighbors
    # 4 more neighbors attached to the outer ring.
    
    centers = []
    # Center
    centers.append((0.0, 0.0))
    
    # 6 neighbors
    for i in range(6):
        angle = np.pi / 3 * i
        x = np.sqrt(3) * np.cos(angle)
        y = 1.5 * np.sin(angle) # Wait, vertical distance is 1.5?
        # Let's re-verify hexagon geometry.
        # Flat-top hexagon:
        # Vertices at angles 0, 60, 120...
        # Distance from center to vertex (radius) = 1.
        # Distance from center to edge (apothem) = sqrt(3)/2.
        # If we place hexagons flat-top:
        # Horizontal distance between centers in a row: 2 * apothem = sqrt(3).
        # Vertical distance between rows: 3/2 * side = 1.5? No.
        # Let's use pointy-top for easier row logic?
        # Pointy-top:
        # Height = 2. Width = sqrt(3).
        # Horizontal spacing: sqrt(3).
        # Vertical spacing: 1.5.
        # Offset: sqrt(3)/2.
        
    # Let's stick to Pointy-Top for standard grid logic.
    # Side length s = 1.
    # dx = sqrt(3)
    # dy = 1.5
    # offset = sqrt(3)/2
    
    dx = np.sqrt(3)
    dy = 1.5
    
    # Configuration:
    # Row 0: 3 hexes at x = -dx, 0, dx. y = 0.
    # Row 1: 4 hexes at x = -dx - dx/2, -dx/2, dx/2, dx + dx/2. y = dy.
    # Row 2: 3 hexes at x = -dx, 0, dx. y = 2*dy.
    # Row 3: 1 hex at x = 0. y = 3*dy.
    # Total: 3+4+3+1 = 11. This is a valid, connected, dense cluster.
    
    rows = [
        (0, 3, 0),      # y_index, count, x_offset_index (0-centered)
        (1, 4, 0),
        (2, 3, 0),
        (3, 1, 0)
    ]
    
    inner_centers = []
    for y_idx, count, _ in rows:
        y = y_idx * dy
        # Calculate x positions centered around 0
        # If count is odd, center is 0. If even, center is between -0.5 and 0.5?
        # For honeycomb, x positions are k * dx + (y_idx % 2) * (dx/2)
        # Let's just generate them relative to a local origin and then center the whole thing.
        
        # Simpler: Just list the coordinates for this specific 11-shape.
        # Row 0 (y=0): x = -dx, 0, dx
        # Row 1 (y=dy): x = -1.5*dx, -0.5*dx, 0.5*dx, 1.5*dx
        # Row 2 (y=2dy): x = -dx, 0, dx
        # Row 3 (y=3dy): x = 0
        
        row0 = [(-dx, 0), (0, 0), (dx, 0)]
        row1 = [(-1.5*dx, dy), (-0.5*dx, dy), (0.5*dx, dy), (1.5*dx, dy)]
        row2 = [(-dx, 2*dy), (0, 2*dy), (dx, 2*dy)]
        row3 = [(0, 3*dy)]
        
        inner_centers.extend(row0)
        inner_centers.extend(row1)
        inner_centers.extend(row2)
        inner_centers.extend(row3)
        
    # Center the cluster
    cx = np.mean([c[0] for c in inner_centers])
    cy = np.mean([c[1] for c in inner_centers])
    inner_centers = [(x - cx, y - cy) for x, y in inner_centers]
    
    # 2. Define the function to minimize: Outer Hexagon Side Length
    # The outer hexagon is regular, side length S, center (cx_out, cy_out), rotation theta.
    # We need the smallest S such that all inner hexagons are inside.
    # Since inner hexagons can rotate, we assume they are aligned to minimize projection?
    # Actually, the task says "each inner hexagon may translate and rotate".
    # To minimize the container, we should align inner hexagons to the container or optimize them too.
    # However, optimizing 11 rotations + 11 translations + 2 container center + 1 container rotation + 1 S is high dimensional.
    # Let's fix inner rotations to 0 (aligned with each other) and optimize container.
    # Or better: The "width" of a hexagon in any direction is fixed if we consider the convex hull?
    # No, the width varies.
    # But if we assume the inner hexagons are all aligned (angle=0), we can compute the required S.
    # The condition for a point (or shape) to be inside a regular hexagon of side S, center C, rotation R:
    # It is easier to check if the inner hexagons are inside by checking their vertices.
    # But we want to MINIMIZE S.
    
    # Let's define the objective function for Nelder-Mead.
    # Variables: [cx_out, cy_out, theta_out, S]
    # We want to minimize S.
    # Constraint: All vertices of all inner hexagons must be inside the outer hexagon.
    
    def get_inner_vertices():
        verts = []
        for cx_in, cy_in in inner_centers:
            # Pointy-top hexagon vertices, side 1, angle 0
            # Angles: 30, 90, 150, 210, 270, 330 degrees?
            # Pointy top: vertices at 30, 90, 150, 210, 270, 330 from horizontal?
            # No, pointy top has vertices at 90, 150, 210, 270, 330, 30.
            # Let's use standard math angles: 0, 60, 120, 180, 240, 300 is flat-top.
            # Pointy-top: 30, 90, 150, 210, 270, 330.
            angles = np.deg2rad([30, 90, 150, 210, 270, 330])
            for a in angles:
                vx = cx_in + np.cos(a)
                vy = cy_in + np.sin(a)
                verts.append((vx, vy))
        return np.array(verts)

    all_inner_verts = get_inner_vertices()
    
    def is_point_in_hexagon(px, py, cx_h, cy_h, S, theta_h):
        # Translate point to hexagon center
        dx = px - cx_h
        dy = py - cy_h
        # Rotate point by -theta_h
        cos_t = np.cos(-theta_h)
        sin_t = np.sin(-theta_h)
        rx = dx * cos_t - dy * sin_t
        ry = dx * sin_t + dy * cos_t
        
        # Check if (rx, ry) is inside regular hexagon of side S centered at origin.
        # Flat-top hexagon (since we rotated the frame, let's assume the hexagon is flat-top in local coords)
        # Wait, if theta_h is the rotation of the hexagon, and we rotate the point by -theta_h,
        # the hexagon is now axis-aligned (flat-top or pointy-top depending on definition).
        # Let's assume the base hexagon (theta=0) is Flat-Top.
        # Flat-Top vertices: (S, 0), (S/2, S*sqrt(3)/2), (-S/2, S*sqrt(3)/2), (-S, 0), (-S/2, -S*sqrt(3)/2), (S/2, -S*sqrt(3)/2)
        # Conditions for Flat-Top hexagon of side S:
        # |x| <= S
        # |y| <= S * sqrt(3) / 2
        # |y| <= -sqrt(3) * |x| + S * sqrt(3)  (Slope -sqrt(3) for top right? No)
        # Let's use the support function or simple inequalities.
        # For flat-top:
        # 1. |x| <= S
        # 2. |y| <= sqrt(3)/2 * S
        # 3. |y| <= -sqrt(3) * |x| + sqrt(3) * S  ?? No.
        
        # Let's use the 6 half-planes.
        # Normal vectors for flat-top hexagon:
        # (1, 0), (1/2, sqrt(3)/2), (-1/2, sqrt(3)/2), (-1, 0), (-1/2, -sqrt(3)/2), (1/2, -sqrt(3)/2)
        # Distance to each edge must be <= apothem = S * sqrt(3) / 2.
        
        normals = [
            (1, 0),
            (0.5, np.sqrt(3)/2),
            (-0.5, np.sqrt(3)/2),
            (-1, 0),
            (-0.5, -np.sqrt(3)/2),
            (0.5, -np.sqrt(3)/2)
        ]
        
        apothem = S * np.sqrt(3) / 2
        
        for nx, ny in normals:
            dist = abs(rx * nx + ry * ny)
            if dist > apothem + 1e-9:
                return False
        return True

    def objective(params):
        cx_out, cy_out, theta_out, S = params
        # Check all inner vertices
        for px, py in all_inner_verts:
            if not is_point_in_hexagon(px, py, cx_out, cy_out, S, theta_out):
                return S + 1000.0  # Penalty
        return S

    # Initial guess
    # Bounding box of inner centers
    xs = [c[0] for c in inner_centers]
    ys = [c[1] for c in inner_centers]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    # Approximate S
    width = max_x - min_x
    height = max_y - min_y
    # For flat-top hex, width = 2*S, height = sqrt(3)*S
    S_guess = max(width / 2, height / np.sqrt(3)) + 1.0
    
    x0 = [(min_x + max_x) / 2, (min_y + max_y) / 2, 0.0, S_guess]
    
    # Optimize
    best_result = None
    best_score = float('inf')
    
    # Use Nelder-Mead
    try:
        # We can run multiple restarts if time permits
        restarts = 5
        for i in range(restarts):
            if time.time() > deadline:
                break
                
            if i == 0:
                x_start = x0
            else:
                # Perturb
                x_start = [x0[0] + np.random.uniform(-1, 1), 
                           x0[1] + np.random.uniform(-1, 1), 
                           np.random.uniform(0, 2*np.pi), 
                           S_guess * np.random.uniform(0.8, 1.2)]
                
            res = minimize(objective, x_start, method='Nelder-Mead', 
                           options={'maxiter': 1000, 'xatol': 1e-6, 'fatol': 1e-6})
            
            if res.fun < best_score:
                best_score = res.fun
                best_result = res.x
                
    except Exception as e:
        # Fallback
        pass
        
    if best_result is None:
        # Fallback to initial guess
        cx_out, cy_out, theta_out, S = x0
    else:
        cx_out, cy_out, theta_out, S = best_result
        
    # Ensure S is not too small (validity)
    S = max(S, 2.0) # 11 hexagons need at least side 2
    
    # Construct inner list
    inner = []
    for cx_in, cy_in in inner_centers:
        inner.append([cx_in, cy_in, 0.0])
        
    return inner, (cx_out, cy_out), S, np.rad2deg(theta_out)
# EVOLVE-BLOCK-END
