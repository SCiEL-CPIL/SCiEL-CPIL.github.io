import numpy as np
import time
from scipy.optimize import minimize_scalar
import math

N = 12
SQRT3 = np.sqrt(3.0)

def construct():
    start_time = time.time()
    
    # 1. Define the rigid cluster of 12 hexagons
    # We use a "truncated hexagon of side 3" shape.
    # A full hexagon of side 3 (in hex grid units) has 19 hexes.
    # We remove 7 hexes to get 12.
    # Specifically, we remove 1 from each of the 6 corners (leaving 13) 
    # and 1 from the middle of one side (leaving 12).
    # This shape is very compact.
    
    # Generate hex centers in axial coordinates (q, r)
    # Then convert to Cartesian.
    # Axial to Cartesian:
    # x = size * sqrt(3) * (q + r/2)
    # y = size * 3/2 * r
    # For unit hexes, size=1.
    
    def axial_to_cartesian(q, r):
        x = SQRT3 * (q + r / 2.0)
        y = 1.5 * r
        return x, y

    # Define the set of (q, r) for the 12 hexes
    # Full hex side 3:
    # max(abs(q), abs(r), abs(-q-r)) <= 2
    # Let's list them and filter.
    centers_axial = []
    for q in range(-2, 3):
        for r in range(-2, 3):
            if abs(q) <= 2 and abs(r) <= 2 and abs(-q - r) <= 2:
                centers_axial.append((q, r))
                
    # This gives 19 centers.
    # We need to remove 7.
    # The corners are (2,0), (0,2), (-2,2), (-2,0), (0,-2), (2,-2).
    # Removing these 6 corners leaves 13.
    # We need to remove 1 more. To maintain symmetry as much as possible, 
    # we might remove one from the middle of a side, e.g., (2, -1) or (1, 2).
    # Let's try removing the 6 corners and one side-middle.
    
    corners = [(2,0), (0,2), (-2,2), (-2,0), (0,-2), (2,-2)]
    # Side middles are like (2, -1), (1, -2), etc.
    # Let's pick (2, -1) to remove.
    to_remove = set(corners)
    to_remove.add((2, -1))
    
    final_centers_axial = [c for c in centers_axial if c not in to_remove]
    
    # Verify count
    if len(final_centers_axial) != 12:
        # Fallback to a 3x4 grid if the logic fails
        # But the logic should be correct. 19 - 6 - 1 = 12.
        pass

    # Convert to Cartesian
    centers_cart = [axial_to_cartesian(q, r) for q, r in final_centers_axial]
    
    # Center the cluster at the origin
    cx = np.mean([c[0] for c in centers_cart])
    cy = np.mean([c[1] for c in centers_cart])
    centers_cart = [(c[0] - cx, c[1] - cy) for c in centers_cart]

    # 2. Generate vertices for each hexagon in the cluster
    # A hexagon of side 1 centered at (x,y) with angle 0 has vertices:
    # (1, 0), (0.5, sqrt(3)/2), (-0.5, sqrt(3)/2), (-1, 0), (-0.5, -sqrt(3)/2), (0.5, -sqrt(3)/2)
    # Rotated by theta.
    
    def get_hex_vertices(cx, cy, side, angle_rad):
        angles = angle_rad + np.radians(np.arange(0, 360, 60))
        return np.column_stack([
            cx + side * np.cos(angles),
            cy + side * np.sin(angles)
        ])

    # The inner hexagons can also rotate. In the optimal packing, they are usually aligned with the grid.
    # We assume the inner hexagons are aligned (angle 0) relative to the cluster frame.
    # The cluster frame itself will be rotated.
    
    # 3. Function to calculate the minimum enclosing hexagon side length
    # for a given set of points (vertices of the cluster) and outer hexagon rotation.
    
    def min_enclosing_hex_side(points, outer_angle_rad):
        """
        points: Nx2 array of points (vertices of the cluster).
        outer_angle_rad: rotation of the outer hexagon.
        Returns the minimum side length S of a regular hexagon (centered at origin, 
        rotated by outer_angle_rad) that contains all points.
        """
        max_required = 0.0
        theta = outer_angle_rad
        # Outer hexagon normal directions
        # The support function of a regular hexagon with side S centered at origin, 
        # rotated by theta, in direction phi is:
        # h(phi) = S * sqrt(3)/2 * max(1, |cos(phi - theta - k*pi/3)|) ... 
        # Actually, simpler: 
        # The distance from center to edge in direction phi is d(phi).
        # For a regular hexagon with side S, apothem a = S * sqrt(3)/2.
        # The distance to the boundary in direction phi is a / cos(delta), where delta is 
        # the angle between phi and the nearest normal to an edge.
        # The normals are at theta + k*pi/3.
        # So delta = min_k | (phi - (theta + k*pi/3)) mod pi - pi/2 |? No.
        # Let's use the projection method.
        # A point (x,y) is inside if for all 6 edge normals n_k, dot((x,y), n_k) <= a.
        # So we need a >= max_k dot((x,y), n_k).
        # Side S = a / (sqrt(3)/2) = 2a / sqrt(3).
        
        # Normals for the outer hexagon
        normals = []
        for k in range(6):
            angle = theta + np.radians(k * 60)
            normals.append(np.array([np.cos(angle), np.sin(angle)]))
        
        max_proj = 0.0
        for p in points:
            for n in normals:
                proj = np.dot(p, n)
                if proj > max_proj:
                    max_proj = proj
        
        apothem_required = max_proj
        side_required = apothem_required * 2.0 / SQRT3
        return side_required

    # 4. Optimization
    # We rotate the entire cluster by alpha.
    # The outer hexagon rotation beta is also a variable.
    # However, the system is invariant under simultaneous rotation of cluster and outer hexagon.
    # So we can fix the outer hexagon angle to 0 and just rotate the cluster.
    # OR fix the cluster angle to 0 and rotate the outer hexagon.
    # Let's fix outer hexagon angle to 0 and rotate the cluster by alpha.
    
    # Pre-compute cluster vertices in local frame (alpha=0)
    local_vertices = []
    for i in range(12):
        cx, cy = centers_cart[i]
        verts = get_hex_vertices(cx, cy, 1.0, 0.0)
        local_vertices.append(verts)
    
    # Flatten all vertices
    all_local_verts = np.vstack(local_vertices)
    
    # Function to evaluate side length for a given cluster rotation alpha
    def objective(alpha):
        # Rotate all vertices by alpha
        cos_a = np.cos(alpha)
        sin_a = np.sin(alpha)
        rot_matrix = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
        rotated_verts = all_local_verts @ rot_matrix.T
        
        # Outer hexagon is fixed at angle 0
        # Normals for angle 0: 0, 60, 120, 180, 240, 300 degrees
        normals = []
        for k in range(6):
            angle = np.radians(k * 60)
            normals.append(np.array([np.cos(angle), np.sin(angle)]))
        
        max_proj = 0.0
        for p in rotated_verts:
            for n in normals:
                proj = np.dot(p, n)
                if proj > max_proj:
                    max_proj = proj
        
        apothem_required = max_proj
        side_required = apothem_required * 2.0 / SQRT3
        return side_required

    # 5. Search for optimal alpha
    # The function is periodic with period 60 degrees (pi/3).
    # We can search in [0, pi/3].
    
    # Use a fine grid search first, then refine.
    best_side = float('inf')
    best_alpha = 0.0
    
    # Grid search
    alphas = np.linspace(0, np.pi/3, 300)
    for a in alphas:
        s = objective(a)
        if s < best_side:
            best_side = s
            best_alpha = a
    
    # Refine with Nelder-Mead or Golden Section
    # Since it's 1D, Golden Section is good.
    def golden_section_search(f, a, b, tol=1e-6):
        gr = (np.sqrt(5) + 1) / 2
        c = b - (b - a) / gr
        d = a + (b - a) / gr
        fc = f(c)
        fd = f(d)
        while abs(b - a) > tol:
            if fc < fd:
                b = d
                d = c
                fd = fc
                c = b - (b - a) / gr
                fc = f(c)
            else:
                a = c
                c = d
                fc = fd
                d = a + (b - a) / gr
                fd = f(d)
        return (a + b) / 2, f((a + b) / 2)
    
    # Find local minimum around best_alpha from grid
    # Determine bracket from grid
    idx = np.argmin([objective(a) for a in alphas])
    a_left = alphas[max(0, idx-1)]
    a_right = alphas[min(len(alphas)-1, idx+1)]
    
    best_alpha_refined, best_side_refined = golden_section_search(objective, a_left, a_right)
    
    if best_side_refined < best_side:
        best_side = best_side_refined
        best_alpha = best_alpha_refined

    # 6. Construct the final answer
    # We need to return:
    # inner: list of (x, y, angle_degrees) for 12 hexes
    # center: (x, y) of outer hexagon
    # side: side length of outer hexagon
    # angle: rotation of outer hexagon
    
    # The outer hexagon is centered at (0,0) and rotated by 0 degrees.
    # The inner hexagons are rotated by best_alpha.
    
    inner_hexes = []
    for i in range(12):
        cx, cy = centers_cart[i]
        # Rotate center by best_alpha
        cos_a = np.cos(best_alpha)
        sin_a = np.sin(best_alpha)
        rx = cx * cos_a - cy * sin_a
        ry = cx * sin_a + cy * cos_a
        # The hexagon itself is rotated by best_alpha (since it was aligned to 0 in local frame)
        angle_deg = np.degrees(best_alpha)
        inner_hexes.append((rx, ry, angle_deg))
        
    outer_center = (0.0, 0.0)
    outer_side = best_side
    outer_angle = 0.0
    
    return inner_hexes, outer_center, outer_side, outer_angle

# EVOLVE-BLOCK-END
