"""Pack 12 unit hexagons inside a regular hexagon, minimizing outer side length."""
import math
import time
import numpy as np
from scipy.optimize import minimize, differential_evolution

N = 12

# EVOLVE-BLOCK-START

def hex_vertices(cx, cy, angle_deg, side=1.0):
    """Return vertices of a regular hexagon centered at (cx,cy) with given rotation."""
    a = math.radians(angle_deg)
    verts = []
    for i in range(6):
        theta = a + i * math.pi / 3
        verts.append((cx + side * math.cos(theta), cy + side * math.sin(theta)))
    return np.array(verts)


def polygon_area_signed(verts):
    """Shoelace formula for signed area."""
    n = len(verts)
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += verts[i][0] * verts[j][1] - verts[j][0] * verts[i][1]
    return 0.5 * area


def point_in_polygon(pt, poly):
    """Ray casting point-in-polygon test."""
    x, y = pt
    n = len(poly)
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = poly[i]
        xj, yj = poly[j]
        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
            inside = not inside
        j = i
    return inside


def polygons_overlap_sat(v1, v2, tol=1e-9):
    """Check if two convex polygons overlap using SAT. Returns True if they overlap."""
    polys = [v1, v2]
    for poly in polys:
        for i in range(len(poly)):
            j = (i + 1) % len(poly)
            # Edge normal
            edge = poly[j] - poly[i]
            normal = np.array([-edge[1], edge[0]])
            # Project both polygons onto normal
            proj1 = v1 @ normal
            proj2 = v2 @ normal
            min1, max1 = proj1.min(), proj1.max()
            min2, max2 = proj2.min(), proj2.max()
            # If there's a gap, no overlap
            if max1 < min2 - tol or max2 < min1 - tol:
                return False
    return True


def min_distance_polygons(v1, v2):
    """Approximate minimum distance between two convex polygons (0 if overlapping)."""
    if polygons_overlap_sat(v1, v2):
        return 0.0
    # Check all edge-to-edge distances
    min_d = float('inf')
    for i in range(len(v1)):
        for j in range(len(v2)):
            d = np.linalg.norm(v1[i] - v2[j])
            min_d = min(min_d, d)
    return min_d


def compute_min_outer_side(inner_hexes, outer_center, outer_angle_deg):
    """
    Given inner hexagons (list of [cx, cy, angle_deg]) and outer hexagon params,
    find the minimum outer side length such that all inner hexagons fit inside.
    We compute the max distance from outer center to any vertex of any inner hexagon,
    then convert to side length.
    """
    ox, oy = outer_center
    oa = math.radians(outer_angle_deg)
    # Outer hexagon vertices for a unit side
    outer_unit_verts = []
    for i in range(6):
        theta = oa + i * math.pi / 3
        outer_unit_verts.append((math.cos(theta), math.sin(theta)))
    outer_unit_verts = np.array(outer_unit_verts)
    
    # For each inner hexagon vertex, find the minimum scale s such that
    # the vertex is inside the outer hexagon of side s centered at outer_center
    # A point p is inside hexagon(center, side, angle) iff for all edges,
    # the dot product with the inward normal <= side * inradius_factor
    
    # The inradius of a regular hexagon with side s is s * sqrt(3)/2
    # The apothem (distance from center to edge) = s * sqrt(3)/2
    
    # For a regular hexagon with vertices at angles theta_i = angle + i*pi/3,
    # the edge between vertex i and i+1 has inward normal pointing from edge to center.
    # The constraint is: for each edge, dot(p - center, inward_normal_i) <= apothem
    
    # Inward normals: for edge from v_i to v_{i+1}, the outward normal is perpendicular to edge
    # For regular hexagon, inward normal to edge i is at angle theta_i + pi/6 + pi (pointing inward)
    # Actually, let's think: edge i connects vertices at angles theta_i and theta_{i+1} = theta_i + pi/3
    # Midpoint of edge i is at angle theta_i + pi/6 from center, at distance apothem
    # Inward normal to edge i points from edge midpoint toward center, i.e., at angle theta_i + pi/6 + pi
    
    max_scale = 0.0
    for hex_data in inner_hexes:
        cx, cy, ang = hex_data
        verts = hex_vertices(cx, cy, ang)
        for vx, vy in verts:
            # Vector from outer center to this vertex
            dx, dy = vx - ox, vy - oy
            # For the outer hexagon with side s, the constraint is:
            # For each of the 6 edges, dot((dx,dy), inward_normal_k) <= s * sqrt(3)/2
            # where inward_normal_k is unit vector
            # So s >= dot((dx,dy), inward_normal_k) / (sqrt(3)/2) for each k
            # s must satisfy all 6, so s >= max_k(dot((dx,dy), inward_normal_k) / (sqrt(3)/2))
            # But also s >= 0, and we need the point to be inside, so all constraints must hold.
            # The minimum s for this point is max over k of max(0, dot((dx,dy), inward_normal_k)) / (sqrt(3)/2)
            
            # Inward normal to edge k: edge k is between vertices at angles oa + k*pi/3 and oa + (k+1)*pi/3
            # The outward normal to edge k is at angle oa + k*pi/3 + pi/6
            # The inward normal is the negative: at angle oa + k*pi/3 + pi/6 + pi
            # Actually for a regular hexagon, the apothem direction to edge k is at angle oa + k*pi/3 + pi/6
            # and the inward normal (from edge toward center) is at angle oa + k*pi/3 + pi/6 + pi
            
            required_s = 0.0
            for k in range(6):
                # Direction from center to edge k midpoint
                edge_mid_angle = oa + k * math.pi / 3 + math.pi / 6
                # Inward normal: from edge toward center, opposite to edge_mid_angle
                inward_nx = -math.cos(edge_mid_angle)
                inward_ny = -math.sin(edge_mid_angle)
                dot_val = dx * inward_nx + dy * inward_ny
                # Constraint: dot_val <= s * sqrt(3)/2
                # So s >= dot_val / (sqrt(3)/2) if dot_val > 0
                if dot_val > 0:
                    s_req = dot_val / (math.sqrt(3) / 2)
                    required_s = max(required_s, s_req)
            max_scale = max(max_scale, required_s)
    
    return max_scale


def objective(params, outer_center, outer_angle_deg):
    """
    Objective: return the minimum outer side length given inner hexagon positions.
    params: flattened array of [cx1, cy1, ang1, cx2, cy2, ang2, ...]
    """
    inner_hexes = []
    for i in range(N):
        cx = params[3*i]
        cy = params[3*i+1]
        ang = params[3*i+2]
        inner_hexes.append([cx, cy, ang])
    
    # Check for overlaps between inner hexagons
    verts_list = [hex_vertices(h[0], h[1], h[2]) for h in inner_hexes]
    for i in range(N):
        for j in range(i+1, N):
            if polygons_overlap_sat(verts_list[i], verts_list[j], tol=1e-7):
                # Overlap penalty: large value
                return 1e6
    
    # Compute minimum outer side
    s = compute_min_outer_side(inner_hexes, outer_center, outer_angle_deg)
    return s


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    # Initial layout: hexagonal lattice arrangement
    # For 12 hexagons, use a hexagonal number pattern
    # Center + ring of 6 + ring of 9? No, 1+6=7, 1+6+12=19. 
    # Let's use a more compact arrangement.
    # 12 hexagons: could be 3x4 grid or hexagonal cluster
    
    # Start with a hexagonal lattice: central hexagon surrounded by others
    # Hexagonal lattice spacing for unit hexagons: distance between centers = sqrt(3) * side = sqrt(3)
    # for adjacent hexagons in the same row, or 2*sqrt(3)/2 = sqrt(3) for nearest neighbors
    
    # Actually for regular hexagons of side 1 tiling:
    # Horizontal spacing: 3/2 (for pointy-top) or sqrt(3) (for flat-top)
    # Let's use flat-top hexagons (angle 0 means flat top)
    
    # For flat-top hexagons:
    # Width = 2 * side = 2
    # Height = sqrt(3) * side = sqrt(3)
    # Horizontal spacing between centers in same row: 3/2 * side = 1.5
    # Vertical spacing between rows: sqrt(3) * side = sqrt(3)
    # Odd rows offset by 1.5 horizontally
    
    # Let's try a compact hexagonal cluster
    # Row 0: 3 hexagons
    # Row 1: 3 hexagons  
    # Row 2: 3 hexagons
    # Row 3: 3 hexagons
    # That's 12 in a 4x3 grid
    
    sqrt3 = math.sqrt(3)
    
    # Initial positions: 4 rows x 3 columns of flat-top hexagons
    initial_params = np.zeros(3 * N)
    idx = 0
    for row in range(4):
        for col in range(3):
            cx = col * 1.5
            cy = row * sqrt3
            if row % 2 == 1:
                cx += 0.75
            # Center the layout
            cx -= 1.5  # center 3 columns
            cy -= 1.5 * sqrt3  # center 4 rows
            initial_params[idx] = cx
            initial_params[idx+1] = cy
            initial_params[idx+2] = 0.0
            idx += 3
    
    outer_center = (0.0, 0.0)
    outer_angle_deg = 0.0
    
    # Try multiple initial configurations
    best_result = None
    best_score = float('inf')
    
    configs = []
    
    # Config 1: 4x3 grid (flat-top)
    configs.append(initial_params.copy())
    
    # Config 2: Hexagonal cluster
    # Center + 6 around + 5 more
    hex_cluster = np.zeros(3 * N)
    # Center
    hex_cluster[0] = 0.0
    hex_cluster[1] = 0.0
    hex_cluster[2] = 0.0
    # Ring 1: 6 hexagons at distance sqrt(3)
    for i in range(6):
        ang = i * math.pi / 3
        idx = 3 * (i + 1)
        hex_cluster[idx] = sqrt3 * math.cos(ang)
        hex_cluster[idx+1] = sqrt3 * math.sin(ang)
        hex_cluster[idx+2] = 0.0
    # Ring 2: 5 more hexagons (partial ring)
    for i in range(5):
        ang = i * 2 * math.pi / 5 + math.pi / 5  # offset to avoid symmetry issues
        idx = 3 * (i + 7)
        hex_cluster[idx] = 2 * sqrt3 * math.cos(ang)
        hex_cluster[idx+1] = 2 * sqrt3 * math.sin(ang)
        hex_cluster[idx+2] = 0.0
    configs.append(hex_cluster)
    
    # Config 3: 2 rows of 6 (compact rectangle)
    rect_params = np.zeros(3 * N)
    for row in range(2):
        for col in range(6):
            cx = col * 1.5 - 2.25
            cy = row * sqrt3 - sqrt3/2
            idx = 3 * (row * 6 + col)
            rect_params[idx] = cx
            rect_params[idx+1] = cy
            rect_params[idx+2] = 0.0
    configs.append(rect_params)
    
    # Config 4: 3 rows of 4
    rect34 = np.zeros(3 * N)
    for row in range(3):
        for col in range(4):
            cx = col * 1.5 - 1.5
            cy = row * sqrt3 - sqrt3
            idx = 3 * (row * 4 + col)
            rect34[idx] = cx
            rect34[idx+1] = cy
            rect34[idx+2] = 0.0
    configs.append(rect34)
    
    for config in configs:
        if time.time() > deadline:
            break
        
        # Evaluate initial score
        score = objective(config, outer_center, outer_angle_deg)
        
        if score < best_score:
            best_score = score
            best_result = config.copy()
        
        # Local optimization using Nelder-Mead
        if score < 1e5:  # Only optimize if not heavily overlapping
            try:
                res = minimize(
                    objective,
                    config,
                    args=(outer_center, outer_angle_deg),
                    method='Nelder-Mead',
                    options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-10}
                )
                if res.fun < best_score:
                    best_score = res.fun
                    best_result = res.x.copy()
            except Exception:
                pass
        
        # Also try with slight rotation of outer hexagon
        for outer_ang in [30.0, 60.0]:
            if time.time() > deadline:
                break
            score2 = objective(config, outer_center, outer_ang)
            if score2 < best_score:
                best_score = score2
                best_result = config.copy()
                outer_angle_deg = outer_ang
    
    # If we have time, try differential evolution for global search
    if time.time() < deadline - 30:
        try:
            # Define bounds
            bounds = [(-5, 5), (-5, 5), (-180, 180)] * N
            def de_objective(params):
                return objective(params, outer_center, outer_angle_deg)
            
            de_res = differential_evolution(
                de_objective,
                bounds,
                seed=42,
                maxiter=50,
                tol=1e-6,
                polish=False,
                update_interval=10
            )
            if de_res.fun < best_score:
                best_score = de_res.fun
                best_result = de_res.x.copy()
        except Exception:
            pass
    
    # Final local polish on best result
    if best_result is not None and time.time() < deadline - 10:
        try:
            res = minimize(
                objective,
                best_result,
                args=(outer_center, outer_angle_deg),
                method='Nelder-Mead',
                options={'maxiter': 5000, 'xatol': 1e-9, 'fatol': 1e-12}
            )
            if res.fun < best_score:
                best_score = res.fun
                best_result = res.x.copy()
        except Exception:
            pass
    
    # Construct output
    if best_result is None:
        best_result = initial_params
    
    inner = []
    for i in range(N):
        cx = best_result[3*i]
        cy = best_result[3*i+1]
        ang = best_result[3*i+2]
        inner.append([cx, cy, ang])
    
    # Use the computed minimum side length
    side = compute_min_outer_side(inner, outer_center, outer_angle_deg)
    
    return inner, (0.0, 0.0), side, outer_angle_deg

# EVOLVE-BLOCK-END
