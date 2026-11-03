import math
import numpy as np
import time
from scipy.optimize import minimize

N = 11
APOTHEM = math.sqrt(3) / 2.0
START_TIME = time.time()
TIME_LIMIT = 150.0
SAFETY_MARGIN = 10.0
EPS = 1e-9
MIN_DIST = math.sqrt(3) + 1e-6

def hex_vertices(cx, cy, s, angle):
    # Regular hexagon vertices
    # Side length s, center (cx, cy), rotation angle
    # Vertices are at distance s from center
    # Standard orientation: flat top? Or pointy top?
    # Axial to cartesian used in seed:
    # x = sqrt(3) * (q + r/2)
    # y = 1.5 * r
    # This corresponds to a hexagon with flat tops (vertices at 0, 60, ... degrees? No.)
    # Let's check the distance between (0,0) and (1,0) in axial:
    # (0,0) -> (0,0)
    # (1,0) -> (sqrt(3), 0)
    # Distance is sqrt(3).
    # For side length 1, the distance between centers of adjacent hexagons is 2 * apothem = sqrt(3).
    # So the hexagons are "flat topped" if we consider the standard axial coords.
    # Actually, let's just define the support function or use a simple distance check.
    # For the objective, we need the distance from center to the furthest vertex along the container normals.
    # A regular hexagon of side 1 has circumradius 1.
    # The distance from center to any point on the boundary in direction theta is:
    # r(theta) = apothem / cos((theta - nearest_vertex_angle) mod pi/3)
    # Or simpler: The support function of a hexagon.
    pass

def support_function_hex(theta, angle_offset):
    # Returns the distance from center to the boundary of a unit hexagon in direction theta
    # Assuming "flat top" orientation where vertices are at 0, pi/3, ...
    # Wait, axial coords:
    # (1,0) is at (sqrt(3), 0). Vector is horizontal.
    # The hexagon at (0,0) has a vertex pointing towards (1,0)?
    # If vertices are at 0, 60, 120... then vertex at 0 is (1,0) relative to center? No, circumradius is 1.
    # So vertex at (1,0).
    # Then the face opposite to (1,0) is at x = -1.
    # The face parallel to y-axis?
    # Let's stick to the property: distance between centers of non-overlapping hexagons must be >= sqrt(3).
    # This is true for ANY orientation if they are aligned? No.
    # If one is rotated 30 degrees, the distance to touch is different.
    # However, the optimal packing of regular polygons usually involves aligned orientations or specific symmetries.
    # For simplicity and robustness, we will assume all inner hexagons have the SAME orientation as the outer one (or 0 relative to each other) initially, 
    # but allow the WHOLE cluster to rotate. 
    # Actually, the problem says "each inner hexagon may translate and rotate".
    # But mixing rotations is complex. Let's assume all inner hexagons are parallel (angle 0) for the LIO, 
    # and we optimize the global rotation. This is a strong constraint but often optimal for hexagons.
    
    # Support function of a unit hexagon (side 1) with vertices at angles 0, pi/3, ...
    # The boundary is composed of 6 segments.
    # Distance from center to edge (apothem) is sqrt(3)/2.
    # Angle to vertex is 0.
    # The support function h(theta) = apothem / cos((theta % (pi/3)) - pi/6)
    # Let's verify: at theta=0 (vertex), cos(-pi/6) = sqrt(3)/2. h = (sqrt(3)/2) / (sqrt(3)/2) = 1. Correct (circumradius).
    # At theta=pi/6 (mid-edge), cos(0) = 1. h = sqrt(3)/2. Correct (apothem).
    
    theta_mod = np.abs(theta % (math.pi / 3.0))
    # Map to [-pi/6, pi/6]
    if theta_mod > math.pi / 6.0:
        theta_mod = math.pi / 3.0 - theta_mod
    return (math.sqrt(3) / 2.0) / np.cos(theta_mod - math.pi / 6.0)

def construct():
    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
        # Fallback: return a valid, though suboptimal, configuration
        # Simple grid
        pts = []
        for q in range(-1, 2):
            for r in range(-1, 2):
                if abs(q) + abs(r) + abs(q+r) <= 2: # Hex shape of radius 2? No, that's 7.
                    pass
        # Just return the sparse one from seed to be safe
        sparse_11 = [
            (0, 0),
            (2, 0), (-2, 0),
            (0, 2), (0, -2),
            (2, 2), (-2, -2),
            (2, -2), (-2, 2),
            (4, 0),
            (0, 4)
        ]
        inner = []
        for q, r in sparse_11:
            x = math.sqrt(3) * (q + r / 2.0)
            y = 1.5 * r
            inner.append([float(x), float(y), 0.0])
        # Estimate side
        xs = [p[0] for p in inner]
        ys = [p[1] for p in inner]
        cx = (max(xs)+min(xs))/2
        cy = (max(ys)+min(ys))/2
        # Rough side
        max_r = max(math.sqrt((x-cx)**2 + (y-cy)**2) for x,y in inner)
        side = (max_r + APOTHEM) / APOTHEM
        return inner, [cx, cy], side, 0.0

    # 1. Generate Initial Candidates
    # We need 11 points.
    # Candidate 1: Axial grid "flower" + extras
    # Candidate 2: Linear strip
    # Candidate 3: Random
    
    candidates = []
    
    # Grid based
    axial_grid = []
    for q in range(-2, 3):
        for r in range(-2, 3):
            axial_grid.append((q, r))
    
    # Filter to 11 closest to center
    axial_grid.sort(key=lambda p: p[0]**2 + p[1]**2 + p[0]*p[1])
    base_axial = axial_grid[:11]
    
    # Convert to cartesian
    def axial_to_cartesian(axial_list):
        pts = []
        for q, r in axial_list:
            x = math.sqrt(3) * (q + r / 2.0)
            y = 1.5 * r
            pts.append([x, y])
        return np.array(pts)
    
    candidates.append(axial_to_cartesian(base_axial))
    
    # Another candidate: 3x3 grid minus corners?
    # Let's just use the same base for LIO, as LIO should fix overlaps if they exist in the initial guess (by pushing apart).
    # However, if initial guess has heavy overlap, LIO might get stuck.
    # Let's create a "spread out" random candidate too.
    
    np.random.seed(42)
    rand_pts = np.random.uniform(-5, 5, (11, 2))
    candidates.append(rand_pts)
    
    best_global_S = float('inf')
    best_global_pts = None
    best_global_beta = 0.0
    best_global_center = None

    for init_pts in candidates:
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            break
            
        # Objective Function for LIO
        # Variables: x_0, y_0, ..., x_10, y_10, beta (rotation of container)
        # Total 23 variables.
        
        def objective(params):
            # Time check inside objective is expensive but necessary for safety
            if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
                return 1e10
            
            pts = params[:22].reshape(11, 2)
            beta = params[22]
            
            # 1. Overlap Penalty
            # Min distance between centers of two unit hexagons (parallel) is sqrt(3)
            # If they touch, dist = sqrt(3).
            # Penalty = sum( max(0, (sqrt(3) - dist))^2 )
            penalty = 0.0
            min_dist_sq = 3.0 # (sqrt(3))^2
            for i in range(11):
                for j in range(i + 1, 11):
                    dx = pts[i, 0] - pts[j, 0]
                    dy = pts[i, 1] - pts[j, 1]
                    dist_sq = dx*dx + dy*dy
                    if dist_sq < min_dist_sq:
                        # Strong penalty to push them apart
                        # Use linear penalty for stability? No, quadratic is standard.
                        # dist = sqrt(dist_sq). 
                        # To avoid sqrt, we can use (min_dist - dist) approx?
                        # Let's just use sqrt, it's only 55 pairs.
                        dist = math.sqrt(dist_sq)
                        diff = math.sqrt(3.0) - dist
                        if diff > 0:
                            penalty += diff * diff * 10.0 # Weight 10 to prioritize non-overlap
            
            # 2. Container Size
            # We want to minimize the side length S of the outer hexagon.
            # The outer hexagon has center C and rotation beta.
            # To minimize S, we should center the cluster optimally.
            # For a given beta, the optimal center C is such that the max extent in all 6 normal directions is minimized.
            # This is a linear programming problem, but we can approximate it.
            # Alternatively, we can include C in the variables?
            # If we include C (2 vars), total 25 vars.
            # Let's include Cx, Cy.
            
            # Wait, I defined params as 23. Let's change to 25: 22 coords + 1 beta + 2 center?
            # No, if we optimize C, we can just compute the bounding box of pts relative to C.
            # Actually, for a fixed set of points and fixed beta, the optimal C is the center of the "minimal enclosing hexagon".
            # This is equivalent to finding C that minimizes max_i max_k | (P_i - C) . n_k |
            # where n_k are the 6 normals.
            # This is a convex optimization.
            # For simplicity in the objective, let's estimate C as the centroid, and refine if needed?
            # No, that's bad.
            # Let's add Cx, Cy to the variables.
            
            # RE-DEFINE OBJECTIVE VARS:
            # 0-21: x,y for 11 hexes
            # 22: beta
            # 23: cx
            # 24: cy
            
            # Since I'm re-defining the structure, let's restart the objective logic clearly.
            # I will implement the objective inside the loop with the correct dimensionality.
            
            return penalty # Placeholder, will rewrite below
        
        # Corrected Objective Setup
        # Variables: 22 (coords) + 1 (beta) + 2 (center) = 25
        
        def lio_objective(params):
            if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
                return 1e10
            
            pts = params[:22].reshape(11, 2)
            beta = params[22]
            cx = params[23]
            cy = params[24]
            
            # Overlap Penalty
            penalty = 0.0
            for i in range(11):
                for j in range(i + 1, 11):
                    dx = pts[i, 0] - pts[j, 0]
                    dy = pts[i, 1] - pts[j, 1]
                    dist_sq = dx*dx + dy*dy
                    if dist_sq < 3.0:
                        dist = math.sqrt(dist_sq)
                        diff = math.sqrt(3.0) - dist
                        if diff > 0:
                            penalty += diff * diff * 50.0
            
            # Container Size
            # Normals for outer hexagon
            # Outer hexagon rotation is beta.
            # Normals are perpendicular to sides.
            # If hexagon vertices are at beta, beta+pi/3..., then normals are at beta+pi/6, ...
            base_normal_angle = beta + math.pi / 6.0
            normals = []
            for k in range(6):
                ang = base_normal_angle + k * math.pi / 3.0
                normals.append([math.cos(ang), math.sin(ang)])
            normals = np.array(normals)
            
            # Relative positions
            rel_pts = pts - np.array([cx, cy])
            
            # Projection of each point onto each normal
            # proj[i, k] = dot(P_i - C, n_k)
            proj = rel_pts @ normals.T
            
            # For the outer hexagon to contain the inner hexagons,
            # The outer boundary in direction n_k must be at distance D from center.
            # The inner hexagon i extends in direction n_k by support_function_hex(n_k_angle, 0)
            # Wait, the inner hexagons are aligned (angle 0).
            # The support function of an inner hexagon in direction theta is:
            # h(theta) = APOTHEM / cos((theta mod pi/3) - pi/6)
            
            # We need D >= max_i ( proj[i, k] + h(theta_k) ) for all k
            # And D >= max_i ( -proj[i, k] + h(theta_k + pi) ) ... wait, support is symmetric?
            # h(theta) is the distance to the boundary in direction theta.
            # The condition is: C + D * n_k must be outside P_i + hex_i.
            # Distance from C to P_i is proj.
            # The "radius" of P_i in direction n_k is h(theta_k).
            # So D must be >= |proj| + h(theta_k)?
            # No.
            # The outer hexagon face is at distance D from center C, normal n_k.
            # The inner hexagon P_i must satisfy: (P_i - C) . n_k + h_i(n_k) <= D
            # AND
            # -(P_i - C) . n_k + h_i(-n_k) <= D  (for the opposite face)
            
            # Since h(theta) is not symmetric in a simple way?
            # h(theta) for hexagon:
            # Vertices at 0, 60...
            # h(0) = 1. h(60) = 1. h(30) = 0.866.
            # It is symmetric around 0 and 60.
            # h(-theta) = h(theta) for this specific shape?
            # h(-30) = APOTHEM / cos(-30 - 30) = APOTHEM / cos(-60) = APOTHEM / 0.5 = sqrt(3) = 1.732?
            # Wait.
            # Formula: APOTHEM / cos((theta % (pi/3)) - pi/6)
            # theta = -pi/6 (-30 deg).
            # -pi/6 % pi/3 = 2pi/6 = pi/3? No.
            # Python % for floats: -0.5 % 1.0 = 0.5.
            # -pi/6 % pi/3 = pi/6.
            # cos(pi/6 - pi/6) = cos(0) = 1.
            # h = APOTHEM = 0.866.
            # So h(-30) = 0.866.
            # h(30) = 0.866.
            # It is symmetric.
            
            # So for each normal direction n_k (angle theta_k):
            # Required D_k = max_i ( (P_i - C) . n_k + h(theta_k) )
            # AND
            # Required D_k = max_i ( -(P_i - C) . n_k + h(theta_k) )  # Because h(-theta) = h(theta)
            
            # Actually, the constraint is:
            # D >= max_i ( (P_i - C) . n_k + h(theta_k) )
            # D >= max_i ( (C - P_i) . n_k + h(theta_k) )  # Which is max_i ( -(P_i - C) . n_k + h(theta_k) )
            
            # So D must be at least the max of both.
            # Let M_k = max( max_i(proj[i,k] + h_k), max_i(-proj[i,k] + h_k) )
            # Then D = max_k M_k.
            # Side Length S = (D + APOTHEM) / APOTHEM
            
            # Calculate h_k for the 6 normals
            # theta_k is the angle of the normal.
            # The inner hexagons are at angle 0.
            # So we need h(theta_k) for the 6 specific angles.
            
            h_vals = []
            for k in range(6):
                theta_k = base_normal_angle + k * math.pi / 3.0
                # Calculate support function of unit hex (angle 0) in direction theta_k
                # h(theta) = APOTHEM / cos((theta % (pi/3)) - pi/6)
                # We need to be careful with the modulo for negative angles or angles > pi/3.
                # Let's use the geometric property:
                # The hexagon is invariant under rotation by 60 deg.
                # So h(theta) = h(theta mod pi/3) ?? No.
                # h is periodic with pi/3.
                # Within [0, pi/3], it goes from 1 to 0.866 to 1.
                # Let's just implement the function carefully.
                
                def calc_h(theta):
                    # Map theta to [0, pi/3)
                    t = theta % (math.pi / 3.0)
                    # t is in [0, pi/3)
                    # The "center" of the interval is pi/6.
                    # Distance from center: t - pi/6.
                    # Range: [-pi/6, pi/6)
                    return (math.sqrt(3) / 2.0) / np.cos(t - math.pi / 6.0)
                
                h_vals.append(calc_h(theta_k))
            
            h_vals = np.array(h_vals)
            
            # proj is 11x6
            # proj_plus = proj + h_vals (broadcast)
            # proj_minus = -proj + h_vals
            
            proj_plus = proj + h_vals
            proj_minus = -proj + h_vals
            
            # Max over points (axis 0)
            max_plus = np.max(proj_plus, axis=0)
            max_minus = np.max(proj_minus, axis=0)
            
            M = np.maximum(max_plus, max_minus)
            D = np.max(M)
            
            S = (D + APOTHEM) / APOTHEM
            
            return penalty + S
        
        # Initial Guess
        # pts from candidate
        # beta = 0
        # center = centroid of pts
        cx_init = np.mean(init_pts[:, 0])
        cy_init = np.mean(init_pts[:, 1])
        
        x0 = np.concatenate([init_pts.flatten(), [0.0, cx_init, cy_init]])
        
        # Optimize
        try:
            res = minimize(lio_objective, x0, method='L-BFGS-B', 
                           options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-12})
            
            if res.fun < best_global_S:
                # Check validity
                pts_opt = res.x[:22].reshape(11, 2)
                # Verify no overlaps
                valid = True
                for i in range(11):
                    for j in range(i + 1, 11):
                        dx = pts_opt[i, 0] - pts_opt[j, 0]
                        dy = pts_opt[i, 1] - pts_opt[j, 1]
                        if dx*dx + dy*dy < 3.0 - 1e-5:
                            valid = False
                            break
                    if not valid: break
                
                if valid:
                    best_global_S = float(res.fun)
                    best_global_pts = pts_opt
                    best_global_beta = float(res.x[22])
                    best_global_center = (float(res.x[23]), float(res.x[24]))
                    
        except Exception as e:
            pass

    if best_global_pts is None:
        # Fallback
        best_global_pts = candidates[0]
        best_global_beta = 0.0
        cx = np.mean(best_global_pts[:, 0])
        cy = np.mean(best_global_pts[:, 1])
        best_global_center = (cx, cy)
        # Calculate S
        # Just use a large estimate
        best_global_S = 10.0

    # Finalize Output
    cx, cy = best_global_center
    angle_deg = math.degrees(best_global_beta)
    
    # Recalculate S accurately for the return value
    # We need to return the side length.
    # We can just use the objective's S part, but let's recalculate to be sure.
    
    def get_final_side(pts, cx, cy, beta):
        base_normal_angle = beta + math.pi / 6.0
        normals = []
        for k in range(6):
            ang = base_normal_angle + k * math.pi / 3.0
            normals.append([math.cos(ang), math.sin(ang)])
        normals = np.array(normals)
        
        rel_pts = pts - np.array([cx, cy])
        proj = rel_pts @ normals.T
        
        def calc_h(theta):
            t = theta % (math.pi / 3.0)
            return (math.sqrt(3) / 2.0) / np.cos(t - math.pi / 6.0)
        
        h_vals = np.array([calc_h(base_normal_angle + k * math.pi / 3.0) for k in range(6)])
        
        proj_plus = proj + h_vals
        proj_minus = -proj + h_vals
        
        max_plus = np.max(proj_plus, axis=0)
        max_minus = np.max(proj_minus, axis=0)
        
        M = np.maximum(max_plus, max_minus)
        D = np.max(M)
        
        return (D + APOTHEM) / APOTHEM

    final_side = get_final_side(best_global_pts, cx, cy, best_global_beta)
    
    inner = []
    for x, y in best_global_pts:
        inner.append([float(x), float(y), 0.0]) # Inner hexagons are aligned (angle 0)
        
    center = [float(cx), float(cy)]
    side = float(final_side)
    rotation = float(angle_deg)
    
    return inner, center, side, rotation

