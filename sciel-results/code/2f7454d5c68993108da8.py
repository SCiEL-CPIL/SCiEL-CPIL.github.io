import math
import time
import numpy as np
from scipy.optimize import minimize
from shapely.geometry import Polygon

N = 12

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin for 150s limit

    # Precompute unit hexagon vertices for angle 0 (pointy-top, vertex at 0 deg)
    # Vertices at 0, 60, 120, 180, 240, 300 degrees
    unit_verts_0 = np.array([[math.cos(math.radians(a)), math.sin(math.radians(a))] for a in range(0, 360, 60)])

    def get_hex_verts(cx, cy, side, angle_deg):
        angle_rad = math.radians(angle_deg)
        cos_a = math.cos(angle_rad)
        sin_a = math.sin(angle_rad)
        rot = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
        verts = unit_verts_0 @ rot.T * side
        verts += np.array([cx, cy])
        return verts

    def calc_min_outer_side(positions, angles, outer_angle=0.0):
        """
        Calculate the minimum side length of a regular hexagon centered at (0,0) 
        with rotation outer_angle that contains all inner hexagons.
        """
        theta_rad = math.radians(outer_angle)
        # Normals are at angles theta + 30 + k*60
        normals = []
        for k in range(6):
            ang = theta_rad + math.pi/6 + k * math.pi/3
            normals.append([math.cos(ang), math.sin(ang)])
        normals = np.array(normals) # (6, 2)
        
        max_proj = 0.0
        
        for i in range(N):
            cx, cy = positions[i]
            ang = angles[i]
            verts = get_hex_verts(cx, cy, 1.0, ang)
            
            # Project all vertices onto the normals
            projs = verts @ normals.T # (6, 6)
            max_proj_i = np.max(projs)
            if max_proj_i > max_proj:
                max_proj = max_proj_i
        
        return max_proj * 2.0 / math.sqrt(3)

    def check_disjoint(positions, angles):
        """
        Check if inner hexagons are disjoint using Shapely.
        Returns True if disjoint, False if any overlap.
        """
        polys = []
        for i in range(N):
            cx, cy = positions[i]
            ang = angles[i]
            verts = get_hex_verts(cx, cy, 1.0, ang)
            polys.append(Polygon(verts))
        
        for i in range(N):
            for j in range(i+1, N):
                if polys[i].intersects(polys[j]):
                    return False
        return True

    def sat_distance(p1_verts, p2_verts):
        """
        Calculate the minimum separation distance between two convex polygons
        using the Separating Axis Theorem. Returns 0 if they intersect.
        """
        min_dist = float('inf')
        
        # Get unique edge normals for both polygons
        normals = []
        for verts in [p1_verts, p2_verts]:
            for i in range(len(verts)):
                v1 = verts[i]
                v2 = verts[(i + 1) % len(verts)]
                edge = v2 - v1
                # Normal is perpendicular to edge
                n = np.array([-edge[1], edge[0]])
                n = n / (np.linalg.norm(n) + 1e-9)
                # Check if this normal is already present (to avoid duplicates)
                is_dup = False
                for existing in normals:
                    if np.dot(n, existing) > 0.9999:
                        is_dup = True
                        break
                if not is_dup:
                    normals.append(n)
        
        for n in normals:
            # Project all vertices of both polygons onto the normal
            proj1 = p1_verts @ n
            proj2 = p2_verts @ n
            
            min1, max1 = np.min(proj1), np.max(proj1)
            min2, max2 = np.min(proj2), np.max(proj2)
            
            # Check for overlap
            if max1 < min2 or max2 < min1:
                # They are separated along this axis
                dist = min(min2 - max1, min1 - max2)
                if dist < min_dist:
                    min_dist = dist
                # If we found a separating axis, the distance is exactly this (for convex shapes)
                # But we must check all axes to find the MINIMUM distance if they are not intersecting?
                # Actually, for SAT, if they are disjoint, the distance is the minimum gap over all axes.
                # If they intersect, the "distance" is 0.
                # However, if they are disjoint, any separating axis gives a valid distance, 
                # but the true minimum distance is the minimum over all axes? 
                # No, the minimum distance between two convex polygons is the minimum over all axes 
                # of the gap, provided the axis is a "critical" axis. 
                # For simplicity and robustness in optimization, we return the minimum gap found.
                # If they intersect, min_dist will remain 0 (or negative if we allowed it, but we clamp).
        
        # If no separating axis was found (min_dist is still inf or they overlap), return 0
        # Wait, the logic above: if they are disjoint, we will find at least one separating axis.
        # The distance is the minimum gap over all potential separating axes? 
        # Actually, the minimum distance between two convex polygons is the minimum of the gaps 
        # over the set of all edge normals of both polygons.
        
        if min_dist == float('inf'):
            return 0.0 # They intersect
        return min_dist

    def constraints_func(params):
        """
        Returns a list of constraint violations.
        Constraint: distance between hex i and j >= 0.
        """
        positions = np.array(params[0:2*N]).reshape(N, 2)
        angles = np.array(params[2*N:3*N])
        
        verts_list = []
        for i in range(N):
            verts_list.append(get_hex_verts(positions[i][0], positions[i][1], 1.0, angles[i]))
            
        cons = []
        for i in range(N):
            for j in range(i+1, N):
                dist = sat_distance(verts_list[i], verts_list[j])
                # We want dist >= 0. 
                # In SLSQP, 'ineq' means func >= 0.
                cons.append(dist)
        return cons

    def objective(params):
        """
        Minimize the outer side length.
        """
        positions = np.array(params[0:2*N]).reshape(N, 2)
        angles = np.array(params[2*N:3*N])
        return calc_min_outer_side(positions, angles, 0.0)

    # Strategy:
    # 1. Construct a deterministic initial configuration on a hexagonal lattice.
    #    12 hexagons fit nicely in a 4x3 or similar arrangement.
    #    A common packing for 12 hexagons is a hexagonal number arrangement or a rectangular grid.
    #    Let's try a 4x3 grid of flat-top hexagons.
    #    Flat-top hexagon width = sqrt(3), height = 2.
    #    Horizontal spacing = sqrt(3) * 1.5 (for offset rows) or sqrt(3) (for aligned)?
    #    For a simple rectangular grid of pointy-top hexagons:
    #    Width = sqrt(3), Height = 1.5.
    #    But hexagons tessellate.
    #    Let's use the standard honeycomb coordinates.
    
    sqrt3 = math.sqrt(3)
    
    # Honeycomb layout for 12 hexagons:
    # Row 0: 4 hexagons
    # Row 1: 4 hexagons
    # Row 2: 4 hexagons
    # This is a 4x3 block.
    
    coords = []
    for r in range(3):
        for c in range(4):
            x = c * sqrt3 + (r % 2) * (sqrt3 / 2)
            y = r * 1.5
            coords.append([x, y])
            
    # Center the configuration
    cx = np.mean([c[0] for c in coords])
    cy = np.mean([c[1] for c in coords])
    
    init_positions = np.array(coords) - np.array([cx, cy])
    init_angles = np.zeros(N)
    
    x0 = np.concatenate([init_positions.flatten(), init_angles])
    
    # Define constraints for SLSQP
    # SLSQP expects a list of dicts or a single function returning an array.
    # We will use a list of individual constraints for clarity, though it's slower.
    # Alternatively, we can use a single function returning an array of violations.
    
    def all_constraints(params):
        positions = np.array(params[0:2*N]).reshape(N, 2)
        angles = np.array(params[2*N:3*N])
        
        verts_list = []
        for i in range(N):
            verts_list.append(get_hex_verts(positions[i][0], positions[i][1], 1.0, angles[i]))
            
        violations = []
        for i in range(N):
            for j in range(i+1, N):
                dist = sat_distance(verts_list[i], verts_list[j])
                violations.append(dist)
        return np.array(violations)

    constraints = [{'type': 'ineq', 'fun': all_constraints}]
    
    # Bounds:
    # Positions should stay within a reasonable box, e.g., [-5, 5]
    # Angles in [0, 60]
    bounds = []
    for i in range(N):
        bounds.append((-5, 5)) # x
        bounds.append((-5, 5)) # y
        bounds.append((0, 60)) # angle

    best_result = x0
    best_score = objective(x0)
    
    # Run SLSQP
    try:
        if time.time() < deadline:
            res = minimize(
                objective, 
                x0, 
                method='SLSQP', 
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 200, 'ftol': 1e-6}
            )
            if res.fun < best_score:
                best_score = res.fun
                best_result = res.x
    except Exception as e:
        print(f"SLSQP failed: {e}")

    # If SLSQP didn't improve or failed, try a few local perturbations with Nelder-Mead
    # just in case SLSQP got stuck in a bad local minimum or constraint handling was tricky.
    
    np.random.seed(42)
    for _ in range(5):
        if time.time() > deadline - 10:
            break
            
        # Perturb the best result found so far
        x0_perturb = best_result + np.random.normal(0, 0.1, best_result.shape)
        # Ensure angles are within bounds
        x0_perturb[2*N:3*N] = np.clip(x0_perturb[2*N:3*N], 0, 60)
        
        try:
            res_nm = minimize(
                objective, 
                x0_perturb, 
                method='Nelder-Mead', 
                options={'maxiter': 100, 'xatol': 1e-4, 'fatol': 1e-4}
            )
            # Check validity of NM result
            pos_nm = np.array(res_nm.x[0:2*N]).reshape(N, 2)
            ang_nm = np.array(res_nm.x[2*N:3*N])
            if check_disjoint(pos_nm, ang_nm):
                side_nm = calc_min_outer_side(pos_nm, ang_nm, 0.0)
                if side_nm < best_score:
                    best_score = side_nm
                    best_result = res_nm.x
        except:
            pass

    # Final Check and Formatting
    if best_result is None or best_score > 1e5:
        # Absolute fallback: Honeycomb
        sqrt3 = math.sqrt(3)
        coords = [
            (-2, 0), (0, 0), (2, 0),
            (-1, sqrt3), (1, sqrt3), (3, sqrt3),
            (-2, 2*sqrt3), (0, 2*sqrt3), (2, 2*sqrt3),
            (-1, 3*sqrt3), (1, 3*sqrt3), (3, 3*sqrt3)
        ]
        cx = sum(c[0] for c in coords) / N
        cy = sum(c[1] for c in coords) / N
        
        inner = [[c[0]-cx, c[1]-cy, 0.0] for c in coords]
        pos = np.array([[c[0]-cx, c[1]-cy] for c in coords])
        ang = np.zeros(N)
        side = calc_min_outer_side(pos, ang, 0.0)
        return inner, (0.0, 0.0), side, 0.0

    # Extract results
    positions = np.array(best_result[0:2*N]).reshape(N, 2)
    angles = np.array(best_result[2*N:3*N])
    
    inner = []
    for i in range(N):
        inner.append([positions[i][0], positions[i][1], angles[i]])
    
    # Final validation
    if not check_disjoint(positions, angles):
        # Fallback to honeycomb if somehow invalid
        sqrt3 = math.sqrt(3)
        coords = [
            (-2, 0), (0, 0), (2, 0),
            (-1, sqrt3), (1, sqrt3), (3, sqrt3),
            (-2, 2*sqrt3), (0, 2*sqrt3), (2, 2*sqrt3),
            (-1, 3*sqrt3), (1, 3*sqrt3), (3, 3*sqrt3)
        ]
        cx = sum(c[0] for c in coords) / N
        cy = sum(c[1] for c in coords) / N
        inner = [[c[0]-cx, c[1]-cy, 0.0] for c in coords]
        pos = np.array([[c[0]-cx, c[1]-cy] for c in coords])
        ang = np.zeros(N)
        final_side = calc_min_outer_side(pos, ang, 0.0)
        return inner, (0.0, 0.0), final_side, 0.0

    final_side = calc_min_outer_side(positions, angles, 0.0)
    
    return inner, (0.0, 0.0), final_side, 0.0

# EVOLVE-BLOCK-END
