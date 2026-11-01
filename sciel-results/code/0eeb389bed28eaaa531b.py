import numpy as np
import time
from scipy.optimize import minimize
import itertools

N = 12

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit

    def hexagon_vertices(cx, cy, side, angle_deg):
        angle_rad = np.radians(angle_deg)
        angles = angle_rad + np.radians(np.arange(0, 360, 60))
        return np.column_stack([
            cx + side * np.cos(angles),
            cy + side * np.sin(angles)
        ])

    def get_outer_side_length(vertices_array, center, outer_angle_deg):
        """
        Calculate the minimum side length of a regular hexagon with given center and angle
        that contains all points in vertices_array.
        
        For a regular hexagon with side S, center C, and angle A:
        The hexagon is the intersection of 3 strips defined by the normals to the sides.
        The normals are at angles A, A+60, A+120.
        The distance from center to each side (apothem) is S * sqrt(3) / 2.
        
        For a point P, the distance from C along the normal direction n is d = dot(P-C, n).
        The condition for P to be inside is |d| <= S * sqrt(3) / 2.
        So S >= 2 * |dot(P-C, n)| / sqrt(3) for all normals n and all points P.
        """
        cx, cy = center
        outer_angle_rad = np.radians(outer_angle_deg)
        
        # Normals to the sides of the outer hexagon
        # Sides are perpendicular to the apothem directions.
        # Apothem directions are at angles: outer_angle, outer_angle+60, outer_angle+120
        # Wait, standard hexagon vertices are at 0, 60, ...
        # If vertices are at angle A, the sides are between them.
        # The normal to the side between vertex A and A+60 is at A+30.
        # Let's verify.
        # Vertices: (S, 0), (S/2, S*sqrt(3)/2).
        # Midpoint of side: (3S/4, S*sqrt(3)/4).
        # Vector from center to midpoint: (3S/4, S*sqrt(3)/4).
        # Angle: arctan( (sqrt(3)/4) / (3/4) ) = arctan(1/sqrt(3)) = 30 degrees.
        # So normals are at 30, 90, 150 degrees relative to the vertex angle 0.
        
        normal_angles = np.radians([outer_angle_deg + 30, outer_angle_deg + 90, outer_angle_deg + 150])
        normals = np.column_stack([np.cos(normal_angles), np.sin(normal_angles)])
        
        # Translate points to center
        pts = vertices_array - np.array(center)
        
        max_s = 0.0
        for n in normals:
            # Project all points onto normal
            projections = pts.dot(n)
            # We need S * sqrt(3)/2 >= max(|proj|)
            max_abs_proj = np.max(np.abs(projections))
            s_req = 2.0 * max_abs_proj / np.sqrt(3.0)
            if s_req > max_s:
                max_s = s_req
                
        return max_s

    def check_validity(positions, angles):
        """
        positions: (N, 2) array of (x, y)
        angles: (N,) array of angles in degrees
        Returns True if all are pairwise disjoint.
        """
        vertices_list = []
        for i in range(N):
            x, y = positions[i]
            ang = angles[i]
            verts = hexagon_vertices(x, y, 1.0, ang)
            vertices_list.append(verts)
        
        for i in range(N):
            for j in range(i + 1, N):
                if not polygons_disjoint_sat(vertices_list[i], vertices_list[j]):
                    return False
        return True

    def polygons_disjoint_sat(p1, p2, eps=1e-5):
        """
        Returns True if two convex polygons are disjoint (allowing touching with eps tolerance).
        """
        # Precompute edges and normals for p1 and p2
        # For efficiency in repeated calls, we could cache, but N=12 is small.
        
        def get_axes(p):
            axes = []
            for i in range(len(p)):
                j = (i + 1) % len(p)
                edge = p[j] - p[i]
                normal = np.array([-edge[1], edge[0]])
                norm = np.linalg.norm(normal)
                if norm > 1e-10:
                    normal /= norm
                    axes.append(normal)
            return axes

        axes1 = get_axes(p1)
        axes2 = get_axes(p2)
        
        all_axes = axes1 + axes2
        
        for n in all_axes:
            proj1 = p1.dot(n)
            proj2 = p2.dot(n)
            
            min1, max1 = np.min(proj1), np.max(proj1)
            min2, max2 = np.min(proj2), np.max(proj2)
            
            # Check for gap
            if max1 < min2 - eps or max2 < min1 - eps:
                return True
        
        return False

    # --- Initialization: Compact Honeycomb Cluster ---
    # A regular hexagonal packing of unit hexagons.
    # Axial coordinates (q, r)
    # x = sqrt(3) * (q + r/2)
    # y = 1.5 * r
    
    # We need 12 hexagons.
    # Center (0,0)
    # Ring 1: 6 hexagons
    # Ring 2: 12 hexagons. We only need 5 more to reach 12 total.
    # Let's pick the 5 closest ones from Ring 2 to maintain compactness.
    
    coords = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        # Ring 2 candidates, sorted by distance to origin (all are same dist, so pick a contiguous arc)
        (2, 0), (1, 1), (0, 2), (-1, 2), (-2, 1)
    ]
    
    init_positions = []
    init_angles = []
    for q, r in coords:
        x = np.sqrt(3) * (q + r / 2.0)
        y = 1.5 * r
        init_positions.append([x, y])
        init_angles.append(0.0)
        
    init_positions = np.array(init_positions)
    init_angles = np.array(init_angles)
    
    # Verify initial validity
    if not check_validity(init_positions, init_angles):
        # Fallback: Spread out slightly if there's an issue (shouldn't happen for hex lattice)
        # Just in case, perturb
        init_positions += np.random.normal(0, 0.1, (N, 2))
        if not check_validity(init_positions, init_angles):
             # Emergency fallback: Linear array
             init_positions = np.array([[i*2.0, 0.0] for i in range(N)])
             init_angles = np.zeros(N)

    # --- Optimization Strategy ---
    # 1. Optimize the outer container (center, angle) for the fixed inner configuration.
    #    This is a 3D optimization problem.
    # 2. Optionally, perform local search on inner positions to see if we can shrink the bounding box.
    #    However, the honeycomb is likely locally optimal for the shape.
    #    Let's focus on optimizing the outer hexagon fit first.
    
    # Objective for outer hexagon
    def objective_outer(params):
        if time.time() > deadline:
            return 1e10
        cx, cy, angle = params
        # Compute all vertices of inner hexagons
        verts = []
        for i in range(N):
            x, y = init_positions[i]
            ang = init_angles[i]
            v = hexagon_vertices(x, y, 1.0, ang)
            verts.append(v)
        all_verts = np.vstack(verts)
        
        return get_outer_side_length(all_verts, (cx, cy), angle)
    
    # Initial guess for outer: (0, 0, 0)
    x0_outer = [0.0, 0.0, 0.0]
    
    try:
        res_outer = minimize(objective_outer, x0_outer, method='Nelder-Mead', 
                             options={'maxiter': 200, 'xatol': 1e-6, 'fatol': 1e-6})
        best_center, best_angle = res_outer.x
        best_side = res_outer.fun
    except Exception:
        best_center, best_angle = x0_outer
        best_side = objective_outer(x0_outer)

    # --- Refinement: Local Search on Inner Positions ---
    # We can try to nudge inner hexagons to reduce the max side length.
    # This is a 24D problem. We can use a simplified local search.
    # Since the honeycomb is tight, moving them might cause overlaps.
    # Let's define a combined objective: Minimize Outer Side Length, subject to Disjointness.
    # If disjointness fails, return a large penalty.
    
    def objective_full(params):
        if time.time() > deadline:
            return 1e10
        positions = params[:2*N].reshape((N, 2))
        angles = params[2*N:]
        
        if not check_validity(positions, angles):
            return 1e10
            
        verts = []
        for i in range(N):
            x, y = positions[i]
            ang = angles[i]
            v = hexagon_vertices(x, y, 1.0, ang)
            verts.append(v)
        all_verts = np.vstack(verts)
        
        return get_outer_side_length(all_verts, best_center, best_angle)

    # Start from the valid honeycomb configuration
    x0_full = np.concatenate([init_positions.flatten(), init_angles])
    
    # We will run a limited Nelder-Mead on the full configuration.
    # To ensure we don't get stuck in invalid regions, we start from a valid point.
    # Nelder-Mead doesn't handle constraints, so it might jump into invalid regions.
    # We rely on the penalty function.
    
    try:
        res_full = minimize(objective_full, x0_full, method='Nelder-Mead',
                            options={'maxiter': 300, 'xatol': 1e-5, 'fatol': 1e-5})
        
        if res_full.fun < best_side:
            best_params = res_full.x
            best_side = res_full.fun
        else:
            best_params = x0_full
            
    except Exception:
        best_params = x0_full

    # Extract final positions
    final_positions = best_params[:2*N].reshape((N, 2))
    final_angles = best_params[2*N:]
    
    # Final check for validity and compute exact side length
    if not check_validity(final_positions, final_angles):
        # Revert to initial valid configuration if optimization broke it
        final_positions = init_positions
        final_angles = init_angles
        verts = []
        for i in range(N):
            x, y = final_positions[i]
            ang = final_angles[i]
            v = hexagon_vertices(x, y, 1.0, ang)
            verts.append(v)
        all_verts = np.vstack(verts)
        best_side = get_outer_side_length(all_verts, best_center, best_angle)
    else:
        # Recalculate best side with final positions to be precise
        verts = []
        for i in range(N):
            x, y = final_positions[i]
            ang = final_angles[i]
            v = hexagon_vertices(x, y, 1.0, ang)
            verts.append(v)
        all_verts = np.vstack(verts)
        # Re-optimize outer for these specific positions just in case
        def obj_final_outer(params):
            cx, cy, angle = params
            return get_outer_side_length(all_verts, (cx, cy), angle)
        
        try:
            res_final_outer = minimize(obj_final_outer, best_center, method='Nelder-Mead',
                                       options={'maxiter': 50, 'xatol': 1e-7, 'fatol': 1e-7})
            best_center = res_final_outer.x[:2]
            best_angle = res_final_outer.x[2]
            best_side = res_final_outer.fun
        except Exception:
            pass # Keep previous best

    inner = []
    for i in range(N):
        x = final_positions[i, 0]
        y = final_positions[i, 1]
        ang = final_angles[i]
        inner.append([float(x), float(y), float(ang)])
        
    return inner, (float(best_center[0]), float(best_center[1])), float(best_side), float(best_angle)

# EVOLVE-BLOCK-END
