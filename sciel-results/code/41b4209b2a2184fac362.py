import numpy as np
import time
from scipy.optimize import minimize

def construct():
    start_time = time.time()
    DEADLINE = 140.0

    def get_hex_vertices(center, angle_deg):
        # Unit hexagon vertices
        # Angle is rotation of the hexagon
        # Vertices at 0, 60, ... relative to local x-axis?
        # Standard definition: flat top means vertices at 30, 90, ... ?
        # If angle=30, vertices are at 30, 90, ...
        # Let's stick to the definition used in the current program:
        # "Inner hexagons have angle 30 degrees (flat-top)"
        # Vertices at 30, 90, 150, 210, 270, 330.
        # This corresponds to a rotation of 30 degrees from "pointy-top" (0, 60...)
        # Or just define vertices directly.
        
        theta = np.deg2rad(angle_deg)
        # Base vertices for "pointy top" (vertex at 90 deg? No, vertex at 0 deg?)
        # Let's use the formula:
        # Vertex k is at angle (k * 60 + angle_deg)
        # If angle_deg = 30, vertices are at 30, 90, ...
        # This matches the current program's implication.
        
        angles = np.deg2rad(np.arange(6) * 60 + angle_deg)
        xs = np.cos(angles)
        ys = np.sin(angles)
        
        return np.column_stack((center[0] + xs, center[1] + ys))

    def calc_min_side(centers, inner_angles, outer_angle):
        # Collect all vertices
        all_vertices = []
        for i in range(len(centers)):
            verts = get_hex_vertices(centers[i], inner_angles[i])
            all_vertices.append(verts)
        
        all_vertices = np.vstack(all_vertices)
        
        # Outer hexagon normals
        # Outer hexagon rotated by outer_angle
        # Sides are perpendicular to normals at angles: outer_angle + 0, 60, ... ?
        # For a flat-top hexagon (angle 30), the sides are at angles 0, 60... (normals)
        # Wait.
        # If vertices are at 30, 90..., the sides are between them.
        # Side 1: 30 to 90. Midpoint 60. Normal 60.
        # Side 2: 90 to 150. Midpoint 120. Normal 120.
        # So normals are at 60, 120, 180, 240, 300, 0.
        # Generally, if vertices are at $\alpha + k 60$, normals are at $\alpha + 30 + k 60$.
        
        theta_outer = np.deg2rad(outer_angle)
        normal_angles = np.deg2rad(np.arange(6) * 60 + outer_angle + 30)
        normals = np.column_stack((np.cos(normal_angles), np.sin(normal_angles)))
        
        # Calculate max dot product
        # all_vertices: (N, 2)
        # normals: (6, 2)
        # We want max over all v, n of v . n
        
        # Project all vertices onto all normals
        # (N, 1, 2) dot (1, 6, 2) -> (N, 6)
        dots = all_vertices @ normals.T
        max_dot = np.max(dots)
        
        # Constraint: max_dot <= S * apothem = S * sqrt(3)/2
        # S >= max_dot / (sqrt(3)/2) = 2 * max_dot / sqrt(3)
        
        return 2.0 * max_dot / np.sqrt(3)

    def objective(params):
        r1, r2, phi1, phi2, phi_outer, theta_inner = params
        
        centers = []
        angles = []
        
        # Center
        centers.append(np.array([0.0, 0.0]))
        angles.append(theta_inner)
        
        # Ring 1
        for k in range(6):
            ang = np.deg2rad(k * 60 + phi1)
            x = r1 * np.cos(ang)
            y = r1 * np.sin(ang)
            centers.append(np.array([x, y]))
            angles.append(theta_inner)
            
        # Ring 2
        for k in range(4):
            ang = np.deg2rad(30 + k * 60 + phi2)
            x = r2 * np.cos(ang)
            y = r2 * np.sin(ang)
            centers.append(np.array([x, y]))
            angles.append(theta_inner)
            
        return calc_min_side(centers, angles, phi_outer)

    # Initial guess
    x0 = [np.sqrt(3), 3.0, 0.0, 0.0, 30.0, 30.0]
    
    # Constraints?
    # We should ensure no overlap.
    # Nelder-Mead doesn't support constraints well.
    # We can use a penalty method or just rely on the geometry that if they overlap, 
    # the "fitting" hexagon might still be small, but it's invalid.
    # However, if they overlap, the problem is invalid.
    # We need to enforce non-overlap.
    
    # Let's add a penalty for overlap.
    # Distance between centers must be >= sqrt(3) if they are touching?
    # No, if rotated, the min distance can be smaller?
    # For two identical regular hexagons, the min distance between centers for non-overlap
    # is achieved when a vertex of one touches a side of the other?
    # Actually, for regular hexagons, the "Minkowski sum" is a larger hexagon.
    # The minimum distance between centers such that they don't overlap is the inradius of the Minkowski sum?
    # No, it's the distance such that the boundaries touch.
    # For two unit hexagons, the min center distance is $\sqrt{3}$ if aligned.
    # If rotated, it can be smaller?
    # Let's assume we keep them aligned (theta_inner same for all).
    # Then min distance is $\sqrt{3}$.
    
    def check_overlap(centers, r1, r2, phi1, phi2):
        # Check all pairs
        # This is expensive in objective, but N calls are small.
        # 11 hexes -> 55 pairs.
        # We can approximate or just use distance > sqrt(3) - eps.
        # This is sufficient for aligned hexagons.
        
        min_dist = np.inf
        for i in range(len(centers)):
            for j in range(i+1, len(centers)):
                d = np.linalg.norm(centers[i] - centers[j])
                if d < min_dist:
                    min_dist = d
        return min_dist

    def penalized_objective(params):
        val = objective(params)
        r1, r2, phi1, phi2, phi_outer, theta_inner = params
        
        centers = []
        centers.append(np.array([0.0, 0.0]))
        for k in range(6):
            ang = np.deg2rad(k * 60 + phi1)
            centers.append(np.array([r1 * np.cos(ang), r1 * np.sin(ang)]))
        for k in range(4):
            ang = np.deg2rad(30 + k * 60 + phi2)
            centers.append(np.array([r2 * np.cos(ang), r2 * np.sin(ang)]))
            
        min_d = check_overlap(centers, r1, r2, phi1, phi2)
        
        if min_d < np.sqrt(3) - 1e-5:
            # Penalty proportional to violation
            penalty = (np.sqrt(3) - min_d) * 100.0
            return val + penalty
            
        return val

    # Run optimization
    # Use Nelder-Mead
    res = minimize(penalized_objective, x0, method='Nelder-Mead', 
                   options={'maxiter': 5000, 'xatol': 1e-4, 'fatol': 1e-4})
    
    best_params = res.x
    best_val = res.fun
    
    # Refine with tighter tolerance?
    # Or just use the result.
    
    # Ensure validity of the best solution found
    # If the penalty was triggered, we might have an invalid solution.
    # Let's check the final solution.
    
    r1, r2, phi1, phi2, phi_outer, theta_inner = best_params
    
    centers = []
    angles = []
    centers.append(np.array([0.0, 0.0]))
    angles.append(theta_inner)
    for k in range(6):
        ang = np.deg2rad(k * 60 + phi1)
        centers.append(np.array([r1 * np.cos(ang), r1 * np.sin(ang)]))
        angles.append(theta_inner)
    for k in range(4):
        ang = np.deg2rad(30 + k * 60 + phi2)
        centers.append(np.array([r2 * np.cos(ang), r2 * np.sin(ang)]))
        angles.append(theta_inner)
        
    min_d = check_overlap(centers, r1, r2, phi1, phi2)
    
    if min_d < np.sqrt(3) - 1e-4:
        # Fall back to original solution if invalid
        # Original solution
        centers = []
        angles = []
        centers.append((0.0, 0.0))
        for k in range(6):
            ang = np.deg2rad(k * 60)
            centers.append((np.sqrt(3)*np.cos(ang), np.sqrt(3)*np.sin(ang)))
        for k in range(4):
            ang = np.deg2rad(30 + k * 60)
            centers.append((3.0*np.cos(ang), 3.0*np.sin(ang)))
        angles = [30.0]*11
        phi_outer = 30.0
        side = calc_min_side([np.array(c) for c in centers], angles, phi_outer)
    else:
        side = calc_min_side(centers, angles, phi_outer)
        
    inner = []
    for i in range(len(centers)):
        inner.append([centers[i][0], centers[i][1], angles[i]])
        
    return inner, (0.0, 0.0), side, phi_outer

