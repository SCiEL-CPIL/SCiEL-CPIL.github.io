import numpy as np
import time
import random
from scipy.optimize import differential_evolution, minimize

N = 12

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s limit

    # Generate 12 disjoint hexagon centers on a hexagonal lattice
    # Side length s=1, distance between touching centers d = sqrt(3)
    sqrt3 = np.sqrt(3)
    
    # Basis vectors for hex lattice centers
    # u = (sqrt(3), 0)
    # v = (sqrt(3)/2, 3/2)
    # Note: 3/2 = sqrt(3) * sqrt(3) / 2.
    
    centers = []
    # 1 center
    centers.append((0.0, 0.0))
    
    # 6 neighbors (Ring 1)
    # Angles: 0, 60, 120, 180, 240, 300
    for i in range(6):
        theta = i * np.pi / 3
        centers.append((sqrt3 * np.cos(theta), sqrt3 * np.sin(theta)))
        
    # Need 5 more from Ring 2 (distance 2*sqrt3)
    # Ring 2 has 12 positions.
    # Let's pick 5 consecutive ones to keep the shape compact.
    # Ring 2 angles are offset by 30 deg from Ring 1?
    # Lattice points at distance 2*sqrt3:
    # Vectors: 2*u, 2*v, 2*(u-v), -2*u, -2*v, -2*(u-v) are 6 points? No.
    # Let's just generate lattice points and pick the closest 5 to the centroid (0,0) excluding the first 7.
    
    # Generate a larger set of lattice points
    lattice_points = []
    for i in range(-3, 4):
        for j in range(-3, 4):
            x = i * sqrt3 + j * sqrt3 / 2
            y = j * 3 / 2
            lattice_points.append((x, y))
            
    # Sort by distance to origin
    lattice_points.sort(key=lambda p: p[0]**2 + p[1]**2)
    
    # We already have 7. We need 5 more.
    # The first 7 in sorted order are the center and 6 neighbors.
    # Let's verify.
    # Distances: 0, sqrt3 (6 times), 2*sqrt3 (12 times)...
    # So lattice_points[0] is (0,0).
    # lattice_points[1:7] are the 6 neighbors.
    # lattice_points[7:12] are the next 5 closest.
    
    selected_centers = [p for p in lattice_points[:12]]
    
    # Convert to numpy array
    inner_base = np.array(selected_centers)
    
    # Function to compute the minimum outer side length given rotations
    def compute_side_length(phi, theta_out):
        """
        phi: global rotation of the inner cluster
        theta_out: rotation of the outer hexagon
        Returns the minimum side length L of the outer hexagon.
        """
        # Rotate inner centers
        cos_phi = np.cos(phi)
        sin_phi = np.sin(phi)
        rot_matrix = np.array([[cos_phi, -sin_phi], [sin_phi, cos_phi]])
        rotated_centers = inner_base @ rot_matrix.T
        
        # The outer hexagon is centered at the origin (we can shift the whole system, 
        # but minimizing the max radial extent from the centroid is optimal for a symmetric container 
        # if we allow translation. Here we fix outer center at 0 for simplicity, 
        # but actually, the optimal outer center is the centroid of the cluster? 
        # For a regular hexagon, the "radius" (apothem) must cover the max projection.
        # The side length L is related to the apothem a by a = L * sqrt(3) / 2.
        # So we want to minimize max projection / (sqrt(3)/2).
        
        # Outer hexagon normals
        # Vertices at angles theta_out + i*pi/3.
        # Normals at angles theta_out + pi/6 + i*pi/3.
        
        min_L = float('inf')
        
        # We can optimize the outer center (cx, cy) too, but for a convex container,
        # the optimal center to minimize the max distance to vertices is the center of the minimum enclosing circle,
        # which for a hexagon constraint is slightly different.
        # However, simply centering the outer hexagon at the centroid of the inner cluster is a good heuristic.
        # Let's compute the centroid of the rotated centers.
        centroid = np.mean(rotated_centers, axis=0)
        
        # For each of the 6 faces of the outer hexagon, we need the outer hexagon to contain the cluster.
        # The distance from the outer hexagon center to the face (apothem) must be >= max projection of cluster points onto the face normal.
        # Let's assume the outer hexagon is centered at the centroid.
        # If we allow the outer center to move, we can reduce L.
        # Let's optimize the outer center (cx, cy) as well?
        # To keep it fast, let's fix outer center to centroid.
        
        cx, cy = centroid
        
        # Normals of the outer hexagon faces
        for k in range(6):
            normal_angle = theta_out + np.pi / 6 + k * np.pi / 3
            normal = np.array([np.cos(normal_angle), np.sin(normal_angle)])
            
            # Project all inner hexagon vertices? No, just centers is not enough.
            # We must check the vertices of the inner hexagons.
            # This makes it expensive.
            
            # Alternative: The inner hexagons are regular.
            # The support function of a rotated hexagon is known.
            # h(theta) = r * cos(theta - alpha) ... ?
            # For a regular hexagon of side 1, the support function h(u) in direction u is:
            # If u is aligned with a vertex, h = 1 (distance to vertex).
            # If u is aligned with a face normal, h = sqrt(3)/2 (apothem).
            # General formula: h(theta) = (sqrt(3)/2) * cos(pi/6 - (theta mod pi/3 - pi/6)) ?
            # Actually, simpler: The vertices are at distance 1 from center.
            # The support function of a regular hexagon (side 1) in direction angle alpha is:
            # h(alpha) = (1/sqrt(3)) * cos(pi/6 - (alpha mod pi/3 - pi/6)) * 2 ? No.
            
            # Let's just compute vertices for the 12 hexagons. It's 12*6 = 72 points.
            # 6 normals * 72 points = 432 dot products. Very fast.
            
            max_proj = -float('inf')
            for i in range(12):
                cx_i, cy_i = rotated_centers[i]
                # Inner hexagon rotation is fixed relative to cluster?
                # In the lattice, they are all aligned.
                # So inner hexagon angle is phi (same as cluster rotation).
                # Vertices of inner hexagon i:
                # Center: (cx_i, cy_i)
                # Angle: phi
                # Side: 1
                for m in range(6):
                    v_angle = phi + m * np.pi / 3
                    vx = cx_i + np.cos(v_angle)
                    vy = cy_i + np.sin(v_angle)
                    proj = (vx - cx) * normal[0] + (vy - cy) * normal[1]
                    if proj > max_proj:
                        max_proj = proj
            
            # The apothem of the outer hexagon must be at least max_proj
            # Apothem a = L * sqrt(3) / 2
            # So L = 2 * max_proj / sqrt(3)
            L_required = 2.0 * max_proj / sqrt3
            if L_required < min_L:
                min_L = L_required
                
        return min_L

    def objective(params):
        phi = params[0]
        theta_out = params[1]
        # We can also optimize outer center offset?
        # For now, assume outer center is centroid.
        return compute_side_length(phi, theta_out)

    # Optimization
    # Bounds: 0 to 2*pi for both
    bounds = [(0, 2 * np.pi), (0, 2 * np.pi)]
    
    best_params = [0.0, 0.0]
    best_val = objective(best_params)
    
    # Use Differential Evolution for global search
    if time.time() < deadline - 10.0:
        try:
            res = differential_evolution(
                objective, 
                bounds, 
                seed=42, 
                maxiter=100, 
                popsize=10,
                tol=1e-6,
                polish=True
            )
            if res.fun < best_val:
                best_val = res.fun
                best_params = res.x
        except Exception:
            pass

    # Local refinement
    if time.time() < deadline - 2.0:
        try:
            result = minimize(objective, best_params, method='Nelder-Mead', 
                              options={'maxiter': 200, 'xatol': 1e-8, 'fatol': 1e-10})
            if result.fun < best_val:
                best_val = result.fun
                best_params = result.x
        except Exception:
            pass

    # Construct the final output
    phi = best_params[0]
    theta_out = best_params[1]
    
    inner = []
    for i in range(12):
        cx, cy = inner_base[i]
        # Rotate center
        cos_phi = np.cos(phi)
        sin_phi = np.sin(phi)
        rx = cx * cos_phi - cy * sin_phi
        ry = cx * sin_phi + cy * cos_phi
        # Angle of the hexagon itself is also phi (since they are aligned in the lattice)
        inner.append([float(rx), float(ry), float(np.degrees(phi))])
        
    # Outer hexagon center: centroid of the rotated cluster
    rotated_centers = np.array([[inner[i][0], inner[i][1]] for i in range(12)])
    center_x = np.mean(rotated_centers[:, 0])
    center_y = np.mean(rotated_centers[:, 1])
    center = (float(center_x), float(center_y))
    
    # Recalculate side length accurately for return
    # We need to return the side length that *just* fits.
    # The objective function returned the minimum L required.
    # We should use that value.
    side = float(best_val)
    
    angle = float(np.degrees(theta_out))
    
    return inner, center, side, angle
# EVOLVE-BLOCK-END
