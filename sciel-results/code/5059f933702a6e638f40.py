import numpy as np
import time
import itertools
from scipy.optimize import minimize

N = 12

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s limit

    def hexagon_vertices(cx, cy, side, angle_deg):
        angle_rad = np.radians(angle_deg)
        angles = angle_rad + np.radians(np.arange(0, 360, 60))
        return np.column_stack([
            cx + side * np.cos(angles),
            cy + side * np.sin(angles)
        ])

    def required_outer_side(verts_flat):
        """
        Compute the exact side length of the smallest regular hexagon (angle 0, centered at origin)
        that contains all points in verts_flat.
        """
        x = verts_flat[:, 0]
        y = verts_flat[:, 1]
        sqrt3 = np.sqrt(3.0)
        
        # Linear forms corresponding to the 6 normals of a regular hexagon
        L1 = x + y / sqrt3
        L2 = y * 2.0 / sqrt3
        L3 = -x + y / sqrt3
        L4 = -x - y / sqrt3
        L5 = -y * 2.0 / sqrt3
        L6 = x - y / sqrt3
        
        all_L = np.stack([L1, L2, L3, L4, L5, L6], axis=1)
        return np.max(all_L)

    def are_disjoint(v1, v2):
        """
        Check if two convex polygons (hexagons) are disjoint using Separating Axis Theorem.
        """
        for i in range(6):
            edge = v1[(i + 1) % 6] - v1[i]
            normal = np.array([-edge[1], edge[0]])
            norm = np.linalg.norm(normal)
            if norm < 1e-10:
                continue
            normal /= norm
            
            proj1 = v1.dot(normal)
            proj2 = v2.dot(normal)
            
            if np.max(proj1) < np.min(proj2) or np.max(proj2) < np.min(proj1):
                return True
        
        for i in range(6):
            edge = v2[(i + 1) % 6] - v2[i]
            normal = np.array([-edge[1], edge[0]])
            norm = np.linalg.norm(normal)
            if norm < 1e-10:
                continue
            normal /= norm
            
            proj1 = v1.dot(normal)
            proj2 = v2.dot(normal)
            
            if np.max(proj1) < np.min(proj2) or np.max(proj2) < np.min(proj1):
                return True
                
        return False

    def is_valid_configuration(positions, angles):
        """
        Check if all hexagons are disjoint.
        positions: (N, 2)
        angles: (N,)
        """
        verts = []
        for i in range(N):
            verts.append(hexagon_vertices(positions[i, 0], positions[i, 1], 1.0, angles[i]))
        
        for i in range(N):
            for j in range(i + 1, N):
                if not are_disjoint(verts[i], verts[j]):
                    return False
        return True

    # Define a dense honeycomb cluster of 12 unit hexagons.
    # We use axial coordinates (q, r).
    # Distance between centers in a honeycomb of side 1 is sqrt(3).
    # Basis vectors:
    # u = (sqrt(3), 0)
    # v = (sqrt(3)/2, 1.5)
    # Let's pick a compact set of 12.
    # Center (0,0). 6 neighbors. 5 more from the next ring.
    # This forms a "hexagon of hexagons" shape (though 12 doesn't form a perfect larger hexagon, 
    # it forms a compact blob).
    
    coords = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (1, 1), (0, 2), (-1, 2), (-2, 1)
    ]
    
    base_positions = np.zeros((N, 2))
    for i, (q, r) in enumerate(coords):
        base_positions[i, 0] = np.sqrt(3) * (q + r / 2.0)
        base_positions[i, 1] = 1.5 * r

    # Pre-verify the base configuration is valid (disjoint)
    # Since it's a subset of the infinite tiling, they are disjoint.
    # But let's double check with SAT just in case of floating point issues or if I picked overlapping coords.
    if not is_valid_configuration(base_positions, np.zeros(N)):
        # Fallback to a simple grid if the cluster is invalid (should not happen)
        base_positions = np.array([[i * 2.0, (i % 3) * 2.0] for i in range(N)])
        if not is_valid_configuration(base_positions, np.zeros(N)):
             # Last resort: spread out
             base_positions = np.array([[i * 4.0, 0.0] for i in range(N)])

    def evaluate_cluster(theta_deg, cx, cy):
        """
        Evaluate the side length of the outer hexagon for a cluster rotated by theta
        and translated to (cx, cy).
        Returns (side_length, is_valid).
        """
        theta_rad = np.radians(theta_deg)
        c, s = np.cos(theta_rad), np.sin(theta_rad)
        R = np.array([[c, -s], [s, c]])
        
        # Rotate base positions
        rotated = (base_positions @ R.T) + np.array([cx, cy])
        
        # Calculate vertices for all hexagons
        # Since all hexagons have the same rotation angle theta, 
        # we can compute vertices efficiently.
        verts_list = []
        for i in range(N):
            verts_list.append(hexagon_vertices(rotated[i, 0], rotated[i, 1], 1.0, theta_deg))
        
        verts_flat = np.vstack(verts_list)
        side = required_outer_side(verts_flat)
        
        # Check validity
        # We can skip SAT if we trust the base configuration and rigid body motion preserves disjointness.
        # Rigid body motion (rotation + translation) preserves relative positions, so if base is disjoint, rotated is disjoint.
        # So is_valid is always True for this strategy, assuming base is valid.
        
        return side, True

    def objective(params):
        """
        Objective for optimizer. params = [theta_deg, cx, cy]
        """
        theta_deg, cx, cy = params
        side, valid = evaluate_cluster(theta_deg, cx, cy)
        if not valid:
            return 1e10
        return side

    best_side = float('inf')
    best_params = None
    best_positions = None
    best_angles = None

    # Strategy: Multi-start optimization
    # We search over theta in [0, 30) degrees due to 6-fold symmetry of the outer hexagon constraint?
    # Actually, the outer hexagon is fixed at angle 0. The inner cluster can rotate.
    # The objective function is periodic with period 60 degrees?
    # If we rotate the cluster by 60 degrees, the set of points is the same set of hexagons, just permuted?
    # No, the positions are specific. But the bounding box of a hexagon is invariant under 60 deg rotation?
    # Yes, the support function of a regular hexagon is periodic with 60 deg.
    # So we only need to search theta in [0, 60).
    
    # Grid search over theta
    # Let's do a coarse grid first, then refine.
    
    coarse_steps = 30 # 2 degree steps
    theta_vals = np.linspace(0, 60, coarse_steps, endpoint=False)
    
    # For each theta, we need to find the optimal translation (cx, cy).
    # The optimal translation likely centers the bounding box of the rotated cluster at the origin.
    # We can compute the centroid of the rotated cluster and set (cx, cy) to -centroid initially,
    # then use Nelder-Mead to fine-tune.
    
    for theta in theta_vals:
        if time.time() > deadline:
            break
            
        # Initial guess for translation: Center of mass of the rotated cluster at origin
        theta_rad = np.radians(theta)
        c, s = np.cos(theta_rad), np.sin(theta_rad)
        R = np.array([[c, -s], [s, c]])
        rotated_base = base_positions @ R.T
        centroid = np.mean(rotated_base, axis=0)
        init_cx, init_cy = -centroid[0], -centroid[1]
        
        init_params = [theta, init_cx, init_cy]
        
        # Refine with Nelder-Mead
        try:
            res = minimize(
                objective, 
                init_params, 
                method='Nelder-Mead', 
                options={'maxiter': 200, 'xatol': 1e-4, 'fatol': 1e-6, 'adaptive': True}
            )
            
            opt_theta, opt_cx, opt_cy = res.x
            opt_side, opt_valid = evaluate_cluster(opt_theta, opt_cx, opt_cy)
            
            if opt_valid and opt_side < best_side:
                best_side = opt_side
                best_params = [opt_theta, opt_cx, opt_cy]
                
                # Reconstruct positions for return
                theta_rad = np.radians(opt_theta)
                c, s = np.cos(theta_rad), np.sin(theta_rad)
                R = np.array([[c, -s], [s, c]])
                final_rotated = (base_positions @ R.T) + np.array([opt_cx, opt_cy])
                best_positions = final_rotated
                best_angles = np.full(N, opt_theta)
                
        except Exception:
            pass

    # If the grid search didn't find anything (shouldn't happen), fallback to base config
    if best_positions is None:
        best_positions = base_positions
        best_angles = np.zeros(N)
        verts_flat = np.vstack([hexagon_vertices(base_positions[i,0], base_positions[i,1], 1.0, 0.0) for i in range(N)])
        best_side = required_outer_side(verts_flat)

    # Final check and construction of return values
    # The interface requires: inner (x, y, angle), center (x, y), side, angle
    # Center is (0,0) for the outer hexagon.
    # Angle is 0 for the outer hexagon.
    
    inner = []
    for i in range(N):
        inner.append([float(best_positions[i, 0]), float(best_positions[i, 1]), float(best_angles[i])])
        
    return inner, (0.0, 0.0), float(best_side), 0.0

# EVOLVE-BLOCK-END
