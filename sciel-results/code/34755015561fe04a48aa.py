import numpy as np
import time
import itertools
from scipy.optimize import minimize, least_squares

N = 12
SQRT3 = np.sqrt(3.0)

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
        
        # Linear forms corresponding to the 6 normals of a regular hexagon
        L1 = x + y / SQRT3
        L2 = y * 2.0 / SQRT3
        L3 = -x + y / SQRT3
        L4 = -x - y / SQRT3
        L5 = -y * 2.0 / SQRT3
        L6 = x - y / SQRT3
        
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

    def get_cluster_coords(shape_id):
        """
        Returns axial coordinates (q, r) for specific compact cluster shapes of 12 hexagons.
        These shapes represent different topological contact graphs.
        """
        # Shape 0: The "blob" from the current program (Center + 6 + 5)
        # This is a valid packing.
        if shape_id == 0:
            return [
                (0, 0),
                (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
                (2, 0), (1, 1), (0, 2), (-1, 2), (-2, 1)
            ]
        
        # Shape 1: A "bar" or "snake" shape. 
        # (0,0), (1,0), (2,0), (3,0), (4,0), (5,0)
        # (0,1), (1,1), (2,1), (3,1), (4,1), (5,1)
        # This is a 2x6 rectangle.
        # Is it valid? Yes, it's a subset of the tiling.
        if shape_id == 1:
            coords = []
            for q in range(6):
                coords.append((q, 0))
                coords.append((q, 1))
            return coords

        # Shape 2: A "cross" or "plus" shape with extensions.
        # Center (0,0).
        # Ring 1: (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1) [6]
        # Ring 2: Add (2,0), (2,-1) [2] ... wait, need 5 more.
        # Let's try a "T" shape or similar.
        # (0,0)
        # (1,0), (-1,0), (0,1), (0,-1) [4] -> Total 5
        # (2,0), (-2,0), (0,2), (0,-2) [4] -> Total 9
        # (1,1), (1,-1) [2] -> Total 11
        # (-1,1) [1] -> Total 12
        if shape_id == 2:
            return [
                (0, 0),
                (1, 0), (-1, 0), (0, 1), (0, -1),
                (2, 0), (-2, 0), (0, 2), (0, -2),
                (1, 1), (1, -1),
                (-1, 1)
            ]
            
        # Shape 3: "Triangle" cluster.
        # Row 0: (0,0), (1,0), (2,0), (3,0) [4]
        # Row 1: (0,1), (1,1), (2,1) [3]
        # Row 2: (0,2), (1,2) [2]
        # Row 3: (0,3) [1]
        # Total: 4+3+2+1 = 10. Need 2 more.
        # Add (3,1) and (3,2)?
        # (3,1) is adjacent to (2,1) and (3,0)? (3,1) - (2,1) is valid. (3,1)-(3,0) is not axial neighbor?
        # Axial neighbors of (3,0): (4,0), (2,0), (3,1), (3,-1), (2,1), (4,-1). Yes (3,1) is neighbor.
        # (3,2) neighbor of (3,1)? Yes.
        if shape_id == 3:
            return [
                (0, 0), (1, 0), (2, 0), (3, 0),
                (0, 1), (1, 1), (2, 1), (3, 1),
                (0, 2), (1, 2), (3, 2),
                (0, 3)
            ]

        # Fallback: Shape 0
        return get_cluster_coords(0)

    def axial_to_xy(q, r):
        x = SQRT3 * (q + r / 2.0)
        y = 1.5 * r
        return np.array([x, y])

    def solve_topology(coords, time_limit=5.0):
        """
        For a given set of axial coordinates (topology), solve for the optimal
        rotation and translation to minimize the outer hexagon side length.
        We assume the internal angles are fixed at 0 (standard tiling orientation) 
        because rotating individual hexagons in a tight cluster usually increases 
        the bounding box or causes overlap unless the cluster is very loose.
        However, the problem allows rotation. 
        For a "rigid body" cluster formed by a tiling subset, the internal 
        relative angles are 0. 
        If we rotate the whole cluster, all angles are theta.
        If we allow individual rotations, the "topology" (contact graph) might break 
        or change. 
        The prompt asks to solve for exact geometric realization of each topology.
        For a tiling subset, the realization is unique up to rigid motion (if we 
        assume edge-to-edge contact). 
        If we allow "sliding" (non-contact), it's no longer a fixed topology in 
        the rigid sense. 
        So we stick to rigid clusters formed by tilings.
        """
        
        base_positions = np.array([axial_to_xy(q, r) for q, r in coords])
        
        # Center the base positions
        centroid = np.mean(base_positions, axis=0)
        base_positions -= centroid
        
        best_side = float('inf')
        best_params = None
        best_positions = None
        best_angles = None
        
        # Search over global rotation theta
        # Due to symmetry of the outer hexagon (60 deg) and inner cluster (likely lower symmetry),
        # we search [0, 60).
        
        # Coarse grid
        coarse_steps = 20 # 3 deg steps
        theta_vals = np.linspace(0, 60, coarse_steps, endpoint=False)
        
        for theta in theta_vals:
            if time.time() > start_time + time_limit:
                break
                
            theta_rad = np.radians(theta)
            c, s = np.cos(theta_rad), np.sin(theta_rad)
            R = np.array([[c, -s], [s, c]])
            
            rotated = base_positions @ R.T
            
            # The optimal translation centers the "bounding box" of the rotated cluster.
            # For a hexagon constraint, the optimal center is roughly the centroid, 
            # but we can refine it.
            # Let's just use centroid for now, as the shape is convex-ish.
            # Actually, the "center" that minimizes the max support function is the 
            # center of the minimal enclosing hexagon. 
            # For a rigid cluster, translating it doesn't change the relative positions, 
            # so the shape of the union is fixed. The optimal placement in the outer 
            # hexagon is to center the union's "hexagonal bounding box".
            # The required_outer_side function assumes the outer hexagon is centered at (0,0).
            # So we need to find (cx, cy) such that max(L_i(rotated + c)) is minimized.
            # This is equivalent to finding the center of the minimal enclosing hexagon of the set of points.
            
            # Let's use a simple optimization for (cx, cy) for each theta.
            # Or, since the cluster is small, we can just compute the support functions 
            # and set cx, cy to balance the min/max of each L_i.
            
            x = rotated[:, 0]
            y = rotated[:, 1]
            
            # We want to choose cx, cy to minimize max(L_i(x+cx, y+cy)).
            # L1 = x+cx + (y+cy)/S3
            # L2 = 2(y+cy)/S3
            # ...
            # Let's just use Nelder-Mead for (cx, cy) with a tight budget.
            
            def obj_trans(params):
                cx, cy = params
                pts = rotated + np.array([cx, cy])
                # Compute vertices for all hexagons? 
                # No, required_outer_side takes verts_flat.
                # We can approximate the union of hexagons by their vertices? 
                # No, the union of convex sets is not convex. 
                # But the outer hexagon must contain ALL hexagons. 
                # So it must contain all vertices. 
                # Is that sufficient? 
                # A regular hexagon is convex. If it contains all vertices of a convex polygon (the inner hexagon), 
                # does it contain the inner hexagon? Yes.
                # So we just need to contain the vertices of all 12 inner hexagons.
                
                verts_list = []
                for i in range(N):
                    verts_list.append(hexagon_vertices(pts[i, 0], pts[i, 1], 1.0, theta))
                verts_flat = np.vstack(verts_list)
                return required_outer_side(verts_flat)

            # Initial guess: 0,0
            init_params = [0.0, 0.0]
            
            try:
                res = minimize(
                    obj_trans, 
                    init_params, 
                    method='Nelder-Mead', 
                    options={'maxiter': 50, 'xatol': 1e-4, 'fatol': 1e-5, 'adaptive': True}
                )
                
                opt_cx, opt_cy = res.x
                opt_side = obj_trans([opt_cx, opt_cy])
                
                if opt_side < best_side:
                    best_side = opt_side
                    best_params = [theta, opt_cx, opt_cy]
                    
                    final_rotated = rotated + np.array([opt_cx, opt_cy])
                    best_positions = final_rotated
                    best_angles = np.full(N, theta)
                    
            except Exception:
                pass

        return best_side, best_positions, best_angles

    best_global_side = float('inf')
    best_global_positions = None
    best_global_angles = None

    # Enumerate topologies (shapes)
    # We will try a few distinct shapes.
    shape_ids = [0, 1, 2, 3]
    
    # Allocate time per shape
    time_per_shape = (deadline - time.time()) / len(shape_ids)
    if time_per_shape < 2.0:
        time_per_shape = 2.0 # Minimum time to try
    
    for sid in shape_ids:
        if time.time() > deadline:
            break
            
        coords = get_cluster_coords(sid)
        
        # Verify validity of the base tiling (should be always true for these)
        base_pos = np.array([axial_to_xy(q, r) for q, r in coords])
        if not is_valid_configuration(base_pos, np.zeros(N)):
            continue
            
        side, pos, ang = solve_topology(coords, time_limit=time_per_shape)
        
        if pos is not None and side < best_global_side:
            best_global_side = side
            best_global_positions = pos
            best_global_angles = ang

    # Fallback if no shape worked
    if best_global_positions is None:
        coords = get_cluster_coords(0)
        best_global_positions = np.array([axial_to_xy(q, r) for q, r in coords])
        centroid = np.mean(best_global_positions, axis=0)
        best_global_positions -= centroid
        best_global_angles = np.zeros(N)
        verts_flat = np.vstack([hexagon_vertices(best_global_positions[i,0], best_global_positions[i,1], 1.0, 0.0) for i in range(N)])
        best_global_side = required_outer_side(verts_flat)

    # Final check and construction of return values
    inner = []
    for i in range(N):
        inner.append([float(best_global_positions[i, 0]), float(best_global_positions[i, 1]), float(best_global_angles[i])])
        
    return inner, (0.0, 0.0), float(best_global_side), 0.0

# EVOLVE-BLOCK-END
