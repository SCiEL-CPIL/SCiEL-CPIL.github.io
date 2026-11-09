# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def hexagon_packing_11():
    """
    Constructs a packing of 11 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Returns
        inner_hex_data: np.ndarray of shape (11,3), where each row is of the form (x, y, angle_degrees)
        outer_hex_data: np.ndarray of shape (3,) of form (x,y,angle_degree)
        outer_hex_side_length: float
    """
    start_time = time.time()
    deadline = start_time + 580  # safety margin
    
    # Hexagon vertices function
    def hex_vertices(cx, cy, s, a_deg):
        a = np.radians(a_deg)
        angles = a + np.radians(60 * np.arange(6))
        return np.column_stack([cx + s * np.cos(angles), cy + s * np.sin(angles)])
    
    # Check if two hexagons are separated (no intersection)
    def hexagons_separated(v1, v2):
        """Check if two hexagons (as vertex arrays) are separated by > 1e-6 on some edge normal."""
        # Get edge normals for hexagon 1
        for i in range(6):
            e1 = v1[(i+1) % 6] - v1[i]
            n1 = np.array([-e1[1], e1[0]])
            n1 = n1 / (np.linalg.norm(n1) + 1e-15)
            # Project both hexagons onto this normal
            proj1 = v1 @ n1
            proj2 = v2 @ n1
            # Check if intervals are apart
            if proj1.max() < proj2.min() - 1e-6 or proj2.max() < proj1.min() - 1e-6:
                return True
        # Get edge normals for hexagon 2
        for i in range(6):
            e2 = v2[(i+1) % 6] - v2[i]
            n2 = np.array([-e2[1], e2[0]])
            n2 = n2 / (np.linalg.norm(n2) + 1e-15)
            proj1 = v1 @ n2
            proj2 = v2 @ n2
            if proj1.max() < proj2.min() - 1e-6 or proj2.max() < proj1.min() - 1e-6:
                return True
        return False
    
    # Compute minimum side length of enclosing hexagon
    def enclosing_hex_side(inner_positions, inner_angles, outer_center, outer_angle):
        """Find minimum side length of regular hexagon (center, angle) enclosing all inner hexagons."""
        ocx, ocy = outer_center
        oa = np.radians(outer_angle)
        
        # All vertices of all inner hexagons
        all_verts = []
        for i in range(len(inner_positions)):
            cx, cy = inner_positions[i]
            a_deg = inner_angles[i]
            v = hex_vertices(cx, cy, 1.0, a_deg)
            all_verts.append(v)
        all_verts = np.vstack(all_verts)
        
        # Shift by outer center
        all_verts_shifted = all_verts - np.array([ocx, ocy])
        
        min_s = 0.0
        for i in range(6):
            a_i = oa + np.radians(60 * i)
            a_ip1 = oa + np.radians(60 * (i + 1))
            e_unit = np.array([np.cos(a_ip1) - np.cos(a_i), np.sin(a_ip1) - np.sin(a_i)])
            # cross(e_unit, p_j) for all j
            crosses = all_verts_shifted[:, 0] * e_unit[1] - all_verts_shifted[:, 1] * e_unit[0]
            min_cross = np.min(crosses)
            required_s = -2 * min_cross / np.sqrt(3)
            if required_s > min_s:
                min_s = required_s
        
        return max(min_s, 0.0)
    
    # Objective function: minimize outer hexagon side length
    def objective(params):
        if time.time() > deadline:
            return 1e10
        
        # Parse parameters: 11*(x,y,angle) + outer(x,y,angle) = 36 params
        inner_params = params[:33].reshape(11, 3)
        outer_params = params[33:36]
        
        inner_positions = inner_params[:, :2]
        inner_angles = inner_params[:, 2]
        outer_center = outer_params[:2]
        outer_angle = outer_params[2]
        
        # Compute vertices of all inner hexagons
        inner_verts = []
        for i in range(11):
            v = hex_vertices(inner_positions[i, 0], inner_positions[i, 1], 1.0, inner_angles[i])
            inner_verts.append(v)
        
        # Check separation between all pairs
        penalty = 0.0
        for i in range(11):
            for j in range(i+1, 11):
                if not hexagons_separated(inner_verts[i], inner_verts[j]):
                    # Compute minimum overlap
                    min_gap = np.inf
                    for vi in [inner_verts[i], inner_verts[j]]:
                        for k in range(6):
                            e = vi[(k+1)%6] - vi[k]
                            n = np.array([-e[1], e[0]])
                            n = n / (np.linalg.norm(n) + 1e-15)
                            proj_i = inner_verts[i] @ n
                            proj_j = inner_verts[j] @ n
                            gap = max(proj_i.min() - proj_j.max(), proj_j.min() - proj_i.max())
                            if gap < min_gap:
                                min_gap = gap
                    # Penalty for overlap
                    if min_gap < 1e-6:
                        penalty += (1e-6 - min_gap) ** 2
        
        # Compute enclosing hexagon side
        s = enclosing_hex_side(inner_positions, inner_angles, outer_center, outer_angle)
        
        # Also ensure side >= sqrt(11)
        if s < np.sqrt(11):
            s = np.sqrt(11)
        
        return s + 1000 * penalty

    # Initial guess: "Flower + 4" configuration
    # This is a valid packing of 11 unit hexagons.
    sqrt3 = np.sqrt(3)
    centers = [
        [0, 0],
        [2, 0],
        [-2, 0],
        [1, sqrt3],
        [-1, sqrt3],
        [1, -sqrt3],
        [-1, -sqrt3],
        [4, 0],
        [-4, 0],
        [3, sqrt3],
        [3, -sqrt3]
    ]
    
    # Flatten to 1D array for minimize
    x0 = np.zeros(36)
    for i, c in enumerate(centers):
        x0[3*i] = c[0]
        x0[3*i+1] = c[1]
        x0[3*i+2] = 0.0
    x0[33] = 0.0 # outer center x
    x0[34] = 0.0 # outer center y
    x0[35] = 0.0 # outer angle
    
    best_params = x0.copy()
    best_val = objective(x0)
    
    # Optimize using Powell method which is robust for this type of problem
    try:
        result = minimize(objective, x0, method='Powell', 
                         options={'maxiter': 200, 'ftol': 1e-8, 'xtol': 1e-8})
        
        if result.fun < best_val:
            best_val = result.fun
            best_params = result.x
    except Exception:
        pass
    
    # Extract best solution
    inner_params = best_params[:33].reshape(11, 3)
    outer_params = best_params[33:36]
    
    inner_positions = inner_params[:, :2]
    inner_angles = inner_params[:, 2]
    outer_center = outer_params[:2]
    outer_angle = outer_params[2]
    
    # Compute actual enclosing side
    s = enclosing_hex_side(inner_positions, inner_angles, outer_center, outer_angle)
    s = max(s, np.sqrt(11))
    
    inner_hex_data = np.column_stack([inner_positions, inner_angles])
    outer_hex_data = np.array([outer_center[0], outer_center[1], outer_angle])
    
    return inner_hex_data, outer_hex_data, float(s)

# EVOLVE-BLOCK-END
