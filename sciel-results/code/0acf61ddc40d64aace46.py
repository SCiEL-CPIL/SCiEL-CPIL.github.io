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
    
    # Get all 12 edge normals (6 from each hexagon)
    def get_normals(v1, v2):
        normals = []
        for i in range(6):
            e = v1[(i+1) % 6] - v1[i]
            n = np.array([-e[1], e[0]])
            n = n / (np.linalg.norm(n) + 1e-15)
            normals.append(n)
        for i in range(6):
            e = v2[(i+1) % 6] - v2[i]
            n = np.array([-e[1], e[0]])
            n = n / (np.linalg.norm(n) + 1e-15)
            normals.append(n)
        return np.array(normals)
    
    # Smooth separation penalty using log-sum-exp
    def smooth_separation_penalty(v1, v2, beta=50.0):
        """
        Compute a smooth penalty for two hexagons not being separated.
        Uses log-sum-exp to create a smooth surrogate for the max violation.
        """
        normals = get_normals(v1, v2)
        
        # Project both hexagons onto each normal
        proj1 = v1 @ normals.T  # shape (6, 12)
        proj2 = v2 @ normals.T  # shape (6, 12)
        
        # For each normal, compute the gap (positive means separated)
        # gap_i = max(proj1.min() - proj2.max(), proj2.min() - proj1.max())
        gaps = np.maximum(proj1.min(axis=0) - proj2.max(axis=0), 
                          proj2.min(axis=0) - proj1.max(axis=0))
        
        # We want all gaps > 1e-6. Penalty when any gap < 1e-6.
        # Use soft-min: if min_gap < 1e-6, penalty = (1e-6 - soft_min_gap)^2
        # soft_min_gap = -logsumexp(-gaps) / beta
        # But we need to be careful: if all gaps are large positive, we want penalty ~ 0
        
        # Compute violation: v_i = 1e-6 - gap_i (positive when violated)
        violations = 1e-6 - gaps
        
        # Use log-sum-exp to get a smooth max of violations
        # If all violations <= 0, soft_max should be small
        # Clip violations to avoid overflow
        v_clipped = np.clip(violations, -100, 100)
        
        # log-sum-exp for max
        max_v = np.max(v_clipped)
        if max_v > 1e-10:
            # At least one violation exists
            lse = max_v + np.log(np.sum(np.exp(v_clipped - max_v)))
            soft_max = lse / beta
            # Scale to get a reasonable penalty
            penalty = soft_max ** 2
        else:
            # All gaps are sufficient, small penalty
            penalty = 0.0
        
        return penalty
    
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
    
    # Objective function: minimize outer hexagon side length with smooth separation penalty
    def objective(params, beta=50.0):
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
        
        # Smooth separation penalty between all pairs
        penalty = 0.0
        for i in range(11):
            for j in range(i+1, 11):
                penalty += smooth_separation_penalty(inner_verts[i], inner_verts[j], beta)
        
        # Compute enclosing hexagon side
        s = enclosing_hex_side(inner_positions, inner_angles, outer_center, outer_angle)
        
        # Also ensure side >= sqrt(11)
        if s < np.sqrt(11):
            s = np.sqrt(11)
        
        return s + 100 * penalty
    
    # Initial guess: "Flower + 4" configuration
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
    
    # Multi-stage optimization with increasing beta for smoother convergence
    beta_schedule = [10.0, 30.0, 80.0, 200.0]
    
    for beta in beta_schedule:
        if time.time() > deadline - 60:
            break
        
        try:
            # Use Powell for robustness, then try L-BFGS-B for polish
            result = minimize(lambda p: objective(p, beta), best_params, method='Powell', 
                             options={'maxiter': 150, 'ftol': 1e-10, 'xtol': 1e-10})
            
            if result.fun < best_val:
                best_val = result.fun
                best_params = result.x.copy()
                
            # Try L-BFGS-B for a polish step
            if time.time() < deadline - 30:
                result2 = minimize(lambda p: objective(p, beta), best_params, method='L-BFGS-B',
                                  options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-12})
                
                if result2.fun < best_val:
                    best_val = result2.fun
                    best_params = result2.x.copy()
                    
        except Exception:
            pass
    
    # Final high-precision polish pass
    if time.time() < deadline - 20:
        try:
            result_final = minimize(lambda p: objective(p, 500.0), best_params, method='L-BFGS-B',
                                   options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-14})
            
            if result_final.fun < best_val:
                best_val = result_final.fun
                best_params = result_final.x.copy()
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
