import numpy as np
from scipy.optimize import minimize
import time
import itertools

def _hex_vertices(cx, cy, d, angle_deg):
    """Compute vertices of a regular hexagon."""
    a = np.radians(angle_deg)
    angles = a + np.radians(60) * np.arange(6)
    return np.column_stack([
        cx + d * np.cos(angles),
        cy + d * np.sin(angles)
    ])

def _hex_edge_normals(angle_deg):
    """Compute the 6 outward unit edge normals for a regular hexagon."""
    a = np.radians(angle_deg)
    normals_angles = a + np.radians(30) + np.radians(60) * np.arange(6)
    return np.column_stack([np.cos(normals_angles), np.sin(normals_angles)])

def _check_separation(hex1, hex2, tol=1e-6):
    """Check if two hexagons are separated using SAT."""
    v1 = _hex_vertices(hex1[0], hex1[1], 1.0, hex1[2])
    v2 = _hex_vertices(hex2[0], hex2[1], 1.0, hex2[2])
    
    n1 = _hex_edge_normals(hex1[2])
    n2 = _hex_edge_normals(hex2[2])
    
    for normals in [n1, n2]:
        for normal in normals:
            proj1 = v1 @ normal
            proj2 = v2 @ normal
            min1, max1 = proj1.min(), proj1.max()
            min2, max2 = proj2.min(), proj2.max()
            if max1 + tol < min2 or max2 + tol < min1:
                return True
    return False

def _vertex_inside_outer(v, outer_verts):
    """Check if vertex v is inside outer hexagon."""
    for i in range(6):
        p1 = outer_verts[i]
        p2 = outer_verts[(i+1) % 6]
        cross = (p2[0]-p1[0])*(v[1]-p1[1]) - (p2[1]-p1[1])*(v[0]-p1[0])
        if cross < -1e-6:
            return False
    return True

def _validate(inner_data, outer_data, s):
    """Validate the packing."""
    if not np.all(np.isfinite(inner_data)) or not np.all(np.isfinite(outer_data)):
        return False
    if s < np.sqrt(12) - 1e-9:
        return False
    
    outer_verts = _hex_vertices(outer_data[0], outer_data[1], s, outer_data[2])
    
    for i in range(6):
        for j in range(i+1, 6):
            if np.linalg.norm(outer_verts[i] - outer_verts[j]) < 1e-8:
                return False
    
    inner_verts_list = []
    for i in range(12):
        verts = _hex_vertices(inner_data[i,0], inner_data[i,1], 1.0, inner_data[i,2])
        inner_verts_list.append(verts)
        
        for j in range(6):
            p1 = verts[j]
            p2 = verts[(j+1) % 6]
            if abs(np.linalg.norm(p2 - p1) - 1.0) > 1e-8:
                return False
        
        for v in verts:
            if not _vertex_inside_outer(v, outer_verts):
                return False
    
    for i in range(12):
        for j in range(i+1, 12):
            if not _check_separation(inner_data[i], inner_data[j]):
                return False
    
    return True

def _min_enclosing_hex_side(centers, inner_angles=None):
    """
    Find the minimum side length s of a regular hexagon (centered at origin, angle 0)
    that contains all inner hexagons.
    """
    outer_normals = _hex_edge_normals(0.0)
    sqrt3 = np.sqrt(3)
    
    if inner_angles is None:
        inner_angles = np.zeros(len(centers))
    
    max_required_s = 0.0
    for i, center in enumerate(centers):
        inner_angle = inner_angles[i]
        inner_verts = _hex_vertices(0, 0, 1.0, inner_angle)
        for n in outer_normals:
            support = np.max(inner_verts @ n)
            required_s = (center @ n + support) / (sqrt3 / 2)
            if required_s > max_required_s:
                max_required_s = required_s
    
    return max_required_s

def hexagon_packing_12():
    start_time = time.time()
    deadline = start_time + 580
    
    sqrt3 = np.sqrt(3)
    
    # Initial placement: honeycomb grid
    positions = []
    for col in range(4):
        for row in range(3):
            x = col * 1.5
            if col % 2 == 0:
                y = row * sqrt3
            else:
                y = (row + 0.5) * sqrt3
            positions.append([x, y, 0.0])
    
    xs = [p[0] for p in positions]
    ys = [p[1] for p in positions]
    cx = (min(xs) + max(xs)) / 2
    cy = (min(ys) + max(ys)) / 2
    
    for p in positions:
        p[0] -= cx
        p[1] -= cy
    
    inner_data = np.array(positions)
    centers = inner_data[:, :2].copy()
    inner_angles = np.zeros(12)
    
    s = _min_enclosing_hex_side(centers, inner_angles) * 1.001
    outer_data = np.array([0.0, 0.0, 0.0])
    
    if not _validate(inner_data, outer_data, s):
        s = 6.0
        outer_data = np.array([0.0, 0.0, 0.0])
    
    best_value = 1.0 / s
    best_result = (inner_data.copy(), outer_data.copy(), s)
    
    # Smooth objective using log-sum-exp for smooth max
    def smooth_max(x, beta=50):
        """Smooth approximation of max using log-sum-exp."""
        x_max = np.max(x)
        if len(x) == 1:
            return x_max
        return x_max + np.log(np.sum(np.exp(beta * (x - x_max)))) / beta
    
    def objective(params):
        inner_xy = params[:24].reshape(12, 2)
        s_val = params[24]
        
        if s_val < np.sqrt(12):
            return 1e6
        
        outer_normals = _hex_edge_normals(0.0)
        unit_hex_verts = _hex_vertices(0, 0, 1.0, 0.0)
        
        # Support values for unit hex in each outer normal direction
        h_vals = np.array([np.max(unit_hex_verts @ n) for n in outer_normals])
        
        penalty = 0.0
        
        # Containment: for each inner hex, for each outer normal, 
        # center@n + h_val <= s * sqrt3/2
        apothem = s_val * sqrt3 / 2
        
        for i in range(12):
            center = inner_xy[i]
            projs = center @ outer_normals.T  # (6,)
            required = projs + h_vals  # (6,)
            violations = required - apothem  # (6,)
            # Smooth max of violations
            max_viol = smooth_max(violations, beta=30)
            if max_viol > 0:
                penalty += max_viol ** 2
        
        # Separation between inner hexagons
        for i in range(12):
            v1 = _hex_vertices(inner_xy[i,0], inner_xy[i,1], 1.0, 0.0)
            n1 = _hex_edge_normals(0.0)
            
            for j in range(i+1, 12):
                v2 = _hex_vertices(inner_xy[j,0], inner_xy[j,1], 1.0, 0.0)
                
                min_gap = np.inf
                separated = False
                for normal in n1:
                    proj1 = v1 @ normal
                    proj2 = v2 @ normal
                    min1, max1 = proj1.min(), proj1.max()
                    min2, max2 = proj2.min(), proj2.max()
                    
                    gap = min(min2 - max1, min1 - max2)
                    if gap > 1e-6:
                        separated = True
                        break
                    if gap < min_gap:
                        min_gap = gap
                
                if not separated:
                    if min_gap < 0:
                        penalty += min_gap ** 2 * 100
                    else:
                        # Small positive gap, add small penalty to encourage separation
                        penalty += max(0, 1e-4 - min_gap) ** 2
        
        return s_val + 500 * penalty
    
    # Initial params
    init_params = np.zeros(25)
    init_params[:24] = inner_data[:, :2].flatten()
    init_params[24] = s
    
    n_restarts = 20
    for restart in range(n_restarts):
        if time.time() > deadline - 30:
            break
        
        if restart == 0:
            params0 = init_params.copy()
        elif restart < 5:
            # Small perturbations
            perturbation = 0.1 * np.random.randn(24)
            params0 = init_params.copy()
            params0[:24] += perturbation
            params0[24] = s + 0.1 * np.random.randn()
        elif restart < 10:
            # Medium perturbations
            perturbation = 0.3 * np.random.randn(24)
            params0 = init_params.copy()
            params0[:24] += perturbation
            params0[24] = s + 0.2 * np.random.randn()
        else:
            # Larger perturbations to escape local minima
            perturbation = 0.5 * np.random.randn(24)
            params0 = init_params.copy()
            params0[:24] += perturbation
            params0[24] = s + 0.3 * np.random.randn()
        
        try:
            result = minimize(
                objective,
                params0,
                method='Nelder-Mead',
                options={
                    'maxiter': 5000,
                    'xatol': 1e-8,
                    'fatol': 1e-8,
                    'adaptive': True
                }
            )
            
            if result.fun < 200:
                final_params = result.x
                inner_xy_final = final_params[:24].reshape(12, 2)
                s_final = final_params[24]
                
                if s_final > np.sqrt(12):
                    inner_final = np.zeros((12, 3))
                    inner_final[:, 0] = inner_xy_final[:, 0]
                    inner_final[:, 1] = inner_xy_final[:, 1]
                    inner_final[:, 2] = 0.0
                    outer_final = np.array([0.0, 0.0, 0.0])
                    
                    # Recompute s exactly
                    s_exact = _min_enclosing_hex_side(inner_xy_final, inner_angles) * 1.0001
                    
                    if _validate(inner_final, outer_final, s_exact):
                        value = 1.0 / s_exact
                        if value > best_value:
                            best_value = value
                            best_result = (inner_final, outer_final, s_exact)
        except Exception:
            pass
    
    # Final polish: try to tighten s by direct minimization
    if time.time() < deadline - 10:
        current_centers = best_result[0][:, :2].copy()
        
        def tight_objective(centers_flat):
            c = centers_flat.reshape(12, 2)
            s_val = _min_enclosing_hex_side(c, inner_angles)
            penalty = 0.0
            for i in range(12):
                v1 = _hex_vertices(c[i,0], c[i,1], 1.0, 0.0)
                for j in range(i+1, 12):
                    v2 = _hex_vertices(c[j,0], c[j,1], 1.0, 0.0)
                    n = _hex_edge_normals(0.0)
                    min_gap = np.inf
                    for normal in n:
                        proj1 = v1 @ normal
                        proj2 = v2 @ normal
                        gap = min(proj2.min() - proj1.max(), proj1.min() - proj2.max())
                        if gap < min_gap:
                            min_gap = gap
                    if min_gap < 0:
                        penalty += min_gap ** 2
            return s_val + 1000 * penalty
        
        try:
            polish_result = minimize(
                tight_objective,
                current_centers.flatten(),
                method='Nelder-Mead',
                options={
                    'maxiter': 3000,
                    'xatol': 1e-10,
                    'fatol': 1e-10,
                    'adaptive': True
                }
            )
            
            if polish_result.fun < 500:
                final_centers = polish_result.x.reshape(12, 2)
                s_exact = _min_enclosing_hex_side(final_centers, inner_angles) * 1.0001
                
                inner_final = np.zeros((12, 3))
                inner_final[:, 0] = final_centers[:, 0]
                inner_final[:, 1] = final_centers[:, 1]
                inner_final[:, 2] = 0.0
                outer_final = np.array([0.0, 0.0, 0.0])
                
                if _validate(inner_final, outer_final, s_exact):
                    value = 1.0 / s_exact
                    if value > best_value:
                        best_value = value
                        best_result = (inner_final, outer_final, s_exact)
        except Exception:
            pass
    
    return best_result
