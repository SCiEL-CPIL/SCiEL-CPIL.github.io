import numpy as np
import time
from scipy.optimize import minimize


def _hex_vertices(cx, cy, d, a_deg):
    """Return 6 vertices of a regular hexagon with center (cx,cy), side d, angle a_deg."""
    a = np.deg2rad(a_deg)
    angles = a + np.deg2rad(60.0 * np.arange(6))
    return np.column_stack((cx + d * np.cos(angles), cy + d * np.sin(angles)))


def _check_validity(inner, outer, s):
    """Validate packing; return (is_valid, error_msg)."""
    if not np.all(np.isfinite(inner)) or not np.all(np.isfinite(outer)):
        return False, "non-finite values"
    if s < np.sqrt(12.0) - 1e-12:
        return False, "s < sqrt(12)"
    
    # Outer hexagon vertices must be distinct
    ov = _hex_vertices(outer[0], outer[1], s, outer[2])
    for i in range(6):
        for j in range(i + 1, 6):
            if np.linalg.norm(ov[i] - ov[j]) < 1e-10:
                return False, "outer vertices not distinct"
    
    # Check unit edge length for inner hexagons
    for row in inner:
        cx, cy, a = row
        hv = _hex_vertices(cx, cy, 1.0, a)
        for i in range(6):
            edge_len = np.linalg.norm(hv[(i + 1) % 6] - hv[i])
            if abs(edge_len - 1.0) > 1e-8:
                return False, f"unit edge length differs: {edge_len}"
    
    # Check inner-inner separation using SAT
    def _separated(v1, v2):
        """Check if two hexagons are separated (non-overlapping) using SAT."""
        # Get all edges from both hexagons
        edges = []
        for i in range(6):
            edges.append(v1[(i + 1) % 6] - v1[i])
            edges.append(v2[(i + 1) % 6] - v2[i])
        
        for edge in edges:
            # Normal direction
            normal = np.array([-edge[1], edge[0]])
            if np.linalg.norm(normal) < 1e-12:
                continue
            normal /= np.linalg.norm(normal)
            
            # Project both hexagons onto normal
            proj1 = v1 @ normal
            proj2 = v2 @ normal
            min1, max1 = np.min(proj1), np.max(proj1)
            min2, max2 = np.min(proj2), np.max(proj2)
            
            # Check if intervals are separated by more than 1e-6
            gap1 = min2 - max1  # v2 to the right of v1
            gap2 = min1 - max2  # v1 to the right of v2
            if gap1 > 1e-6 or gap2 > 1e-6:
                return True  # Found a separating axis
        
        return False  # No separating axis found -> overlapping
    
    inner_verts = [_hex_vertices(row[0], row[1], 1.0, row[2]) for row in inner]
    for i in range(12):
        for j in range(i + 1, 12):
            if not _separated(inner_verts[i], inner_verts[j]):
                return False, f"inner hexagons {i} and {j} overlap"
    
    # Check inner vertices inside outer hexagon
    for row in inner:
        cx, cy, a = row
        hv = _hex_vertices(cx, cy, 1.0, a)
        for v in hv:
            # For every outer edge (p1, p2) in vertex order, cross(p2-p1, v-p1) >= -1e-6
            for k in range(6):
                p1 = ov[k]
                p2 = ov[(k + 1) % 6]
                edge = p2 - p1
                to_v = v - p1
                cross = edge[0] * to_v[1] - edge[1] * to_v[0]
                if cross < -1e-6:
                    return False, f"inner vertex outside outer hexagon"
    
    return True, ""


def _objective(params, s_fixed=None):
    """
    Objective function for optimization.
    params: 12*3 = 36 values: for each inner hex, (x, y, angle_deg)
    We also optimize the outer side length s if s_fixed is None.
    Actually, let's fix the outer center at origin and optimize its side length too.
    
    Let me restructure: params = [x1,y1,a1, x2,y2,a2, ..., x12,y12,a12, s]
    Total: 12*3 + 1 = 37 parameters.
    """
    # This is complex; let's use a simpler approach with penalty-based optimization.
    pass


def hexagon_packing_12():
    """
    Constructs a packing of 12 disjoint unit regular hexagons inside a larger regular hexagon,
    maximizing 1/outer_hex_side_length.
    """
    start_time = time.time()
    deadline = start_time + 580.0  # Safety margin before 630s limit
    
    # Initial guess: symmetric arrangement
    # 12 hexagons in a hexagonal arrangement
    inner = np.array([
        [0.0, 0.0, 0.0],
        [2.5, 0.0, 0.0],
        [-2.5, 0.0, 0.0],
        [1.25, 2.1650635, 0.0],
        [-1.25, 2.1650635, 0.0],
        [1.25, -2.1650635, 0.0],
        [-1.25, -2.1650635, 0.0],
        [3.75, 2.1650635, 0.0],
        [-3.75, 2.1650635, 0.0],
        [3.75, -2.1650635, 0.0],
        [-3.75, -2.1650635, 0.0],
        [0.0, -4.330127, 0.0],
    ])
    outer = np.array([0.0, 0.0, 0.0])
    s = 8.0
    
    # Validate initial guess
    valid, msg = _check_validity(inner, outer, s)
    if not valid:
        print(f"Initial guess invalid: {msg}")
        # Try a larger s
        s = 10.0
        valid, msg = _check_validity(inner, outer, s)
        if not valid:
            print(f"Still invalid: {msg}")
            # Fallback: just return a very large outer hex
            outer_hex_data = np.array([0.0, 0.0, 0.0])
            s = 20.0
            return inner, outer_hex_data, s
    
    best_s = s
    best_inner = inner.copy()
    best_outer = outer.copy()
    
    # Try to optimize using scipy
    try:
        def obj_and_penalty(params):
            """
            params: 37 values: 12*(x,y,a) + s
            Returns: s + penalties
            """
            if params[-1] < np.sqrt(12.0) - 1e-9:
                return 1e6
            
            inner_p = params[:36].reshape(12, 3)
            s_p = params[-1]
            outer_p = np.array([0.0, 0.0, 0.0])  # Fix outer center at origin
            
            # Penalty for overlaps
            inner_verts = [_hex_vertices(row[0], row[1], 1.0, row[2]) for row in inner_p]
            penalty = 0.0
            
            for i in range(12):
                for j in range(i + 1, 12):
                    # Simple distance-based penalty (not exact, but helps gradient)
                    dist = np.linalg.norm(inner_verts[i][0] - inner_verts[j][0])
                    if dist < 2.0:  # Rough check
                        penalty += max(0, 2.0 - dist) ** 2
            
            # Penalty for vertices outside outer hex
            ov = _hex_vertices(0.0, 0.0, s_p, 0.0)
            for row in inner_p:
                hv = _hex_vertices(row[0], row[1], 1.0, row[2])
                for v in hv:
                    for k in range(6):
                        p1 = ov[k]
                        p2 = ov[(k + 1) % 6]
                        edge = p2 - p1
                        to_v = v - p1
                        cross = edge[0] * to_v[1] - edge[1] * to_v[0]
                        if cross < 0:
                            penalty += cross ** 2
            
            return s_p + penalty
        
        x0 = np.concatenate([inner.flatten(), [s]])
        
        result = minimize(obj_and_penalty, x0, method='Nelder-Mead',
                          options={'maxiter': 5000, 'xatol': 1e-6, 'fatol': 1e-6})
        
        if result.fun < best_s:
            inner_opt = result.x[:36].reshape(12, 3)
            s_opt = result.x[-1]
            outer_opt = np.array([0.0, 0.0, 0.0])
            
            # Validate
            valid, msg = _check_validity(inner_opt, outer_opt, s_opt)
            if valid:
                best_inner = inner_opt
                best_outer = outer_opt
                best_s = s_opt
                print(f"Optimized: s = {best_s:.6f}, 1/s = {1/best_s:.6f}")
            else:
                print(f"Optimized result invalid: {msg}")
        
        if time.time() > deadline:
            return best_inner, best_outer, best_s
        
        # Try more iterations if time permits
        if time.time() < deadline - 60:
            result2 = minimize(obj_and_penalty, result.x, method='Nelder-Mead',
                              options={'maxiter': 10000, 'xatol': 1e-8, 'fatol': 1e-8})
            
            if result2.fun < best_s:
                inner_opt = result2.x[:36].reshape(12, 3)
                s_opt = result2.x[-1]
                outer_opt = np.array([0.0, 0.0, 0.0])
                
                valid, msg = _check_validity(inner_opt, outer_opt, s_opt)
                if valid:
                    best_inner = inner_opt
                    best_outer = outer_opt
                    best_s = s_opt
                    print(f"Further optimized: s = {best_s:.6f}, 1/s = {1/best_s:.6f}")
    
    except Exception as e:
        print(f"Optimization error: {e}")
    
    # Final validation
    valid, msg = _check_validity(best_inner, best_outer, best_s)
    if not valid:
        print(f"Final result invalid: {msg}, falling back to safe configuration")
        # Safe fallback
        outer_hex_data = np.array([0.0, 0.0, 0.0])
        s = 20.0
        return inner, outer_hex_data, s
    
    return best_inner, best_outer, best_s


# EVOLVE-BLOCK-END
