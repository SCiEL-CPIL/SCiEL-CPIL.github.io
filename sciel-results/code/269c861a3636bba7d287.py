import math
import time
import numpy as np
import itertools
from scipy.optimize import linprog

N = 11
_START_TIME = time.time()
_DEADLINE = _START_TIME + 145.0
_SQRT3 = math.sqrt(3)

def _hex_vertices_local():
    """Returns vertices of a unit hexagon centered at (0,0) with angle 0."""
    angles = np.arange(6) * math.pi / 3.0
    return np.column_stack([np.cos(angles), np.sin(angles)])

def _generate_cluster_vertices(shape_idx, theta_cluster_deg):
    """
    Generate the 66 vertices (11 hexes * 6) for a given cluster shape and rotation.
    Shape 0: 1 center + 6 ring1 + 4 ring2 (consecutive)
    Shape 1: 1 center + 6 ring1 + 4 ring2 (symmetric pairs)
    """
    # Lattice vectors
    u = np.array([_SQRT3, 0.0])
    v = np.array([_SQRT3 / 2, 1.5])
    
    centers = []
    # Center
    centers.append(np.array([0.0, 0.0]))
    
    # Ring 1
    ring1_coords = [(1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1)]
    for i, j in ring1_coords:
        centers.append(i * u + j * v)
        
    # Ring 2 selection
    ring2_coords = [(2, 0), (1, 1), (0, 2), (-1, 2), (-2, 1), (-2, 0), 
                    (-1, -1), (0, -2), (1, -2), (2, -1)]
    
    if shape_idx == 0:
        # 4 consecutive: (2,0), (1,1), (0,2), (-1,2)
        selected = ring2_coords[0:4]
    elif shape_idx == 1:
        # Symmetric pairs: (2,0), (-2,0) and (1,1), (-1,-1)
        selected = [ring2_coords[0], ring2_coords[5], ring2_coords[1], ring2_coords[6]]
    else:
        # 3 consecutive + 1 opposite: (2,0), (1,1), (0,2) and (-2,0)
        selected = [ring2_coords[0], ring2_coords[1], ring2_coords[2], ring2_coords[5]]
        
    for i, j in selected:
        centers.append(i * u + j * v)
        
    # Ensure we have exactly 11 centers
    if len(centers) != 11:
        # Fallback to first 11 if logic fails
        centers = centers[:11]
        
    # Local vertices for one hexagon
    local_verts = _hex_vertices_local()
    
    # All vertices
    all_verts = np.zeros((11 * 6, 2))
    for idx, c in enumerate(centers):
        # Vertices of this hexagon: c + local_verts
        all_verts[idx*6:(idx+1)*6] = c + local_verts
        
    # Rotate by theta_cluster
    theta = math.radians(theta_cluster_deg)
    R = np.array([[math.cos(theta), -math.sin(theta)], 
                  [math.sin(theta), math.cos(theta)]])
    all_verts = all_verts @ R.T
    
    return all_verts

def _min_enclosing_hex_side_lp(verts, outer_angle_deg):
    """
    Find minimum side length of a regular hexagon (angle outer_angle_deg)
    containing all points in verts.
    Returns (cx, cy, side) or None.
    """
    M = len(verts)
    if M == 0:
        return 0.0, 0.0, 0.0

    base_angle = math.radians(outer_angle_deg + 30)
    angles = base_angle + np.arange(6) * math.pi / 3.0
    normals = np.column_stack([np.cos(angles), np.sin(angles)])  # (6, 2)

    # v_dot_n: (M, 6)
    v_dot_n = verts @ normals.T  # (M, 6)

    max_dots = np.max(v_dot_n, axis=0)  # (6,)
    min_dots = np.min(v_dot_n, axis=0)  # (6,)

    A_rows = []
    b_vals = []

    for k in range(6):
        nx, ny = normals[k]
        # Constraint 1: -nx*cx - ny*cy - R <= -max_dots[k]
        A_rows.append([-nx, -ny, -1.0])
        b_vals.append(-max_dots[k])
        # Constraint 2: nx*cx + ny*cy - R <= min_dots[k]
        A_rows.append([nx, ny, -1.0])
        b_vals.append(min_dots[k])

    A = np.array(A_rows)
    b = np.array(b_vals)

    # R >= 0
    A = np.vstack([A, [[0.0, 0.0, -1.0]]])
    b = np.append(b, 0.0)

    c_obj = [0.0, 0.0, 1.0]
    try:
        res = linprog(c_obj, A_ub=A, b_ub=b,
                      bounds=[(None, None), (None, None), (0, None)],
                      method='highs')
        if res.success:
            cx, cy, R = res.x
            side = R / (_SQRT3 / 2)
            return cx, cy, side
    except Exception:
        pass
    return None

def construct():
    global _START_TIME, _DEADLINE

    if time.time() > _DEADLINE - 5.0:
        # Fallback: simple row
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0

    best_side = float('inf')
    best_theta_c = 0.0
    best_theta_h = 0.0
    best_center = (0.0, 0.0)
    best_shape_idx = 0

    shapes = [0, 1, 2]
    
    # Coarse Search
    for shape_idx in shapes:
        if time.time() > _DEADLINE - 20.0:
            break
            
        # Precompute vertices for this shape at theta_c = 0? 
        # No, theta_c changes. But we can compute base centers once.
        # Actually, _generate_cluster_vertices does the rotation.
        
        for theta_c in np.arange(0, 60, 2.0):
            if time.time() > _DEADLINE - 15.0:
                break
                
            verts = _generate_cluster_vertices(shape_idx, float(theta_c))
            
            for theta_h in np.arange(0, 60, 1.0):
                if time.time() > _DEADLINE - 10.0:
                    break
                    
                res = _min_enclosing_hex_side_lp(verts, float(theta_h))
                if res:
                    cx, cy, side = res
                    if side < best_side:
                        best_side = side
                        best_theta_c = float(theta_c)
                        best_theta_h = float(theta_h)
                        best_center = (cx, cy)
                        best_shape_idx = shape_idx

    # Refinement
    if best_side < float('inf'):
        # Refine theta_c
        def objective_c(params):
            tc = params[0] % 60.0
            verts = _generate_cluster_vertices(best_shape_idx, float(tc))
            # For each tc, we need the best th. 
            # This is expensive. Instead, do a nested search.
            # For now, just refine theta_h for the current best theta_c
            return 0.0 # Placeholder
            
        # Nested Golden Section for theta_h given best_theta_c
        verts_best = _generate_cluster_vertices(best_shape_idx, best_theta_c)
        
        def obj_h(params):
            th = params[0] % 60.0
            res = _min_enclosing_hex_side_lp(verts_best, float(th))
            if res:
                return res[2]
            return 1e10

        # Golden Section Search for theta_h
        a, b = max(0.0, best_theta_h - 2.0), min(60.0, best_theta_h + 2.0)
        gr = (math.sqrt(5.0) - 1.0) / 2.0
        c = b - gr * (b - a)
        d = a + gr * (b - a)
        fc = obj_h([c])
        fd = obj_h([d])
        for _ in range(100):
            if time.time() > _DEADLINE - 5.0:
                break
            if fc < fd:
                b = d
                d = c
                fd = fc
                c = b - gr * (b - a)
                fc = obj_h([c])
            else:
                a = c
                c = d
                fc = fd
                d = a + gr * (b - a)
                fd = obj_h([d])
        best_theta_h = (a + b) / 2.0 % 60.0
        res = _min_enclosing_hex_side_lp(verts_best, best_theta_h)
        if res:
            best_center = (res[0], res[1])
            best_side = res[2]

        # Now refine theta_c with the new best theta_h
        def obj_c(params):
            tc = params[0] % 60.0
            verts = _generate_cluster_vertices(best_shape_idx, float(tc))
            res = _min_enclosing_hex_side_lp(verts, best_theta_h)
            if res:
                return res[2]
            return 1e10

        a, b = max(0.0, best_theta_c - 2.0), min(60.0, best_theta_c + 2.0)
        c = b - gr * (b - a)
        d = a + gr * (b - a)
        fc = obj_c([c])
        fd = obj_c([d])
        for _ in range(100):
            if time.time() > _DEADLINE - 2.0:
                break
            if fc < fd:
                b = d
                d = c
                fd = fc
                c = b - gr * (b - a)
                fc = obj_c([c])
            else:
                a = c
                c = d
                fc = fd
                d = a + gr * (b - a)
                fd = obj_c([d])
        best_theta_c = (a + b) / 2.0 % 60.0
        
        # Final evaluation
        verts_final = _generate_cluster_vertices(best_shape_idx, best_theta_c)
        res = _min_enclosing_hex_side_lp(verts_final, best_theta_h)
        if res:
            best_center = (res[0], res[1])
            best_side = res[2]

    # Construct return values
    # inner: list of [x, y, angle]
    # The centers are in local coords, rotated by best_theta_c
    # We need the actual centers in the global frame.
    # _generate_cluster_vertices returns vertices. We need centers.
    
    # Recalculate centers
    u = np.array([_SQRT3, 0.0])
    v = np.array([_SQRT3 / 2, 1.5])
    centers_local = [np.array([0.0, 0.0])]
    ring1_coords = [(1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1)]
    for i, j in ring1_coords:
        centers_local.append(i * u + j * v)
    ring2_coords = [(2, 0), (1, 1), (0, 2), (-1, 2), (-2, 1), (-2, 0), 
                    (-1, -1), (0, -2), (1, -2), (2, -1)]
    if best_shape_idx == 0:
        selected = ring2_coords[0:4]
    elif best_shape_idx == 1:
        selected = [ring2_coords[0], ring2_coords[5], ring2_coords[1], ring2_coords[6]]
    else:
        selected = [ring2_coords[0], ring2_coords[1], ring2_coords[2], ring2_coords[5]]
    for i, j in selected:
        centers_local.append(i * u + j * v)
        
    theta_rad = math.radians(best_theta_c)
    R = np.array([[math.cos(theta_rad), -math.sin(theta_rad)], 
                  [math.sin(theta_rad), math.cos(theta_rad)]])
                  
    inner = []
    for c in centers_local:
        c_global = c @ R.T
        inner.append([float(c_global[0]), float(c_global[1]), best_theta_c])
        
    return inner, best_center, best_side, best_theta_h

if __name__ == '__main__':
    inner, center, side, angle = construct()
    print(f"Side: {side}, Angle: {angle}, Center: {center}")
