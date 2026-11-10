# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time


def _hex_vertices(cx, cy, s, a_deg):
    """Compute 6 vertices of a regular hexagon."""
    a = np.deg2rad(a_deg)
    angles = a + np.deg2rad(60 * np.arange(6))
    return np.column_stack([cx + s * np.cos(angles), cy + s * np.sin(angles)])


def _check_validity(inner, outer, side, tol=1e-6):
    """Check if packing is valid. Returns True if valid."""
    # All finite
    if not (np.all(np.isfinite(inner)) and np.all(np.isfinite(outer))):
        return False
    if not np.isfinite(side) or side < np.sqrt(11) - 1e-9:
        return False

    # Shift by outer center
    ox, oy, oa = outer
    inner_shifted = inner.copy()
    inner_shifted[:, 0] -= ox
    inner_shifted[:, 1] -= oy

    # Check edge lengths of inner hexagons (should be 1 within 1e-8)
    for i in range(len(inner_shifted)):
        cx, cy, ang = inner_shifted[i]
        verts = _hex_vertices(cx, cy, 1.0, ang)
        for j in range(6):
            v1 = verts[j]
            v2 = verts[(j + 1) % 6]
            d = np.linalg.norm(v2 - v1)
            if abs(d - 1.0) > 1e-8:
                return False

    # Check all inner hexagons are inside outer hexagon
    outer_verts = _hex_vertices(0, 0, side, oa)  # shifted to origin
    n_outer = len(outer_verts)
    for i in range(n_outer):
        p1 = outer_verts[i]
        p2 = outer_verts[(i + 1) % n_outer]
        edge = p2 - p1
        normal = np.array([-edge[1], edge[0]])
        # For CCW polygon, interior is to the left, so cross(p-p1, edge) >= 0
        # Actually: cross(edge, p - p1) >= 0 for CCW
        for k in range(len(inner_shifted)):
            cx, cy, ang = inner_shifted[k]
            verts = _hex_vertices(cx, cy, 1.0, ang)
            for v in verts:
                cross_val = np.cross(edge, v - p1)
                if cross_val < -tol:
                    return False

    # Check pairwise separation of inner hexagons
    n_inner = len(inner_shifted)
    for i in range(n_inner):
        for j in range(i + 1, n_inner):
            ci = inner_shifted[i]
            cj = inner_shifted[j]
            vi = _hex_vertices(ci[0], ci[1], 1.0, ci[2])
            vj = _hex_vertices(cj[0], cj[1], 1.0, cj[2])
            
            # Check if they are separated on at least one of the 12 edge normals
            separated = False
            # Edge normals of hex i
            for k in range(6):
                e = vi[(k + 1) % 6] - vi[k]
                nrm = np.array([-e[1], e[0]])
                nrm = nrm / np.linalg.norm(nrm)
                
                proj_i = np.dot(vi, nrm)
                proj_j = np.dot(vj, nrm)
                
                min_i = np.min(proj_i)
                max_i = np.max(proj_i)
                min_j = np.min(proj_j)
                max_j = np.max(proj_j)
                
                # Check if intervals are apart (gap > tol)
                if max_i + tol < min_j or max_j + tol < min_i:
                    separated = True
                    break
            
            if not separated:
                # Also check normals of hex j
                for k in range(6):
                    e = vj[(k + 1) % 6] - vj[k]
                    nrm = np.array([-e[1], e[0]])
                    nrm = nrm / np.linalg.norm(nrm)
                    
                    proj_i = np.dot(vi, nrm)
                    proj_j = np.dot(vj, nrm)
                    
                    min_i = np.min(proj_i)
                    max_i = np.max(proj_i)
                    min_j = np.min(proj_j)
                    max_j = np.max(proj_j)
                    
                    if max_i + tol < min_j or max_j + tol < min_i:
                        separated = True
                        break
            
            if not separated:
                return False

    return True


def hexagon_packing_11():
    """
    Constructs a packing of 11 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Returns
        inner_hex_data: np.ndarray of shape (11,3), where each row is of the form (x, y, angle_degrees)
        outer_hex_data: np.ndarray of shape (3,) of form (x,y,angle_degree)
        outer_hex_side_length: float
    """
    deadline = time.time() + 600.0
    safety_margin = 10.0  # seconds

    # Honeycomb initial guess
    sqrt3 = np.sqrt(3)
    
    # 1 center
    # 6 surrounding
    # 4 more from next ring
    # Positions (x, y)
    # Center: (0,0)
    # Ring 1:
    # (sqrt3, 0)
    # (sqrt3/2, 1.5)
    # (-sqrt3/2, 1.5)
    # (-sqrt3, 0)
    # (-sqrt3/2, -1.5)
    # (sqrt3/2, -1.5)
    
    # Ring 2 candidates (neighbors of Ring 1 that are not Ring 0 or 1)
    # Neighbors of (sqrt3, 0):
    # (2*sqrt3, 0)
    # (1.5*sqrt3, 1.5)
    # (1.5*sqrt3, -1.5)
    
    # Let's pick 4 from Ring 2 to make it compact.
    # Symmetric choice:
    # (2*sqrt3, 0)
    # (1.5*sqrt3, 1.5)
    # (1.5*sqrt3, -1.5)
    # And one more? Maybe (-2*sqrt3, 0)? Or (-1.5*sqrt3, 1.5)?
    # To balance, let's try:
    # (2*sqrt3, 0)
    # (-2*sqrt3, 0)
    # (1.5*sqrt3, 1.5)
    # (-1.5*sqrt3, -1.5)
    
    inner_init = np.array([
        [0, 0],
        [sqrt3, 0],
        [sqrt3/2, 1.5],
        [-sqrt3/2, 1.5],
        [-sqrt3, 0],
        [-sqrt3/2, -1.5],
        [sqrt3/2, -1.5],
        [2*sqrt3, 0],
        [-2*sqrt3, 0],
        [1.5*sqrt3, 1.5],
        [-1.5*sqrt3, -1.5]
    ])
    
    angles_init = np.zeros(11)
    outer_angle_init = 0.0
    
    # Objective: Minimize side length S
    # Params: 11*2 (pos) + 11 (angles) + 1 (outer angle) = 34
    # We fix outer center at (0,0) by shifting inner positions.
    
    def compute_min_side(inner_pos, inner_angles, outer_angle_deg):
        oa = np.deg2rad(outer_angle_deg)
        normal_angles = oa + np.deg2rad(30 + 60 * np.arange(6))
        normals = np.column_stack([np.cos(normal_angles), np.sin(normal_angles)])
        
        all_verts = []
        for i in range(11):
            cx, cy = inner_pos[i]
            ang = inner_angles[i]
            verts = _hex_vertices(cx, cy, 1.0, ang)
            all_verts.append(verts)
        all_verts = np.vstack(all_verts)
        
        max_dots = np.max(all_verts @ normals.T, axis=0)
        min_s = np.max(max_dots) / (np.sqrt(3) / 2)
        return min_s

    def objective(params):
        inner_pos = params[:22].reshape(11, 2)
        inner_angles = params[22:33]
        outer_angle = params[33]
        
        # Check overlap
        for i in range(11):
            for j in range(i + 1, 11):
                vi = _hex_vertices(inner_pos[i, 0], inner_pos[i, 1], 1.0, inner_angles[i])
                vj = _hex_vertices(inner_pos[j, 0], inner_pos[j, 1], 1.0, inner_angles[j])
                
                # Quick check
                d = np.linalg.norm(inner_pos[i] - inner_pos[j])
                if d < 1.7:
                    return 1e6
                
                separated = False
                for k in range(6):
                    e = vi[(k + 1) % 6] - vi[k]
                    nrm = np.array([-e[1], e[0]])
                    nrm = nrm / np.linalg.norm(nrm)
                    proj_i = np.dot(vi, nrm)
                    proj_j = np.dot(vj, nrm)
                    if np.max(proj_i) + 1e-6 < np.min(proj_j) or np.max(proj_j) + 1e-6 < np.min(proj_i):
                        separated = True
                        break
                if not separated:
                    for k in range(6):
                        e = vj[(k + 1) % 6] - vj[k]
                        nrm = np.array([-e[1], e[0]])
                        nrm = nrm / np.linalg.norm(nrm)
                        proj_i = np.dot(vi, nrm)
                        proj_j = np.dot(vj, nrm)
                        if np.max(proj_i) + 1e-6 < np.min(proj_j) or np.max(proj_j) + 1e-6 < np.min(proj_i):
                            separated = True
                            break
                if not separated:
                    return 1e6
        
        s = compute_min_side(inner_pos, inner_angles, outer_angle)
        return s

    best_result = None
    best_score = 0.0
    
    params_init = np.concatenate([inner_init.flatten(), angles_init, [outer_angle_init]])
    
    try:
        result = minimize(
            objective,
            params_init,
            method='Nelder-Mead',
            options={'maxiter': 2000, 'xatol': 1e-5, 'fatol': 1e-7}
        )
        
        if result.fun < 1e5:
            inner_pos_opt = result.x[:22].reshape(11, 2)
            inner_angles_opt = result.x[22:33]
            outer_angle_opt = result.x[33]
            
            inner_opt = np.column_stack([inner_pos_opt, inner_angles_opt])
            outer_opt = np.array([0, 0, outer_angle_opt])
            side_opt = result.fun
            
            if _check_validity(inner_opt, outer_opt, side_opt):
                best_result = (inner_opt, outer_opt, side_opt)
                best_score = 1.0 / side_opt
    except Exception:
        pass

    if best_result is None:
        # Fallback to seed
        inner_hex_data = np.array(
            [
                [0, 0, 0],
                [-2.5, 0, 0],
                [2.5, 0, 0],
                [-1.25, 2.17, 0],
                [1.25, 2.17, 0],
                [-1.25, -2.17, 0],
                [1.25, -2.17, 0],
                [-3.75, 2.17, 0],
                [3.75, 2.17, 0],
                [-3.75, -2.17, 0],
                [3.75, -2.17, 0],
            ]
        )
        outer_hex_data = np.array([0, 0, 0.0])
        outer_hex_side_length = 8.0
        return (inner_hex_data, outer_hex_data, outer_hex_side_length)

    return best_result


# EVOLVE-BLOCK-END
