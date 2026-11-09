# EVOLVE-BLOCK-START
import numpy as np
import math
import time

def _hex_vertices(cx, cy, d, angle_deg):
    """Compute vertices of a regular hexagon."""
    a = np.deg2rad(angle_deg)
    angles = a + np.deg2rad(60.0 * np.arange(6))
    return np.column_stack([cx + d * np.cos(angles), cy + d * np.sin(angles)])

def _is_valid(inner, outer, s, tol=1e-6, edge_tol=1e-8):
    """Check validity of a packing configuration."""
    if not np.all(np.isfinite(inner)) or not np.all(np.isfinite(outer)) or not np.isfinite(s):
        return False
    if s < math.sqrt(12) - 1e-9:
        return False
    
    ocx, ocy, oang = outer[0], outer[1], outer[2]
    outer_verts = _hex_vertices(ocx, ocy, s, oang)
    
    # Check outer vertices are distinct
    for i in range(6):
        for j in range(i+1, 6):
            if np.linalg.norm(outer_verts[i] - outer_verts[j]) < 1e-10:
                return False
    
    # Check inner hexagons have unit edges
    for i in range(12):
        cx, cy, ang = inner[i]
        verts = _hex_vertices(cx, cy, 1.0, ang)
        for k in range(6):
            v1 = verts[k]
            v2 = verts[(k+1)%6]
            edge_len = np.linalg.norm(v2 - v1)
            if abs(edge_len - 1.0) > edge_tol:
                return False
    
    # Check separation between inner hexagons using SAT
    for i in range(12):
        for j in range(i+1, 12):
            v1 = _hex_vertices(inner[i][0], inner[i][1], 1.0, inner[i][2])
            v2 = _hex_vertices(inner[j][0], inner[j][1], 1.0, inner[j][2])
            
            normals = []
            for verts in [v1, v2]:
                for k in range(6):
                    p1 = verts[k]
                    p2 = verts[(k+1)%6]
                    edge = p2 - p1
                    n = np.array([-edge[1], edge[0]])
                    n_norm = np.linalg.norm(n)
                    if n_norm > 1e-12:
                        n = n / n_norm
                        normals.append(n)
            
            separated = False
            for n in normals:
                proj1 = np.dot(v1, n)
                proj2 = np.dot(v2, n)
                min1, max1 = proj1.min(), proj1.max()
                min2, max2 = proj2.min(), proj2.max()
                gap = min(max1, max2) - max(min1, min2)
                if gap > -tol:
                    separated = True
                    break
            
            if not separated:
                return False
    
    # Check inner vertices inside outer hexagon
    for i in range(12):
        verts = _hex_vertices(inner[i][0], inner[i][1], 1.0, inner[i][2])
        for v in verts:
            inside = True
            for k in range(6):
                p1 = outer_verts[k]
                p2 = outer_verts[(k+1)%6]
                edge = p2 - p1
                rel = v - p1
                cross = edge[0] * rel[1] - edge[1] * rel[0]
                if cross < -1e-6:
                    inside = False
                    break
            if not inside:
                return False
    
    return True

def _compute_min_s(inner, outer_angle_deg, outer_center):
    """Compute the minimum outer side length needed to contain all inner hexagons."""
    oang = np.deg2rad(outer_angle_deg)
    
    all_verts = []
    for i in range(12):
        verts = _hex_vertices(inner[i][0], inner[i][1], 1.0, inner[i][2])
        all_verts.append(verts)
    all_verts = np.vstack(all_verts)
    
    # n_k angles in radians
    n_angles = oang + np.deg2rad(30.0) + np.deg2rad(60.0 * np.arange(6))
    normals = np.column_stack([np.cos(n_angles), np.sin(n_angles)])
    
    # For each vertex v, compute max_k dot(v - C, n_k)
    dots = all_verts @ normals.T - outer_center @ normals.T  # shape (72, 6)
    max_dots = dots.max(axis=0)  # shape (6,)
    s_min = (2.0 / math.sqrt(3.0)) * max_dots.max()
    
    return s_min

def _objective(params):
    """Objective: minimize s. Params: 24 center coords + 12 angles + 3 outer params = 39 params."""
    cx_cy = params[:24].reshape(12, 2)
    inner_angles = params[24:36]
    outer_center = params[36:38]
    outer_angle = params[38]
    
    inner = np.zeros((12, 3))
    inner[:, 0] = cx_cy[:, 0]
    inner[:, 1] = cx_cy[:, 1]
    inner[:, 2] = inner_angles
    
    s = _compute_min_s(inner, outer_angle, outer_center)
    
    if not np.isfinite(s) or s < math.sqrt(12) - 1e-9:
        return s + 100.0
    
    outer = np.array([outer_center[0], outer_center[1], outer_angle])
    
    if not _is_valid(inner, outer, s):
        return s + 100.0
    
    return s

def hexagon_packing_12():
    start_time = time.time()
    deadline = 610.0
    
    best_s = float('inf')
    best_inner = None
    best_outer = None
    
    # Initial configuration: hexagonal packing
    r = 2 * math.sqrt(3)  # center-to-center distance for touching
    
    initial_positions = []
    # Center
    initial_positions.append((0, 0))
    # Ring 1: 6 hexagons
    for k in range(6):
        angle = math.pi / 3 * k
        initial_positions.append((r * math.cos(angle), r * math.sin(angle)))
    # Ring 2: 5 more hexagons
    for k in range(0, 12, 2):
        if len(initial_positions) >= 12:
            break
        angle = math.pi / 3 * k + math.pi / 6
        initial_positions.append((2 * r * math.cos(angle), 2 * r * math.sin(angle)))
    
    # Ensure we have exactly 12
    while len(initial_positions) < 12:
        initial_positions.append((0, 0))
    initial_positions = initial_positions[:12]
    
    # Initial params: 24 centers + 12 angles + 3 outer = 39
    initial_params = np.zeros(39)
    for i in range(12):
        initial_params[2*i] = initial_positions[i][0]
        initial_params[2*i+1] = initial_positions[i][1]
    # Angles default to 0, outer center (0,0), outer angle 0
    
    def _update_best(params):
        nonlocal best_s, best_inner, best_outer
        cx_cy = params[:24].reshape(12, 2)
        inner_angles = params[24:36]
        outer_center = params[36:38]
        outer_angle = params[38]
        
        inner = np.zeros((12, 3))
        inner[:, 0] = cx_cy[:, 0]
        inner[:, 1] = cx_cy[:, 1]
        inner[:, 2] = inner_angles
        
        s = _compute_min_s(inner, outer_angle, outer_center)
        outer = np.array([outer_center[0], outer_center[1], outer_angle])
        
        if s < best_s and _is_valid(inner, outer, s):
            best_s = s
            best_inner = inner.copy()
            best_outer = outer.copy()
    
    # Phase 1: Differential Evolution with global bounds
    bounds = [(-8, 8)] * 24  # centers
    bounds += [(-180, 180)] * 12  # inner angles
    bounds += [(-5, 5), (-5, 5)]  # outer center
    bounds += [(-30, 30)]  # outer angle
    
    def cb1(xk, convergence):
        _update_best(xk)
        return time.time() > deadline
    
    try:
        from scipy.optimize import differential_evolution
        res = differential_evolution(
            _objective,
            bounds,
            maxiter=200,
            tol=1e-8,
            mut=(0.5, 1.5),
            recomb=0.7,
            seed=42,
            polish=False,
            callback=cb1
        )
        _update_best(res.x)
    except Exception as e:
        print(f"DE failed: {e}")
    
    # Phase 2: Local refinement with Nelder-Mead
    if best_inner is not None and time.time() < deadline - 60:
        best_x = np.concatenate([best_inner[:, :2].flatten(), best_inner[:, 2],
                                  best_outer[:2], [best_outer[2]]])
        try:
            from scipy.optimize import minimize
            res2 = minimize(
                _objective,
                best_x,
                method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-11, 'fatol': 1e-11, 'adaptive': True}
            )
            _update_best(res2.x)
        except Exception as e:
            print(f"Nelder-Mead failed: {e}")
    
    # Phase 3: Second DE with tighter bounds around best
    if best_inner is not None and time.time() < deadline - 40:
        best_x = np.concatenate([best_inner[:, :2].flatten(), best_inner[:, 2],
                                  best_outer[:2], [best_outer[2]]])
        tight_bounds = []
        for i in range(24):
            tight_bounds.append((best_x[i] - 0.5, best_x[i] + 0.5))
        for i in range(12):
            tight_bounds.append((best_x[24+i] - 30, best_x[24+i] + 30))
        tight_bounds.append((best_x[36] - 1.0, best_x[36] + 1.0))
        tight_bounds.append((best_x[37] - 1.0, best_x[37] + 1.0))
        tight_bounds.append((best_x[38] - 15, best_x[38] + 15))
        
        def cb3(xk, convergence):
            _update_best(xk)
            return time.time() > deadline
        
        try:
            from scipy.optimize import differential_evolution
            res3 = differential_evolution(
                _objective, tight_bounds,
                maxiter=100, tol=1e-10,
                mut=(0.3, 1.0), recomb=0.8,
                seed=123, polish=False,
                callback=cb3
            )
            _update_best(res3.x)
        except Exception as e:
            print(f"Refined DE failed: {e}")
    
    # Phase 4: Final local refinement
    if best_inner is not None and time.time() < deadline - 15:
        best_x = np.concatenate([best_inner[:, :2].flatten(), best_inner[:, 2],
                                  best_outer[:2], [best_outer[2]]])
        try:
            from scipy.optimize import minimize
            res4 = minimize(
                _objective, best_x,
                method='Nelder-Mead',
                options={'maxiter': 3000, 'xatol': 1e-13, 'fatol': 1e-13, 'adaptive': True}
            )
            _update_best(res4.x)
        except Exception as e:
            print(f"Final Nelder-Mead failed: {e}")
    
    # Fallback
    if best_inner is None:
        inner = np.array([
            [0, 0, 0],
            [-2.5, 0, 0], [2.5, 0, 0],
            [-1.25, 2.165, 0], [1.25, 2.165, 0],
            [-1.25, -2.165, 0], [1.25, -2.165, 0],
            [-3.75, 2.165, 0], [3.75, 2.165, 0],
            [-3.75, -2.165, 0], [3.75, -2.165, 0],
            [0, -4, 0],
        ])
        outer = np.array([0, 0, 0])
        s = 8.0
        return inner, outer, s
    
    return best_inner, best_outer, best_s

# EVOLVE-BLOCK-END
