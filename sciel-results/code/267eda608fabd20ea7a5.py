import numpy as np
import math
import time
from scipy.optimize import minimize

N = 12
DEADLINE = 145.0

def _hex_vertices(cx, cy, side=1.0, rot_deg=0.0):
    angles = np.radians(np.array([0, 60, 120, 180, 240, 300])) + np.radians(rot_deg)
    x = cx + side * np.cos(angles)
    y = cy + side * np.sin(angles)
    return np.column_stack((x, y))

def _edge_normals(v):
    """Compute outward normals for each edge of a polygon."""
    n = len(v)
    normals = np.zeros((n, 2))
    for i in range(n):
        p1 = v[i]
        p2 = v[(i + 1) % n]
        e = p2 - p1
        norm = np.linalg.norm(e)
        if norm > 1e-12:
            # Outward normal (right-hand rule: rotate edge vector by -90 degrees)
            normals[i] = np.array([e[1], -e[0]]) / norm
    return normals

def _sat_overlap_fast(v1, v2):
    """
    Returns negative value if overlapping (penetration depth), positive if separated.
    Uses SAT with precomputed normals.
    """
    n1 = len(v1)
    n2 = len(v2)
    normals1 = _edge_normals(v1)
    normals2 = _edge_normals(v2)
    all_normals = np.vstack([normals1, normals2])
    
    min_pen = float('inf')
    for nrm in all_normals:
        proj1 = v1 @ nrm
        min1, max1 = np.min(proj1), np.max(proj1)
        proj2 = v2 @ nrm
        min2, max2 = np.min(proj2), np.max(proj2)
        
        if max1 < min2 or max2 < min1:
            # Separated on this axis
            return max(min2 - max1, min1 - max2)
        
        overlap = min(max1, max2) - max(min1, min2)
        min_pen = min(min_pen, overlap)
    
    return -min_pen

def _outer_hex_side_fast(pts, outer_rot_deg):
    """
    Minimum side length of regular hexagon centered at origin with given rotation
    that encloses all points in pts.
    """
    angles = np.radians(np.array([30, 90, 150, 210, 270, 330]) + outer_rot_deg)
    normals = np.column_stack((np.cos(angles), np.sin(angles)))
    proj = pts @ normals.T
    max_proj = np.max(np.abs(proj))
    return (2.0 / np.sqrt(3.0)) * max_proj

def _make_hex_grid(rows, cols, rot=0.0, gap=0.1):
    pts = []
    spacing_x = 2.0 + gap
    spacing_y = np.sqrt(3) + gap * 0.5
    for r in range(rows):
        y = r * spacing_y
        for c in range(cols):
            x = c * spacing_x
            if r % 2 == 1:
                x += spacing_x / 2.0
            pts.append([x, y, rot])
    pts = np.array(pts)
    pts[:, 0] -= np.mean(pts[:, 0])
    pts[:, 1] -= np.mean(pts[:, 1])
    return pts

def _make_ring(gap=0.1):
    pts = []
    d1 = np.sqrt(3) + gap
    d2 = 2 * np.sqrt(3) + gap * 1.5
    pts.append([0, 0, 0.0])
    for k in range(6):
        ang = np.radians(60 * k)
        pts.append([d1 * np.cos(ang), d1 * np.sin(ang), 0.0])
    for k in range(5):
        ang = np.radians(60 * k + 30)
        pts.append([d2 * np.cos(ang), d2 * np.sin(ang), 0.0])
    pts = np.array(pts)
    pts[:, 0] -= np.mean(pts[:, 0])
    pts[:, 1] -= np.mean(pts[:, 1])
    return pts

def _check_disjoint(inner):
    n = len(inner)
    verts = [_hex_vertices(inner[i][0], inner[i][1], 1.0, inner[i][2]) for i in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            d = _sat_overlap_fast(verts[i], verts[j])
            if d < -1e-9:
                return False
    return True

def construct():
    t0 = time.time()
    
    def time_left():
        return DEADLINE - (time.time() - t0)

    # Precompute rotation angles for inner hexes (0, 30, 60, etc. are equivalent for regular hexes)
    # Actually, for a regular hexagon, rotating by 60 degrees gives the same shape.
    # Rotating by 30 degrees flips "pointy-top" to "flat-top" orientation.
    # We only need to consider 0 and 30 degrees of rotation (plus multiples of 60).
    # But to be general, we allow any angle in the optimizer.
    
    def objective(params):
        if time.time() - t0 > DEADLINE - 5:
            return 1e6
        
        inner_params = params[:36].reshape(12, 3)
        ox = params[36]
        oy = params[37]
        oang = params[38]
        
        verts_list = []
        for i in range(12):
            cx, cy, rot = inner_params[i]
            v = _hex_vertices(cx, cy, 1.0, rot)
            verts_list.append(v)
        
        # Overlap penalty
        penalty = 0.0
        for i in range(12):
            for j in range(i + 1, 12):
                d = _sat_overlap_fast(verts_list[i], verts_list[j])
                if d < 0:
                    penalty += (-d) ** 2 * 100.0
        
        # Calculate outer side needed
        all_pts = np.vstack(verts_list)
        all_pts[:, 0] -= ox
        all_pts[:, 1] -= oy
        
        L = _outer_hex_side_fast(all_pts, oang)
        
        return L + penalty

    # Initial configurations
    configs = []
    configs.append(_make_hex_grid(4, 3, gap=0.2))
    configs.append(_make_hex_grid(3, 4, gap=0.2))
    configs.append(_make_hex_grid(2, 6, gap=0.2))
    configs.append(_make_hex_grid(6, 2, gap=0.2))
    configs.append(_make_ring(gap=0.2))
    
    # Rotated variants
    base43 = _make_hex_grid(4, 3, gap=0.2)
    rotated43 = base43.copy()
    rotated43[:, 2] = 30.0
    configs.append(rotated43)
    
    base34 = _make_hex_grid(3, 4, gap=0.2)
    rotated34 = base34.copy()
    rotated34[:, 2] = 30.0
    configs.append(rotated34)
    
    # Add some random starts
    np.random.seed(42)
    for _ in range(3):
        pts = np.random.randn(12, 3) * 3.0
        pts[:, 2] = np.random.choice([0, 30, 60], size=12)
        pts[:, 0] -= np.mean(pts[:, 0])
        pts[:, 1] -= np.mean(pts[:, 1])
        configs.append(pts)
    
    best_L = float('inf')
    best_params = None
    
    outer_rots = [0.0, 30.0, 15.0]
    
    for cfg in configs:
        if time_left() < 10:
            break
        
        for orot in outer_rots:
            if time_left() < 5:
                break
            
            init_params = np.concatenate((cfg.flatten(), [0.0, 0.0, orot]))
            
            # Quick validity check
            inner_init = init_params[:36].reshape(12, 3)
            verts_init = [_hex_vertices(inner_init[i, 0], inner_init[i, 1], 1.0, inner_init[i, 2]) for i in range(12)]
            valid = True
            for i in range(12):
                for j in range(i + 1, 12):
                    if _sat_overlap_fast(verts_init[i], verts_init[j]) < -1e-9:
                        valid = False
                        break
                if not valid:
                    break
            
            if valid:
                L_val = objective(init_params)
                if L_val < best_L:
                    best_L = L_val
                    best_params = init_params.copy()
            
            # Optimize with Nelder-Mead (robust for non-smooth SAT boundaries)
            try:
                res = minimize(
                    objective,
                    init_params,
                    method='Nelder-Mead',
                    options={
                        'maxiter': 300,
                        'xatol': 1e-5,
                        'fatol': 1e-5,
                        'adaptive': True
                    }
                )
                
                res_inner = res.x[:36].reshape(12, 3)
                res_ox = res.x[36]
                res_oy = res.x[37]
                res_oang = res.x[38]
                
                verts_res = [_hex_vertices(res_inner[i, 0], res_inner[i, 1], 1.0, res_inner[i, 2]) for i in range(12)]
                valid = True
                for i in range(12):
                    for j in range(i + 1, 12):
                        if _sat_overlap_fast(verts_res[i], verts_res[j]) < -1e-9:
                            valid = False
                            break
                    if not valid:
                        break
                
                if valid:
                    all_pts = np.vstack(verts_res)
                    all_pts[:, 0] -= res_ox
                    all_pts[:, 1] -= res_oy
                    L_res = _outer_hex_side_fast(all_pts, res_oang)
                    
                    if L_res < best_L:
                        best_L = L_res
                        best_params = res.x.copy()
                        
            except Exception:
                pass

    if best_params is None:
        cfg = _make_hex_grid(4, 3, gap=0.5)
        best_params = np.concatenate((cfg.flatten(), [0.0, 0.0, 0.0]))
        best_L = objective(best_params)

    # Extract final solution
    inner = []
    for i in range(12):
        x = best_params[i * 3]
        y = best_params[i * 3 + 1]
        r = best_params[i * 3 + 2]
        inner.append([float(x), float(y), float(r)])
    
    outer_center = (float(best_params[36]), float(best_params[37]))
    outer_angle = float(best_params[38])
    
    # Final verification
    if not _check_disjoint(inner):
        cfg = _make_hex_grid(4, 3, gap=0.5)
        inner = [[float(p[0]), float(p[1]), float(p[2])] for p in cfg]
        outer_center = (0.0, 0.0)
        outer_angle = 0.0
        all_pts = np.vstack([_hex_vertices(p[0], p[1], 1.0, p[2]) for p in inner])
        outer_side = _outer_hex_side_fast(all_pts, outer_angle)
        return inner, outer_center, outer_side, outer_angle
    
    all_pts = np.vstack([_hex_vertices(p[0], p[1], 1.0, p[2]) for p in inner])
    all_pts[:, 0] -= outer_center[0]
    all_pts[:, 1] -= outer_center[1]
    outer_side = _outer_hex_side_fast(all_pts, outer_angle)
    
    # Tiny epsilon for numerical safety
    outer_side *= 1.00001
    
    return inner, outer_center, outer_side, outer_angle
