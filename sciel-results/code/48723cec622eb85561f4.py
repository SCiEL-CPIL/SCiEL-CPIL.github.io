import numpy as np
import time
from scipy.optimize import minimize

N = 12
DEADLINE = 145.0

def _hex_vertices(cx, cy, side=1.0, rot_deg=0.0):
    angles = np.radians(np.array([0, 60, 120, 180, 240, 300])) + np.radians(rot_deg)
    x = cx + side * np.cos(angles)
    y = cy + side * np.sin(angles)
    return np.column_stack((x, y))

def _sat_overlap(v1, v2):
    """
    Returns penetration depth if overlapping (negative), or separation distance if not.
    Uses Separating Axis Theorem for convex polygons.
    """
    axes = []
    for v in (v1, v2):
        for i in range(len(v)):
            p1 = v[i]
            p2 = v[(i+1) % len(v)]
            e = p2 - p1
            n = np.array([-e[1], e[0]])
            norm = np.linalg.norm(n)
            if norm > 1e-12:
                n = n / norm
                axes.append(n)
    
    min_overlap = float('inf')
    for n in axes:
        proj1 = v1 @ n
        min1, max1 = np.min(proj1), np.max(proj1)
        proj2 = v2 @ n
        min2, max2 = np.min(proj2), np.max(proj2)
        
        if max1 < min2 or max2 < min1:
            gap = min2 - max1 if min2 > max1 else min1 - max2
            return gap  # positive = separated
        
        overlap = min(max1, max2) - max(min1, min2)
        min_overlap = min(min_overlap, overlap)
    
    return -min_overlap  # negative = overlapping

def _outer_hex_side_fast(points, outer_rot_deg):
    angles = np.radians(np.array([30, 90, 150, 210, 270, 330]) + outer_rot_deg)
    normals = np.column_stack((np.cos(angles), np.sin(angles)))
    proj = points @ normals.T
    max_proj = np.max(np.abs(proj))
    return (2.0 / np.sqrt(3.0)) * max_proj

def _check_disjoint(inner, tol=1e-9):
    n = len(inner)
    for i in range(n):
        for j in range(i + 1, n):
            v1 = _hex_vertices(inner[i, 0], inner[i, 1], 1.0, inner[i, 2])
            v2 = _hex_vertices(inner[j, 0], inner[j, 1], 1.0, inner[j, 2])
            dist = _sat_overlap(v1, v2)
            if dist < -tol:
                return False
    return True

def _make_hex_grid(rows, cols, rot_offset=0.0, gap=0.0):
    pts = []
    sqrt3 = np.sqrt(3)
    spacing_x = 2.0 + gap
    spacing_y = sqrt3 + gap * 0.5
    for r in range(rows):
        y = r * spacing_y
        for c in range(cols):
            x = c * spacing_x
            if r % 2 == 1:
                x += spacing_x / 2.0
            pts.append([x, y, rot_offset])
    pts = np.array(pts)
    pts[:, 0] -= np.mean(pts[:, 0])
    pts[:, 1] -= np.mean(pts[:, 1])
    return pts

def _make_ring(gap=0.0):
    pts = []
    d1 = np.sqrt(3) + gap
    d2 = 2 * np.sqrt(3) + gap
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

def _make_packed_cluster():
    pts = []
    sqrt3 = np.sqrt(3)
    pts.append([0, 0, 0.0])
    for k in range(6):
        ang = np.radians(60 * k)
        pts.append([sqrt3 * np.cos(ang), sqrt3 * np.sin(ang), 0.0])
    for k in range(5):
        ang = np.radians(60 * k + 30)
        pts.append([2 * sqrt3 * np.cos(ang), 2 * sqrt3 * np.sin(ang), 0.0])
    pts = np.array(pts)
    pts[:, 0] -= np.mean(pts[:, 0])
    pts[:, 1] -= np.mean(pts[:, 1])
    return pts

def _smooth_overlap_penalty(params, beta=30.0):
    """
    Compute smooth overlap penalty using SAT-based distances.
    """
    xys_rots = params[:3*N].reshape(N, 3)
    centers = xys_rots[:, :2]
    
    vertices_list = []
    for i in range(N):
        cx, cy, rot = xys_rots[i]
        v = _hex_vertices(cx, cy, 1.0, rot)
        vertices_list.append(v)
    
    penetrations = []
    for i in range(N):
        for j in range(i+1, N):
            d = np.linalg.norm(centers[i] - centers[j])
            if d < 2.5:
                d_sat = _sat_overlap(vertices_list[i], vertices_list[j])
                if d_sat < 0:
                    penetrations.append(-d_sat)
    
    if len(penetrations) == 0:
        return 0.0
    
    p_arr = np.array(penetrations)
    scaled = 2.0 * p_arr / beta
    max_s = np.max(scaled)
    if max_s > 500:
        return np.sum(p_arr**2)
    lse = max_s + np.log(np.sum(np.exp(scaled - max_s)))
    return beta * lse

def _objective(params, t0, deadline):
    """
    Objective: outer side length + smooth overlap penalty.
    params: [x0, y0, a0, ..., x11, y11, a11, oang]
    """
    if time.time() - t0 > deadline:
        return 1e6
    
    xys_rots = params[:3*N].reshape(N, 3)
    outer_rot = params[3*N]
    
    penalty = _smooth_overlap_penalty(params, beta=30.0)
    
    all_points = np.vstack([_hex_vertices(x, y, 1.0, r) for x, y, r in xys_rots])
    L = _outer_hex_side_fast(all_points, outer_rot)
    
    return L + 100.0 * penalty

def _sequential_displacement(pts, max_iters=150, step_factor=0.5, t0=None, deadline=None):
    """
    Sequentially resolve overlaps by pushing hexagons apart.
    """
    pts = pts.copy()
    for it in range(max_iters):
        if t0 is not None and time.time() - t0 > deadline:
            break
        
        moved = False
        for i in range(12):
            for j in range(i + 1, 12):
                v1 = _hex_vertices(pts[i, 0], pts[i, 1], 1.0, pts[i, 2])
                v2 = _hex_vertices(pts[j, 0], pts[j, 1], 1.0, pts[j, 2])
                
                # Find minimum penetration axis
                axes = []
                for v in (v1, v2):
                    for k in range(len(v)):
                        p1 = v[k]
                        p2 = v[(k+1) % len(v)]
                        e = p2 - p1
                        n = np.array([-e[1], e[0]])
                        norm = np.linalg.norm(n)
                        if norm > 1e-12:
                            n = n / norm
                            axes.append(n)
                
                min_overlap = float('inf')
                best_axis = None
                for n in axes:
                    proj1 = v1 @ n
                    min1, max1 = np.min(proj1), np.max(proj1)
                    proj2 = v2 @ n
                    min2, max2 = np.min(proj2), np.max(proj2)
                    
                    if max1 < min2 or max2 < min1:
                        continue
                    
                    overlap = min(max1, max2) - max(min1, min2)
                    if overlap < min_overlap:
                        min_overlap = overlap
                        best_axis = n
                
                if best_axis is not None and min_overlap > 1e-12:
                    push = (min_overlap * step_factor) / 2.0
                    # Determine push direction: from i to j or j to i
                    d_center = np.array([pts[j, 0] - pts[i, 0], pts[j, 1] - pts[i, 1]])
                    if np.dot(d_center, best_axis) > 0:
                        pts[i, 0] -= push * best_axis[0]
                        pts[i, 1] -= push * best_axis[1]
                        pts[j, 0] += push * best_axis[0]
                        pts[j, 1] += push * best_axis[1]
                    else:
                        pts[i, 0] += push * best_axis[0]
                        pts[i, 1] += push * best_axis[1]
                        pts[j, 0] -= push * best_axis[0]
                        pts[j, 1] -= push * best_axis[1]
                    moved = True
        
        if not moved:
            break
    
    return pts

def construct():
    t0 = time.time()
    
    def time_left():
        return DEADLINE - (time.time() - t0)
    
    # Generate diverse initial configurations
    configs = [
        _make_hex_grid(4, 3, gap=0.1),
        _make_hex_grid(3, 4, gap=0.1),
        _make_hex_grid(2, 6, gap=0.1),
        _make_hex_grid(6, 2, gap=0.1),
        _make_ring(gap=0.1),
        _make_hex_grid(4, 3, rot_offset=30, gap=0.1),
        _make_hex_grid(3, 4, rot_offset=30, gap=0.1),
        _make_packed_cluster(),
    ]
    
    np.random.seed(42)
    for _ in range(8):
        pts = np.random.randn(12, 3) * 2.5
        pts[:, 2] = np.random.uniform(0, 60, 12)
        pts[:, 0] -= np.mean(pts[:, 0])
        pts[:, 1] -= np.mean(pts[:, 1])
        configs.append(pts)
    
    best_L = float('inf')
    best_params = None
    
    outer_rots = [0.0, 30.0, 15.0, 45.0, 22.5, 37.5]
    
    for cfg in configs:
        if time_left() < 12:
            break
        
        # Phase 1: Sequential displacement to resolve overlaps
        resolved = _sequential_displacement(cfg, max_iters=100, step_factor=0.4, t0=t0, deadline=DEADLINE - 5)
        
        # Center the configuration
        resolved[:, 0] -= np.mean(resolved[:, 0])
        resolved[:, 1] -= np.mean(resolved[:, 1])
        
        # Check if resolved config is valid
        if _check_disjoint(resolved):
            for oang in outer_rots[:2]:
                all_pts = np.vstack([_hex_vertices(resolved[i,0], resolved[i,1], 1.0, resolved[i,2]) for i in range(12)])
                L = _outer_hex_side_fast(all_pts, oang)
                if L < best_L:
                    best_L = L
                    best_params = np.concatenate((resolved.flatten(), [oang]))
        
        # Phase 2: L-BFGS-B optimization with smooth penalty
        for oang in outer_rots[:4]:
            if time_left() < 8:
                break
            
            init_params = np.concatenate((resolved.flatten(), [oang]))
            
            try:
                res = minimize(
                    _objective,
                    init_params,
                    args=(t0, DEADLINE - 3),
                    method='L-BFGS-B',
                    options={
                        'maxiter': 150,
                        'ftol': 1e-9,
                        'gtol': 1e-6,
                        'maxfun': 300
                    }
                )
                
                if res.fun < 1e5:
                    inner = res.x[:36].reshape(12, 3)
                    oang_res = res.x[36]
                    if _check_disjoint(inner):
                        all_pts = np.vstack([_hex_vertices(inner[i,0], inner[i,1], 1.0, inner[i,2]) for i in range(12)])
                        L = _outer_hex_side_fast(all_pts, oang_res)
                        
                        if L < best_L:
                            best_L = L
                            best_params = res.x.copy()
            except Exception:
                pass
        
        # Phase 3: Nelder-Mead refinement from best L-BFGS-B result
        if best_params is not None and time_left() > 10:
            try:
                res_nm = minimize(
                    _objective,
                    best_params,
                    args=(t0, DEADLINE - 3),
                    method='Nelder-Mead',
                    options={
                        'maxiter': 200,
                        'xatol': 1e-8,
                        'fatol': 1e-8,
                        'adaptive': True
                    }
                )
                
                if res_nm.fun < 1e5:
                    inner = res_nm.x[:36].reshape(12, 3)
                    oang_res = res_nm.x[36]
                    if _check_disjoint(inner):
                        all_pts = np.vstack([_hex_vertices(inner[i,0], inner[i,1], 1.0, inner[i,2]) for i in range(12)])
                        L = _outer_hex_side_fast(all_pts, oang_res)
                        
                        if L < best_L:
                            best_L = L
                            best_params = res_nm.x.copy()
            except Exception:
                pass
    
    # Final refinement pass on best solution
    if best_params is not None and time_left() > 8:
        for _ in range(3):
            if time_left() < 5:
                break
            try:
                res_final = minimize(
                    _objective,
                    best_params,
                    args=(t0, DEADLINE - 2),
                    method='L-BFGS-B',
                    options={
                        'maxiter': 100,
                        'ftol': 1e-10,
                        'gtol': 1e-7,
                        'maxfun': 200
                    }
                )
                
                if res_final.fun < 1e5:
                    inner = res_final.x[:36].reshape(12, 3)
                    oang_res = res_final.x[36]
                    if _check_disjoint(inner):
                        all_pts = np.vstack([_hex_vertices(inner[i,0], inner[i,1], 1.0, inner[i,2]) for i in range(12)])
                        L = _outer_hex_side_fast(all_pts, oang_res)
                        
                        if L < best_L:
                            best_L = L
                            best_params = res_final.x.copy()
            except Exception:
                break
    
    # Fallback
    if best_params is None:
        cfg = _make_hex_grid(4, 3, gap=0.5)
        best_params = np.concatenate((cfg.flatten(), [0.0]))
        inner = [[float(p[0]), float(p[1]), float(p[2])] for p in cfg]
        outer_center = (0.0, 0.0)
        outer_angle = 0.0
        all_pts = np.vstack([_hex_vertices(p[0], p[1], 1.0, p[2]) for p in inner])
        outer_side = _outer_hex_side_fast(all_pts, outer_angle)
        return inner, outer_center, outer_side, outer_angle
    
    # Extract final solution
    inner = []
    for i in range(12):
        x = best_params[i * 3]
        y = best_params[i * 3 + 1]
        r = best_params[i * 3 + 2]
        inner.append([float(x), float(y), float(r)])
    
    outer_center = (0.0, 0.0)
    outer_angle = float(best_params[36])
    
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
    outer_side *= 1.0001
    
    return inner, outer_center, outer_side, outer_angle
