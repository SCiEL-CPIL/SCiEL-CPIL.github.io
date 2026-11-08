"""Improved packing of 12 unit hexagons inside a regular hexagon using binary search + Nelder-Mead."""
import time
import numpy as np
from scipy.optimize import minimize

N = 12
DEADLINE = 145.0  # seconds, safety margin before 150s kill


def _hex_verts(cx, cy, s, angle_deg):
    """Return vertices of a regular hexagon centered at (cx,cy), side s, rotated angle_deg."""
    a = np.radians(angle_deg)
    angles = a + np.arange(6) * np.pi / 3.0
    xs = cx + s * np.cos(angles)
    ys = cy + s * np.sin(angles)
    return np.column_stack([xs, ys])


def _hex_overlap_depth(cx1, cy1, a1, cx2, cy2, a2, s=1.0):
    """Compute minimum translation distance to separate two unit hexagons.
    Returns 0 if no overlap, positive if overlapping."""
    dist = np.sqrt((cx2-cx1)**2 + (cy2-cy1)**2)
    if dist > 4.0:  # max possible overlap distance
        return 0.0
    
    v1 = _hex_verts(cx1, cy1, s, a1)
    v2 = _hex_verts(cx2, cy2, s, a2)
    
    def get_normals(verts):
        normals = []
        n = len(verts)
        for i in range(n):
            p1 = verts[i]
            p2 = verts[(i+1) % n]
            edge = p2 - p1
            normal = np.array([-edge[1], edge[0]])
            norm = np.linalg.norm(normal)
            if norm < 1e-15:
                continue
            normal = normal / norm
            center = np.mean(verts, axis=0)
            if np.dot(normal, center - p1) < 0:
                normal = -normal
            normals.append(normal)
        return np.array(normals)
    
    n1 = get_normals(v1)
    n2 = get_normals(v2)
    all_normals = np.vstack([n1, n2])
    
    min_overlap = float('inf')
    
    for normal in all_normals:
        proj1 = v1 @ normal
        proj2 = v2 @ normal
        
        min1, max1 = proj1.min(), proj1.max()
        min2, max2 = proj2.min(), proj2.max()
        
        if max1 <= min2 or max2 <= min1:
            return 0.0
        
        overlap = min(max1, max2) - max(min1, min2)
        min_overlap = min(min_overlap, overlap)
    
    if min_overlap == float('inf'):
        return 0.0
    
    return min_overlap


def _containment_penalty(verts, outer_side, outer_angle=0.0):
    """Compute how much vertices violate containment in outer hexagon. Returns sum of squared violations."""
    a = np.radians(outer_angle)
    apothem = outer_side * np.sqrt(3) / 2.0
    normal_angles = a + (np.pi / 6.0) + np.arange(6) * np.pi / 3.0
    
    penalty = 0.0
    for v in verts:
        dx = v[0]
        dy = v[1]
        for na in normal_angles:
            proj = dx * np.cos(na) + dy * np.sin(na)
            if proj > apothem:
                viol = proj - apothem
                penalty += viol * viol
                break
    return penalty


def _disjointness_penalty(x, y, angles, inner_verts_list):
    """Compute pairwise overlap penalty between inner hexagons."""
    n = len(inner_verts_list)
    penalty = 0.0
    for i in range(n):
        for j in range(i+1, n):
            depth = _hex_overlap_depth(x[i], y[i], angles[i], x[j], y[j], angles[j])
            if depth > 0:
                penalty += depth * depth
    return penalty


def _objective(params, outer_side):
    """Objective function: total penalty (containment + disjointness)."""
    # params: [x1, y1, a1, x2, y2, a2, ..., xN, yN, aN]
    x = params[0::3]
    y = params[1::3]
    angles = params[2::3]
    
    # Compute all vertices
    inner_verts_list = []
    for i in range(N):
        verts = _hex_verts(x[i], y[i], 1.0, angles[i])
        inner_verts_list.append(verts)
    
    # Containment penalty
    contain_pen = 0.0
    for verts in inner_verts_list:
        contain_pen += _containment_penalty(verts, outer_side)
    
    # Disjointness penalty
    disp_pen = _disjointness_penalty(x, y, angles, inner_verts_list)
    
    return contain_pen + disp_pen


def _params_to_inner(params):
    """Convert parameter vector to inner list format."""
    inner = []
    for i in range(N):
        inner.append([params[3*i], params[3*i+1], params[3*i+2]])
    return inner


def _inner_to_params(inner):
    """Convert inner list to parameter vector."""
    params = []
    for p in inner:
        params.extend([p[0], p[1], p[2]])
    return np.array(params)


def _init_hex_grid(rows, cols, sx, sy, ox=0, oy=0):
    """Create hexagonal grid initialization."""
    inner = []
    count = 0
    for r in range(rows):
        for c in range(cols):
            if count >= N:
                break
            xx = ox + c * sx + (r % 2) * sx / 2
            yy = oy + r * sy
            inner.append([xx, yy, 0.0])
            count += 1
    return inner


def _try_pack(outer_side, init_params, time_budget):
    """Try to pack with given outer_side using Nelder-Mead. Returns (params, penalty) or None."""
    start = time.time()
    
    def obj(p):
        return _objective(p, outer_side)
    
    result = minimize(
        obj, init_params, method='Nelder-Mead',
        options={'maxiter': 5000, 'xatol': 1e-6, 'fatol': 1e-8}
    )
    
    if time.time() - start > time_budget:
        return None
    
    final_pen = result.fun
    if final_pen < 1e-6:
        return result.x, final_pen
    return None


def construct():
    start_time = time.time()
    
    best_side = float('inf')
    best_inner = None
    best_center = (0.0, 0.0)
    best_angle = 0.0
    
    # Initial configurations
    init_configs = []
    
    # Hexagonal grid 3x4
    init_configs.append(_init_hex_grid(3, 4, 2.1, 1.85, -2.1, -1.85))
    # Hexagonal grid 4x3
    init_configs.append(_init_hex_grid(4, 3, 2.1, 1.85, -2.1, -2.7))
    # 2x6 line
    init_configs.append(_init_hex_grid(2, 6, 2.1, 1.85, -5.25, -0.9))
    # 6x2 line
    init_configs.append(_init_hex_grid(6, 2, 2.1, 1.85, -1.05, -5.25))
    
    # Random inits
    for seed in range(8):
        np.random.seed(seed)
        inner = []
        for i in range(N):
            xx = np.random.uniform(-2.5, 2.5)
            yy = np.random.uniform(-2.5, 2.5)
            inner.append([xx, yy, 0.0])
        init_configs.append(inner)
    
    # Binary search on outer side
    lo = 3.0
    hi = 7.0
    
    # First, find a valid packing at hi
    for init_inner in init_configs:
        if time.time() - start_time > DEADLINE - 20:
            break
        init_params = _inner_to_params(init_inner)
        result = _try_pack(hi, init_params, time_budget=15.0)
        if result is not None:
            params, pen = result
            if hi < best_side:
                best_side = hi
                best_inner = _params_to_inner(params)
                best_center = (0.0, 0.0)
                best_angle = 0.0
                break
    
    # If we found a valid packing, binary search for smaller side
    if best_inner is not None:
        for iteration in range(20):
            if time.time() - start_time > DEADLINE - 10:
                break
            
            mid = (lo + hi) / 2.0
            if mid <= best_side:
                break
            
            found = False
            for init_inner in init_configs[:4]:  # Use fewer inits for speed
                if time.time() - start_time > DEADLINE - 10:
                    break
                init_params = _inner_to_params(init_inner)
                result = _try_pack(mid, init_params, time_budget=8.0)
                if result is not None:
                    params, pen = result
                    found = True
                    if mid < best_side:
                        best_side = mid
                        best_inner = _params_to_inner(params)
                        best_center = (0.0, 0.0)
                        best_angle = 0.0
                    break
            
            if found:
                hi = mid
            else:
                lo = mid
    
    # Fallback: conservative valid packing
    if best_inner is None:
        inner = []
        for i in range(N):
            xx = (i - (N-1)/2.0) * 2.1
            yy = 0.0
            inner.append([xx, 0.0, 0.0])
        span = (N-1) * 2.1 + 2.0
        side = span / np.sqrt(3) + 0.5
        return inner, (0.0, 0.0), side, 0.0
    
    return best_inner, best_center, best_side, best_angle
