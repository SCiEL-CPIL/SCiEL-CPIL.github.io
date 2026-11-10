# EVOLVE-BLOCK-START
import numpy as np
import time
from math import sqrt, cos, sin, pi
from scipy.optimize import minimize

def hexagon_vertices(cx, cy, d, a_deg):
    a_rad = np.radians(a_deg)
    angles = a_rad + np.arange(6) * np.pi / 3
    return np.array([[cx + d * cos(angles[i]), cy + d * sin(angles[i])] for i in range(6)])

def check_validity(inner_hex_data, outer_hex_data, s):
    n = 12
    if inner_hex_data.shape != (n, 3) or outer_hex_data.shape != (3,):
        return False, "Wrong shape"
    if not np.all(np.isfinite(inner_hex_data)) or not np.all(np.isfinite(outer_hex_data)):
        return False, "Non-finite"
    if s < sqrt(12):
        return False, "s too small"
    
    ox, oy, oa = outer_hex_data
    outer_verts = hexagon_vertices(ox, oy, s, oa)
    
    # Check distinct outer vertices
    if len(set(map(tuple, np.round(outer_verts, 10)))) < 6:
        return False, "Outer vertices not distinct"
    
    inner_verts = []
    for i in range(n):
        ix, iy, ia = inner_hex_data[i]
        verts = hexagon_vertices(ix, iy, 1, ia)
        inner_verts.append(verts)
    
    # Check separation between inner hexagons using SAT
    def get_axes(verts):
        axes = []
        for i in range(len(verts)):
            p1 = verts[i]
            p2 = verts[(i + 1) % len(verts)]
            edge = p2 - p1
            normal = np.array([-edge[1], edge[0]])
            normal = normal / np.linalg.norm(normal)
            axes.append(normal)
        return axes
    
    def project(verts, axis):
        return np.dot(verts, axis)
    
    axes_pool = []
    for i in range(n):
        axes_pool.extend(get_axes(inner_verts[i]))
    
    for i in range(n):
        for j in range(i + 1, n):
            sep = False
            for axis in axes_pool:
                proj_i = project(inner_verts[i], axis)
                proj_j = project(inner_verts[j], axis)
                min_i, max_i = np.min(proj_i), np.max(proj_i)
                min_j, max_j = np.min(proj_j), np.max(proj_j)
                if max_i + 1e-6 < min_j or max_j + 1e-6 < min_i:
                    sep = True
                    break
            if not sep:
                return False, f"Inner {i} and {j} overlap"
    
    # Check inner vertices inside outer hexagon
    for i in range(n):
        for v in inner_verts[i]:
            inside = True
            for k in range(6):
                p1 = outer_verts[k]
                p2 = outer_verts[(k + 1) % 6]
                e = p2 - p1
                pv = v - p1
                cross = e[0] * pv[1] - e[1] * pv[0]
                if cross < -1e-6:
                    inside = False
                    break
            if not inside:
                return False, f"Vertex of inner {i} outside outer"
    
    return True, ""

def compute_min_s(inner_hex_data, outer_center, outer_angle):
    """Compute minimum s such that all inner vertices are inside the outer hexagon."""
    ox, oy = outer_center
    oa = outer_angle
    oa_rad = np.radians(oa)
    
    # Outer hexagon edges: for each edge, compute the constraint on s
    # A vertex v is inside if for all edges (p1, p2): cross(p2-p1, v-p1) >= 0
    # The edge p1 = (ox + s*cos(a), oy + s*sin(a)), p2 = (ox + s*cos(a+60), oy + s*sin(a+60))
    # cross(p2-p1, v-p1) = cross(s*(cos(a+60)-cos(a), sin(a+60)-sin(a)), v - (ox+s*cos(a), oy+s*sin(a)))
    # This is linear in s for each edge.
    
    all_verts = []
    for i in range(12):
        ix, iy, ia = inner_hex_data[i]
        verts = hexagon_vertices(ix, iy, 1, ia)
        all_verts.extend(verts)
    all_verts = np.array(all_verts)
    
    min_s = sqrt(12)
    for v in all_verts:
        for k in range(6):
            a1 = oa_rad + k * pi / 3
            a2 = oa_rad + (k + 1) * pi / 3
            # p1 = (ox + s*cos(a1), oy + s*sin(a1))
            # p2 = (ox + s*cos(a2), oy + s*sin(a2))
            # edge = s * (cos(a2)-cos(a1), sin(a2)-sin(a1))
            # v - p1 = (vx - ox - s*cos(a1), vy - oy - s*sin(a1))
            # cross(edge, v-p1) = s*(cos(a2)-cos(a1))*(vy-oy-s*sin(a1)) - s*(sin(a2)-sin(a1))*(vx-ox-s*cos(a1))
            # = s * [(cos(a2)-cos(a1))*(vy-oy) - (sin(a2)-sin(a1))*(vx-ox)]
            #   - s^2 * [(cos(a2)-cos(a1))*sin(a1) - (sin(a2)-sin(a1))*cos(a1)]
            # = s * A - s^2 * B
            # where A = (cos(a2)-cos(a1))*(vy-oy) - (sin(a2)-sin(a1))*(vx-ox)
            #       B = (cos(a2)-cos(a1))*sin(a1) - (sin(a2)-sin(a1))*cos(a1)
            # We need s*A - s^2*B >= 0, i.e., s*(A - s*B) >= 0
            # If B > 0: s <= A/B
            # If B < 0: s >= A/B (but s > 0, so this is always satisfied if A/B < 0)
            # If B == 0: need A >= 0 (independent of s)
            
            dx = cos(a2) - cos(a1)
            dy = sin(a2) - sin(a1)
            vx_ox = v[0] - ox
            vy_oy = v[1] - oy
            A = dx * vy_oy - dy * vx_ox
            B = dx * sin(a1) - dy * cos(a1)
            
            if abs(B) < 1e-12:
                if A < -1e-9:
                    return np.inf
            else:
                ratio = A / B
                if B > 1e-12:
                    # s <= ratio
                    if ratio < min_s:
                        min_s = ratio
                else:
                    # B < 0: s >= ratio. If ratio > 0, need s >= ratio
                    if ratio > min_s:
                        min_s = ratio
    
    return min_s

def initial_packing():
    """Create initial honeycomb-like packing of 12 hexagons."""
    centers = []
    # Center
    centers.append([0, 0])
    # Ring 1: 6 hexagons at distance sqrt(3)
    d = sqrt(3)
    for k in range(6):
        a = pi / 3 * k
        centers.append([d * cos(a), d * sin(a)])
    # Ring 2: 5 more hexagons
    # Place them at the "corners" of the ring 2 positions
    # Ring 2 positions are at distance 2*sqrt(3) from center
    r2 = 2 * sqrt(3)
    # Choose 5 out of 6 positions in ring 2
    for k in range(5):
        a = pi / 6 + pi / 3 * k  # offset by 30 degrees
        centers.append([r2 * cos(a), r2 * sin(a)])
    
    inner = np.column_stack([np.array(centers), np.zeros(12)])
    return inner

def optimize_packing(seed=0, time_limit=10.0):
    """Optimize the packing to minimize s."""
    start_time = time.time()
    np.random.seed(seed)
    
    best_inner = initial_packing()
    best_outer = np.array([0.0, 0.0, 0.0])
    best_s = compute_min_s(best_inner, best_outer[:2], best_outer[2])
    
    if not np.isfinite(best_s) or best_s < sqrt(12):
        # Fallback
        best_s = 10.0
    
    # Try different outer angles
    for oa in [0, 30, 60, 90, 120, 150]:
        if time.time() - start_time > time_limit:
            break
        test_outer = np.array([0.0, 0.0, float(oa)])
        s_test = compute_min_s(best_inner, test_outer[:2], test_outer[2])
        if np.isfinite(s_test) and s_test < best_s:
            best_s = s_test
            best_outer = test_outer
    
    # Optimize positions using scipy
    def objective(params):
        if time.time() - start_time > time_limit:
            return 1e10
        # params: 24 values for 12 centers (x, y each)
        new_inner = best_inner.copy()
        for i in range(12):
            new_inner[i, 0] = params[2*i]
            new_inner[i, 1] = params[2*i+1]
        
        s_val = compute_min_s(new_inner, best_outer[:2], best_outer[2])
        if not np.isfinite(s_val):
            return 1e10
        # Also check separation
        inner_verts = []
        for i in range(12):
            verts = hexagon_vertices(new_inner[i,0], new_inner[i,1], 1, new_inner[i,2])
            inner_verts.append(verts)
        
        def get_axes(verts):
            axes = []
            for i in range(6):
                p1 = verts[i]
                p2 = verts[(i+1)%6]
                edge = p2 - p1
                normal = np.array([-edge[1], edge[0]])
                normal = normal / np.linalg.norm(normal)
                axes.append(normal)
            return axes
        
        axes_pool = []
        for i in range(12):
            axes_pool.extend(get_axes(inner_verts[i]))
        
        for i in range(12):
            for j in range(i+1, 12):
                sep = False
                for axis in axes_pool:
                    proj_i = np.dot(inner_verts[i], axis)
                    proj_j = np.dot(inner_verts[j], axis)
                    if max(proj_i) + 1e-6 < min(proj_j) or max(proj_j) + 1e-6 < min(proj_i):
                        sep = True
                        break
                if not sep:
                    return 1e10 + s_val
        
        return s_val
    
    x0 = best_inner[:, :2].flatten()
    
    result = minimize(objective, x0, method='Nelder-Mead', 
                      options={'maxiter': 5000, 'xatol': 1e-6, 'fatol': 1e-6, 'adaptive': True})
    
    if time.time() - start_time < time_limit and result.fun < best_s and np.isfinite(result.fun):
        new_inner = best_inner.copy()
        for i in range(12):
            new_inner[i, 0] = result.x[2*i]
            new_inner[i, 1] = result.x[2*i+1]
        best_inner = new_inner
        best_s = result.fun
    
    return best_inner, best_outer, best_s

def known_valid_packing():
    """Fallback to a valid loose packing."""
    centers = [
        [0, 0],
        [-3, 0], [3, 0],
        [-1.5, 2.6], [1.5, 2.6],
        [-1.5, -2.6], [1.5, -2.6],
        [-4.5, 2.6], [4.5, 2.6],
        [-4.5, -2.6], [4.5, -2.6],
        [0, -5]
    ]
    inner = np.column_stack([np.array(centers), np.zeros(12)])
    outer = np.array([0.0, 0.0, 0.0])
    s = 10.0
    return inner, outer, s

def hexagon_packing_12():
    deadline = time.time() + 620.0
    best_result = None
    best_score = 0
    
    for seed in range(10):
        if time.time() > deadline:
            break
        try:
            remaining = deadline - time.time()
            if remaining < 5:
                break
            t_limit = min(remaining - 2, 15.0)
            inner, outer, s = optimize_packing(seed, time_limit=t_limit)
            valid, _ = check_validity(inner, outer, s)
            if valid and 1.0/s > best_score:
                best_result = (inner, outer, s)
                best_score = 1.0/s
        except Exception:
            pass
    
    if best_result is None:
        best_result = known_valid_packing()
    
    return best_result
# EVOLVE-BLOCK-END
