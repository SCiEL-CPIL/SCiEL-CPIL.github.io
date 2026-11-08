import numpy as np
from scipy.optimize import minimize
import time
import math

N = 12

def _hex_verts(cx, cy, side, angle_deg):
    """Vertices of a regular hexagon centered at (cx, cy) with given side and rotation."""
    a = np.radians(angle_deg)
    verts = np.empty((6, 2))
    for i in range(6):
        theta = a + np.radians(60 * i)
        verts[i, 0] = cx + side * np.cos(theta)
        verts[i, 1] = cy + side * np.sin(theta)
    return verts

def _containment_penalty(inner_verts, outer_verts):
    """Sum of squared distances of inner vertices outside the outer hexagon."""
    penalty = 0.0
    n_out = len(outer_verts)
    for iv in inner_verts:
        for i in range(n_out):
            v1 = outer_verts[i]
            v2 = outer_verts[(i + 1) % n_out]
            edge = v2 - v1
            edge_len_sq = np.dot(edge, edge)
            if edge_len_sq < 1e-12:
                continue
            edge_len = math.sqrt(edge_len_sq)
            # Normal vector pointing inwards
            # Edge direction is v2-v1. Inward normal is 90 deg clockwise if CCW polygon?
            # Regular hexagon vertices are CCW.
            # Edge vector E. Normal N = (E_y, -E_x) points Inwards?
            # Let's check: v1=(1,0), v2=(0,1). Edge = (-1, 1). Normal = (1, 1).
            # Midpoint of edge is (0.5, 0.5). Center is (0,0).
            # Vector from mid to center is (-0.5, -0.5).
            # Normal (1,1) points away from center. So (-1, -1) is inward.
            # So inward normal is (edge_y, -edge_x) normalized?
            # Let's just use distance to line and check sign.
            rel = iv - v1
            cross = edge[0] * rel[1] - edge[1] * rel[0]
            # For CCW polygon, cross > 0 means point is to the left of edge (inside).
            # If cross < 0, point is outside.
            if cross < 0:
                dist = -cross / edge_len
                penalty += dist * dist
    return penalty

def _sat_overlap_depth(h1, h2):
    """Minimum penetration depth between two convex polygons using SAT. Returns 0 if disjoint."""
    min_depth = float('inf')
    axes = []
    for poly in [h1, h2]:
        n = len(poly)
        for i in range(n):
            p1 = poly[i]
            p2 = poly[(i + 1) % n]
            edge = p2 - p1
            norm = math.hypot(edge[0], edge[1])
            if norm < 1e-10:
                continue
            # Normal
            normal = np.array([-edge[1], edge[0]]) / norm
            axes.append(normal)
    
    for axis in axes:
        proj1 = np.dot(h1, axis)
        proj2 = np.dot(h2, axis)
        min1, max1 = np.min(proj1), np.max(proj1)
        min2, max2 = np.min(proj2), np.max(proj2)
        
        if max1 <= min2 + 1e-9 or max2 <= min1 + 1e-9:
            return 0.0
        
        overlap = min(max1, max2) - max(min1, min2)
        if overlap < min_depth:
            min_depth = overlap
    
    return min_depth

def _total_penalty(params, outer_side, outer_angle):
    """Total penalty: containment + disjointness."""
    # params: 2N floats (x1, y1, ..., xN, yN). Rotations are fixed to 0.
    inner_verts_list = []
    for i in range(N):
        cx = params[2*i]
        cy = params[2*i+1]
        inner_verts_list.append(_hex_verts(cx, cy, 1.0, 0.0))
    
    outer_verts = _hex_verts(0.0, 0.0, outer_side, outer_angle)
    
    cont_pen = 0.0
    for iv in inner_verts_list:
        cont_pen += _containment_penalty(iv, outer_verts)
    
    disj_pen = 0.0
    for i in range(N):
        for j in range(i + 1, N):
            depth = _sat_overlap_depth(inner_verts_list[i], inner_verts_list[j])
            if depth > 0:
                disj_pen += depth * depth
    
    return cont_pen + disj_pen

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    best_score = float('inf')
    best_result = None
    
    # Initial configurations
    def build_hexagonal_lattice(nrows, ncols):
        positions = []
        for r in range(nrows):
            for c in range(ncols):
                x = c * 2.0 + (1.0 if r % 2 == 1 else 0.0)
                y = r * math.sqrt(3)
                positions.append((x, y))
        return positions

    configs = []
    
    # 3x4 lattice
    pos_3x4 = build_hexagonal_lattice(3, 4)
    if len(pos_3x4) == N:
        arr = np.array(pos_3x4)
        centroid = arr.mean(axis=0)
        configs.append((arr - centroid).flatten().tolist())
        
    # 2x6 lattice
    pos_2x6 = build_hexagonal_lattice(2, 6)
    if len(pos_2x6) == N:
        arr = np.array(pos_2x6)
        centroid = arr.mean(axis=0)
        configs.append((arr - centroid).flatten().tolist())
        
    # 4x3 lattice
    pos_4x3 = build_hexagonal_lattice(4, 3)
    if len(pos_4x3) == N:
        arr = np.array(pos_4x3)
        centroid = arr.mean(axis=0)
        configs.append((arr - centroid).flatten().tolist())
        
    # Ring config: 1 center, 6 ring, 5 extra
    # This is hard to generate automatically. Let's use a cluster.
    # Compact cluster:
    # Center (0,0)
    # Ring 1: 6 hexes at distance 2
    # Ring 2: We need 5 more.
    # Let's just use random restarts if the lattice fails.
    
    # Add a "flower" + 5 random close to it
    center = np.array([0.0, 0.0])
    ring1 = []
    for i in range(6):
        ang = math.radians(60 * i)
        ring1.append((2.0 * math.cos(ang), 2.0 * math.sin(ang)))
    # 5 points from second ring (distance approx 3.46)
    ring2 = []
    for i in range(5):
        ang = math.radians(60 * i + 30) # Offset to fit in gaps
        ring2.append((3.5 * math.cos(ang), 3.5 * math.sin(ang)))
    cluster_pos = [center] + ring1 + ring2
    arr = np.array(cluster_pos)
    centroid = arr.mean(axis=0)
    configs.append((arr - centroid).flatten().tolist())
    
    # Random configs
    np.random.seed(42)
    for _ in range(5):
        rand_pos = np.random.uniform(-3, 3, (N, 2))
        configs.append(rand_pos.flatten().tolist())
    
    outer_angles = [0.0, 30.0]
    
    for outer_angle in outer_angles:
        for init_params_list in configs:
            if time.time() > deadline - 10.0:
                break
                
            init_params = np.array(init_params_list)
            
            # Start with a large side length
            current_side = 5.0
            current_params = init_params.copy()
            
            # Optimize to fit in current_side
            def objective(p):
                return _total_penalty(p, current_side, outer_angle)
            
            try:
                result = minimize(
                    objective,
                    current_params,
                    method='L-BFGS-B',
                    options={'maxiter': 50, 'maxfun': 500, 'ftol': 1e-10, 'gtol': 1e-8}
                )
                if result.fun < 1e-5:
                    current_params = result.x
                    
                    # Binary search for minimum side length
                    low = 3.0
                    high = current_side
                    
                    for _ in range(10): # 10 refinement steps
                        if time.time() > deadline - 5.0:
                            break
                        mid = (low + high) / 2.0
                        if mid <= low + 1e-4:
                            break
                        
                        def shrink_obj(p, s=mid):
                            return _total_penalty(p, s, outer_angle)
                        
                        try:
                            shrink_result = minimize(
                                shrink_obj,
                                current_params,
                                method='L-BFGS-B',
                                options={'maxiter': 50, 'maxfun': 500, 'ftol': 1e-10, 'gtol': 1e-8}
                            )
                            if shrink_result.fun < 1e-5:
                                high = mid
                                current_params = shrink_result.x
                                current_side = mid
                            else:
                                low = mid
                        except Exception:
                            low = mid
                    
                    if current_side < best_score:
                        best_score = current_side
                        best_result = (current_params, (0.0, 0.0), current_side, outer_angle)
                        
            except Exception:
                pass
    
    # Fallback
    if best_result is None:
        # Use the first config with a large side
        inner = []
        for i in range(N):
            inner.append([float(configs[0][2*i]), float(configs[0][2*i+1]), 0.0])
        return inner, (0.0, 0.0), 6.0, 0.0
    
    opt_params, outer_center, outer_side, outer_angle = best_result
    
    inner = []
    for i in range(N):
        x = float(opt_params[2*i])
        y = float(opt_params[2*i+1])
        inner.append([x, y, 0.0])
    
    # Final validation
    inner_verts_list = []
    for i in range(N):
        cx, cy = opt_params[2*i], opt_params[2*i+1]
        inner_verts_list.append(_hex_verts(cx, cy, 1.0, 0.0))
    outer_verts = _hex_verts(outer_center[0], outer_center[1], outer_side, outer_angle)
    
    cont_pen = sum(_containment_penalty(iv, outer_verts) for iv in inner_verts_list)
    disj_pen = 0.0
    for i in range(N):
        for j in range(i+1, N):
            d = _sat_overlap_depth(inner_verts_list[i], inner_verts_list[j])
            if d > 0:
                disj_pen += d * d
    
    if cont_pen > 1e-3 or disj_pen > 1e-3:
        # Return fallback if invalid
        inner = []
        for i in range(N):
            inner.append([float(configs[0][2*i]), float(configs[0][2*i+1]), 0.0])
        return inner, (0.0, 0.0), 6.0, 0.0
    
    return inner, (float(outer_center[0]), float(outer_center[1])), float(outer_side), float(outer_angle)
