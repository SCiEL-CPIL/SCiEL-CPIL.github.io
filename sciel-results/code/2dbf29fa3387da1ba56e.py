"""Packing 12 unit hexagons inside a regular hexagon using Nelder-Mead with binary search."""
import time
import numpy as np
from scipy.optimize import minimize

N = 12
DEADLINE = 145.0  # safety margin before 150s kill


def _hex_verts(cx, cy, s, angle_deg):
    """Return vertices of a regular hexagon centered at (cx,cy), side s, rotated angle_deg."""
    a = np.radians(angle_deg)
    angles = a + np.arange(6) * np.pi / 3.0
    xs = cx + s * np.cos(angles)
    ys = cy + s * np.sin(angles)
    return np.column_stack([xs, ys])


def _hex_overlap_depth(cx1, cy1, a1, cx2, cy2, a2, s1=1.0, s2=1.0):
    """Compute minimum translation distance to separate two unit hexagons.
    Returns 0 if no overlap, positive if overlapping."""
    dist = np.sqrt((cx2-cx1)**2 + (cy2-cy1)**2)
    if dist > 2.0 * (s1 + s2):
        return 0.0
    
    v1 = _hex_verts(cx1, cy1, s1, a1)
    v2 = _hex_verts(cx2, cy2, s2, a2)
    
    def get_normals(verts):
        normals = []
        n = len(verts)
        center = np.mean(verts, axis=0)
        for i in range(n):
            p1 = verts[i]
            p2 = verts[(i+1) % n]
            edge = p2 - p1
            normal = np.array([-edge[1], edge[0]])
            norm = np.linalg.norm(normal)
            if norm < 1e-15:
                continue
            normal = normal / norm
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


def _containment_violation(verts, outer_center, outer_side, outer_angle):
    """Compute max containment violation for a set of vertices against outer hexagon."""
    cx, cy = outer_center
    a = np.radians(outer_angle)
    apothem = outer_side * np.sqrt(3) / 2.0
    normal_angles = a + (np.pi / 6.0) + np.arange(6) * np.pi / 3.0
    
    max_viol = 0.0
    for v in verts:
        dx = v[0] - cx
        dy = v[1] - cy
        for na in normal_angles:
            proj = dx * np.cos(na) + dy * np.sin(na)
            if proj > apothem:
                viol = proj - apothem
                if viol > max_viol:
                    max_viol = viol
                break
    return max_viol


def _objective(params, outer_side):
    """Objective function: sum of squared violations for packing N unit hexagons in outer hexagon."""
    n = N
    # params layout: [x0, y0, a0, x1, y1, a1, ..., x11, y11, a11]
    positions = params.reshape(n, 3)
    
    penalty = 0.0
    
    # Containment penalty
    for i in range(n):
        ix, iy, ia = positions[i]
        verts = _hex_verts(ix, iy, 1.0, ia)
        viol = _containment_violation(verts, (0.0, 0.0), outer_side, 0.0)
        if viol > 0:
            penalty += viol * viol * 100.0
    
    # Disjointness penalty
    for i in range(n):
        for j in range(i+1, n):
            ix1, iy1, ia1 = positions[i]
            ix2, iy2, ia2 = positions[j]
            depth = _hex_overlap_depth(ix1, iy1, ia1, ix2, iy2, ia2)
            if depth > 0:
                penalty += depth * depth * 100.0
    
    return penalty


def _init_hex_grid(rows, cols, spacing_x, spacing_y, start_x=0, start_y=0):
    """Initialize inner hexagons in a hexagonal grid pattern."""
    inner = []
    count = 0
    for r in range(rows):
        for c in range(cols):
            if count >= N:
                break
            x = start_x + c * spacing_x + (r % 2) * spacing_x / 2
            y = start_y + r * spacing_y
            inner.append([x, y, 0.0])
            count += 1
    return inner


def _try_pack(init_positions, outer_side, time_budget=5.0):
    """Try to pack using Nelder-Mead. Returns (success, final_positions, final_penalty)."""
    start_time = time.time()
    
    x0 = np.array(init_positions).flatten()
    
    # Use Nelder-Mead with adaptive options
    result = minimize(
        _objective,
        x0,
        args=(outer_side,),
        method='Nelder-Mead',
        options={
            'maxiter': 500,
            'xatol': 1e-4,
            'fatol': 1e-4,
            'adaptive': True
        }
    )
    
    elapsed = time.time() - start_time
    if elapsed > time_budget:
        return False, x0, float('inf')
    
    final_penalty = result.fun
    final_positions = result.x.reshape(N, 3)
    
    # Check if we got a valid packing
    success = final_penalty < 1e-6
    return success, final_positions, final_penalty


def construct():
    start_time = time.time()
    
    best_side = float('inf')
    best_positions = None
    
    # Initial configurations
    configs = []
    
    # 3x4 grid
    configs.append(_init_hex_grid(3, 4, 2.2, 2.0, -2.2, -2.0))
    # 4x3 grid
    configs.append(_init_hex_grid(4, 3, 2.2, 2.0, -2.2, -3.0))
    # 2x6 grid
    configs.append(_init_hex_grid(2, 6, 2.2, 2.0, -5.5, -1.0))
    # 6x2 grid
    configs.append(_init_hex_grid(6, 2, 2.2, 2.0, -2.2, -5.5))
    # Random
    np.random.seed(42)
    for seed in range(5):
        np.random.seed(seed)
        inner = []
        for i in range(N):
            x = np.random.uniform(-3, 3)
            y = np.random.uniform(-3, 3)
            inner.append([x, y, 0.0])
        configs.append(inner)
    
    # For each config, find a feasible outer side via binary search
    for init_pos in configs:
        if time.time() - start_time > DEADLINE - 10:
            break
        
        # Estimate initial outer side
        all_x = [p[0] for p in init_pos]
        all_y = [p[1] for p in init_pos]
        span_x = max(all_x) - min(all_x) + 2.0
        span_y = max(all_y) - min(all_y) + 2.0
        lo = max(span_x, span_y) / np.sqrt(3) + 1.0
        hi = lo + 3.0
        
        # Binary search for minimum feasible outer side
        for _ in range(15):
            if time.time() - start_time > DEADLINE - 5:
                break
            
            mid = (lo + hi) / 2.0
            success, _, _ = _try_pack(init_pos, mid, time_budget=3.0)
            
            if success:
                hi = mid
            else:
                lo = mid
        
        # Try to find a valid packing at the best side found
        if hi < best_side:
            success, positions, penalty = _try_pack(init_pos, hi, time_budget=5.0)
            if success and penalty < 1e-6:
                best_side = hi
                best_positions = positions
    
    # If no valid solution found, return a conservative valid packing
    if best_positions is None:
        inner = []
        for i in range(N):
            x = (i - (N-1)/2.0) * 2.1
            y = 0.0
            inner.append([x, 0.0, 0.0])
        span = (N-1) * 2.1 + 2.0
        side = span / np.sqrt(3) + 0.5
        return inner, (0.0, 0.0), side, 0.0
    
    # Convert positions to required format
    inner = []
    for i in range(N):
        inner.append([float(best_positions[i, 0]), float(best_positions[i, 1]), float(best_positions[i, 2])])
    
    return inner, (0.0, 0.0), float(best_side), 0.0
