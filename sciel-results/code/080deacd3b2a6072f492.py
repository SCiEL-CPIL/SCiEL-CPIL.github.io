import math
import time
import numpy as np
from scipy.optimize import linprog, minimize

N = 11
_START_TIME = time.time()
_DEADLINE = _START_TIME + 142.0
_SQRT3 = math.sqrt(3)
_INVSQRT3 = 1.0 / _SQRT3


def _hex_vertices(cx, cy, angle_deg):
    """Vertices of a unit regular hexagon centered at (cx,cy) with rotation angle_deg."""
    a0 = math.radians(angle_deg)
    angles = a0 + np.arange(6) * math.pi / 3.0
    return np.column_stack([cx + np.cos(angles), cy + np.sin(angles)])


def _hex_support_vectorized(angle1_deg, angle2_deg, theta_arr):
    """Vectorized support function for two hexagons at given angles for array of directions."""
    a1 = math.radians(angle1_deg)
    a2 = math.radians(angle2_deg)
    pi3 = math.pi / 3.0
    
    def _support_single(a0, theta):
        diff = (theta - a0) % (2 * math.pi)
        k = np.round(diff / pi3)
        d1 = np.abs(diff - k * pi3)
        d2 = np.abs(diff - (k - 1) * pi3)
        d3 = np.abs(diff - (k + 1) * pi3)
        min_d = np.minimum(np.minimum(d1, d2), d3)
        return np.cos(min_d)
    
    s1 = _support_single(a1, theta_arr)
    s2 = _support_single(a2, theta_arr)
    return s1, s2


def _hex_distance_fast(d, angle1, angle2):
    """Fast minimum distance between two unit hexagons. Returns negative if overlapping."""
    a1 = math.radians(angle1)
    a2 = math.radians(angle2)
    dx, dy = d[0], d[1]
    
    pi3 = math.pi / 3.0
    pi6 = math.pi / 6.0
    
    # All 12 edge normals
    normals1 = a1 + pi6 + np.arange(6) * pi3
    normals2 = a2 + pi6 + np.arange(6) * pi3
    all_normals = np.concatenate([normals1, normals2])
    
    # Compute projections
    cos_n = np.cos(all_normals)
    sin_n = np.sin(all_normals)
    proj = np.abs(dx * cos_n + dy * sin_n)
    
    # Compute supports
    s1, s2 = _hex_support_vectorized(angle1, angle2, all_normals)
    
    sep = proj - s1 - s2
    return float(np.min(sep))





def _enclosing_hex_side(verts, outer_angle_deg):
    """Find minimum side length of regular hexagon with given orientation containing all vertices."""
    base_angle = math.radians(outer_angle_deg + 30)
    angles = base_angle + np.arange(6) * math.pi / 3.0
    normals = np.column_stack([np.cos(angles), np.sin(angles)])
    
    M = len(verts)
    A_rows = []
    b_vals = []
    
    for i in range(M):
        vi = verts[i]
        for k in range(6):
            nk = normals[k]
            A_rows.append([-nk[0], -nk[1], -1.0])
            b_vals.append(-nk[0]*vi[0] - nk[1]*vi[1])
            A_rows.append([nk[0], nk[1], -1.0])
            b_vals.append(nk[0]*vi[0] + nk[1]*vi[1])
    
    A = np.array(A_rows)
    b = np.array(b_vals)
    c_obj = np.array([0.0, 0.0, 1.0])
    bounds = [(None, None), (None, None), (0, None)]
    
    try:
        res = linprog(c_obj, A_ub=A, b_ub=b, bounds=bounds, method='highs')
        if res.status == 0:
            cx, cy, r = res.x
            side = r * 2.0 / _SQRT3
            return side
    except Exception:
        pass
    
    # Fallback: centroid-based
    cx, cy = np.mean(verts, axis=0)
    rel = verts - np.array([cx, cy])
    projections = rel @ normals.T
    max_proj = np.max(projections, axis=0)
    min_proj = np.min(projections, axis=0)
    r = np.max(np.maximum(max_proj, -min_proj))
    if r < 0: r = 0
    side = r * 2.0 / _SQRT3
    return side


def _build_positions(layout_idx):
    """Generate 11 positions for given layout index."""
    sqrt3 = _SQRT3
    scale = 1.001
    u = (sqrt3*scale, 0.0)
    v = (sqrt3/2*scale, 1.5*scale)
    
    if layout_idx == 0:
        # Compact hexagonal: center + 6 ring + 4 from second ring
        target = [(0,0)]
        ring1 = [(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1)]
        target.extend(ring1)
        ring2 = [(2,0),(1,1),(0,2),(-1,2),(-2,1),(-2,0),(-1,-1),(0,-2),(1,-2),(2,-1)]
        ring2_sorted = sorted(ring2, key=lambda p: p[0]**2 + p[1]**2)
        target.extend(ring2_sorted[:4])
        positions = []
        for ti, tj in target:
            x = ti*u[0] + tj*v[0]
            y = ti*u[1] + tj*v[1]
            positions.append((x,y))
        return positions[:N]
    elif layout_idx == 1:
        # Row-based: 3-4-4 pattern
        positions = []
        y1 = -1.5*scale
        for k in range(-1, 2):
            x = k*sqrt3*scale + sqrt3/2*scale
            positions.append((x, y1))
        y2 = 0.0
        for k in range(-1, 3):
            x = k*sqrt3*scale
            positions.append((x, y2))
        y3 = 1.5*scale
        for k in range(-1, 3):
            x = k*sqrt3*scale + sqrt3/2*scale
            positions.append((x, y3))
        return positions[:N]
    elif layout_idx == 2:
        # 4-3-4 pattern
        positions = []
        y1 = -1.5*scale
        for k in range(-2, 2):
            x = k*sqrt3*scale
            positions.append((x, y1))
        y2 = 0.0
        for k in range(-1, 2):
            x = k*sqrt3*scale + sqrt3/2*scale
            positions.append((x, y2))
        y3 = 1.5*scale
        for k in range(-2, 2):
            x = k*sqrt3*scale
            positions.append((x, y3))
        return positions[:N]
    elif layout_idx == 3:
        # 2-4-2-3 pattern
        positions = []
        y1 = -3.0*scale
        for k in range(0, 2):
            x = k*sqrt3*scale - sqrt3/2*scale
            positions.append((x, y1))
        y2 = -1.5*scale
        for k in range(-1, 3):
            x = k*sqrt3*scale
            positions.append((x, y2))
        y3 = 0.0
        for k in range(-1, 1):
            x = k*sqrt3*scale + sqrt3/2*scale
            positions.append((x, y3))
        y4 = 1.5*scale
        for k in range(-1, 2):
            x = k*sqrt3*scale
            positions.append((x, y4))
        return positions[:N]
    elif layout_idx == 4:
        # 3-5-3 pattern
        positions = []
        y1 = -1.5*scale
        for k in range(-1, 2):
            x = k*sqrt3*scale + sqrt3/2*scale
            positions.append((x, y1))
        y2 = 0.0
        for k in range(-2, 3):
            x = k*sqrt3*scale
            positions.append((x, y2))
        y3 = 1.5*scale
        for k in range(-1, 2):
            x = k*sqrt3*scale + sqrt3/2*scale
            positions.append((x, y3))
        return positions[:N]
    elif layout_idx == 5:
        # Center + 6 ring + 4 more
        positions = []
        positions.append((0, 0))
        ring1 = [(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1)]
        for ti, tj in ring1:
            x = ti*u[0] + tj*v[0]
            y = ti*u[1] + tj*v[1]
            positions.append((x, y))
        ring2 = [(2,0),(1,1),(-1,2),(-2,0),(-1,-1),(0,-2)]
        for ti, tj in ring2[:4]:
            x = ti*u[0] + tj*v[0]
            y = ti*u[1] + tj*v[1]
            positions.append((x, y))
        return positions[:N]
    elif layout_idx == 6:
        # 2-3-3-3 pattern
        positions = []
        y1 = -3.0*scale
        for k in range(0, 2):
            x = k*sqrt3*scale - sqrt3/2*scale
            positions.append((x, y1))
        y2 = -1.5*scale
        for k in range(-1, 2):
            x = k*sqrt3*scale
            positions.append((x, y2))
        y3 = 0.0
        for k in range(-1, 2):
            x = k*sqrt3*scale + sqrt3/2*scale
            positions.append((x, y3))
        y4 = 1.5*scale
        for k in range(-1, 2):
            x = k*sqrt3*scale
            positions.append((x, y4))
        return positions[:N]
    elif layout_idx == 7:
        # 4-4-3 pattern
        positions = []
        y1 = -1.5*scale
        for k in range(-2, 2):
            x = k*sqrt3*scale
            positions.append((x, y1))
        y2 = 0.0
        for k in range(-2, 2):
            x = k*sqrt3*scale + sqrt3/2*scale
            positions.append((x, y2))
        y3 = 1.5*scale
        for k in range(-1, 2):
            x = k*sqrt3*scale
            positions.append((x, y3))
        return positions[:N]
    elif layout_idx == 8:
        # 5-6 pattern
        positions = []
        y1 = -0.75*scale
        for k in range(-2, 3):
            x = k*sqrt3*scale
            positions.append((x, y1))
        y2 = 0.75*scale
        for k in range(-3, 3):
            x = k*sqrt3*scale + sqrt3/2*scale
            positions.append((x, y2))
        return positions[:N]
    elif layout_idx == 9:
        # 6-5 pattern
        positions = []
        y1 = -0.75*scale
        for k in range(-3, 3):
            x = k*sqrt3*scale + sqrt3/2*scale
            positions.append((x, y1))
        y2 = 0.75*scale
        for k in range(-2, 3):
            x = k*sqrt3*scale
            positions.append((x, y2))
        return positions[:N]
    elif layout_idx == 10:
        # Diamond-ish
        positions = []
        y0 = -3.0*scale
        positions.append((0, y0))
        y1 = -1.5*scale
        for k in range(-1, 2):
            x = k*sqrt3*scale
            positions.append((x, y1))
        y2 = 0.0
        for k in range(-2, 3):
            x = k*sqrt3*scale
            positions.append((x, y2))
        y3 = 1.5*scale
        for k in range(-1, 2):
            x = k*sqrt3*scale
            positions.append((x, y3))
        y4 = 3.0*scale
        positions.append((0, y4))
        return positions[:N]
    elif layout_idx == 11:
        # 3-3-5 pattern
        positions = []
        y1 = -1.5*scale
        for k in range(-1, 2):
            x = k*sqrt3*scale
            positions.append((x, y1))
        y2 = 0.0
        for k in range(-1, 2):
            x = k*sqrt3*scale + sqrt3/2*scale
            positions.append((x, y2))
        y3 = 1.5*scale
        for k in range(-2, 3):
            x = k*sqrt3*scale
            positions.append((x, y3))
        return positions[:N]
    else:
        # Fallback: compact grid
        coords = []
        for i in range(-3, 4):
            for j in range(-3, 4):
                x = i*u[0] + j*v[0]
                y = i*u[1] + j*v[1]
                coords.append((x, y))
        coords_sorted = sorted(coords, key=lambda t: t[0]**2 + t[1]**2)
        return coords_sorted[:N]


def _objective_fast(params, inner_angle):
    """Objective: enclosing hex side length + penalty for overlaps. Optimized."""
    n_pos = N * 2
    positions = [(params[i], params[i+1]) for i in range(0, n_pos, 2)]
    outer_angle = params[-1]
    
    # Get all vertices
    all_vertices = []
    for px, py in positions:
        verts = _hex_vertices(px, py, inner_angle)
        all_vertices.append(verts)
    all_vertices = np.vstack(all_vertices)
    
    # Enclosing hex side
    side = _enclosing_hex_side(all_vertices, outer_angle)
    
    # Penalty for overlaps - vectorized
    penalty = 0.0
    for i in range(N):
        for j in range(i+1, N):
            di = np.array([positions[j][0] - positions[i][0], positions[j][1] - positions[i][1]])
            dist = _hex_distance_fast(di, inner_angle, inner_angle)
            if dist < 0:
                penalty += dist * dist * 100.0
    
    return side + penalty


def _optimize_layout(positions, inner_angle, outer_angles):
    """Optimize a layout by trying different outer angles and refining positions."""
    best_side = float('inf')
    best_pos = None
    best_outer = 0.0
    
    for outer_angle in outer_angles:
        # Compute enclosing hex for this angle
        all_vertices = np.vstack([_hex_vertices(px, py, inner_angle) for px, py in positions])
        side = _enclosing_hex_side(all_vertices, outer_angle)
        
        if side < best_side:
            best_side = side
            best_pos = positions
            best_outer = outer_angle
    
    # Try to optimize positions with Nelder-Mead
    if time.time() < _DEADLINE - 15.0:
        # Fix the best outer angle and optimize positions
        x0 = np.array([positions[i][0] for i in range(N)] + 
                      [positions[i][1] for i in range(N)] + [best_outer])
        
        def obj(params):
            return _objective_fast(params, inner_angle)
        
        try:
            res = minimize(obj, x0, method='Nelder-Mead', 
                          options={'maxiter': 100, 'xatol': 1e-5, 'fatol': 1e-7})
            if res.fun < best_side:
                best_side = res.fun
                best_pos = [(res.x[i], res.x[i+1]) for i in range(0, N*2, 2)]
                best_outer = res.x[-1]
        except Exception:
            pass
    
    return best_side, best_pos, best_outer


def construct():
    global _START_TIME, _DEADLINE
    
    if time.time() > _DEADLINE - 5.0:
        inner = [[4.0*k, 0.0, 0.0] for k in range(-(N//2), N - N//2)]
        return inner, (0.0, 0.0), 40.0, 0.0
    
    best_side = float('inf')
    best_inner = None
    best_center = (0.0, 0.0)
    best_outer_angle = 0.0
    
    num_layouts = 12
    layouts = [_build_positions(i) for i in range(num_layouts)]
    
    # Outer angles to try
    outer_angles = list(np.arange(0, 30.0, 1.0))
    
    for layout_idx, positions in enumerate(layouts):
        if time.time() > _DEADLINE - 10.0:
            break
        
        inner_angle = 30.0
        
        # First pass: try discrete outer angles
        best_local_side = float('inf')
        best_local_pos = positions
        best_local_angle = 0.0
        
        for outer_angle_deg in outer_angles:
            all_vertices = np.vstack([_hex_vertices(px, py, inner_angle) for px, py in positions])
            side = _enclosing_hex_side(all_vertices, outer_angle_deg)
            if side < best_local_side:
                best_local_side = side
                best_local_pos = positions
                best_local_angle = outer_angle_deg
        
        # Second pass: refine with local optimization
        if time.time() < _DEADLINE - 15.0:
            refined_side, refined_pos, refined_angle = _optimize_layout(
                best_local_pos, inner_angle, 
                list(np.arange(best_local_angle - 5, best_local_angle + 5, 0.5)))
            
            if refined_side < best_local_side:
                best_local_side = refined_side
                best_local_pos = refined_pos
                best_local_angle = refined_angle
        
        if best_local_side < best_side:
            best_side = best_local_side
            best_inner = [[pos[0], pos[1], inner_angle] for pos in best_local_pos]
            best_center = (0.0, 0.0)  # Will be recomputed
            best_outer_angle = best_local_angle
    
    # Compute proper center for the best configuration
    if best_inner is not None:
        all_vertices = np.vstack([_hex_vertices(p[0], p[1], p[2]) for p in best_inner])
        # Find center via LP
        base_angle = math.radians(best_outer_angle + 30)
        angles = base_angle + np.arange(6) * math.pi / 3.0
        normals = np.column_stack([np.cos(angles), np.sin(angles)])
        
        M = len(all_vertices)
        A_rows = []
        b_vals = []
        for i in range(M):
            vi = all_vertices[i]
            for k in range(6):
                nk = normals[k]
                A_rows.append([-nk[0], -nk[1], -1.0])
                b_vals.append(-nk[0]*vi[0] - nk[1]*vi[1])
                A_rows.append([nk[0], nk[1], -1.0])
                b_vals.append(nk[0]*vi[0] + nk[1]*vi[1])
        
        A = np.array(A_rows)
        b = np.array(b_vals)
        c_obj = np.array([0.0, 0.0, 1.0])
        bounds = [(None, None), (None, None), (0, None)]
        
        try:
            res = linprog(c_obj, A_ub=A, b_ub=b, bounds=bounds, method='highs')
            if res.status == 0:
                best_center = (res.x[0], res.x[1])
                best_side = res.x[2] * 2.0 / _SQRT3
        except Exception:
            pass
    
    if best_inner is None:
        inner = [[4.0*k, 0.0, 0.0] for k in range(-(N//2), N - N//2)]
        return inner, (0.0, 0.0), 40.0, 0.0
    
    return best_inner, best_center, best_side, best_outer_angle
