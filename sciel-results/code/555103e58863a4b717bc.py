import math
import time
import numpy as np
from scipy.optimize import linprog, minimize
from scipy.spatial import ConvexHull

N = 11
_START_TIME = time.time()
_DEADLINE = _START_TIME + 148.0
_SQRT3 = math.sqrt(3)
_INVSQRT3 = 1.0 / _SQRT3


def _hex_vertices(cx, cy, angle_deg):
    """Vertices of a unit regular hexagon centered at (cx,cy) with rotation angle_deg."""
    a0 = math.radians(angle_deg)
    angles = a0 + np.arange(6) * math.pi / 3.0
    return np.column_stack([cx + np.cos(angles), cy + np.sin(angles)])


def _hex_vertices_batch(positions, angle_deg):
    """Vertices for multiple hexagons at given positions, all with same rotation."""
    a0 = math.radians(angle_deg)
    angles = a0 + np.arange(6) * math.pi / 3.0
    base = np.column_stack([np.cos(angles), np.sin(angles)])  # (6,2)
    # positions: (N, 2)
    # result: (N*6, 2)
    return positions[:, np.newaxis, :] + base[np.newaxis, :, :]


def _hex_support(cx, cy, angle_deg, theta):
    """Support function of unit hexagon: max dot product with direction theta."""
    a0 = math.radians(angle_deg)
    diff = (theta - a0) % (2 * math.pi)
    k = round(diff / (math.pi / 3.0))
    d1 = abs(diff - k * math.pi / 3.0)
    d2 = abs(diff - (k - 1) * math.pi / 3.0)
    d3 = abs(diff - (k + 1) * math.pi / 3.0)
    min_d = min(d1, d2, d3)
    return math.cos(min_d)


def _hex_distance_same_angle(d, angle_deg):
    """Minimum separation between two unit hexagons with same rotation angle.
    Returns positive if separated, negative if overlapping."""
    a = math.radians(angle_deg)
    dx, dy = d
    # Edge normals: a + pi/6 + k*pi/3 for k=0..5
    min_sep = float('inf')
    for k in range(6):
        theta = a + math.pi / 6.0 + k * math.pi / 3.0
        nx, ny = math.cos(theta), math.sin(theta)
        proj = abs(dx * nx + dy * ny)
        # Support of unit hex in direction theta for same angle
        # diff = (theta - a) % (2pi) = pi/6 + k*pi/3 mod 2pi
        # For k=0: pi/6, k=1: pi/2, k=2: 5pi/6, k=3: 7pi/6 -> pi/6, k=4: 3pi/2 -> pi/2, k=5: 11pi/6 -> 5pi/6
        # cos values: cos(pi/6)=sqrt3/2, cos(pi/2)=0, cos(5pi/6)=-sqrt3/2
        # But support = cos(min_d) where min_d is min angular distance to vertex directions
        # Vertex directions relative to a: 0, pi/3, 2pi/3, pi, 4pi/3, 5pi/3
        # theta - a = pi/6 + k*pi/3
        # min distance to {0, pi/3, ...}: 
        # k=0: pi/6 -> dist to 0 is pi/6, to pi/3 is pi/6 -> min pi/6 -> cos(pi/6)
        # k=1: pi/2 -> dist to pi/3 is pi/6, to 2pi/3 is pi/6 -> min pi/6 -> cos(pi/6)
        # Actually for a regular hexagon, the support in any edge-normal direction equals the apothem = sqrt(3)/2
        s1 = math.sqrt(3) / 2.0
        s2 = math.sqrt(3) / 2.0
        sep = proj - s1 - s2
        if sep < min_sep:
            min_sep = sep
    return min_sep


def _hex_distance(d, angle1, angle2):
    """Minimum distance between two unit hexagons with centers separated by vector d, 
    rotations angle1, angle2. Returns 0 if overlapping, positive if separated."""
    # Two convex polygons are disjoint iff there exists a separating axis.
    # For two regular hexagons, the separating axes are the normals to all edges.
    # Edge normals of hexagon with rotation a: a + pi/6 + k*pi/3 for k=0..5
    # Distance along axis n: |d·n| - support1(n) - support2(n)
    # If for all axes this is >= 0, they're disjoint.
    # The minimum distance is the max over axes of (support1 + support2 - |d·n|) ... no wait.
    # Actually: they overlap iff for ALL axes n: |d·n| < support1(n) + support2(n)
    # They are disjoint iff there EXISTS an axis n: |d·n| >= support1(n) + support2(n)
    # The minimum separation = min over axes of (|d·n| - support1(n) - support2(n))
    # If this min > 0, they're separated by that amount.
    # If this min < 0, they overlap (penetration depth = -min).
    
    a1 = math.radians(angle1)
    a2 = math.radians(angle2)
    d = np.array(d)
    
    # Try all 12 edge normals (6 from each hexagon)
    min_sep = float('inf')
    for k in range(6):
        # Normals from hexagon 1
        theta1 = a1 + math.pi/6.0 + k * math.pi/3.0
        proj = abs(np.dot(d, [math.cos(theta1), math.sin(theta1)]))
        s1 = _hex_support(0, 0, angle1, theta1)
        s2 = _hex_support(0, 0, angle2, theta1)
        sep = proj - s1 - s2
        if sep < min_sep:
            min_sep = sep
        
        # Normals from hexagon 2
        theta2 = a2 + math.pi/6.0 + k * math.pi/3.0
        proj = abs(np.dot(d, [math.cos(theta2), math.sin(theta2)]))
        s1 = _hex_support(0, 0, angle1, theta2)
        s2 = _hex_support(0, 0, angle2, theta2)
        sep = proj - s1 - s2
        if sep < min_sep:
            min_sep = sep
    
    return min_sep


def _enclosing_hex_side(verts, outer_angle_deg):
    """Find minimum side length of regular hexagon with given orientation containing all vertices.
    Uses LP for exact result."""
    base_angle = math.radians(outer_angle_deg + 30)
    angles = base_angle + np.arange(6) * math.pi / 3.0
    normals = np.column_stack([np.cos(angles), np.sin(angles)])  # (6, 2)
    
    M = len(verts)
    # Vectorize constraint construction
    # For each vertex vi and each normal nk:
    #   nk·(c - vi) <= r  =>  nk·c - r <= nk·vi
    #  -nk·(c - vi) <= r  =>  -nk·c - r <= -nk·vi
    # A_ub * x <= b_ub where x = [cx, cy, r]
    # Row for (nk, vi): [nk[0], nk[1], -1] · [cx,cy,r] <= nk·vi
    #                   [-nk[0], -nk[1], -1] · [cx,cy,r] <= -nk·vi
    
    # Build A and b efficiently
    # normals: (6,2), verts: (M,2)
    # For each k in 0..5, for each i in 0..M-1:
    #   A row: [nk[0], nk[1], -1], b: nk·vi
    #   A row: [-nk[0], -nk[1], -1], b: -nk·vi
    
    # Use broadcasting: normals[None,:, :] shape (1,6,2), verts[:,None,:] shape (M,1,2)
    # dot products: (M,6)
    dots = verts @ normals.T  # (M, 6)
    
    # A has 2*M*6 rows, 3 cols
    # b has 2*M*6 rows
    A = np.zeros((2 * M * 6, 3))
    b = np.zeros(2 * M * 6)
    
    idx = 0
    for k in range(6):
        nk = normals[k]
        A[idx:idx+M, 0] = nk[0]
        A[idx:idx+M, 1] = nk[1]
        A[idx:idx+M, 2] = -1.0
        b[idx:idx+M] = dots[:, k]
        idx += M
        A[idx:idx+M, 0] = -nk[0]
        A[idx:idx+M, 1] = -nk[1]
        A[idx:idx+M, 2] = -1.0
        b[idx:idx+M] = -dots[:, k]
        idx += M
    
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


def _enclosing_hex_side_fast(verts, outer_angle_deg):
    """Fast approximation: use centroid as center, compute required radius."""
    base_angle = math.radians(outer_angle_deg + 30)
    angles = base_angle + np.arange(6) * math.pi / 3.0
    normals = np.column_stack([np.cos(angles), np.sin(angles)])
    
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


def _smooth_penalty(dist):
    """Smooth penalty: soft-max of 0 and -dist, scaled."""
    if dist >= 0:
        return 0.0
    # Smooth approximation of max(0, -dist) using sigmoid-like
    # For small |dist|, use quadratic; for large, linear
    d = -dist
    if d < 0.01:
        return d * d * 50.0
    elif d < 0.1:
        return d * d * 50.0 + (d - 0.01) * 0.5
    else:
        return d * 0.5 + 0.0045


def _objective_smooth(params, inner_angle):
    """Smooth objective: enclosing hex side + smooth penalty for overlaps."""
    n_pos = N * 2
    positions = np.array([(params[i], params[i+1]) for i in range(0, n_pos, 2)])
    outer_angle = params[-1]
    
    # Get all vertices using batch computation
    all_vertices = _hex_vertices_batch(positions, inner_angle).reshape(-1, 2)
    
    # Enclosing hex side (fast version for optimization)
    side = _enclosing_hex_side_fast(all_vertices, outer_angle)
    
    # Smooth penalty for overlaps
    penalty = 0.0
    for i in range(N):
        for j in range(i+1, N):
            di = positions[j] - positions[i]
            dist = _hex_distance_same_angle(di, inner_angle)
            if dist < 0:
                penalty += _smooth_penalty(dist)
    
    return side + penalty


def _objective_exact(params, inner_angle):
    """Exact objective: enclosing hex side (LP) + hard penalty for overlaps."""
    n_pos = N * 2
    positions = np.array([(params[i], params[i+1]) for i in range(0, n_pos, 2)])
    outer_angle = params[-1]
    
    all_vertices = _hex_vertices_batch(positions, inner_angle).reshape(-1, 2)
    
    side = _enclosing_hex_side(all_vertices, outer_angle)
    
    penalty = 0.0
    for i in range(N):
        for j in range(i+1, N):
            di = positions[j] - positions[i]
            dist = _hex_distance_same_angle(di, inner_angle)
            if dist < 0:
                penalty += dist * dist * 100.0
    
    return side + penalty


def _optimize_layout(positions, inner_angle, outer_angles):
    """Optimize a layout by trying different outer angles and refining positions."""
    best_side = float('inf')
    best_pos = None
    best_outer = 0.0
    
    # First pass: try discrete outer angles with fast enclosing hex
    pos_arr = np.array(positions)
    for outer_angle in outer_angles:
        all_vertices = _hex_vertices_batch(pos_arr, inner_angle).reshape(-1, 2)
        side = _enclosing_hex_side_fast(all_vertices, outer_angle)
        
        if side < best_side:
            best_side = side
            best_pos = positions
            best_outer = outer_angle
    
    # Second pass: refine with smooth Nelder-Mead
    if time.time() < _DEADLINE - 20.0:
        x0 = np.array([positions[i][0] for i in range(N)] + 
                      [positions[i][1] for i in range(N)] + [best_outer])
        
        try:
            res = minimize(_objective_smooth, x0, args=(inner_angle,),
                          method='Nelder-Mead', 
                          options={'maxiter': 300, 'xatol': 1e-7, 'fatol': 1e-9, 'adaptive': True})
            if res.fun < best_side:
                best_side = res.fun
                best_pos = [(res.x[i], res.x[i+1]) for i in range(0, N*2, 2)]
                best_outer = res.x[-1]
        except Exception:
            pass
    
    # Third pass: high-precision polish with exact LP-based objective
    if time.time() < _DEADLINE - 10.0 and best_pos is not None:
        x0 = np.array([best_pos[i][0] for i in range(N)] + 
                      [best_pos[i][1] for i in range(N)] + [best_outer])
        try:
            res2 = minimize(_objective_exact, x0, args=(inner_angle,),
                           method='Nelder-Mead',
                           options={'maxiter': 200, 'xatol': 1e-9, 'fatol': 1e-11, 'adaptive': True})
            if res2.fun < best_side:
                best_side = res2.fun
                best_pos = [(res2.x[i], res2.x[i+1]) for i in range(0, N*2, 2)]
                best_outer = res2.x[-1]
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
    
    # Outer angles to try (finer grid)
    outer_angles = list(np.arange(0, 30.0, 0.5))
    
    for layout_idx, positions in enumerate(layouts):
        if time.time() > _DEADLINE - 12.0:
            break
        
        inner_angle = 30.0
        
        # Use _optimize_layout which includes fast screening + smooth optimization + polish
        refined_side, refined_pos, refined_angle = _optimize_layout(
            positions, inner_angle, outer_angles)
        
        if refined_side < best_side:
            best_side = refined_side
            best_inner = [[pos[0], pos[1], inner_angle] for pos in refined_pos]
            best_center = (0.0, 0.0)
            best_outer_angle = refined_angle
    
    # Final: compute exact enclosing hex side via LP for the best configuration
    if best_inner is not None:
        pos_arr = np.array([[p[0], p[1]] for p in best_inner])
        all_vertices = _hex_vertices_batch(pos_arr, best_inner[0][2]).reshape(-1, 2)
        exact_side = _enclosing_hex_side(all_vertices, best_outer_angle)
        if exact_side < best_side:
            best_side = exact_side
        
        # Find center via LP
        base_angle = math.radians(best_outer_angle + 30)
        angles = base_angle + np.arange(6) * math.pi / 3.0
        normals = np.column_stack([np.cos(angles), np.sin(angles)])
        
        M = len(all_vertices)
        dots = all_vertices @ normals.T  # (M, 6)
        A = np.zeros((2 * M * 6, 3))
        b = np.zeros(2 * M * 6)
        idx = 0
        for k in range(6):
            nk = normals[k]
            A[idx:idx+M, 0] = nk[0]
            A[idx:idx+M, 1] = nk[1]
            A[idx:idx+M, 2] = -1.0
            b[idx:idx+M] = dots[:, k]
            idx += M
            A[idx:idx+M, 0] = -nk[0]
            A[idx:idx+M, 1] = -nk[1]
            A[idx:idx+M, 2] = -1.0
            b[idx:idx+M] = -dots[:, k]
            idx += M
        
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
