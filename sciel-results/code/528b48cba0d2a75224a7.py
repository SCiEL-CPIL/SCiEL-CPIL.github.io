import math
import time
import numpy as np
from scipy.optimize import linprog, minimize

N = 11
_START_TIME = time.time()
_DEADLINE = _START_TIME + 145.0
_SQRT3 = math.sqrt(3)
_INVSQRT3 = 1.0 / _SQRT3


def _hex_edge_normals(angle_deg):
    """Return 6 edge normal directions for a hexagon with given rotation."""
    a = math.radians(angle_deg)
    return a + math.pi/6.0 + np.arange(6) * math.pi/3.0


def _hex_support(angle_deg, theta):
    """Support function of unit hexagon (centered at origin, rotated by angle_deg) in direction theta."""
    a = math.radians(angle_deg)
    diff = (theta - a) % (2 * math.pi)
    # The hexagon has vertices at angles a + k*pi/3, k=0..5
    # Edge normals are at a + pi/6 + k*pi/3
    # For a given direction theta, find which "sector" it falls in
    # The support is max over vertices of v·n
    # Vertices: cos(a + k*pi/3), sin(a + k*pi/3)
    # For direction theta, support = max_k (cos(theta - a - k*pi/3))
    # This equals cos(min_k |theta - a - k*pi/3|) where the min is over k
    # But we need the closest vertex direction within pi/3
    
    # Actually, for a regular hexagon with circumradius 1:
    # support(theta) = cos(delta) where delta = min angle from theta to any vertex direction
    # Vertex directions: a + k*pi/3
    # The max of cos(theta - a - k*pi/3) over k is cos(min_k |theta - a - k*pi/3|)
    # since cos is decreasing on [0, pi]
    
    # Find the nearest vertex direction
    rel = (theta - a) % (2 * math.pi)
    k = round(rel / (math.pi / 3.0))
    delta = abs(rel - k * math.pi / 3.0)
    return math.cos(delta)


def _hex_distance_same_angle(d, angle_deg):
    """Minimum separation between two unit hexagons with same rotation.
    Positive = separated, negative = overlapping."""
    a = math.radians(angle_deg)
    dx, dy = d
    # For same orientation, the Minkowski difference is a hexagon with edge normals
    # at a + pi/6 + k*pi/3, and the "radius" in each normal direction is 2*apothem = sqrt(3)
    # Distance = min over edge normals of (proj - 2*apothem)
    # But actually, the separation is min over all directions of (|d·n| - support1(n) - support2(n))
    # For same angle: support1(n) = support2(n) = support(angle, n)
    # So separation = min_n (|d·n| - 2*support(angle, n))
    
    min_sep = float('inf')
    for k in range(6):
        theta = a + math.pi/6.0 + k * math.pi/3.0
        nx, ny = math.cos(theta), math.sin(theta)
        proj = abs(dx * nx + dy * ny)
        s = _hex_support(angle_deg, theta)
        sep = proj - 2.0 * s
        if sep < min_sep:
            min_sep = sep
    return min_sep


def _hex_distance(d, angle1, angle2):
    """Minimum separation between two unit hexagons with possibly different rotations."""
    a1 = math.radians(angle1)
    a2 = math.radians(angle2)
    dx, dy = d
    min_sep = float('inf')
    
    # Try all 12 edge normals (6 from each hexagon)
    for k in range(6):
        for angle in [angle1, angle2]:
            a = math.radians(angle)
            theta = a + math.pi/6.0 + k * math.pi/3.0
            nx, ny = math.cos(theta), math.sin(theta)
            proj = abs(dx * nx + dy * ny)
            s1 = _hex_support(angle1, theta)
            s2 = _hex_support(angle2, theta)
            sep = proj - s1 - s2
            if sep < min_sep:
                min_sep = sep
    
    return min_sep


def _hex_vertices(cx, cy, angle_deg):
    """Vertices of a unit regular hexagon centered at (cx,cy) with rotation angle_deg."""
    a0 = math.radians(angle_deg)
    angles = a0 + np.arange(6) * math.pi / 3.0
    return np.column_stack([cx + np.cos(angles), cy + np.sin(angles)])


def _hex_vertices_batch(positions, angle_deg):
    """Vertices for multiple hexagons at given positions, all with same rotation."""
    a0 = math.radians(angle_deg)
    angles = a0 + np.arange(6) * math.pi / 3.0
    base = np.column_stack([np.cos(angles), np.sin(angles)])
    return positions[:, np.newaxis, :] + base[np.newaxis, :, :]


def _enclosing_hex_side_lp(verts, outer_angle_deg):
    """Find minimum side length of regular hexagon with given orientation containing all vertices via LP."""
    base_angle = math.radians(outer_angle_deg + 30)
    angles = base_angle + np.arange(6) * math.pi / 3.0
    normals = np.column_stack([np.cos(angles), np.sin(angles)])
    
    M = len(verts)
    dots = verts @ normals.T  # (M, 6)
    
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
            return res.x[2] * 2.0 / _SQRT3
    except Exception:
        pass
    
    # Fallback
    cx, cy = np.mean(verts, axis=0)
    rel = verts - np.array([cx, cy])
    projections = rel @ normals.T
    max_proj = np.max(projections, axis=0)
    min_proj = np.min(projections, axis=0)
    r = np.max(np.maximum(max_proj, -min_proj))
    if r < 0: r = 0
    return r * 2.0 / _SQRT3


def _enclosing_hex_side_fast(verts, outer_angle_deg):
    """Fast approximation: use centroid as center."""
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
    return r * 2.0 / _SQRT3


def _smooth_repulsion(dist, sharpness=50.0):
    """Smooth repulsive force magnitude when dist < 0 (overlap)."""
    if dist >= 0:
        return 0.0
    d = -dist
    return d * sharpness


def _physics_simulate(positions, inner_angle, outer_angle, max_iters=5000, lr=0.01):
    """Physics-based simulation using repulsive forces to push hexagons apart."""
    pos = np.array(positions, dtype=float)
    n = len(pos)
    
    # Precompute edge normals for the inner hexagons
    inner_normals = _hex_edge_normals(inner_angle)
    inner_cos = np.cos(inner_normals)
    inner_sin = np.sin(inner_normals)
    
    # Precompute support values for inner normals
    inner_support = np.array([_hex_support(inner_angle, theta) for theta in inner_normals])
    
    best_pos = pos.copy()
    best_side = float('inf')
    
    for it in range(max_iters):
        if time.time() > _DEADLINE - 20.0:
            break
        
        # Compute forces on each hexagon
        forces = np.zeros((n, 2))
        
        for i in range(n):
            for j in range(i + 1, n):
                d = pos[j] - pos[i]
                # Compute minimum separation and which normal gives it
                min_sep = float('inf')
                best_nx, best_ny = 0.0, 0.0
                
                for k in range(6):
                    nx = inner_cos[k]
                    ny = inner_sin[k]
                    proj = abs(d[0] * nx + d[1] * ny)
                    s = inner_support[k]
                    sep = proj - 2.0 * s
                    if sep < min_sep:
                        min_sep = sep
                        best_nx = nx
                        best_ny = ny
                
                if min_sep < 0:
                    # Overlapping: apply repulsive force
                    f = _smooth_repulsion(min_sep, sharpness=50.0)
                    # Force direction: along the normal that gives min separation
                    # Determine sign based on which way j is from i
                    dot = d[0] * best_nx + d[1] * best_ny
                    sign = 1.0 if dot >= 0 else -1.0
                    
                    forces[i] += sign * f * np.array([best_nx, best_ny])
                    forces[j] -= sign * f * np.array([best_nx, best_ny])
        
        # Also add a gentle attractive force toward center to keep things compact
        for i in range(n):
            forces[i] -= 0.01 * pos[i]
        
        # Update positions
        pos += lr * forces
        
        # Compute current enclosing hex side
        all_vertices = _hex_vertices_batch(pos, inner_angle).reshape(-1, 2)
        side = _enclosing_hex_side_fast(all_vertices, outer_angle)
        
        if side < best_side:
            best_side = side
            best_pos = pos.copy()
        
        # Check convergence
        if it > 100 and abs(forces).max() < 1e-6:
            break
    
    return best_pos, best_side


def _smooth_penalty(dist, sharpness=20.0):
    """Smooth penalty: soft-max of 0 and -dist."""
    if dist >= 0:
        return 0.0
    d = -dist
    # Smooth approximation using exp
    if d < 0.005:
        return d * d * sharpness
    else:
        return d * 0.5 + 0.0025


def _physics_optimize(positions, inner_angle, outer_angles):
    """Use physics simulation to optimize a layout."""
    best_side = float('inf')
    best_pos = None
    best_outer = 0.0
    
    pos_arr = np.array(positions)
    
    # First pass: try discrete outer angles with fast enclosing hex
    for outer_angle in outer_angles:
        all_vertices = _hex_vertices_batch(pos_arr, inner_angle).reshape(-1, 2)
        side = _enclosing_hex_side_fast(all_vertices, outer_angle)
        
        if side < best_side:
            best_side = side
            best_pos = positions
            best_outer = outer_angle
    
    # Physics simulation with the best outer angle
    if time.time() < _DEADLINE - 25.0:
        sim_pos, sim_side = _physics_simulate(pos_arr, inner_angle, best_outer, max_iters=3000, lr=0.005)
        if sim_side < best_side:
            best_side = sim_side
            best_pos = [tuple(p) for p in sim_pos]
    
    # Nelder-Mead refinement after physics
    if time.time() < _DEADLINE - 15.0 and best_pos is not None:
        x0 = np.array([best_pos[i][0] for i in range(N)] + 
                      [best_pos[i][1] for i in range(N)])
        
        def _objective(params):
            n_pos = N * 2
            positions = np.array([(params[i], params[i+1]) for i in range(0, n_pos, 2)])
            all_vertices = _hex_vertices_batch(positions, inner_angle).reshape(-1, 2)
            side = _enclosing_hex_side_fast(all_vertices, best_outer)
            
            penalty = 0.0
            for i in range(N):
                for j in range(i + 1, N):
                    di = positions[j] - positions[i]
                    dist = _hex_distance_same_angle(di, inner_angle)
                    if dist < 0:
                        penalty += dist * dist * 100.0
            
            return side + penalty
        
        try:
            res = minimize(_objective, x0, method='Nelder-Mead', 
                          options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True})
            if res.fun < best_side:
                best_side = res.fun
                best_pos = [(res.x[i], res.x[i+1]) for i in range(0, N*2, 2)]
        except Exception:
            pass
    
    return best_side, best_pos, best_outer


def _build_positions(layout_idx):
    """Generate 11 positions for given layout index."""
    sqrt3 = _SQRT3
    scale = 1.001
    u = (sqrt3 * scale, 0.0)
    v = (sqrt3 / 2 * scale, 1.5 * scale)
    
    if layout_idx == 0:
        # Compact hexagonal: center + 6 ring + 4 from second ring
        target = [(0, 0)]
        ring1 = [(1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1)]
        target.extend(ring1)
        ring2 = [(2, 0), (1, 1), (0, 2), (-1, 2), (-2, 1), (-2, 0), (-1, -1), (0, -2), (1, -2), (2, -1)]
        ring2_sorted = sorted(ring2, key=lambda p: p[0]**2 + p[1]**2)
        target.extend(ring2_sorted[:4])
        positions = []
        for ti, tj in target:
            x = ti * u[0] + tj * v[0]
            y = ti * u[1] + tj * v[1]
            positions.append((x, y))
        return positions[:N]
    elif layout_idx == 1:
        # Row-based: 3-4-4 pattern
        positions = []
        y1 = -1.5 * scale
        for k in range(-1, 2):
            x = k * sqrt3 * scale + sqrt3 / 2 * scale
            positions.append((x, y1))
        y2 = 0.0
        for k in range(-1, 3):
            x = k * sqrt3 * scale
            positions.append((x, y2))
        y3 = 1.5 * scale
        for k in range(-1, 3):
            x = k * sqrt3 * scale + sqrt3 / 2 * scale
            positions.append((x, y3))
        return positions[:N]
    elif layout_idx == 2:
        # 4-3-4 pattern
        positions = []
        y1 = -1.5 * scale
        for k in range(-2, 2):
            x = k * sqrt3 * scale
            positions.append((x, y1))
        y2 = 0.0
        for k in range(-1, 2):
            x = k * sqrt3 * scale + sqrt3 / 2 * scale
            positions.append((x, y2))
        y3 = 1.5 * scale
        for k in range(-2, 2):
            x = k * sqrt3 * scale
            positions.append((x, y3))
        return positions[:N]
    elif layout_idx == 3:
        # 3-5-3 pattern
        positions = []
        y1 = -1.5 * scale
        for k in range(-1, 2):
            x = k * sqrt3 * scale + sqrt3 / 2 * scale
            positions.append((x, y1))
        y2 = 0.0
        for k in range(-2, 3):
            x = k * sqrt3 * scale
            positions.append((x, y2))
        y3 = 1.5 * scale
        for k in range(-1, 2):
            x = k * sqrt3 * scale + sqrt3 / 2 * scale
            positions.append((x, y3))
        return positions[:N]
    elif layout_idx == 4:
        # Center + 6 ring + 4 more (different selection)
        positions = []
        positions.append((0, 0))
        ring1 = [(1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1)]
        for ti, tj in ring1:
            x = ti * u[0] + tj * v[0]
            y = ti * u[1] + tj * v[1]
            positions.append((x, y))
        ring2 = [(2, 0), (1, 1), (-1, 2), (-2, 0), (-1, -1), (0, -2)]
        for ti, tj in ring2[:4]:
            x = ti * u[0] + tj * v[0]
            y = ti * u[1] + tj * v[1]
            positions.append((x, y))
        return positions[:N]
    elif layout_idx == 5:
        # 4-4-3 pattern
        positions = []
        y1 = -1.5 * scale
        for k in range(-2, 2):
            x = k * sqrt3 * scale
            positions.append((x, y1))
        y2 = 0.0
        for k in range(-2, 2):
            x = k * sqrt3 * scale + sqrt3 / 2 * scale
            positions.append((x, y2))
        y3 = 1.5 * scale
        for k in range(-1, 2):
            x = k * sqrt3 * scale
            positions.append((x, y3))
        return positions[:N]
    elif layout_idx == 6:
        # 5-6 pattern
        positions = []
        y1 = -0.75 * scale
        for k in range(-2, 3):
            x = k * sqrt3 * scale
            positions.append((x, y1))
        y2 = 0.75 * scale
        for k in range(-3, 3):
            x = k * sqrt3 * scale + sqrt3 / 2 * scale
            positions.append((x, y2))
        return positions[:N]
    elif layout_idx == 7:
        # 6-5 pattern
        positions = []
        y1 = -0.75 * scale
        for k in range(-3, 3):
            x = k * sqrt3 * scale + sqrt3 / 2 * scale
            positions.append((x, y1))
        y2 = 0.75 * scale
        for k in range(-2, 3):
            x = k * sqrt3 * scale
            positions.append((x, y2))
        return positions[:N]
    elif layout_idx == 8:
        # Diamond-ish
        positions = []
        y0 = -3.0 * scale
        positions.append((0, y0))
        y1 = -1.5 * scale
        for k in range(-1, 2):
            x = k * sqrt3 * scale
            positions.append((x, y1))
        y2 = 0.0
        for k in range(-2, 3):
            x = k * sqrt3 * scale
            positions.append((x, y2))
        y3 = 1.5 * scale
        for k in range(-1, 2):
            x = k * sqrt3 * scale
            positions.append((x, y3))
        y4 = 3.0 * scale
        positions.append((0, y4))
        return positions[:N]
    elif layout_idx == 9:
        # 3-3-5 pattern
        positions = []
        y1 = -1.5 * scale
        for k in range(-1, 2):
            x = k * sqrt3 * scale
            positions.append((x, y1))
        y2 = 0.0
        for k in range(-1, 2):
            x = k * sqrt3 * scale + sqrt3 / 2 * scale
            positions.append((x, y2))
        y3 = 1.5 * scale
        for k in range(-2, 3):
            x = k * sqrt3 * scale
            positions.append((x, y3))
        return positions[:N]
    elif layout_idx == 10:
        # 2-4-2-3 pattern
        positions = []
        y1 = -3.0 * scale
        for k in range(0, 2):
            x = k * sqrt3 * scale - sqrt3 / 2 * scale
            positions.append((x, y1))
        y2 = -1.5 * scale
        for k in range(-1, 3):
            x = k * sqrt3 * scale
            positions.append((x, y2))
        y3 = 0.0
        for k in range(-1, 1):
            x = k * sqrt3 * scale + sqrt3 / 2 * scale
            positions.append((x, y3))
        y4 = 1.5 * scale
        for k in range(-1, 2):
            x = k * sqrt3 * scale
            positions.append((x, y4))
        return positions[:N]
    elif layout_idx == 11:
        # 2-3-3-3 pattern
        positions = []
        y1 = -3.0 * scale
        for k in range(0, 2):
            x = k * sqrt3 * scale - sqrt3 / 2 * scale
            positions.append((x, y1))
        y2 = -1.5 * scale
        for k in range(-1, 2):
            x = k * sqrt3 * scale
            positions.append((x, y2))
        y3 = 0.0
        for k in range(-1, 2):
            x = k * sqrt3 * scale + sqrt3 / 2 * scale
            positions.append((x, y3))
        y4 = 1.5 * scale
        for k in range(-1, 2):
            x = k * sqrt3 * scale
            positions.append((x, y4))
        return positions[:N]
    else:
        # Fallback: compact grid
        coords = []
        for i in range(-3, 4):
            for j in range(-3, 4):
                x = i * u[0] + j * v[0]
                y = i * u[1] + j * v[1]
                coords.append((x, y))
        coords_sorted = sorted(coords, key=lambda t: t[0]**2 + t[1]**2)
        return coords_sorted[:N]


def _objective_smooth(params, inner_angle, outer_angle):
    """Smooth objective: enclosing hex side + smooth penalty for overlaps."""
    n_pos = N * 2
    positions = np.array([(params[i], params[i+1]) for i in range(0, n_pos, 2)])
    
    all_vertices = _hex_vertices_batch(positions, inner_angle).reshape(-1, 2)
    side = _enclosing_hex_side_fast(all_vertices, outer_angle)
    
    penalty = 0.0
    for i in range(N):
        for j in range(i + 1, N):
            di = positions[j] - positions[i]
            dist = _min_separation_same_angle(di, inner_angle)
            if dist < 0:
                penalty += _smooth_penalty(dist)
    
    return side + penalty


def _objective_exact(params, inner_angle, outer_angle):
    """Exact objective: enclosing hex side (LP) + hard penalty for overlaps."""
    n_pos = N * 2
    positions = np.array([(params[i], params[i+1]) for i in range(0, n_pos, 2)])
    
    all_vertices = _hex_vertices_batch(positions, inner_angle).reshape(-1, 2)
    side = _enclosing_hex_side_lp(all_vertices, outer_angle)
    
    penalty = 0.0
    for i in range(N):
        for j in range(i + 1, N):
            di = positions[j] - positions[i]
            dist = _min_separation_same_angle(di, inner_angle)
            if dist < 0:
                penalty += dist * dist * 100.0
    
    return side + penalty


def _optimize_layout(positions, inner_angle, outer_angles):
    """Optimize a layout by trying different outer angles and refining positions."""
    best_side = float('inf')
    best_pos = None
    best_outer = 0.0
    
    pos_arr = np.array(positions)
    
    # First pass: try discrete outer angles with fast enclosing hex
    for outer_angle in outer_angles:
        all_vertices = _hex_vertices_batch(pos_arr, inner_angle).reshape(-1, 2)
        side = _enclosing_hex_side_fast(all_vertices, outer_angle)
        
        if side < best_side:
            best_side = side
            best_pos = positions
            best_outer = outer_angle
    
    # Second pass: smooth Nelder-Mead optimization
    if time.time() < _DEADLINE - 25.0:
        x0 = np.array([positions[i][0] for i in range(N)] + 
                      [positions[i][1] for i in range(N)])
        
        try:
            res = minimize(_objective_smooth, x0, args=(inner_angle, best_outer),
                          method='Nelder-Mead', 
                          options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True})
            if res.fun < best_side:
                best_side = res.fun
                best_pos = [(res.x[i], res.x[i+1]) for i in range(0, N*2, 2)]
        except Exception:
            pass
    
    # Third pass: high-precision polish with exact LP-based objective
    if time.time() < _DEADLINE - 12.0 and best_pos is not None:
        x0 = np.array([best_pos[i][0] for i in range(N)] + 
                      [best_pos[i][1] for i in range(N)])
        try:
            res2 = minimize(_objective_exact, x0, args=(inner_angle, best_outer),
                           method='Nelder-Mead',
                           options={'maxiter': 300, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True})
            if res2.fun < best_side:
                best_side = res2.fun
                best_pos = [(res2.x[i], res2.x[i+1]) for i in range(0, N*2, 2)]
        except Exception:
            pass
    
    return best_side, best_pos, best_outer


def construct():
    global _START_TIME, _DEADLINE
    
    if time.time() > _DEADLINE - 5.0:
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0
    
    best_side = float('inf')
    best_inner = None
    best_center = (0.0, 0.0)
    best_outer_angle = 0.0
    
    num_layouts = 12
    layouts = [_build_positions(i) for i in range(num_layouts)]
    
    # Outer angles to try (finer grid)
    outer_angles = list(np.arange(0, 30.0, 0.5))
    
    inner_angle = 30.0
    
    for layout_idx, positions in enumerate(layouts):
        if time.time() > _DEADLINE - 15.0:
            break
        
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
        exact_side = _enclosing_hex_side_lp(all_vertices, best_outer_angle)
        if exact_side < best_side:
            best_side = exact_side
        
        # Find center via LP
        base_angle = math.radians(best_outer_angle + 30)
        angles = base_angle + np.arange(6) * math.pi / 3.0
        normals = np.column_stack([np.cos(angles), np.sin(angles)])
        
        M = len(all_vertices)
        dots = all_vertices @ normals.T
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
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0
    
    return best_inner, best_center, best_side, best_outer_angle
