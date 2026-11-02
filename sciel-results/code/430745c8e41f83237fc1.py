import math
import numpy as np
import time
from scipy.optimize import linprog

N = 11
APOTHEM = math.sqrt(3) / 2.0
START_TIME = time.time()
TIME_LIMIT = 145.0

# Unit hexagon vertices (pointy-top, circumradius=1)
_HEX_VERTS = np.array([[math.cos(math.pi/6 + k*math.pi/3), math.sin(math.pi/6 + k*math.pi/3)] for k in range(6)])

def axial_to_cartesian(axial_list):
    pts = []
    for q, r in axial_list:
        x = math.sqrt(3) * (q + r / 2.0)
        y = 1.5 * r
        pts.append((x, y))
    return np.array(pts)

def compute_separation_axes(pts):
    """Compute separation axes for all pairs of hexagons.
    For each pair (i,j), the axis is the direction of (p_j - p_i) normalized.
    The minimum separation distance along this axis must be >= sqrt(3) + eps.
    """
    n_pairs = N * (N - 1) // 2
    axes = np.zeros((n_pairs, 2))
    pair_idx = 0
    for i in range(N):
        for j in range(i + 1, N):
            d = pts[j] - pts[i]
            dist = np.linalg.norm(d)
            if dist < 1e-12:
                axes[pair_idx] = [1.0, 0.0]
            else:
                axes[pair_idx] = d / dist
            pair_idx += 1
    return axes

def build_lp(pts, phi):
    """Build LP to minimize outer hexagon side length L.
    
    Variables: x_0..x_{2N-1} (positions), L (outer side length)
    Total: 2N + 1 variables.
    
    Containment constraints: for each of 6 normals of outer hexagon,
    for each inner hexagon's 6 vertices, the projection must be within L*APOTHEM.
    
    Separation constraints: for each pair, distance along axis >= sqrt(3) + eps.
    """
    cb = math.cos(phi)
    sb = math.sin(phi)
    
    # Rotated hexagon vertices
    R = np.array([[cb, -sb], [sb, cb]])
    rotated_verts = R @ _HEX_VERTS.T  # shape (6, 6)
    
    # Outer hexagon normals (6 directions)
    outer_normals = []
    for k in range(6):
        ang = phi + math.pi / 6.0 + k * math.pi / 3.0
        outer_normals.append([math.cos(ang), math.sin(ang)])
    outer_normals = np.array(outer_normals)  # (6, 2)
    
    n_vars = 2 * N + 1
    c = np.zeros(n_vars)
    c[-1] = 1.0  # minimize L
    
    A_ub = []
    b_ub = []
    
    eps_cont = 1e-9
    
    # Containment constraints
    for k in range(6):
        nx, ny = outer_normals[k]
        for i in range(N):
            for v in range(6):
                vx, vy = rotated_verts[v]
                # Projection of vertex i,v: (x_i + vx)*nx + (y_i + vy)*ny <= L * APOTHEM
                # => nx*x_i + ny*y_i - APOTHEM*L <= -nx*vx - ny*vy
                row = np.zeros(n_vars)
                row[2*i] = nx
                row[2*i+1] = ny
                row[-1] = -APOTHEM
                const = -(nx * vx + ny * vy) + eps_cont
                A_ub.append(row)
                b_ub.append(const)
    
    # Separation constraints
    eps_sep = 1e-6
    sep_dist = math.sqrt(3) + eps_sep
    axes = compute_separation_axes(pts)
    pair_idx = 0
    for i in range(N):
        for j in range(i + 1, N):
            n = axes[pair_idx]
            # (p_j - p_i) . n >= sep_dist
            # => -n_x*x_i - n_y*y_i + n_x*x_j + n_y*y_j >= sep_dist
            # => n_x*x_i + n_y*y_i - n_x*x_j - n_y*y_j <= -sep_dist
            row = np.zeros(n_vars)
            row[2*i] = n[0]
            row[2*i+1] = n[1]
            row[2*j] = -n[0]
            row[2*j+1] = -n[1]
            A_ub.append(row)
            b_ub.append(-sep_dist)
            pair_idx += 1
    
    A_ub = np.array(A_ub)
    b_ub = np.array(b_ub)
    
    bounds = [(None, None)] * (2 * N) + [(0.5, None)]
    
    return c, A_ub, b_ub, bounds

def solve_packing(pts):
    """Solve LP for given initial positions, return (L, positions, phi)."""
    best_L = float('inf')
    best_X = None
    best_phi = 0.0
    
    # Try a few rotations of the outer hexagon
    for phi in [0.0, math.pi/12, math.pi/6, math.pi/4, math.pi/3]:
        if time.time() - START_TIME > TIME_LIMIT - 5.0:
            break
        
        c, A_ub, b_ub, bounds = build_lp(pts, phi)
        
        try:
            res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
            if res.success:
                L_val = res.x[-1]
                if L_val < best_L:
                    best_L = L_val
                    best_X = res.x[:-1].copy()
                    best_phi = phi
        except Exception:
            pass
    
    return best_L, best_X, best_phi

def construct():
    if time.time() - START_TIME > TIME_LIMIT - 10.0:
        return [[0.0, 0.0, 0.0]], [0.0, 0.0], 1.0, 0.0
    
    # Generate candidate initial configurations
    candidates = []
    
    # Dense hexagonal cluster (radius 1 = 7 hexes, add 4 more)
    # Center + 6 neighbors + 4 more at distance 2
    hex_cluster = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (0, 2), (-2, 2), (2, -2)
    ]
    candidates.append(hex_cluster)
    
    # Another cluster variant
    hex_cluster2 = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 1), (-2, 2), (-1, 3), (1, -2)
    ]
    candidates.append(hex_cluster2)
    
    # Row-based configurations
    for rows in range(2, 5):
        for cols in range(2, 6):
            if rows * cols != N:
                continue
            for offset in [0, 1]:
                pts_axial = []
                for r in range(rows):
                    for c in range(cols):
                        q = c + (offset if r % 2 else 0)
                        pts_axial.append((q, r))
                if len(pts_axial) == N:
                    candidates.append(pts_axial)
    
    # Linear arrangements
    for a in range(3, N+1):
        for b in range(1, N - a + 1):
            if a + b != N:
                continue
            pts_axial = []
            for i in range(a):
                pts_axial.append((i, 0))
            for i in range(b):
                pts_axial.append((a - 1 + i, i))
            if len(pts_axial) == N and len(set(pts_axial)) == N:
                candidates.append(pts_axial)
    
    # Triangular-ish
    for r_max in range(2, 5):
        pts_axial = []
        for r in range(r_max + 1):
            for q in range(-r, r + 1):
                if len(pts_axial) < N:
                    pts_axial.append((q, r))
                else:
                    break
            if len(pts_axial) >= N:
                break
        if len(pts_axial) == N:
            candidates.append(pts_axial)
    
    # Compact 3x4 style
    compact1 = [
        (0, 0), (1, 0), (2, 0), (3, 0),
        (0, 1), (1, 1), (2, 1), (3, 1),
        (0, 2), (1, 2), (2, 2)
    ]
    candidates.append(compact1)
    
    compact2 = [
        (0, 0), (1, 0), (2, 0), (3, 0),
        (0, 1), (1, 1), (2, 1),
        (0, 2), (1, 2), (2, 2), (3, 2)
    ]
    candidates.append(compact2)
    
    # More hexagonal
    hex3 = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (-2, 0), (1, 1), (-1, -1)
    ]
    candidates.append(hex3)
    
    hex4 = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 1), (1, 2), (-1, 2), (0, 3)
    ]
    candidates.append(hex4)
    
    # Deduplicate
    unique_candidates = []
    seen = set()
    for cand in candidates:
        key = tuple(sorted(cand))
        if key not in seen:
            seen.add(key)
            unique_candidates.append(cand)
    
    best_L_global = float('inf')
    best_positions_global = None
    best_phi_global = 0.0
    
    for axial in unique_candidates:
        if time.time() - START_TIME > TIME_LIMIT - 5.0:
            break
        
        pts = axial_to_cartesian(axial)
        L, positions, phi = solve_packing(pts)
        
        if L < best_L_global and positions is not None:
            best_L_global = L
            best_positions_global = positions
            best_phi_global = phi
    
    if best_positions_global is None:
        # Fallback: simple grid
        pts = axial_to_cartesian([(i, 0) for i in range(N)])
        L, positions, phi = solve_packing(pts)
        if positions is not None:
            best_L_global = L
            best_positions_global = positions
            best_phi_global = phi
    
    if best_positions_global is None:
        return [[0.0, 0.0, 0.0]], [0.0, 0.0], 1.0, 0.0
    
    # Compute center as centroid
    xs = best_positions_global[0::2]
    ys = best_positions_global[1::2]
    cx = float(np.mean(xs))
    cy = float(np.mean(ys))
    
    angle_deg = float(math.degrees(best_phi_global))
    
    inner = []
    for i in range(N):
        inner.append([float(xs[i]), float(ys[i]), angle_deg])
    
    center = [cx, cy]
    side = float(best_L_global)
    rotation = angle_deg
    
    return inner, center, side, rotation
