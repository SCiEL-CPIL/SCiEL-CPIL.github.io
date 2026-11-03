import math
import time
import numpy as np
from scipy.optimize import minimize

N = 12
DEADLINE = 145.0
SQRT3 = math.sqrt(3)

# Pre-compute unit hexagon properties
_HEX_ANGLES = np.radians(np.arange(0, 360, 60))
_UNIT_HEX_VERTS = np.column_stack((np.cos(_HEX_ANGLES), np.sin(_HEX_ANGLES)))
_EDGE_NORMALS = np.zeros((6, 2))
for _i in range(6):
    _ex = _UNIT_HEX_VERTS[(_i + 1) % 6, 0] - _UNIT_HEX_VERTS[_i, 0]
    _ey = _UNIT_HEX_VERTS[(_i + 1) % 6, 1] - _UNIT_HEX_VERTS[_i, 1]
    _nx, _ny = -_ey, _ex
    _norm = math.sqrt(_nx * _nx + _ny * _ny)
    _EDGE_NORMALS[_i] = [_nx / _norm, _ny / _norm]

_UX = _UNIT_HEX_VERTS[:, 0]
_UY = _UNIT_HEX_VERTS[:, 1]
_ENX = _EDGE_NORMALS[:, 0]
_ENY = _EDGE_NORMALS[:, 1]

# Minkowski Sum of two unit hexagons (for contact detection)
# The Minkowski sum of two regular hexagons of side 1 is a regular dodecagon?
# Actually, sum of two hexagons is a 12-gon if rotated, but if aligned, it's a larger hexagon?
# No, Minkowski sum of two convex polygons is the convex hull of the sums of vertices.
# For two unit hexagons (side 1), the Minkowski sum is a regular hexagon of side 2?
# Let's check.
# Hexagon 1: vertices at angles 0, 60, 120...
# Hexagon 2: vertices at angles 0, 60, 120...
# Sum of vertex 0 (1,0) and vertex 0 (1,0) is (2,0).
# Sum of vertex 0 (1,0) and vertex 60 (0.5, sqrt3/2) is (1.5, sqrt3/2).
# This forms a hexagon with side length 2?
# Distance from (2,0) to (1.5, sqrt3/2): sqrt(0.25 + 0.75) = 1.
# Wait, the Minkowski sum of two unit hexagons is a unit hexagon scaled by 2?
# Let's verify the support function.
# h_H(u) = max dot(v, u).
# h_{H1+H2}(u) = h_H1(u) + h_H2(u).
# If H1=H2, h_{sum} = 2 h_H.
# So the Minkowski sum is indeed a hexagon of side length 2 (if aligned).
# If rotated, it's the convex hull of the 12 sums.

def _get_minkowski_vertices(angle_diff):
    """
    Returns the vertices of the Minkowski sum of two unit hexagons,
    where the second is rotated by angle_diff relative to the first.
    The sum is a convex polygon. We need its support function or vertices.
    Since we only need to check if a point (relative position) is inside,
    and to project points onto the boundary, we can compute the convex hull
    of the 36 possible vertex sums (6x6).
    """
    c = math.cos(angle_diff)
    s = math.sin(angle_diff)
    
    # Vertices of H1
    v1x = _UX
    v1y = _UY
    # Vertices of H2 rotated
    v2x = _UX * c - _UY * s
    v2y = _UX * s + _UY * c
    
    # All sums
    sums = []
    for i in range(6):
        for j in range(6):
            sums.append((v1x[i] + v2x[j], v1y[i] + v2y[j]))
    
    sums = np.array(sums)
    
    # Compute Convex Hull (simple Graham scan or just use scipy if available, but let's stick to stdlib/numpy for speed and simplicity)
    # Actually, for 36 points, a simple hull is fast.
    # Or even simpler: The Minkowski sum of two convex polygons is convex.
    # We can just check if a point is inside by checking against edges.
    # But for optimization, we need the boundary distance.
    
    # Let's just return the points and let the caller compute the hull or use a point-in-polygon test.
    # For efficiency, we'll pre-compute the hull edges for specific angles?
    # No, angle is continuous.
    
    # A faster way: The support function of the sum is h1(u) + h2(u).
    # The distance from the origin to the boundary in direction u is h_sum(u).
    # If we want to know if two hexagons are disjoint, we need to check if
    # the vector d = p2 - p1 is OUTSIDE the Minkowski sum (scaled by 1? No, the Minkowski sum defines the set of vectors that cause intersection).
    # Intersection occurs if d is in (H1 - H2) = H1 + (-H2).
    # -H2 is H2 rotated by 180.
    # So we need to check if d is inside H1 + (-H2).
    
    return sums

def _is_point_in_convex_polygon(pt, verts):
    """
    Check if point pt is inside convex polygon verts.
    verts is (M, 2).
    """
    if len(verts) < 3:
        return False
    n = len(verts)
    for i in range(n):
        p1 = verts[i]
        p2 = verts[(i + 1) % n]
        # Cross product of (p2-p1) and (pt-p1)
        dx1 = p2[0] - p1[0]
        dy1 = p2[1] - p1[1]
        dx2 = pt[0] - p1[0]
        dy2 = pt[1] - p1[1]
        cross = dx1 * dy2 - dy1 * dx2
        if cross < -1e-9:
            return False
    return True

def _hex_disjoint_fast(h1, h2):
    """
    Fast check if two hexagons are disjoint.
    h1, h2: (x, y, angle_rad)
    """
    dx = h1[0] - h2[0]
    dy = h1[1] - h2[1]
    dist_sq = dx * dx + dy * dy
    # Max distance of any point in hexagon from center is 1 (circumradius).
    # If dist > 2, definitely disjoint.
    if dist_sq > 4.0:
        return True
    
    # Check using Separating Axis Theorem (SAT)
    # This is more robust and faster than Minkowski for disjointness.
    a1, a2 = h1[2], h2[2]
    c1, s1 = math.cos(a1), math.sin(a1)
    c2, s2 = math.cos(a2), math.sin(a2)
    
    v1x = h1[0] + _UX * c1 - _UY * s1
    v1y = h1[1] + _UX * s1 + _UY * c1
    v2x = h2[0] + _UX * c2 - _UY * s2
    v2y = h2[1] + _UX * s2 + _UY * c2
    
    en1x = _ENX * c1 - _ENY * s1
    en1y = _ENX * s1 + _ENY * c1
    en2x = _ENX * c2 - _ENY * s2
    en2y = _ENX * s2 + _ENY * c2
    
    # Project onto normals of h1
    for i in range(6):
        nx, ny = en1x[i], en1y[i]
        p1 = v1x * nx + v1y * ny
        p2 = v2x * nx + v2y * ny
        if p1.max() < p2.min() or p2.max() < p1.min():
            return True
            
    # Project onto normals of h2
    for i in range(6):
        nx, ny = en2x[i], en2y[i]
        p1 = v1x * nx + v1y * ny
        p2 = v2x * nx + v2y * ny
        if p1.max() < p2.min() or p2.max() < p1.min():
            return True
            
    return False

def _is_valid(positions, angles_deg):
    angles_rad = np.radians(angles_deg)
    for i in range(N):
        for j in range(i + 1, N):
            h1 = (positions[i, 0], positions[i, 1], angles_rad[i])
            h2 = (positions[j, 0], positions[j, 1], angles_rad[j])
            if not _hex_disjoint_fast(h1, h2):
                return False
    return True

def _compute_min_outer_side(positions, angles_deg):
    angles_rad = np.radians(angles_deg)
    c = np.cos(angles_rad)
    s = np.sin(angles_rad)
    rx = _UX[None, :] * c[:, None] - _UY[None, :] * s[:, None]
    ry = _UX[None, :] * s[:, None] + _UY[None, :] * c[:, None]
    verts_x = rx + positions[:, 0:1]
    verts_y = ry + positions[:, 1:2]
    verts_flat = np.stack((verts_x, verts_y), axis=-1).reshape(-1, 2)
    center = np.mean(verts_flat, axis=0)
    offsets = verts_flat - center

    def eval_phi(phi_deg):
        phi = np.radians(phi_deg)
        normal_angles = phi + np.pi / 6 + np.arange(6) * np.pi / 3
        nx = np.cos(normal_angles)
        ny = np.sin(normal_angles)
        proj = offsets[:, 0:1] * nx[None, :] + offsets[:, 1:2] * ny[None, :]
        abs_proj = np.abs(proj)
        max_projs = np.max(abs_proj, axis=0)
        side_vals = (2.0 / SQRT3) * max_projs
        return float(np.max(side_vals))

    # Coarse scan
    phi_vals = np.radians(np.arange(0, 60, 0.5))
    normal_angles = phi_vals[:, None] + np.pi / 6 + np.arange(6) * np.pi / 3
    nx = np.cos(normal_angles)
    ny = np.sin(normal_angles)
    proj = offsets[:, 0:1, None] * nx[None, :, :] + offsets[:, 1:2, None] * ny[None, :, :]
    abs_proj = np.abs(proj)
    max_projs = np.max(abs_proj, axis=0)
    side_vals = (2.0 / SQRT3) * max_projs
    best_side = float(np.max(side_vals).min())
    best_idx = int(np.argmin(np.max(side_vals, axis=1)))
    best_phi = float(best_idx * 0.5)

    # Fine tune with golden-section
    lo = max(0.0, best_phi - 0.5)
    hi = min(60.0, best_phi + 0.5)
    gr = (math.sqrt(5) - 1) / 2
    x1 = hi - gr * (hi - lo)
    x2 = lo + gr * (hi - lo)
    f1 = eval_phi(x1)
    f2 = eval_phi(x2)
    for _ in range(50):
        if f1 < f2:
            hi = x2
            x2 = x1
            f2 = f1
            x1 = hi - gr * (hi - lo)
            f1 = eval_phi(x1)
        else:
            lo = x1
            x1 = x2
            f1 = f2
            x2 = lo + gr * (hi - lo)
            f2 = eval_phi(x2)
    best_phi = (lo + hi) / 2.0
    best_side = min(best_side, eval_phi(best_phi))

    return best_side, best_phi, float(center[0]), float(center[1])

def _make_honeycomb():
    a1 = np.array([SQRT3, 0.0])
    a2 = np.array([SQRT3 / 2.0, 1.5])
    lattice_coords = [
        (0, 0), (1, 0), (-1, 0), (0, 1), (0, -1),
        (1, -1), (-1, 1), (2, -1), (-2, 1),
        (1, 1), (-1, -1), (2, 0),
    ]
    pos = np.zeros((N, 2))
    ang = np.zeros(N)
    for idx, (i, j) in enumerate(lattice_coords):
        pos[idx] = i * a1 + j * a2
        ang[idx] = 0.0
    return pos, ang

def _make_compact_hex_cluster():
    a1 = np.array([SQRT3, 0.0])
    a2 = np.array([SQRT3 / 2.0, 1.5])
    coords = [
        (0, 0), (1, 0), (-1, 0), (0, 1), (0, -1), (1, -1), (-1, 1),
        (2, -1), (-2, 1), (2, 0), (-2, 0), (0, 2),
    ]
    pos = np.zeros((N, 2))
    ang = np.zeros(N)
    for idx, (i, j) in enumerate(coords):
        pos[idx] = i * a1 + j * a2
        ang[idx] = 0.0
    return pos, ang

def _make_random():
    pos = np.random.uniform(-3, 3, (N, 2))
    ang = np.random.uniform(-30, 30, N)
    return pos, ang

def _resolve_overlaps(pos, ang, max_iters=50):
    """
    Structural Change: Instead of random nudging, we identify overlapping pairs
    and push them apart along the vector connecting their centers, or better,
    we use a simple iterative relaxation that minimizes the sum of penetration depths.
    """
    for _ in range(max_iters):
        if _is_valid(pos, ang):
            break
            
        moved = False
        for i in range(N):
            for j in range(i + 1, N):
                if not _hex_disjoint_fast((pos[i,0], pos[i,1], ang[i]), (pos[j,0], pos[j,1], ang[j])):
                    # Calculate push direction
                    dx = pos[j, 0] - pos[i, 0]
                    dy = pos[j, 1] - pos[i, 1]
                    dist = math.sqrt(dx*dx + dy*dy)
                    if dist < 1e-6:
                        dx = 1.0
                        dy = 0.0
                        dist = 1.0
                    else:
                        dx /= dist
                        dy /= dist
                    
                    # Push apart by a small amount
                    push = 0.1
                    pos[i, 0] -= dx * push
                    pos[i, 1] -= dy * push
                    pos[j, 0] += dx * push
                    pos[j, 1] += dy * push
                    moved = True
        if not moved:
            break
            
    return _is_valid(pos, ang)

def _contact_graph(pos, ang, tolerance=1e-4):
    """
    Identify pairs of hexagons that are 'touching' or very close.
    A pair is 'in contact' if the distance between their boundaries is < tolerance.
    We approximate boundary distance by checking if they intersect, or if the distance
    between centers is close to the minimum possible distance for that orientation.
    
    For a more robust contact detection, we can use the Minkowski sum.
    If d = p2 - p1 is inside the Minkowski sum, they intersect.
    If d is on the boundary, they touch.
    
    For simplicity in this step, we'll consider them in contact if they intersect
    or if the distance between centers is within a small epsilon of the minimum
    separating distance.
    """
    contacts = []
    angles_rad = np.radians(ang)
    for i in range(N):
        for j in range(i + 1, N):
            h1 = (pos[i, 0], pos[i, 1], angles_rad[i])
            h2 = (pos[j, 0], pos[j, 1], angles_rad[j])
            
            # Check intersection
            if not _hex_disjoint_fast(h1, h2):
                contacts.append((i, j))
                continue
            
            # Check proximity
            # We can compute the minimum distance between the two polygons.
            # For now, we'll just use a heuristic: if center distance < 2.0, check closely.
            dx = pos[i, 0] - pos[j, 0]
            dy = pos[i, 1] - pos[j, 1]
            dist = math.sqrt(dx*dx + dy*dy)
            
            if dist < 2.0:
                # SAT to find minimum separation
                a1, a2 = angles_rad[i], angles_rad[j]
                c1, s1 = math.cos(a1), math.sin(a1)
                c2, s2 = math.cos(a2), math.sin(a2)
                
                v1x = pos[i, 0] + _UX * c1 - _UY * s1
                v1y = pos[i, 1] + _UX * s1 + _UY * c1
                v2x = pos[j, 0] + _UX * c2 - _UY * s2
                v2y = pos[j, 1] + _UX * s2 + _UY * c2
                
                en1x = _ENX * c1 - _ENY * s1
                en1y = _ENX * s1 + _ENY * c1
                en2x = _ENX * c2 - _ENY * s2
                en2y = _ENX * s2 + _ENY * c2
                
                min_sep = float('inf')
                
                # Check normals of h1
                for k in range(6):
                    nx, ny = en1x[k], en1y[k]
                    p1 = v1x * nx + v1y * ny
                    p2 = v2x * nx + v2y * ny
                    sep = p2.min() - p1.max()
                    if sep < 0: sep = 0 # Should not happen if disjoint, but safety
                    if sep < min_sep:
                        min_sep = sep
                        
                # Check normals of h2
                for k in range(6):
                    nx, ny = en2x[k], en2y[k]
                    p1 = v1x * nx + v1y * ny
                    p2 = v2x * nx + v2y * ny
                    sep = p1.min() - p2.max()
                    if sep < 0: sep = 0
                    if sep < min_sep:
                        min_sep = sep

                if min_sep < tolerance:
                    contacts.append((i, j))
                    
    return contacts

def _local_search_contact(pos, ang, time_limit, rng):
    """
    High-precision local search that respects contact constraints.
    We treat the configuration as a system of particles.
    We perform a series of small moves, but we penalize overlap heavily.
    To improve on the current program, we will use a more sophisticated
    objective function that includes the outer side length and a
    smooth penalty for overlaps, and use Nelder-Mead or BFGS if possible,
    but since we have many variables, we stick to a refined local search
    with adaptive steps and strict validity checks.
    """
    current_pos = pos.copy()
    current_ang = ang.copy()
    current_side, _, _, _ = _compute_min_outer_side(current_pos, current_ang)
    best_side = current_side
    best_pos = current_pos.copy()
    best_ang = current_ang.copy()

    t_start = time.time()
    step = 0.05
    min_step = 1e-5
    no_improve_count = 0
    max_no_improve = 10

    while time.time() - t_start < time_limit and step > min_step:
        improved = False
        order = rng.permutation(N)
        for idx in order:
            if time.time() - t_start > time_limit:
                break
            
            # Try moving in x, y, angle
            best_move = None
            best_move_side = best_side
            
            for dim in range(3):
                for direction in [1, -1]:
                    new_pos = best_pos.copy()
                    new_ang = best_ang.copy()
                    
                    if dim == 0:
                        new_pos[idx, 0] += direction * step
                    elif dim == 1:
                        new_pos[idx, 1] += direction * step
                    else:
                        new_ang[idx] += direction * step * 10 # Angle step is larger in degrees? 
                        # Let's keep angle step in radians for consistency in calculation, 
                        # but ang is in degrees. So step * 10 degrees.
                        
                    if not _is_valid(new_pos, new_ang):
                        continue
                    
                    new_side, _, _, _ = _compute_min_outer_side(new_pos, new_ang)
                    if new_side < best_move_side - 1e-12:
                        best_move_side = new_side
                        best_move = (dim, direction, new_pos, new_ang)

            if best_move is not None:
                dim, direction, new_pos, new_ang = best_move
                best_pos = new_pos
                best_ang = new_ang
                best_side = best_move_side
                improved = True
                no_improve_count = 0

        if not improved:
            no_improve_count += 1
            if no_improve_count >= max_no_improve:
                step *= 0.7
                no_improve_count = 0
            else:
                step *= 0.9

    return best_side, best_pos, best_ang

def _sa_search(pos0, ang0, time_limit, rng):
    """
    Simulated Annealing with a focus on escaping local minima by
    allowing temporary overlaps (soft constraint) but with a strong
    penalty that scales with the degree of overlap.
    """
    current_pos = pos0.copy()
    current_ang = ang0.copy()
    
    def soft_obj(p, a):
        side, _, _, _ = _compute_min_outer_side(p, a)
        penalty = 0.0
        for i in range(N):
            for j in range(i + 1, N):
                if not _hex_disjoint_fast((p[i,0], p[i,1], a[i]), (p[j,0], p[j,1], a[j])):
                    # Estimate overlap depth
                    dx = p[i, 0] - p[j, 0]
                    dy = p[i, 1] - p[j, 1]
                    dist = math.sqrt(dx*dx + dy*dy)
                    # Approximate min dist is SQRT3
                    if dist < SQRT3:
                        penalty += (SQRT3 - dist) ** 2
                    else:
                        penalty += 0.1 # Small penalty for any intersection
        return side + 50.0 * penalty

    current_obj = soft_obj(current_pos, current_ang)
    best_pos = current_pos.copy()
    best_ang = current_ang.copy()
    best_side = float('inf')
    
    if _is_valid(current_pos, current_ang):
        s, _, _, _ = _compute_min_outer_side(current_pos, current_ang)
        best_side = s

    t_start = time.time()
    T = 0.5
    T_min = 0.0001
    cooling = 0.99

    while time.time() - t_start < time_limit and T > T_min:
        idx = rng.randint(N)
        new_pos = current_pos.copy()
        new_ang = current_ang.copy()
        
        step = 0.1 * (1.0 + T)
        new_pos[idx, 0] += rng.uniform(-step, step)
        new_pos[idx, 1] += rng.uniform(-step, step)
        new_ang[idx] += rng.uniform(-5, 5)
        
        new_obj = soft_obj(new_pos, new_ang)
        
        if new_obj < current_obj or rng.random() < math.exp(-(new_obj - current_obj) / T):
            current_pos = new_pos
            current_ang = new_ang
            current_obj = new_obj
            
            if _is_valid(current_pos, current_ang):
                s, _, _, _ = _compute_min_outer_side(current_pos, current_ang)
                if s < best_side:
                    best_side = s
                    best_pos = current_pos.copy()
                    best_ang = current_ang.copy()
        
        T *= cooling
    
    return best_side, best_pos, best_ang

def construct():
    t0 = time.time()
    rng = np.random.RandomState(42)

    inits = [
        _make_honeycomb(),
        _make_compact_hex_cluster(),
        _make_random(),
        _make_random(),
    ]

    best_side = float('inf')
    best_positions = None
    best_angles = None
    best_phi = 0.0
    best_cx = 0.0
    best_cy = 0.0

    for init_idx, (pos0, ang0) in enumerate(inits):
        if time.time() - t0 > DEADLINE - 20.0:
            break

        # Ensure validity
        if not _is_valid(pos0, ang0):
            _resolve_overlaps(pos0, ang0, max_iters=100)
            if not _is_valid(pos0, ang0):
                continue

        s0, phi0, cx0, cy0 = _compute_min_outer_side(pos0, ang0)
        if s0 < best_side:
            best_side = s0
            best_positions = pos0.copy()
            best_angles = ang0.copy()
            best_phi = phi0
            best_cx = cx0
            best_cy = cy0

        remaining = DEADLINE - 15.0 - (time.time() - t0)
        if remaining < 5.0:
            break
        
        # Phase 1: SA
        sa_budget = min(remaining * 0.4, 15.0)
        if init_idx > 1:
            sa_budget = min(remaining * 0.3, 10.0)
            
        sa_side, sa_pos, sa_ang = _sa_search(pos0, ang0, sa_budget, rng)
        
        if sa_side < best_side:
            best_side = sa_side
            best_positions = sa_pos
            best_angles = sa_ang
            best_side, best_phi, best_cx, best_cy = _compute_min_outer_side(best_positions, best_angles)

        # Phase 2: Local Search
        remaining2 = DEADLINE - 15.0 - (time.time() - t0)
        if remaining2 < 5.0:
            break
            
        ls_budget = min(remaining2 * 0.6, 20.0)
        ls_side, ls_pos, ls_ang = _local_search_contact(best_positions, best_angles, ls_budget, rng)
        
        if ls_side < best_side:
            best_side = ls_side
            best_positions = ls_pos
            best_angles = ls_ang
            best_side, best_phi, best_cx, best_cy = _compute_min_outer_side(best_positions, best_angles)

    # Final verification
    if best_positions is None or not _is_valid(best_positions, best_angles):
        pos0, ang0 = _make_honeycomb()
        _resolve_overlaps(pos0, ang0, max_iters=500)
        best_positions = pos0.copy()
        best_angles = ang0.copy()
        best_side, best_phi, best_cx, best_cy = _compute_min_outer_side(best_positions, best_angles)

    inner = []
    for i in range(N):
        inner.append([float(best_positions[i, 0]), float(best_positions[i, 1]), float(best_angles[i])])

    center = (float(best_cx), float(best_cy))
    side = float(best_side)
    angle = float(best_phi)

    return inner, center, side, angle
