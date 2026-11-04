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

def _hex_edge_normals(rot_deg):
    base = np.radians(np.array([30, 90, 150, 210, 270, 330])) + np.radians(rot_deg)
    return np.column_stack((np.cos(base), np.sin(base)))

def _minkowski_hex_distance(d, rot1, rot2):
    """
    Compute the minimum distance from point d to the Minkowski sum of two unit hexagons
    with rotations rot1 and rot2. The Minkowski sum of two regular hexagons is a convex
    polygon. We compute the support function and find penetration.
    Returns: negative if d is inside (overlap), positive if outside (separated).
    """
    # Minkowski sum of two unit hexagons with rotations r1, r2
    # The vertices of the Minkowski sum are the sums of edge-parallel vertices
    # For regular hexagons, the Minkowski sum is a 12-gon (or hexagon if same rotation)
    
    # Approach: check if d is inside the Minkowski sum by checking all edge constraints
    # The Minkowski sum H1 + H2 has the property that its support function is h(d) = h1(d) + h2(d)
    # where h1, h2 are support functions of H1, H2
    
    # For a regular hexagon with rotation r, the support function in direction u is:
    # h(u) = max over vertices v of u·v
    # For unit hexagon with rotation r, vertices are at angles r, r+60, ..., r+300
    # h(u) = max_k cos(angle(u) - (r + 60k))
    
    # The Minkowski sum contains d iff for all directions u: u·d <= h1(u) + h2(u)
    # Equivalently, d is outside iff exists u such that u·d > h1(u) + h2(u)
    
    # To find penetration depth, we check the 12 critical directions (edge normals of both hexagons)
    angles1 = np.radians(np.array([30, 90, 150, 210, 270, 330])) + np.radians(rot1)
    angles2 = np.radians(np.array([30, 90, 150, 210, 270, 330])) + np.radians(rot2)
    
    all_angles = np.concatenate([angles1, angles2])
    normals = np.column_stack((np.cos(all_angles), np.sin(all_angles)))
    
    # Support function of unit hexagon in direction n:
    # h(n) = max over the 6 vertex directions of n · v_k
    # For unit hexagon with rotation r, vertex k is at angle r + 60k
    # v_k = (cos(r+60k), sin(r+60k))
    # n · v_k = cos(angle(n) - r - 60k)
    
    def _support(n, rot):
        ang_n = np.arctan2(n[1], n[0])
        vals = np.cos(ang_n - np.radians(np.array([0, 60, 120, 180, 240, 300]) - rot))
        return np.max(vals)
    
    min_slack = float('inf')
    for n in normals:
        h1 = _support(n, rot1)
        h2 = _support(n, rot2)
        slack = h1 + h2 - d @ n
        if slack < 0:
            return slack  # d is outside the Minkowski sum
        if slack < min_slack:
            min_slack = slack
    
    return -min_slack  # d is inside, penetration depth is min_slack

def _minkowski_push_direction(d, rot1, rot2):
    """
    Return the direction and magnitude to push hex2 (centered at d relative to hex1)
    to resolve overlap. d = center2 - center1.
    Returns: (direction_unit_vector, penetration_depth)
    If no overlap, returns (None, 0)
    """
    angles1 = np.radians(np.array([30, 90, 150, 210, 270, 330])) + np.radians(rot1)
    angles2 = np.radians(np.array([30, 90, 150, 210, 270, 330])) + np.radians(rot2)
    
    all_angles = np.concatenate([angles1, angles2])
    normals = np.column_stack((np.cos(all_angles), np.sin(all_angles)))
    
    def _support(n, rot):
        ang_n = np.arctan2(n[1], n[0])
        vals = np.cos(ang_n - np.radians(np.array([0, 60, 120, 180, 240, 300]) - rot))
        return np.max(vals)
    
    min_slack = float('inf')
    best_normal = None
    for n in normals:
        h1 = _support(n, rot1)
        h2 = _support(n, rot2)
        slack = h1 + h2 - d @ n
        if slack < 0:
            return None, 0.0  # No overlap
        if slack < min_slack:
            min_slack = slack
            best_normal = n
    
    # The pushing direction is along best_normal (to increase d·n)
    return best_normal, min_slack

def _outer_hex_side_fast(points, outer_rot_deg):
    angles = np.radians(np.array([30, 90, 150, 210, 270, 330]) + outer_rot_deg)
    normals = np.column_stack((np.cos(angles), np.sin(angles)))
    proj = points @ normals.T
    max_proj = np.max(np.abs(proj))
    return (2.0 / np.sqrt(3.0)) * max_proj

def _check_disjoint(inner):
    n = len(inner)
    for i in range(n):
        for j in range(i + 1, n):
            d = inner[j, 0] - inner[i, 0], inner[j, 1] - inner[i, 1]
            dvec = np.array([inner[j, 0] - inner[i, 0], inner[j, 1] - inner[i, 1]])
            dist = _minkowski_hex_distance(dvec, inner[i, 2], inner[j, 2])
            if dist < -1e-9:
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

def _sequential_displacement(pts, max_iters=200, step_factor=0.5, t0=None, deadline=None):
    """
    Sequentially resolve overlaps by pushing hexagons apart along Minkowski sum normals.
    pts: (12, 3) array of [x, y, angle_degrees]
    Returns: modified pts array
    """
    pts = pts.copy()
    for it in range(max_iters):
        if t0 is not None and time.time() - t0 > deadline:
            break
        
        moved = False
        for i in range(12):
            for j in range(i + 1, 12):
                dvec = np.array([pts[j, 0] - pts[i, 0], pts[j, 1] - pts[i, 1]])
                direction, depth = _minkowski_push_direction(dvec, pts[i, 2], pts[j, 2])
                if direction is not None:
                    # Push both apart by depth/2 along direction
                    push = (depth * step_factor) / 2.0
                    pts[i, 0] -= push * direction[0]
                    pts[i, 1] -= push * direction[1]
                    pts[j, 0] += push * direction[0]
                    pts[j, 1] += push * direction[1]
                    moved = True
        
        if not moved:
            break
    
    return pts

def _objective_minkowski(p, t0, deadline):
    """
    Objective: outer side length + overlap penalty using Minkowski sum distances.
    p: [x0, y0, a0, ..., x11, y11, a11, ox, oy, oang]
    """
    if time.time() - t0 > deadline:
        return 1e6
    
    inner = p[:36].reshape(12, 3)
    ox = p[36]
    oy = p[37]
    oang = p[38]
    
    # Overlap penalty using Minkowski sum
    penalty = 0.0
    for i in range(12):
        for j in range(i + 1, 12):
            dvec = np.array([inner[j, 0] - inner[i, 0], inner[j, 1] - inner[i, 1]])
            dist = _minkowski_hex_distance(dvec, inner[i, 2], inner[j, 2])
            if dist < 0:
                penalty += (-dist) ** 2 * 50.0
    
    # Calculate outer side
    verts_list = []
    for i in range(12):
        cx, cy, rot = inner[i]
        v = _hex_vertices(cx, cy, 1.0, rot)
        verts_list.append(v)
    
    all_pts = np.vstack(verts_list)
    all_pts[:, 0] -= ox
    all_pts[:, 1] -= oy
    
    L = _outer_hex_side_fast(all_pts, oang)
    
    return L + penalty

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
    for _ in range(6):
        pts = np.random.randn(12, 3) * 2.5
        pts[:, 2] = np.random.uniform(0, 60, 12)
        pts[:, 0] -= np.mean(pts[:, 0])
        pts[:, 1] -= np.mean(pts[:, 1])
        configs.append(pts)
    
    best_L = float('inf')
    best_params = None
    
    for cfg in configs:
        if time_left() < 15:
            break
        
        # Phase 1: Sequential displacement to resolve overlaps
        resolved = _sequential_displacement(cfg, max_iters=100, step_factor=0.4, t0=t0, deadline=DEADLINE - 5)
        
        # Center the configuration
        resolved[:, 0] -= np.mean(resolved[:, 0])
        resolved[:, 1] -= np.mean(resolved[:, 1])
        
        # Phase 2: Try Nelder-Mead refinement from the resolved state
        for oang in [0.0, 30.0]:
            if time_left() < 8:
                break
            
            init_params = np.concatenate((resolved.flatten(), [0.0, 0.0, oang]))
            
            try:
                res = minimize(
                    _objective_minkowski,
                    init_params,
                    args=(t0, DEADLINE - 3),
                    method='Nelder-Mead',
                    options={
                        'maxiter': 150,
                        'xatol': 1e-7,
                        'fatol': 1e-7,
                        'adaptive': True
                    }
                )
                
                if res.fun < best_L:
                    inner = res.x[:36].reshape(12, 3)
                    if _check_disjoint(inner):
                        ox = res.x[36]
                        oy = res.x[37]
                        oang = res.x[38]
                        all_pts = np.vstack([_hex_vertices(inner[i,0], inner[i,1], 1.0, inner[i,2]) for i in range(12)])
                        all_pts[:, 0] -= ox
                        all_pts[:, 1] -= oy
                        L = _outer_hex_side_fast(all_pts, oang)
                        
                        if L < best_L:
                            best_L = L
                            best_params = res.x.copy()
            except Exception:
                pass
        
        # Also try the sequentially-displaced config directly (no NM refinement)
        if time_left() < 5:
            break
        for oang in [0.0, 30.0]:
            if time_left() < 3:
                break
            inner = resolved
            if _check_disjoint(inner):
                all_pts = np.vstack([_hex_vertices(inner[i,0], inner[i,1], 1.0, inner[i,2]) for i in range(12)])
                all_pts[:, 0] -= 0.0
                all_pts[:, 1] -= 0.0
                L = _outer_hex_side_fast(all_pts, oang)
                if L < best_L:
                    best_L = L
                    best_params = np.concatenate((inner.flatten(), [0.0, 0.0, oang]))
    
    # Fallback
    if best_params is None:
        cfg = _make_hex_grid(4, 3, gap=0.5)
        best_params = np.concatenate((cfg.flatten(), [0.0, 0.0, 0.0]))
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
    
    outer_center = (float(best_params[36]), float(best_params[37]))
    outer_angle = float(best_params[38])
    
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
