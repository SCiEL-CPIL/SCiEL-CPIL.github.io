import numpy as np
import time
from scipy.optimize import differential_evolution, minimize

N = 12
DEADLINE = 140.0

def _hex_vertices(cx, cy, side=1.0, rot_deg=0.0):
    angles = np.radians(np.array([0, 60, 120, 180, 240, 300])) + np.radians(rot_deg)
    x = cx + side * np.cos(angles)
    y = cy + side * np.sin(angles)
    return np.column_stack((x, y))

def _hex_edge_normals(rot_deg):
    base = np.radians(np.array([30, 90, 150, 210, 270, 330])) + np.radians(rot_deg)
    return np.column_stack((np.cos(base), np.sin(base)))

def _sat_overlap_fast(v1, v2, n1, n2):
    """Returns minimum separation (positive if disjoint, negative if overlapping)."""
    min_overlap = float('inf')
    for k in range(6):
        n = n1[k]
        proj1 = v1 @ n
        proj2 = v2 @ n
        min1, max1 = np.min(proj1), np.max(proj1)
        min2, max2 = np.min(proj2), np.max(proj2)
        if max1 < min2 or max2 < min1:
            gap = min2 - max1 if min2 > max1 else min1 - max2
            return gap
        overlap = min(max1, max2) - max(min1, min2)
        if overlap < min_overlap:
            min_overlap = overlap
    for k in range(6):
        n = n2[k]
        proj1 = v1 @ n
        proj2 = v2 @ n
        min1, max1 = np.min(proj1), np.max(proj1)
        min2, max2 = np.min(proj2), np.max(proj2)
        if max1 < min2 or max2 < min1:
            gap = min2 - max1 if min2 > max1 else min1 - max2
            return gap
        overlap = min(max1, max2) - max(min1, min2)
        if overlap < min_overlap:
            min_overlap = overlap
    return -min_overlap

def _outer_hex_side_fast(points, outer_rot_deg):
    angles = np.radians(np.array([30, 90, 150, 210, 270, 330]) + outer_rot_deg)
    normals = np.column_stack((np.cos(angles), np.sin(angles)))
    proj = points @ normals.T
    max_proj = np.max(np.abs(proj))
    return (2.0 / np.sqrt(3.0)) * max_proj

def _check_disjoint(inner, eps=1e-9):
    n = len(inner)
    verts = []
    norms = []
    for i in range(n):
        x, y, rot = inner[i]
        v = _hex_vertices(x, y, 1.0, rot)
        nm = _hex_edge_normals(rot)
        verts.append(v)
        norms.append(nm)
    for i in range(n):
        for j in range(i + 1, n):
            d = _sat_overlap_fast(verts[i], verts[j], norms[i], norms[j])
            if d < -eps:
                return False
    return True

def _compute_L(inner, outer_rot_deg=0.0):
    """Compute the outer hexagon side length for a given inner configuration."""
    all_pts = np.vstack([_hex_vertices(inner[i,0], inner[i,1], 1.0, inner[i,2]) for i in range(12)])
    ox = np.mean(all_pts[:, 0])
    oy = np.mean(all_pts[:, 1])
    all_pts_centered = all_pts.copy()
    all_pts_centered[:, 0] -= ox
    all_pts_centered[:, 1] -= oy
    return _outer_hex_side_fast(all_pts_centered, outer_rot_deg), (ox, oy)

def _objective(p, t0):
    """Objective function for optimization: outer side length + collision penalty."""
    if time.time() - t0 > DEADLINE:
        return 1e6
    
    inner = p[:36].reshape(12, 3)
    oang = p[36]
    
    verts_list = []
    norms_list = []
    for i in range(12):
        cx, cy, rot = inner[i]
        v = _hex_vertices(cx, cy, 1.0, rot)
        nm = _hex_edge_normals(rot)
        verts_list.append(v)
        norms_list.append(nm)
    
    penalty = 0.0
    for i in range(12):
        for j in range(i + 1, 12):
            d = _sat_overlap_fast(verts_list[i], verts_list[j], norms_list[i], norms_list[j])
            if d < 0:
                penalty += (-d) ** 2 * 50.0
    
    all_pts = np.vstack(verts_list)
    ox = np.mean(all_pts[:, 0])
    oy = np.mean(all_pts[:, 1])
    all_pts_centered = all_pts.copy()
    all_pts_centered[:, 0] -= ox
    all_pts_centered[:, 1] -= oy
    
    L = _outer_hex_side_fast(all_pts_centered, oang)
    
    return L + penalty

def _polish(inner, t0, oang_init=0.0, max_iter=500):
    """Local optimization using Nelder-Mead."""
    if time.time() - t0 > DEADLINE - 5:
        return None
    
    all_pts = np.vstack([_hex_vertices(inner[i,0], inner[i,1], 1.0, inner[i,2]) for i in range(12)])
    ox = np.mean(all_pts[:, 0])
    oy = np.mean(all_pts[:, 1])
    
    init_params = np.concatenate((inner.flatten(), [oang_init]))
    
    try:
        res = minimize(
            _objective,
            init_params,
            args=(t0,),
            method='Nelder-Mead',
            options={
                'maxiter': max_iter,
                'xatol': 1e-10,
                'fatol': 1e-10,
                'adaptive': True
            }
        )
        
        if res.fun < 1e5:
            inner_opt = res.x[:36].reshape(12, 3)
            oang_opt = res.x[36]
            if _check_disjoint(inner_opt):
                L_opt, center_opt = _compute_L(inner_opt, oang_opt)
                return (L_opt, inner_opt, center_opt, oang_opt)
    except Exception:
        pass
    
    return None

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
    if len(pts) > 12:
        pts = pts[:12]
    elif len(pts) < 12:
        while len(pts) < 12:
            pts = np.vstack([pts, [0, 0, rot_offset]])
    pts = np.array(pts)
    pts[:, 0] -= np.mean(pts[:, 0])
    pts[:, 1] -= np.mean(pts[:, 1])
    return pts

def _make_ring(gap=0.0):
    pts = []
    sqrt3 = np.sqrt(3)
    d1 = sqrt3 + gap
    d2 = 2 * sqrt3 + gap
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

def _make_packed_cluster(gap=0.0):
    """Create a tightly packed cluster of 12 hexagons in a hexagonal arrangement."""
    pts = []
    sqrt3 = np.sqrt(3)
    # Center hexagon
    pts.append([0, 0, 0.0])
    # 6 neighbors at distance sqrt(3)
    for k in range(6):
        ang = np.radians(60 * k)
        pts.append([sqrt3 * np.cos(ang), sqrt3 * np.sin(ang), 0.0])
    # 5 more at distance 2*sqrt(3), offset by 30 degrees
    for k in range(5):
        ang = np.radians(60 * k + 30)
        pts.append([2 * sqrt3 * np.cos(ang), 2 * sqrt3 * np.sin(ang), 0.0])
    pts = np.array(pts)
    pts[:, 0] -= np.mean(pts[:, 0])
    pts[:, 1] -= np.mean(pts[:, 1])
    return pts

def _make_strip(gap=0.0):
    """Create a strip configuration."""
    pts = []
    sqrt3 = np.sqrt(3)
    for i in range(12):
        x = i * (2.0 + gap)
        y = 0 if i % 2 == 0 else sqrt3 * 0.5
        pts.append([x, y, 0.0])
    pts = np.array(pts)
    pts[:, 0] -= np.mean(pts[:, 0])
    pts[:, 1] -= np.mean(pts[:, 1])
    return pts

def _make_diamond(gap=0.0):
    """Create a diamond/rhombus arrangement."""
    pts = []
    sqrt3 = np.sqrt(3)
    # 4-3-2-1-2-3-4 pattern
    rows = [4, 3, 2, 1, 2, 3, 4]
    y_offset = 0
    for r, count in enumerate(rows):
        x_start = -(count - 1) * (1.0 + gap/2)
        for c in range(count):
            x = x_start + c * (2.0 + gap)
            y = y_offset
            pts.append([x, y, 0.0])
        y_offset += sqrt3 * 0.5 + gap * 0.25
    pts = np.array(pts[:12])
    pts[:, 0] -= np.mean(pts[:, 0])
    pts[:, 1] -= np.mean(pts[:, 1])
    return pts

def _make_compact_hex(gap=0.0):
    """Create a compact hexagonal packing with optimized spacing."""
    pts = []
    sqrt3 = np.sqrt(3)
    # Use a hexagonal grid pattern: 3-4-3-2 or similar
    # Center at (0,0)
    pts.append([0, 0, 0.0])
    # Ring 1: 6 hexagons
    for k in range(6):
        ang = np.radians(60 * k)
        pts.append([sqrt3 * np.cos(ang), sqrt3 * np.sin(ang), 0.0])
    # Ring 2: 6 hexagons (but we only need 5 more)
    for k in range(5):
        ang = np.radians(60 * k + 30)
        pts.append([2 * sqrt3 * np.cos(ang), 2 * sqrt3 * np.sin(ang), 0.0])
    pts = np.array(pts)
    pts[:, 0] -= np.mean(pts[:, 0])
    pts[:, 1] -= np.mean(pts[:, 1])
    return pts

def _make_random_config(seed=None):
    """Generate a random configuration with varied rotations."""
    if seed is not None:
        np.random.seed(seed)
    
    pts = np.random.randn(12, 3) * 4.0
    rot_pattern = np.random.choice(['uniform', 'alternating', 'radial', 'cluster'])
    if rot_pattern == 'uniform':
        pts[:, 2] = np.random.uniform(0, 60, 12)
    elif rot_pattern == 'alternating':
        pts[:, 2] = np.where(np.arange(12) % 2 == 0, 0.0, 30.0) + np.random.uniform(-5, 5, 12)
    elif rot_pattern == 'radial':
        cx = np.mean(pts[:, 0])
        cy = np.mean(pts[:, 1])
        angles = np.arctan2(pts[:, 1] - cy, pts[:, 0] - cx)
        pts[:, 2] = np.degrees(angles) % 60
    else:
        # Cluster: similar rotations for nearby hexagons
        for i in range(12):
            pts[i, 2] = np.random.uniform(0, 60)
    
    pts[:, 0] -= np.mean(pts[:, 0])
    pts[:, 1] -= np.mean(pts[:, 1])
    return pts

def _physics_relax(pts, max_iter=300, dt=0.005, cooling=0.99):
    """Physics-based relaxation to resolve collisions and pack tighter."""
    n = len(pts)
    verts_list = []
    norms_list = []
    for i in range(n):
        x, y, rot = pts[i]
        v = _hex_vertices(x, y, 1.0, rot)
        nm = _hex_edge_normals(rot)
        verts_list.append(v)
        norms_list.append(nm)
    
    forces = np.zeros((n, 3))
    temp = 0.5
    
    for it in range(max_iter):
        if time.time() > DEADLINE:
            break
        
        forces = np.zeros((n, 3))
        
        # Compute pairwise forces
        for i in range(n):
            for j in range(i + 1, n):
                d = _sat_overlap_fast(verts_list[i], verts_list[j], norms_list[i], norms_list[j])
                if d < 0:
                    # Overlapping - push apart
                    overlap = -d
                    dx = pts[j, 0] - pts[i, 0]
                    dy = pts[j, 1] - pts[i, 1]
                    dist = np.sqrt(dx * dx + dy * dy)
                    if dist > 1e-9:
                        fx = (dx / dist) * overlap * 5.0
                        fy = (dy / dist) * overlap * 5.0
                    else:
                        fx = overlap * 5.0
                        fy = 0.0
                    forces[i, 0] -= fx
                    forces[i, 1] -= fy
                    forces[j, 0] += fx
                    forces[j, 1] += fy
        
        # Add small random noise for exploration
        for i in range(n):
            forces[i, 0] += np.random.randn() * temp * 0.05
            forces[i, 1] += np.random.randn() * temp * 0.05
            forces[i, 2] += np.random.randn() * temp * 0.02
        
        # Update positions
        for i in range(n):
            pts[i, 0] += forces[i, 0] * dt
            pts[i, 1] += forces[i, 1] * dt
            pts[i, 2] += forces[i, 2] * dt
            # Recompute vertices
            verts_list[i] = _hex_vertices(pts[i, 0], pts[i, 1], 1.0, pts[i, 2])
            norms_list[i] = _hex_edge_normals(pts[i, 2])
        
        temp *= cooling
    
    return pts

def construct():
    t0 = time.time()
    
    def time_left():
        return DEADLINE - (time.time() - t0)
    
    # Multiple initial configurations
    configs = [
        _make_hex_grid(4, 3, gap=0.05),
        _make_hex_grid(3, 4, gap=0.05),
        _make_hex_grid(2, 6, gap=0.05),
        _make_hex_grid(6, 2, gap=0.05),
        _make_ring(gap=0.05),
        _make_packed_cluster(gap=0.05),
        _make_strip(gap=0.05),
        _make_diamond(gap=0.05),
        _make_compact_hex(gap=0.05),
        _make_hex_grid(4, 3, rot_offset=30, gap=0.05),
        _make_hex_grid(3, 4, rot_offset=30, gap=0.05),
        _make_hex_grid(2, 6, rot_offset=30, gap=0.05),
        _make_ring(gap=0.1),
        _make_packed_cluster(gap=0.1),
    ]
    
    # Add random configurations
    np.random.seed(42)
    for i in range(20):
        configs.append(_make_random_config(seed=42 + i))
    
    best_L = float('inf')
    best_inner = None
    best_center = None
    best_angle = None
    
    # Phase 1: Physics relaxation on each config, then evaluate
    for idx, cfg in enumerate(configs):
        if time_left() < 30:
            break
        
        relaxed = _physics_relax(cfg.copy(), max_iter=200, dt=0.005, cooling=0.99)
        
        # Evaluate with different outer rotations
        for oang in [0.0, 30.0, 15.0, 45.0]:
            if time_left() < 10:
                break
            
            if _check_disjoint(relaxed):
                L, center = _compute_L(relaxed, oang)
                if L < best_L:
                    best_L = L
                    best_inner = relaxed.copy()
                    best_center = center
                    best_angle = oang
    
    # Phase 2: Local optimization from best config
    if best_inner is not None and time_left() > 40:
        for oang in [best_angle, 0.0, 30.0]:
            if time_left() < 15:
                break
            
            result = _polish(best_inner.copy(), t0, oang, max_iter=400)
            if result is not None:
                L_opt, inner_opt, center_opt, angle_opt = result
                if L_opt < best_L:
                    best_L = L_opt
                    best_inner = inner_opt
                    best_center = center_opt
                    best_angle = angle_opt
    
    # Phase 3: Try more random starts with physics + optimization
    if time_left() > 30:
        for i in range(15):
            if time_left() < 15:
                break
            
            pts = _make_random_config(seed=100 + i)
            relaxed = _physics_relax(pts.copy(), max_iter=150, dt=0.005, cooling=0.99)
            
            for oang in [0.0, 30.0]:
                if time_left() < 10:
                    break
                
                if _check_disjoint(relaxed):
                    L, center = _compute_L(relaxed, oang)
                    if L < best_L:
                        best_L = L
                        best_inner = relaxed.copy()
                        best_center = center
                        best_angle = oang
                        
                        # Quick polish
                        if time_left() > 20:
                            result = _polish(best_inner.copy(), t0, oang, max_iter=200)
                            if result is not None:
                                L_opt, inner_opt, center_opt, angle_opt = result
                                if L_opt < best_L:
                                    best_L = L_opt
                                    best_inner = inner_opt
                                    best_center = center_opt
                                    best_angle = angle_opt
    
    # Phase 4: Differential evolution for global optimization
    if time_left() > 50:
        try:
            bounds = [(-10, 10)] * 24 + [(0, 60)] * 12 + [(0, 60)]
            
            def de_objective(p):
                if time.time() - t0 > DEADLINE - 5:
                    return 1e6
                return _objective(p, t0)
            
            de_result = differential_evolution(
                de_objective,
                bounds,
                maxiter=50,
                tol=1e-8,
                popsize=15,
                mutation=(0.3, 1.0),
                recombination=0.8,
                seed=42,
                polish=False,
                init='sohal',
                update_interval=1
            )
            
            if de_result.fun < 1e5:
                inner_de = de_result.x[:36].reshape(12, 3)
                oang_de = de_result.x[36]
                if _check_disjoint(inner_de):
                    L_de, center_de = _compute_L(inner_de, oang_de)
                    if L_de < best_L:
                        best_L = L_de
                        best_inner = inner_de
                        best_center = center_de
                        best_angle = oang_de
                        
                        # Polish the DE result
                        if time_left() > 20:
                            result = _polish(best_inner.copy(), t0, best_angle, max_iter=300)
                            if result is not None:
                                L_opt, inner_opt, center_opt, angle_opt = result
                                if L_opt < best_L:
                                    best_L = L_opt
                                    best_inner = inner_opt
                                    best_center = center_opt
                                    best_angle = angle_opt
        except Exception:
            pass
    
    # Final polish if time allows
    if best_inner is not None and time_left() > 15:
        result = _polish(best_inner.copy(), t0, best_angle, max_iter=300)
        if result is not None:
            L_opt, inner_opt, center_opt, angle_opt = result
            if L_opt < best_L:
                best_L = L_opt
                best_inner = inner_opt
                best_center = center_opt
                best_angle = angle_opt
    
    if best_inner is None:
        # Fallback: use a known valid configuration
        cfg = _make_hex_grid(4, 3, gap=0.5)
        inner = [[float(p[0]), float(p[1]), float(p[2])] for p in cfg]
        outer_center = (0.0, 0.0)
        outer_angle = 0.0
        all_pts = np.vstack([_hex_vertices(p[0], p[1], 1.0, p[2]) for p in inner])
        outer_side = _outer_hex_side_fast(all_pts, outer_angle)
        return inner, outer_center, outer_side, outer_angle
    
    inner = []
    for i in range(12):
        x = best_inner[i, 0]
        y = best_inner[i, 1]
        r = best_inner[i, 2]
        inner.append([float(x), float(y), float(r)])
    
    outer_center = (float(best_center[0]), float(best_center[1]))
    outer_angle = float(best_angle)
    
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
