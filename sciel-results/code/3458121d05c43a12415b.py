"""
Packing of 12 unit hexagons in a minimal outer hexagon.
Uses simulated annealing with a fast bounding-hexagon objective and
SAT-based validity checking. Multiple restarts with diverse initial
configurations, followed by local refinement.
"""
import math
import time
import random
import numpy as np

N = 12
TIME_LIMIT = 140.0  # Safety margin below 150s
INRADIUS = math.sqrt(3) / 2.0
MIN_DIST = 2.0 * INRADIUS  # sqrt(3)

def _hex_verts(cx, cy, angle_deg):
    """Return 6 vertices of a unit hexagon centered at (cx,cy) rotated by angle_deg."""
    a = math.radians(angle_deg)
    ca, sa = math.cos(a), math.sin(a)
    verts = []
    for k in range(6):
        theta = math.radians(60 * k)
        x = math.cos(theta)
        y = math.sin(theta)
        verts.append((cx + ca * x - sa * y, cy + sa * x + ca * y))
    return verts

def _project(verts, axis):
    ax, ay = axis
    vals = [vx * ax + vy * ay for vx, vy in verts]
    return min(vals), max(vals)

def _axes(verts):
    axes = []
    n = len(verts)
    for i in range(n):
        x1, y1 = verts[i]
        x2, y2 = verts[(i + 1) % n]
        dx, dy = x2 - x1, y2 - y1
        length = math.hypot(dx, dy)
        if length < 1e-12:
            continue
        axes.append((-dy / length, dx / length))
    return axes

def _overlap(vertsA, vertsB):
    """SAT overlap check. Returns True if overlapping."""
    axes = _axes(vertsA) + _axes(vertsB)
    for axis in axes:
        minA, maxA = _project(vertsA, axis)
        minB, maxB = _project(vertsB, axis)
        if maxA <= minB + 1e-9 or maxB <= minA + 1e-9:
            return False
    return True

def _is_valid(cx, cy, ang):
    """Check all pairs are disjoint using SAT."""
    verts_list = [_hex_verts(cx[i], cy[i], ang[i]) for i in range(N)]
    axes_list = [_axes(v) for v in verts_list]
    for i in range(N):
        for j in range(i + 1, N):
            overlap = True
            for axis in axes_list[i]:
                minA, maxA = _project(verts_list[i], axis)
                minB, maxB = _project(verts_list[j], axis)
                if maxA <= minB + 1e-9 or maxB <= minA + 1e-9:
                    overlap = False
                    break
            if overlap:
                for axis in axes_list[j]:
                    minA, maxA = _project(verts_list[i], axis)
                    minB, maxB = _project(verts_list[j], axis)
                    if maxA <= minB + 1e-9 or maxB <= minA + 1e-9:
                        overlap = False
                        break
            if overlap:
                return False
    return True

def _bounding_side(cx, cy, ang, center, outer_angle_deg):
    """Compute minimum side length of outer hexagon with given center and rotation."""
    a = math.radians(outer_angle_deg)
    normals = []
    for k in range(6):
        na = a + math.radians(90 + 60 * k)
        normals.append((math.cos(na), math.sin(na)))
    
    cx0, cy0 = center
    apothem = INRADIUS
    max_val = 0.0
    for i in range(N):
        verts = _hex_verts(cx[i], cy[i], ang[i])
        for (vx, vy) in verts:
            for (nx, ny) in normals:
                val = (nx * (vx - cx0) + ny * (vy - cy0)) / apothem
                if val > max_val:
                    max_val = val
    return max_val

def _best_outer(cx, cy, ang):
    """Find best outer hexagon (center + rotation) for given inner config."""
    # Center at centroid of all vertices
    all_vx, all_vy = [], []
    for i in range(N):
        verts = _hex_verts(cx[i], cy[i], ang[i])
        for (vx, vy) in verts:
            all_vx.append(vx)
            all_vy.append(vy)
    cx0 = sum(all_vx) / len(all_vx)
    cy0 = sum(all_vy) / len(all_vy)
    
    best_side = float('inf')
    best_angle = 0.0
    for angle_deg in range(0, 60, 3):
        side = _bounding_side(cx, cy, ang, (cx0, cy0), angle_deg)
        if side < best_side:
            best_side = side
            best_angle = angle_deg
    
    # Refine
    for angle_deg in np.arange(best_angle - 3, best_angle + 3, 0.3):
        side = _bounding_side(cx, cy, ang, (cx0, cy0), angle_deg)
        if side < best_side:
            best_side = side
            best_angle = angle_deg
    
    return cx0, cy0, best_side, best_angle

def _quick_score(cx, cy, ang):
    """Fast approximate score without center optimization."""
    normals = [(math.cos(math.radians(90 + 60 * k)), math.sin(math.radians(90 + 60 * k))) for k in range(6)]
    max_val = 0.0
    for i in range(N):
        a = math.radians(ang[i])
        ca, sa = math.cos(a), math.sin(a)
        for k in range(6):
            theta = math.radians(60 * k)
            vx = cx[i] + ca * math.cos(theta) - sa * math.sin(theta)
            vy = cy[i] + sa * math.cos(theta) + ca * math.sin(theta)
            for (nx, ny) in normals:
                val = (nx * vx + ny * vy) / INRADIUS
                if val > max_val:
                    max_val = val
    return max_val

def _make_honeycomb():
    """Honeycomb: 1 center + 6 ring + 5 second ring."""
    pts = [[0.0, 0.0, 0.0]]
    r1 = 2 * math.sqrt(3)
    for i in range(6):
        angle = math.radians(60 * i)
        pts.append([r1 * math.cos(angle), r1 * math.sin(angle), 0.0])
    r2 = r1 * math.sqrt(3)
    for deg in [30, 90, 150, 210, 270]:
        rad = math.radians(deg)
        pts.append([r2 * math.cos(rad), r2 * math.sin(rad), 0.0])
    return pts[:12]

def _make_lattice():
    """3-4-3-2 lattice."""
    pts = []
    rh = math.sqrt(3)
    cw = 2.0
    for c in range(3):
        pts.append([c * cw - cw, -rh, 0.0])
    for c in range(4):
        pts.append([c * cw - cw - cw/2, 0.0, 0.0])
    for c in range(3):
        pts.append([c * cw - cw, rh, 0.0])
    pts.append([-cw*2 - cw/2 - cw, 0.0, 0.0])
    pts.append([cw*2 - cw/2 + cw, 0.0, 0.0])
    return pts

def _make_ring():
    """1 center + 11 in ring."""
    pts = [[0.0, 0.0, 0.0]]
    r = 2 * math.sqrt(3)
    for i in range(11):
        angle = math.radians(360 * i / 11 + 15)
        pts.append([r * math.cos(angle), r * math.sin(angle), 0.0])
    return pts

def _make_random():
    pts = []
    for _ in range(N):
        pts.append([random.uniform(-4, 4), random.uniform(-4, 4), random.uniform(0, 60)])
    return pts

def _simulated_annealing(cfg, t_start, t_end, seed):
    """Run SA on a configuration. Returns best config found."""
    rng = random.Random(seed)
    cx = np.array([p[0] for p in cfg], dtype=float)
    cy = np.array([p[1] for p in cfg], dtype=float)
    ang = np.array([p[2] for p in cfg], dtype=float)
    
    # Check initial validity
    if not _is_valid(cx, cy, ang):
        return None, float('inf')
    
    score = _quick_score(cx, cy, ang)
    best_cx = cx.copy()
    best_cy = cy.copy()
    best_ang = ang.copy()
    best_score = score
    
    T = 0.3
    T_min = 0.001
    cooling = 0.9995
    
    while time.time() < t_end and T > T_min:
        i = rng.randrange(N)
        dx = rng.gauss(0, 0.3)
        dy = rng.gauss(0, 0.3)
        da = rng.gauss(0, 15)
        
        new_cx = cx.copy()
        new_cy = cy.copy()
        new_ang = ang.copy()
        new_cx[i] += dx
        new_cy[i] += dy
        new_ang[i] = (new_ang[i] + da) % 60
        
        if not _is_valid(new_cx, new_cy, new_ang):
            T *= cooling
            continue
        
        new_score = _quick_score(new_cx, new_cy, new_ang)
        delta = new_score - score
        
        if delta < 0 or rng.random() < math.exp(-delta / T):
            cx, cy, ang = new_cx, new_cy, new_ang
            score = new_score
            if score < best_score:
                best_score = score
                best_cx = cx.copy()
                best_cy = cy.copy()
                best_ang = ang.copy()
        
        T *= cooling
    
    return (best_cx, best_cy, best_ang), best_score

def _local_refine(cfg, t_end):
    """Nelder-Mead style local refinement using coordinate descent."""
    cx = np.array([p[0] for p in cfg], dtype=float)
    cy = np.array([p[1] for p in cfg], dtype=float)
    ang = np.array([p[2] for p in cfg], dtype=float)
    
    if not _is_valid(cx, cy, ang):
        return None
    
    best_score = _quick_score(cx, cy, ang)
    step = 0.1
    
    for iteration in range(200):
        if time.time() > t_end:
            break
        improved = False
        for i in range(N):
            for param_idx, arr in enumerate([cx, cy]):
                for direction in [1, -1]:
                    new_arr = arr.copy()
                    new_arr[i] += direction * step
                    new_cfg = list(new_arr) + list(cy if param_idx == 0 else cx) + list(ang)
                    # Quick validity check
                    ncx = new_arr if param_idx == 0 else cx
                    ncy = cy if param_idx == 0 else new_arr
                    if _is_valid(ncx, ncy, ang):
                        ns = _quick_score(ncx, ncy, ang)
                        if ns < best_score:
                            if param_idx == 0:
                                cx = new_arr
                            else:
                                cy = new_arr
                            best_score = ns
                            improved = True
            # Angle
            for direction in [1, -1]:
                new_ang = ang.copy()
                new_ang[i] = (new_ang[i] + direction * step * 2) % 60
                if _is_valid(cx, cy, new_ang):
                    ns = _quick_score(cx, cy, new_ang)
                    if ns < best_score:
                        ang = new_ang
                        best_score = ns
                        improved = True
        if not improved:
            step *= 0.7
            if step < 0.005:
                break
    
    return (cx, cy, ang)

def construct():
    t0 = time.time()
    random.seed(42)
    np.random.seed(42)
    
    initializers = [_make_honeycomb, _make_lattice, _make_ring, _make_random]
    
    best_cfg = None
    best_score = float('inf')
    
    # Phase 1: Find valid initial configs and evaluate
    valid_cfgs = []
    for init_func in initializers:
        if time.time() - t0 > TIME_LIMIT - 100:
            break
        cfg = init_func()
        cx = np.array([p[0] for p in cfg])
        cy = np.array([p[1] for p in cfg])
        ang = np.array([p[2] for p in cfg])
        if _is_valid(cx, cy, ang):
            s = _quick_score(cx, cy, ang)
            valid_cfgs.append((cfg, s))
            if s < best_score:
                best_score = s
                best_cfg = cfg[:]
    
    # Add some random valid configs
    for _ in range(5):
        if time.time() - t0 > TIME_LIMIT - 80:
            break
        cfg = _make_random()
        cx = np.array([p[0] for p in cfg])
        cy = np.array([p[1] for p in cfg])
        ang = np.array([p[2] for p in cfg])
        if _is_valid(cx, cy, ang):
            s = _quick_score(cx, cy, ang)
            valid_cfgs.append((cfg, s))
            if s < best_score:
                best_score = s
                best_cfg = cfg[:]
    
    if best_cfg is None:
        # Fallback: scaled lattice
        cfg = _make_lattice()
        for p in cfg:
            p[0] *= 1.3
            p[1] *= 1.3
        best_cfg = cfg
        best_score = _quick_score([p[0] for p in cfg], [p[1] for p in cfg], [p[2] for p in cfg])
    
    # Phase 2: Simulated annealing on best configs
    t_sa_end = t0 + TIME_LIMIT - 30
    for cfg, s in sorted(valid_cfgs, key=lambda x: x[1])[:4]:
        if time.time() > t_sa_end:
            break
        result, sa_score = _simulated_annealing(cfg, time.time(), t_sa_end, seed=random.randint(0, 10000))
        if result is not None and sa_score < best_score:
            best_score = sa_score
            best_cfg = [[result[0][i], result[1][i], result[2][i]] for i in range(N)]
    
    # Phase 3: Local refinement
    t_ref_end = t0 + TIME_LIMIT - 15
    if time.time() < t_ref_end:
        refined = _local_refine(best_cfg, t_ref_end)
        if refined is not None:
            rcx, rcy, rang = refined
            if _is_valid(rcx, rcy, rang):
                rs = _quick_score(rcx, rcy, rang)
                if rs < best_score:
                    best_score = rs
                    best_cfg = [[rcx[i], rcy[i], rang[i]] for i in range(N)]
    
    # Phase 4: Final evaluation with proper outer hex optimization
    cx = np.array([p[0] for p in best_cfg])
    cy = np.array([p[1] for p in best_cfg])
    ang = np.array([p[2] for p in best_cfg])
    
    if not _is_valid(cx, cy, ang):
        # Fallback
        cfg = _make_lattice()
        for p in cfg:
            p[0] *= 1.3
            p[1] *= 1.3
        cx = np.array([p[0] for p in cfg])
        cy = np.array([p[1] for p in cfg])
        ang = np.array([p[2] for p in cfg])
        best_cfg = cfg
    
    cx0, cy0, side, outer_angle = _best_outer(cx, cy, ang)
    
    inner = [[best_cfg[i][0], best_cfg[i][1], best_cfg[i][2]] for i in range(N)]
    return inner, (cx0, cy0), side, outer_angle
