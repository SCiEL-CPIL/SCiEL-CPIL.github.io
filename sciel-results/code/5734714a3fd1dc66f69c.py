import numpy as np
import math
import time
from scipy.optimize import minimize

N = 12
DEADLINE = time.time() + 140.0  # safety margin before 150s kill


def hex_vertices(cx, cy, s, angle_deg):
    """Vertices of a regular hexagon centered at (cx,cy), side s, rotation angle_deg."""
    a = math.radians(angle_deg)
    verts = []
    for i in range(6):
        theta = a + i * math.pi / 3.0
        verts.append((cx + s * math.cos(theta), cy + s * math.sin(theta)))
    return verts


def point_in_convex_polygon(p, poly):
    """Check if point p is inside convex polygon poly (list of (x,y))."""
    x, y = p
    n = len(poly)
    sign = 0
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        cross = (x2 - x1) * (y - y1) - (y2 - y1) * (x - x1)
        if cross > 1e-10:
            if sign < 0:
                return False
            sign = 1
        elif cross < -1e-10:
            if sign > 0:
                return False
            sign = -1
    return True


def polygons_overlap_sat(v1, v2, eps=1e-9):
    """SAT: return True if two convex polygons overlap (including touching)."""
    polys = [v1, v2]
    for poly in polys:
        n = len(poly)
        for i in range(n):
            x1, y1 = poly[i]
            x2, y2 = poly[(i + 1) % n]
            # edge normal
            ex = x2 - x1
            ey = y2 - y1
            # normalize
            length = math.sqrt(ex * ex + ey * ey)
            if length < 1e-12:
                continue
            nx = ey / length
            ny = -ex / length
            # project both polygons onto this normal
            min1 = max1 = None
            min2 = max2 = None
            for px, py in v1:
                proj = px * nx + py * ny
                if min1 is None or proj < min1:
                    min1 = proj
                if max1 is None or proj > max1:
                    max1 = proj
            for px, py in v2:
                proj = px * nx + py * ny
                if min2 is None or proj < min2:
                    min2 = proj
                if max2 is None or proj > max2:
                    max2 = proj
            # check separation
            if max1 < min2 - eps or max2 < min1 - eps:
                return False  # separated
    return True  # overlap


def compute_min_outer_side(inner, outer_center=(0.0, 0.0), outer_angle=0.0):
    """Compute minimum side length of outer hexagon (at given center/angle) containing all inner hexes."""
    cx, cy = outer_center
    best = 0.0
    for row in inner:
        v = hex_vertices(row[0], row[1], 1.0, row[2])
        for (px, py) in v:
            # Transform to outer hexagon's local frame
            dx = px - cx
            dy = py - cy
            a = math.radians(outer_angle)
            ca, sa = math.cos(a), math.sin(a)
            lx = ca * dx + sa * dy
            ly = -sa * dx + ca * dy
            # For a flat-top regular hexagon with side S centered at origin:
            # Constraints: |lx| <= S, |lx/2 + sqrt(3)/2 * ly| <= S, |lx/2 - sqrt(3)/2 * ly| <= S
            c1 = abs(lx)
            c2 = abs(lx / 2.0 + math.sqrt(3) / 2.0 * ly)
            c3 = abs(lx / 2.0 - math.sqrt(3) / 2.0 * ly)
            S = max(c1, c2, c3)
            if S > best:
                best = S
    return best


def check_validity(inner, outer_center, outer_side, outer_angle):
    """Full validity check: all inner hexes inside outer, no pairwise overlaps."""
    v_outer = hex_vertices(outer_center[0], outer_center[1], outer_side, outer_angle)
    for row in inner:
        v = hex_vertices(row[0], row[1], 1.0, row[2])
        for p in v:
            if not point_in_convex_polygon(p, v_outer):
                return False
    for i in range(N):
        for j in range(i + 1, N):
            vi = hex_vertices(inner[i][0], inner[i][1], 1.0, inner[i][2])
            vj = hex_vertices(inner[j][0], inner[j][1], 1.0, inner[j][2])
            if polygons_overlap_sat(vi, vj, eps=-1e-7):
                return False
    return True


def build_ring12():
    """Build 12 hexagons in a ring: 6 inner ring + 6 outer ring, alternating."""
    pts = []
    # Center hexagon
    pts.append([0.0, 0.0, 0.0])
    # 6 hexagons around center at distance sqrt(3) (edge-to-edge for flat-top)
    d1 = math.sqrt(3)
    for i in range(6):
        angle = i * math.pi / 3.0
        x = d1 * math.cos(angle)
        y = d1 * math.sin(angle)
        pts.append([x, y, 0.0])
    # 5 more hexagons in second ring
    d2 = 2 * math.sqrt(3)
    for i in range(5):
        angle = (i + 0.5) * math.pi / 3.0
        x = d2 * math.cos(angle)
        y = d2 * math.sin(angle)
        pts.append([x, y, 0.0])
    return pts


def build_hex6_plus6():
    """Two clusters of 6 hexagons side by side."""
    pts = []
    # Cluster 1: center + 5 around
    d1 = math.sqrt(3)
    pts.append([0.0, 0.0, 0.0])
    for i in range(5):
        angle = i * 2 * math.pi / 5.0
        x = d1 * math.cos(angle)
        y = d1 * math.sin(angle)
        pts.append([x, y, 0.0])
    # Cluster 2: offset by some distance
    offset_x = 3.5
    pts.append([offset_x, 0.0, 0.0])
    for i in range(5):
        angle = i * 2 * math.pi / 5.0
        x = offset_x + d1 * math.cos(angle)
        y = d1 * math.sin(angle)
        pts.append([x, y, 0.0])
    return pts


def build_rect3x4():
    """3x4 grid of hexagons in honeycomb pattern."""
    pts = []
    dx = 1.5
    dy = math.sqrt(3) / 2.0
    for r in range(4):
        for c in range(3):
            x = c * 2 * dx + (0 if r % 2 == 0 else dx)
            y = r * 2 * dy
            pts.append([x, y, 0.0])
    # center
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    cx = (min(xs) + max(xs)) / 2
    cy = (min(ys) + max(ys)) / 2
    for p in pts:
        p[0] -= cx
        p[1] -= cy
    return pts


def smooth_max(vals, k=10.0):
    """Smooth approximation of max."""
    m = max(vals)
    s = 0.0
    for v in vals:
        s += math.exp(k * (v - m))
    return m + math.log(s) / k


def free_placement_objective(vars, outer_angle=0.0):
    """
    Objective for free placement: minimize outer hex side.
    vars: [x1,y1,a1, x2,y2,a2, ..., x12,y12,a12]
    Returns outer side length + overlap penalty.
    """
    pos = []
    for i in range(N):
        x, y, a = vars[3*i], vars[3*i+1], vars[3*i+2]
        pos.append([x, y, a])
    
    # Compute outer side
    outer_side = compute_min_outer_side(pos, (0.0, 0.0), outer_angle)
    
    # Overlap penalty
    penalty = 0.0
    for i in range(N):
        vi = hex_vertices(pos[i][0], pos[i][1], 1.0, pos[i][2])
        for j in range(i+1, N):
            vj = hex_vertices(pos[j][0], pos[j][1], 1.0, pos[j][2])
            if polygons_overlap_sat(vi, vj, eps=-1e-8):
                # Compute a penalty based on how much they overlap
                # Use center distance as proxy
                dx = pos[i][0] - pos[j][0]
                dy = pos[i][1] - pos[j][1]
                dist = math.sqrt(dx*dx + dy*dy)
                # Min non-overlap distance for two unit hexes is sqrt(3)
                if dist < math.sqrt(3):
                    penalty += (math.sqrt(3) - dist) ** 2 * 100.0
    return outer_side + penalty


def optimize_topology(initial_inner, outer_angle=0.0, max_time=30.0):
    """Optimize positions and angles of inner hexagons."""
    start = time.time()
    x0 = []
    for p in initial_inner:
        x0.extend([p[0], p[1], p[2]])
    x0 = np.array(x0)
    
    best_x = x0.copy()
    best_obj = free_placement_objective(x0, outer_angle)
    
    # Try L-BFGS-B
    try:
        def obj(vars):
            return free_placement_objective(vars, outer_angle)
        
        res = minimize(obj, x0, method='L-BFGS-B', 
                      options={'maxiter': 500, 'maxfun': 20000})
        if time.time() - start < max_time:
            if res.fun < best_obj:
                best_obj = res.fun
                best_x = res.x.copy()
    except Exception:
        pass
    
    # Try Nelder-Mead if time allows
    if time.time() - start < max_time - 5:
        try:
            def obj(vars):
                return free_placement_objective(vars, outer_angle)
            res = minimize(obj, best_x, method='Nelder-Mead',
                          options={'maxiter': 300, 'xatol': 1e-6, 'fatol': 1e-6})
            if res.fun < best_obj:
                best_obj = res.fun
                best_x = res.x.copy()
        except Exception:
            pass
    
    # Convert back to inner list
    inner = []
    for i in range(N):
        inner.append([best_x[3*i], best_x[3*i+1], best_x[3*i+2]])
    return inner, best_obj


def construct():
    start = time.time()
    
    # Try multiple topologies
    topologies = [build_ring12, build_hex6_plus6, build_rect3x4]
    
    best_inner = None
    best_side = float('inf')
    best_outer_center = (0.0, 0.0)
    best_outer_angle = 0.0
    
    # Also try different outer angles
    outer_angles = [0.0, 30.0, 60.0]
    
    for angle in outer_angles:
        if time.time() - start > DEADLINE - 20:
            break
        for build_func in topologies:
            if time.time() - start > DEADLINE - 20:
                break
            initial_inner = build_func()
            inner, obj_val = optimize_topology(initial_inner, outer_angle=angle, 
                                                max_time=min(25.0, DEADLINE - (time.time() - start) - 10))
            
            # Evaluate
            side = compute_min_outer_side(inner, (0.0, 0.0), angle)
            if check_validity(inner, (0.0, 0.0), side, angle):
                if side < best_side:
                    best_side = side
                    best_inner = inner
                    best_outer_center = (0.0, 0.0)
                    best_outer_angle = angle
    
    # If no valid solution found, use a safe large side
    if best_inner is None:
        inner = build_ring12()
        side = compute_min_outer_side(inner, (0.0, 0.0), 0.0)
        # Expand until valid
        while not check_validity(inner, (0.0, 0.0), side, 0.0) and side < 50:
            side *= 1.1
        return inner, (0.0, 0.0), side, 0.0
    
    # Final refinement: try to shrink by optimizing with tighter constraints
    if time.time() - start < DEADLINE - 10:
        # Try a few more optimization iterations on the best
        inner2, _ = optimize_topology(best_inner, outer_angle=best_outer_angle, 
                                       max_time=min(15.0, DEADLINE - (time.time() - start) - 5))
        side2 = compute_min_outer_side(inner2, (0.0, 0.0), best_outer_angle)
        if check_validity(inner2, (0.0, 0.0), side2, best_outer_angle):
            if side2 < best_side:
                best_inner = inner2
                best_side = side2
    
    return best_inner, best_outer_center, best_side, best_outer_angle
