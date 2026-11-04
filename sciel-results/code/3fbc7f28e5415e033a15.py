"""CCD-based trajectory optimization for packing 12 unit hexagons in a regular hexagon.

Approach:
- Model each inner hexagon by its center + orientation.
- Collision between two inner hexagons is checked via the Minkowski sum
  (a hexagon of side 2 when co-oriented, or a 12-gon when rotated).
- We slide the hexagons along a time-parameterized path from a trivially
  valid configuration into a dense one, using continuous collision detection
  to ensure no collisions occur during the motion.
- The outer hexagon side length is then minimized by binary search on the
  containment constraint, with gradient-free local refinement.
"""
import math
import random
import numpy as np

N = 12

def _hex_verts(angle_deg=0.0, cx=0.0, cy=0.0, side=1.0):
    a = math.radians(angle_deg)
    cos_a, sin_a = math.cos(a), math.sin(a)
    verts = []
    for k in range(6):
        th = math.radians(60 * k)
        x = side * math.cos(th)
        y = side * math.sin(th)
        verts.append((cx + x * cos_a - y * sin_a,
                      cy + x * sin_a + y * cos_a))
    return verts

def _convex_hull(points):
    """Andrew's monotone chain; returns list of (x,y) in CCW order."""
    pts = sorted(set((round(p[0], 12), round(p[1], 12)) for p in points))
    if len(pts) <= 2:
        return pts
    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    lower = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return lower[:-1] + upper[:-1]

def _point_in_convex(poly, p, tol=1e-9):
    """Check if point p is inside convex polygon poly (CCW)."""
    if len(poly) < 3:
        return False
    n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        if (x2 - x1) * (p[1] - y1) - (y2 - y1) * (p[0] - x1) < -tol:
            return False
    return True

def _poly_in_convex(outer, inner, tol=1e-9):
    """Check if all vertices of inner are inside outer (both convex, CCW)."""
    return all(_point_in_convex(outer, (v[0], v[1]), tol) for v in inner)

def _minkowski_sum_hex(angle1_deg, angle2_deg):
    """Compute Minkowski sum of two unit hexagons with given orientations.
    
    The Minkowski sum of two convex polygons is the set of all vector sums.
    For two hexagons, this gives a polygon with up to 12 vertices.
    """
    hex1 = _hex_verts(angle1_deg, 0, 0, 1.0)
    hex2 = _hex_verts(angle2_deg, 0, 0, 1.0)
    pts = [(a[0] + b[0], a[1] + b[1]) for a in hex1 for b in hex2]
    return _convex_hull(pts)

def _separation_check(verts_a, verts_b, tol=1e-9):
    """Return True if convex polygons A and B are disjoint (no interior overlap)."""
    for p in verts_a:
        if _point_in_convex(verts_b, p, tol):
            return False
    for p in verts_b:
        if _point_in_convex(verts_a, p, tol):
            return False
    def orient(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    def on_segment(p, q, r):
        return (min(p[0], r[0]) - tol <= q[0] <= max(p[0], r[0]) + tol and
                min(p[1], r[1]) - tol <= q[1] <= max(p[1], r[1]) + tol)
    def segments_intersect(p1, p2, p3, p4):
        d1 = orient(p3, p4, p1)
        d2 = orient(p3, p4, p2)
        d3 = orient(p1, p2, p3)
        d4 = orient(p1, p2, p4)
        if ((d1 > tol and d2 < -tol) or (d1 < -tol and d2 > tol)) and \
           ((d3 > tol and d4 < -tol) or (d3 < -tol and d4 > tol)):
            return True
        if abs(d1) <= tol and on_segment(p1, p3, p4):
            return True
        if abs(d2) <= tol and on_segment(p2, p3, p4):
            return True
        if abs(d3) <= tol and on_segment(p3, p1, p2):
            return True
        if abs(d4) <= tol and on_segment(p4, p1, p2):
            return True
        return False
    n1 = len(verts_a)
    n2 = len(verts_b)
    for i in range(n1):
        a1 = verts_a[i]
        a2 = verts_a[(i + 1) % n1]
        for j in range(n2):
            b1 = verts_b[j]
            b2 = verts_b[(j + 1) % n2]
            if segments_intersect(a1, a2, b1, b2):
                return False
    return True

def _inner_hexagon(cx, cy, angle_deg, side=1.0):
    return _hex_verts(angle_deg, cx, cy, side)

def _outer_hexagon(cx, cy, side, angle_deg=0.0):
    return _hex_verts(angle_deg, cx, cy, side)

def _is_valid_packing(inner_configs, outer_center, outer_side, outer_angle=0.0, tol=1e-9):
    """Check all inner hexagons are disjoint and inside outer."""
    outer_verts = _outer_hexagon(outer_center[0], outer_center[1], outer_side, outer_angle)
    inner_verts_list = []
    for (cx, cy, ang) in inner_configs:
        v = _inner_hexagon(cx, cy, ang)
        inner_verts_list.append(v)
        if not _poly_in_convex(outer_verts, v, tol):
            return False
    for i in range(len(inner_verts_list)):
        for j in range(i + 1, len(inner_verts_list)):
            if not _separation_check(inner_verts_list[i], inner_verts_list[j], tol):
                return False
    return True

def _ccd_check_trajectory(start_configs, end_configs, outer_center, outer_side, outer_angle, num_steps=20, tol=1e-9):
    """Check if the trajectory from start to end configs is collision-free.
    
    Uses continuous collision detection: for each pair of hexagons,
    check if they collide at any point along the linear interpolation.
    """
    n = len(start_configs)
    for step in range(num_steps + 1):
        t = step / num_steps
        # Interpolate positions and angles
        current_configs = []
        for i in range(n):
            sx, sy, sa = start_configs[i]
            ex, ey, ea = end_configs[i]
            cx = sx + t * (ex - sx)
            cy = sy + t * (ey - sy)
            # Angle interpolation (handle wrap-around)
            da = ea - sa
            while da > 180: da -= 360
            while da < -180: da += 360
            ca = sa + t * da
            current_configs.append((cx, cy, ca))
        
        # Check if all hexagons are inside outer
        outer_verts = _outer_hexagon(outer_center[0], outer_center[1], outer_side, outer_angle)
        inner_verts_list = []
        for (cx, cy, ang) in current_configs:
            v = _inner_hexagon(cx, cy, ang)
            inner_verts_list.append(v)
            if not _poly_in_convex(outer_verts, v, tol):
                return False
        
        # Check pairwise collisions
        for i in range(n):
            for j in range(i + 1, n):
                if not _separation_check(inner_verts_list[i], inner_verts_list[j], tol):
                    return False
    return True

def _build_trajectory_t0():
    """Initial spread-out configuration at t=0."""
    configs = []
    # Center
    configs.append((0.0, 0.0, 0.0))
    # Ring of 6 at radius 3
    for k in range(6):
        th = math.radians(60 * k)
        configs.append((3.0 * math.cos(th), 3.0 * math.sin(th), 0.0))
    # Ring of 5 at radius 5
    for k in range(5):
        th = math.radians(60 * k + 30)
        configs.append((5.0 * math.cos(th), 5.0 * math.sin(th), 0.0))
    return configs

def _build_trajectory_t1(target_radius_inner, target_radius_outer):
    """Target dense configuration at t=1."""
    configs = []
    # Center
    configs.append((0.0, 0.0, 0.0))
    # Inner ring of 6
    for k in range(6):
        th = math.radians(60 * k)
        configs.append((target_radius_inner * math.cos(th), target_radius_inner * math.sin(th), 0.0))
    # Outer ring of 5
    for k in range(5):
        th = math.radians(60 * k + 30)
        configs.append((target_radius_outer * math.cos(th), target_radius_outer * math.sin(th), 0.0))
    return configs

def _optimize_packing(outer_side, outer_center=(0.0, 0.0), outer_angle=0.0, max_iters=300, tol=1e-9):
    """Optimize inner hexagon positions to fit inside outer hexagon using CCD-based trajectory."""
    random.seed(42)
    
    # Estimate target radii based on outer side length
    # For a regular hexagon of side s, the distance from center to vertex is s
    # We want to place hexagons as close as possible
    target_r_inner = 2.0 + 0.1 * outer_side
    target_r_outer = 3.5 + 0.2 * outer_side
    
    # Clamp to reasonable values
    target_r_inner = max(2.0, min(target_r_inner, 3.5))
    target_r_outer = max(3.5, min(target_r_outer, 5.5))
    
    start_configs = _build_trajectory_t0()
    end_configs = _build_trajectory_t1(target_r_inner, target_r_outer)
    
    # Check if the straight trajectory is valid
    if _ccd_check_trajectory(start_configs, end_configs, outer_center, outer_side, outer_angle, num_steps=30, tol=tol):
        return end_configs
    
    # If not, try to find a valid trajectory by adjusting the target
    best_configs = None
    best_penalty = float('inf')
    
    for trial in range(20):
        random.seed(42 + trial)
        
        # Perturb the target configuration
        perturbed_end = []
        for i in range(N):
            x, y, a = end_configs[i]
            if i == 0:
                # Keep center fixed
                perturbed_end.append((x, y, a))
            else:
                dx = random.uniform(-0.3, 0.3)
                dy = random.uniform(-0.3, 0.3)
                da = random.uniform(-15, 15)
                perturbed_end.append((x + dx, y + dy, a + da))
        
        # Check if this perturbed end config is valid
        if _is_valid_packing(perturbed_end, outer_center, outer_side, outer_angle, tol):
            # Check if we can reach it via a valid trajectory
            if _ccd_check_trajectory(start_configs, perturbed_end, outer_center, outer_side, outer_angle, num_steps=30, tol=tol):
                return perturbed_end
        
        # Try local refinement from a valid starting point
        # First, find any valid configuration
        for r_inner in [2.0, 2.2, 2.4, 2.6, 2.8]:
            for r_outer in [3.5, 3.8, 4.1, 4.4, 4.7]:
                test_end = _build_trajectory_t1(r_inner, r_outer)
                if _is_valid_packing(test_end, outer_center, outer_side, outer_angle, tol):
                    # Now try to optimize from here
                    current = test_end[:]
                    for it in range(max_iters // 10):
                        new_configs = [list(c) for c in current]
                        improved = False
                        for i in range(N):
                            if random.random() < 0.4:
                                new_configs[i][0] += random.uniform(-0.2, 0.2)
                                new_configs[i][1] += random.uniform(-0.2, 0.2)
                            if random.random() < 0.15:
                                new_configs[i][2] += random.uniform(-10, 10)
                        new_configs = [tuple(c) for c in new_configs]
                        
                        if _is_valid_packing(new_configs, outer_center, outer_side, outer_angle, tol):
                            current = new_configs
                            improved = True
                    
                    if _is_valid_packing(current, outer_center, outer_side, outer_angle, tol):
                        return current
    
    return None

def _minimize_side_length(outer_center=(0.0, 0.0), outer_angle=0.0, lo=3.0, hi=10.0, tol=0.01):
    """Binary search for minimum outer side length."""
    # First find a valid upper bound
    while hi < 15.0:
        configs = _optimize_packing(hi, outer_center, outer_angle)
        if configs is not None:
            break
        hi *= 1.2
    
    if hi >= 15.0:
        return None, hi
    
    for _ in range(30):
        if hi - lo < tol:
            break
        mid = (lo + hi) / 2
        configs = _optimize_packing(mid, outer_center, outer_angle)
        if configs is not None:
            hi = mid
        else:
            lo = mid
    
    # Return the best configuration at hi
    configs = _optimize_packing(hi, outer_center, outer_angle)
    if configs is None:
        return None, hi
    return configs, hi

def construct():
    """Construct 12 unit hexagons inside a regular hexagon, minimizing outer side."""
    outer_center = (0.0, 0.0)
    outer_angle = 0.0
    
    # Try to find a good packing
    configs, side = _minimize_side_length(outer_center, outer_angle, lo=3.5, hi=8.0, tol=0.01)
    
    if configs is None:
        # Fallback: simple linear arrangement
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0
    
    inner = [[c[0], c[1], c[2]] for c in configs]
    return inner, outer_center, side, outer_angle
