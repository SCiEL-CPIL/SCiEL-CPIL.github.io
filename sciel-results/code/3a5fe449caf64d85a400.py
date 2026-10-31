"""Hexagon packing: 11 unit hexagons inside a regular hexagon, minimize outer side."""
import time
import math
import random
import numpy as np

N = 11
DEADLINE = time.time() + 140.0  # safety margin before 150s kill


def hexagon_vertices(cx, cy, side, rot_deg):
    """Vertices of a regular hexagon (flat-top, rot=0)."""
    a = math.radians(rot_deg)
    v = []
    for k in range(6):
        ang = a + math.radians(60.0 * k)
        v.append((cx + side * math.cos(ang), cy + side * math.sin(ang)))
    return v


def rot_points(pts, a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[p[0] * c - p[1] * s, p[0] * s + p[1] * c] for p in pts])


def sat_overlap(a, b):
    """Separating-axis test for two convex polygons (lists of (x,y))."""
    for poly in (a, b):
        n = len(poly)
        for i in range(n):
            x1, y1 = poly[i]
            x2, y2 = poly[(i + 1) % n]
            ax, ay = y2 - y1, x1 - x2  # edge normal
            min_a, max_a = None, None
            min_b, max_b = None, None
            for p in a:
                d = p[0] * ax + p[1] * ay
                if min_a is None or d < min_a:
                    min_a = d
                if max_a is None or d > max_a:
                    max_a = d
            for p in b:
                d = p[0] * ax + p[1] * ay
                if min_b is None or d < min_b:
                    min_b = d
                if max_b is None or d > max_b:
                    max_b = d
            if max_a < min_b or max_b < min_a:
                return False
    return True


def outer_contains(o, inner):
    """True if every vertex of inner hexagon is inside (or on) outer hexagon."""
    for vx, vy in inner:
        if not sat_overlap([(vx, vy)], o):
            return False
    return True


def feasible(inner, outer):
    """inner: list of (cx,cy,rot); outer: (cx,cy,side,rot). True if disjoint & contained."""
    o = hexagon_vertices(outer[0], outer[1], outer[2], outer[3])
    verts = [hexagon_vertices(cx, cy, 1.0, r) for (cx, cy, r) in inner]
    for v in verts:
        if not outer_contains(o, v):
            return False
    for i in range(len(verts)):
        for j in range(i + 1, len(verts)):
            if sat_overlap(verts[i], verts[j]):
                return False
    return True


def side_for(inner, center, rot):
    """Minimal outer side (centered at 'center', rotation rot) that contains all inner."""
    best = 0.0
    for (cx, cy, r) in inner:
        rv = hexagon_vertices(0.0, 0.0, 1.0, r)
        for p in rv:
            # transform into outer frame (outer centered at 'center')
            x = p[0] + cx - center[0]
            y = p[1] + cy - center[1]
            # rotate by -rot to align with outer axes
            a = -math.radians(rot)
            xr = x * math.cos(a) - y * math.sin(a)
            yr = x * math.sin(a) + y * math.cos(a)
            d = max(abs(xr), abs(yr) * math.sqrt(3.0) / 3.0)
            if d > best:
                best = d
    return best


def objective(inner, center, rot):
    return side_for(inner, center, rot)


def random_init(scale):
    return [[random.uniform(-scale, scale), random.uniform(-scale, scale),
             random.choice([0.0, 30.0])] for _ in range(N)]


def hex_lattice_init():
    """Center + 6 ring + 4 of second ring, hexagonal lattice, side-1 hexagons."""
    h = math.sqrt(3.0) / 2.0
    pts = [(0.0, 0.0)]
    for k in range(6):
        ang = math.radians(60.0 * k)
        pts.append((3.0 * math.cos(ang), 3.0 * math.sin(ang)))
    # second ring: pick 4 adjacent positions to keep it compact
    for k in range(4):
        ang = math.radians(60.0 * k)
        pts.append((6.0 * math.cos(ang), 6.0 * math.sin(ang)))
    return [[x, y, 0.0] for (x, y) in pts]


def optimize(inner, center, rot, max_iter, step, step_decay, max_steps):
    """Simulated-annealing-like local search minimizing outer side."""
    cur = [list(p) for p in inner]
    cur_obj = objective(cur, center, rot)
    best = [list(p) for p in cur]
    best_obj = cur_obj
    T = step
    for it in range(max_iter):
        if time.time() > DEADLINE:
            break
        # propose a random perturbation of a random hexagon
        i = random.randrange(N)
        cand = [list(p) for p in cur]
        dx = random.uniform(-step, step)
        dy = random.uniform(-step, step)
        cand[i][0] += dx
        cand[i][1] += dy
        if random.random() < 0.4:
            cand[i][2] = random.choice([0.0, 30.0, cand[i][2]])
        new_obj = objective(cand, center, rot)
        if new_obj < cur_obj or random.random() < math.exp(-(new_obj - cur_obj) / max(T, 1e-6)):
            cur = cand
            cur_obj = new_obj
            if cur_obj < best_obj - 1e-9:
                best = [list(p) for p in cur]
                best_obj = cur_obj
        T *= step_decay
        if T < 1e-4:
            T = 1e-4
    return best, best_obj


def try_center_rot(inner):
    """Try to improve by adjusting outer center and rotation around the inner centroid."""
    cx = sum(p[0] for p in inner) / N
    cy = sum(p[1] for p in inner) / N
    best_obj = objective(inner, (0.0, 0.0), 0.0)
    best_center = (0.0, 0.0)
    best_rot = 0.0
    # grid over center offsets and rotation
    for dc in (-0.3, 0.0, 0.3):
        for dr in (-0.3, 0.0, 0.3):
            for rot in (0.0, 15.0, 30.0):
                obj = objective(inner, (cx + dc, cy + dr), rot)
                if obj < best_obj - 1e-9:
                    best_obj = obj
                    best_center = (cx + dc, cy + dr)
                    best_rot = rot
    return best_center, best_rot, best_obj


def construct():
    global DEADLINE
    random.seed(12345)
    # Start from the known-good hexagonal lattice configuration
    inner = hex_lattice_init()
    center, rot, obj = try_center_rot(inner)
    best_inner = [list(p) for p in inner]
    best_center = center
    best_rot = rot
    best_obj = obj

    # Local search from the lattice start
    for trial in range(6):
        if time.time() > DEADLINE:
            break
        start = [list(p) for p in best_inner]
        opt_inner, opt_obj = optimize(start, best_center, best_rot,
                                      max_iter=4000, step=1.2, step_decay=0.985, max_steps=4000)
        if opt_obj < best_obj - 1e-9:
            best_inner = opt_inner
            best_obj = opt_obj
        # refine center/rotation
        c, r, o2 = try_center_rot(best_inner)
        if o2 < best_obj - 1e-9:
            best_center, best_rot, best_obj = c, r, o2

    # Occasionally try a random restart to escape local minima
    restarts = 0
    while time.time() < DEADLINE - 5.0 and restarts < 8:
        inner2 = random_init(4.0)
        opt_inner, opt_obj = optimize(inner2, (0.0, 0.0), 0.0,
                                      max_iter=2500, step=2.0, step_decay=0.99, max_steps=2500)
        c, r, o2 = try_center_rot(opt_inner)
        if o2 < best_obj - 1e-9:
            best_inner, best_center, best_rot, best_obj = opt_inner, c, r, o2
        restarts += 1

    # Final polish
    if time.time() < DEADLINE - 2.0:
        opt_inner, opt_obj = optimize(best_inner, best_center, best_rot,
                                      max_iter=3000, step=0.5, step_decay=0.99, max_steps=3000)
        if opt_obj < best_obj - 1e-9:
            best_inner, best_obj = opt_inner, opt_obj
        c, r, o2 = try_center_rot(best_inner)
        if o2 < best_obj - 1e-9:
            best_center, best_rot, best_obj = c, r, o2

    return best_inner, best_center, best_obj, best_rot
