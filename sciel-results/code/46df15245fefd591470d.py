"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import math
import time
import random
import numpy as np

N = 12

# EVOLVE-BLOCK-START
def construct():
    t0 = time.time()
    deadline = t0 + 140.0          # 150s hard limit, keep 10s margin

    # ---------- geometry helpers ----------
    def hex_corners(cx, cy, side, angle_deg):
        a = math.radians(angle_deg)
        return [(cx + side * math.cos(a + i * math.pi / 3),
                 cy + side * math.sin(a + i * math.pi / 3)) for i in range(6)]

    def point_in_hex(px, py, cx, cy, side, angle_deg):
        a = math.radians(angle_deg)
        ca, sa = math.cos(a), math.sin(a)
        dx, dy = px - cx, py - cy
        lx = ca * dx + sa * dy
        ly = -sa * dx + ca * dy
        ap = side * math.sqrt(3.0) / 2.0
        return (abs(lx) <= side and abs(ly) <= ap and abs(lx + ly) <= side)

    def enclosing_side(pts, angle_deg):
        a = math.radians(angle_deg)
        ca, sa = math.cos(a), math.sin(a)
        xs, ys = [], []
        for (x, y) in pts:
            xs.append(ca * x + sa * y)
            ys.append(-sa * x + ca * y)
        xs = np.array(xs); ys = np.array(ys)
        m1 = np.max(xs); m2 = np.max(ys); m3 = np.max(xs + ys)
        M1 = np.min(xs); M2 = np.min(ys); M3 = np.min(xs + ys)
        s = max(m1 - M1, m2 - M2, m3 - M3) / math.sqrt(3.0)
        return s, ((m1 + M1) / 2.0, (m2 + M2) / 2.0)

    def enclosing_side_center(pts, angle_deg):
        a = math.radians(angle_deg)
        ca, sa = math.cos(a), math.sin(a)
        xs = np.array([ca * x + sa * y for (x, y) in pts])
        ys = np.array([-sa * x + ca * y for (x, y) in pts])
        m1, m2, m3 = np.max(xs), np.max(ys), np.max(xs + ys)
        M1, M2, M3 = np.min(xs), np.min(ys), np.min(xs + ys)
        s = max(m1 - M1, m2 - M2, m3 - M3) / math.sqrt(3.0)
        cxl = (m1 + M1) / 2.0
        cyl = (m2 + M2) / 2.0
        return s, (ca * cxl - sa * cyl, sa * cxl + ca * cyl)

    def all_inside(pts, cx, cy, side, angle_deg):
        for (x, y) in pts:
            if not point_in_hex(x, y, cx, cy, side, angle_deg):
                return False
        return True

    def is_valid(pts, cx, cy, side, angle_deg):
        # containment check
        if not all_inside(pts, cx, cy, side, angle_deg):
            return False
        # disjointness check (center distances >= 2 + eps)
        n = len(pts)
        for i in range(n):
            for j in range(i + 1, n):
                dx = pts[i][0] - pts[j][0]
                dy = pts[i][1] - pts[j][1]
                if dx * dx + dy * dy < (2.0 - 1e-9) ** 2:
                    return False
        return True

    def overlap_penalty(pts, cx, cy, side, angle_deg):
        pen = 0.0
        for i in range(len(pts)):
            for j in range(i + 1, len(pts)):
                dx = pts[i][0] - pts[j][0]
                dy = pts[i][1] - pts[j][1]
                d = math.hypot(dx, dy)
                if d < 2.0:
                    pen += (2.0 - d) ** 2
        # containment penalty
        for (x, y) in pts:
            if not point_in_hex(x, y, cx, cy, side, angle_deg):
                pen += 10.0
        return pen

    # ---------- initialization ----------
    def init_hex_lattice():
        # 12 hexagons in a hexagonal (honeycomb) lattice, side spacing = 2
        # rings: center (1) + ring1 (6) + ring2 (5 of 12)
        pts = [(0.0, 0.0)]
        # ring 1: 6 neighbors
        for i in range(6):
            ang = i * math.pi / 3.0
            pts.append((2.0 * math.cos(ang), 2.0 * math.sin(ang)))
        # ring 2: 12 positions, take 5
        for i in range(12):
            ang = (i + 0.5) * math.pi / 6.0
            pts.append((4.0 * math.cos(ang), 4.0 * math.sin(ang)))
            if len(pts) == N:
                break
        return [list(p) for p in pts]

    def init_grid():
        # 4 x 3 grid
        pts = []
        for row in range(3):
            for col in range(4):
                x = (col - 1.5) * 2.2
                y = (row - 1.0) * 2.2
                pts.append([x, y])
        return pts

    def init_random():
        pts = []
        for _ in range(N):
            pts.append([random.uniform(-6, 6), random.uniform(-6, 6)])
        return pts

    # ---------- separation (resolve overlaps) ----------
    def separate(pts, iters=200):
        for _ in range(iters):
            moved = False
            for i in range(len(pts)):
                for j in range(i + 1, len(pts)):
                    dx = pts[i][0] - pts[j][0]
                    dy = pts[i][1] - pts[j][1]
                    d = math.hypot(dx, dy)
                    if d < 2.0:
                        if d < 1e-9:
                            ang = random.uniform(0, 2 * math.pi)
                            dx, dy = math.cos(ang), math.sin(ang)
                            d = 1e-9
                        push = (2.0 - d) / 2.0 + 1e-6
                        ux, uy = dx / d, dy / d
                        pts[i][0] += push * ux
                        pts[i][1] += push * uy
                        pts[j][0] -= push * ux
                        pts[j][1] -= push * uy
                        moved = True
            if not moved:
                break
        return pts

    # ---------- optimize one configuration ----------
    def optimize(pts):
        pts = separate(pts, iters=300)
        best = None
        # iterate: compute enclosing hex, then re-separate within it
        for it in range(30):
            s, c = enclosing_side_center(pts, 0.0)
            # add a small margin so points are strictly inside for separation
            sm = s + 0.5
            # separate points (free) then check
            pts2 = separate(pts, iters=200)
            s2, c2 = enclosing_side_center(pts2, 0.0)
            # recenter
            cx, cy = c2
            for p in pts2:
                p[0] -= cx
                p[1] -= cy
            s2, _ = enclosing_side_center(pts2, 0.0)
            # try to fit within s2 by scaling positions slightly inward if needed
            if best is None or s2 < best[0]:
                best = (s2, [list(p) for p in pts2])
            pts = pts2
        # refine orientation of enclosing hexagon
        pts = best[1]
        best_s = best[0]
        best_c = (0.0, 0.0)
        for ang in np.linspace(0, 60, 25):
            s, c = enclosing_side_center(pts, ang)
            if s < best_s:
                best_s = s
                best_c = c
        # final recenter
        for p in pts:
            p[0] -= best_c[0]
            p[1] -= best_c[1]
        s, c = enclosing_side_center(pts, 0.0)
        return s, pts, c

    # ---------- search ----------
    inners = [init_hex_lattice(), init_grid(), init_random(), init_random()]
    best = None
    for init in inners:
        if time.time() > deadline:
            break
        try:
            s, pts, c = optimize(init)
        except Exception:
            continue
        if best is None or s < best[0]:
            best = (s, pts, c)
        # extra random restarts while time allows
        while time.time() < deadline - 5.0:
            try:
                s2, pts2, c2 = optimize(init_random())
            except Exception:
                break
            if best is None or s2 < best[0]:
                best = (s2, pts2, c2)

    # ---------- fallback ----------
    if best is None:
        # trivial linear arrangement (always valid)
        pts = [[4.0 * k, 0.0] for k in range(-(N // 2), N - N // 2)]
        s, c = enclosing_side_center(pts, 0.0)
        for p in pts:
            p[0] -= c[0]; p[1] -= c[1]
        s, c = enclosing_side_center(pts, 0.0)
        best = (s, pts, c)

    s, pts, c = best
    inner = [[p[0], p[1], 0.0] for p in pts]
    return inner, (c[0], c[1]), s, 0.0
# EVOLVE-BLOCK-END
