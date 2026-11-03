import math
import time
import random

N = 12

# EVOLVE-BLOCK-START
def _hex_vertices(cx, cy, ang_deg):
    a = math.radians(ang_deg)
    ca, sa = math.cos(a), math.sin(a)
    out = []
    for k in range(6):
        t = math.radians(60.0 * k)
        lx = math.cos(t)
        ly = math.sin(t)
        out.append((cx + ca * lx - sa * ly, cy + sa * lx + ca * ly))
    return out

def _convex_hull(pts):
    pts = sorted(set((round(x, 12), round(y, 12)) for x, y in pts))
    if len(pts) <= 1:
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

def _side_from_hull(hull, ang_deg):
    a = math.radians(ang_deg)
    ca, sa = math.cos(a), math.sin(a)
    def proj(p):
        x, y = p
        return (x * ca - y * sa, x * sa + y * ca)
    maxp = [0.0] * 6
    minp = [0.0] * 6
    for p in hull:
        u, v = proj(p)
        for i in range(6):
            t = math.radians(60.0 * i)
            d = u * math.cos(t) + v * math.sin(t)
            if d > maxp[i]:
                maxp[i] = d
            if d < minp[i]:
                minp[i] = d
    side = 0.0
    for i in range(6):
        j = (i - 1) % 6
        w = maxp[i] - minp[j]
        if w > side:
            side = w
    return side

def _build_state():
    xs = [0.0]
    ys = [0.0]
    angs = [0.0]
    for k in range(6):
        t = math.radians(60.0 * k)
        xs.append(2.0 * math.cos(t))
        ys.append(2.0 * math.sin(t))
        angs.append(0.0)
    for k in range(6):
        t = math.radians(60.0 * k + 30.0)
        r = 2.0 * math.sqrt(3.0)
        xs.append(r * math.cos(t))
        ys.append(r * math.sin(t))
        angs.append(0.0)
    return xs, ys, angs

def construct():
    start = time.time()
    deadline = start + 140.0
    rng = random.Random(12345)

    xs, ys, angs = _build_state()
    verts = []
    for i in range(N):
        verts.extend(_hex_vertices(xs[i], ys[i], angs[i]))
    hull = _convex_hull(verts)
    side = _side_from_hull(hull, 0.0)

    best_side = side
    best_xs = xs[:]
    best_ys = ys[:]
    best_angs = angs[:]

    T = 0.25
    T_min = 0.0004
    cool = 0.985

    while T > T_min:
        if time.time() > deadline:
            break
        i = rng.randrange(N)
        dx = rng.gauss(0.0, 0.18)
        dy = rng.gauss(0.0, 0.18)
        dth = rng.gauss(0.0, 8.0)
        nx = xs[i] + dx
        ny = ys[i] + dy
        nth = angs[i] + dth
        new_verts = []
        for j in range(N):
            if j == i:
                new_verts.extend(_hex_vertices(nx, ny, nth))
            else:
                new_verts.extend(_hex_vertices(xs[j], ys[j], angs[j]))
        new_hull = _convex_hull(new_verts)
        new_side = _side_from_hull(new_hull, 0.0)
        dE = new_side - side
        if dE <= 0 or rng.random() < math.exp(-dE / T):
            xs[i] = nx
            ys[i] = ny
            angs[i] = nth
            verts = new_verts
            hull = new_hull
            side = new_side
            if side < best_side - 1e-9:
                best_side = side
                best_xs = xs[:]
                best_ys = ys[:]
                best_angs = angs[:]
        T *= cool

    if time.time() < deadline - 2.0:
        base_xs = best_xs[:]
        base_ys = best_ys[:]
        base_angs = best_angs[:]
        best_rot_side = best_side
        for trial in range(24):
            rot = (trial - 12) * 2.5
            cx = sum(base_xs) / N
            cy = sum(base_ys) / N
            a = math.radians(rot)
            ca, sa = math.cos(a), math.sin(a)
            rx = []
            ry = []
            for px, py in zip(base_xs, base_ys):
                ddx = px - cx
                ddy = py - cy
                rx.append(cx + ca * ddx - sa * ddy)
                ry.append(cy + sa * ddx + ca * ddy)
            verts = []
            for j in range(N):
                verts.extend(_hex_vertices(rx[j], ry[j], base_angs[j] + rot))
            h = _convex_hull(verts)
            s = _side_from_hull(h, 0.0)
            if s < best_rot_side - 1e-9:
                best_rot_side = s
                best_xs = rx[:]
                best_ys = ry[:]
                best_angs = [a_ + rot for a_ in base_angs]
        best_side = best_rot_side

    inner = []
    for i in range(N):
        inner.append([best_xs[i], best_ys[i], best_angs[i]])
    return inner, (0.0, 0.0), best_side, 0.0
# EVOLVE-BLOCK-END
