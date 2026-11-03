"""Packing 12 unit hexagons in a regular hexagon - minimize outer side length."""
import numpy as np
import time
import math
import random

N = 12
START_TIME = time.time()
TIME_LIMIT = 140.0

def hex_vertices(cx, cy, side, angle_deg):
    angle = math.radians(angle_deg)
    return np.array([(cx + side * math.cos(angle + i * math.pi / 3),
                     cy + side * math.sin(angle + i * math.pi / 3)) for i in range(6)])

def polygons_overlap_sat(p1, p2, eps=1e-8):
    for poly in (p1, p2):
        n = len(poly)
        for i in range(n):
            j = (i + 1) % n
            ex, ey = poly[j][0] - poly[i][0], poly[j][1] - poly[i][1]
            nx, ny = -ey, ex
            d1 = np.dot(p1, [nx, ny])
            d2 = np.dot(p2, [nx, ny])
            if d1.max() + eps < d2.min() or d2.max() + eps < d1.min():
                return False
    return True

def min_outer_side_fast(pts, cx, cy, theta_deg):
    """Fast: min side of regular hex (center cx,cy, rot theta) containing pts."""
    theta = math.radians(theta_deg)
    apothem_factor = math.sqrt(3) / 2
    max_proj = 0.0
    for i in range(6):
        a = theta + i * math.pi / 3
        ca, sa = math.cos(a), math.sin(a)
        for px, py in pts:
            proj = (px - cx) * ca + (py - cy) * sa
            ap = abs(proj)
            if ap > max_proj:
                max_proj = ap
    return max_proj / apothem_factor

def get_all_pts(positions, angles):
    pts = []
    for k in range(N):
        v = hex_vertices(positions[k][0], positions[k][1], 1.0, angles[k])
        pts.extend(v.tolist())
    return pts

def optimize_outer(positions, angles):
    """Find best (center, side, angle) for outer hexagon."""
    pts = get_all_pts(positions, angles)
    best_side = float('inf')
    best_c = (0.0, 0.0)
    best_a = 0.0
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    cx0, cy0 = sum(xs)/len(xs), sum(ys)/len(ys)
    for theta_deg in range(0, 30, 3):
        cx, cy = cx0, cy0
        for _ in range(15):
            # Compute current side and find worst point/direction
            theta = math.radians(theta_deg)
            max_proj = 0.0
            worst_px, worst_py = cx, cy
            worst_ca, worst_sa = 1.0, 0.0
            for i in range(6):
                a = theta + i * math.pi / 3
                ca, sa = math.cos(a), math.sin(a)
                for px, py in pts:
                    proj = (px - cx) * ca + (py - cy) * sa
                    if abs(proj) > max_proj:
                        max_proj = abs(proj)
                        worst_px, worst_py = px, py
                        worst_ca, worst_sa = ca if proj > 0 else -ca, sa if proj > 0 else -sa
            # Move center toward worst point
            cx += 0.15 * (worst_px - cx)
            cy += 0.15 * (worst_py - cy)
        side = min_outer_side_fast(pts, cx, cy, theta_deg)
        if side < best_side:
            best_side = side
            best_c = (cx, cy)
            best_a = theta_deg
    return best_c, best_side, best_a

def no_overlap(positions, angles):
    polys = [hex_vertices(positions[i][0], positions[i][1], 1.0, angles[i]) for i in range(N)]
    for i in range(N):
        for j in range(i+1, N):
            if polygons_overlap_sat(polys[i], polys[j]):
                return False
    return True

def init_config():
    pos, ang = [], []
    s3 = math.sqrt(3)
    for r in range(3):
        for c in range(4):
            x = c * s3 + (s3/2 if r % 2 else 0)
            y = r * 1.5
            pos.append([x, y, 0.0])
            ang.append(90.0)
    xs = [p[0] for p in pos]
    ys = [p[1] for p in pos]
    cx, cy = (min(xs)+max(xs))/2, (min(ys)+max(ys))/2
    for p in pos:
        p[0] -= cx
        p[1] -= cy
    return pos, ang

def construct():
    global START_TIME, TIME_LIMIT
    pos, ang = init_config()
    if not no_overlap(pos, ang):
        for i in range(N):
            pos[i][0] *= 1.3
            pos[i][1] *= 1.3
    
    c, s, a = optimize_outer(pos, ang)
    best_s, best_pos, best_ang, best_c, best_a = s, [p[:] for p in pos], ang[:], c, a
    
    T, T_min, cool = 0.5, 0.005, 0.9995
    step = 0.3
    np.random.seed(42)
    random.seed(42)
    it = 0
    while T > T_min:
        if time.time() - START_TIME > TIME_LIMIT:
            break
        it += 1
        np2 = [p[:] for p in pos]
        an2 = ang[:]
        idx = random.randint(0, N-1)
        np2[idx][0] += random.gauss(0, step)
        np2[idx][1] += random.gauss(0, step)
        an2[idx] = (ang[idx] + random.gauss(0, step*15)) % 360
        if not no_overlap(np2, an2):
            T *= cool
            continue
        c2, s2, a2 = optimize_outer(np2, an2)
        d = s2 - s
        if d < 0 or random.random() < math.exp(-d / T):
            pos, ang, c, s, a = np2, an2, c2, s2, a2
            if s < best_s:
                best_s, best_pos, best_ang, best_c, best_a = s, [p[:] for p in pos], ang[:], c, a
        T *= cool
    
    inner = [[p[0], p[1], a] for p, a in zip(best_pos, best_ang)]
    return inner, best_c, best_s, best_a

if __name__ == "__main__":
    r = construct()
    print(f"Outer side: {r[2]}")
