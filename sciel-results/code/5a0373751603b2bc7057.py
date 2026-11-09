"""Pack 11 unit hexagons inside a regular hexagon, minimizing outer side length."""
import numpy as np
import math
import time
import random

N = 11
SQRT3 = math.sqrt(3)

def construct():
    start_time = time.time()
    deadline = start_time + 145.0

    def hex_verts(cx, cy, side, angle_deg):
        a = math.radians(angle_deg)
        return [(cx + side * math.cos(a + k * math.pi / 3),
                 cy + side * math.sin(a + k * math.pi / 3)) for k in range(6)]

    def outer_side(positions, outer_angle_deg):
        """Minimum outer hex side to contain all inner hexes. Outer center at origin."""
        a = math.radians(outer_angle_deg)
        ca, sa = math.cos(-a), math.sin(-a)
        normals = [(math.cos(math.radians(30 + k * 60)), math.sin(math.radians(30 + k * 60))) for k in range(6)]
        max_proj = 0.0
        for ix, iy, iang in positions:
            for vx, vy in hex_verts(ix, iy, 1.0, iang):
                rx = vx * ca - vy * sa
                ry = vx * sa + vy * ca
                for nx, ny in normals:
                    p = rx * nx + ry * ny
                    if p > max_proj:
                        max_proj = p
        return max_proj / (SQRT3 / 2) + 1e-10

    def smooth_outer_side(positions, outer_angle_deg, beta=50.0):
        """Smooth approximation of outer side using log-sum-exp."""
        a = math.radians(outer_angle_deg)
        ca, sa = math.cos(-a), math.sin(-a)
        normals = [(math.cos(math.radians(30 + k * 60)), math.sin(math.radians(30 + k * 60))) for k in range(6)]
        projs = []
        for ix, iy, iang in positions:
            for vx, vy in hex_verts(ix, iy, 1.0, iang):
                rx = vx * ca - vy * sa
                ry = vx * sa + vy * ca
                for nx, ny in normals:
                    projs.append(rx * nx + ry * ny)
        if not projs:
            return 0.0
        m = max(projs)
        s = sum(math.exp(p - m) for p in projs)
        smooth_max = m + math.log(s) / beta
        return smooth_max / (SQRT3 / 2) + 1e-10

    def pair_distance(p1, p2):
        """Min distance between two hexes via SAT. Returns 0 if overlapping."""
        v1 = hex_verts(p1[0], p1[1], 1.0, p1[2])
        v2 = hex_verts(p2[0], p2[1], 1.0, p2[2])
        axes = []
        for verts in (v1, v2):
            for i in range(6):
                x1, y1 = verts[i]
                x2, y2 = verts[(i + 1) % 6]
                dx, dy = x2 - x1, y2 - y1
                L = math.sqrt(dx * dx + dy * dy)
                if L > 1e-12:
                    axes.append((-dy / L, dx / L))
        dx_c = p2[0] - p1[0]
        dy_c = p2[1] - p1[1]
        dc = math.sqrt(dx_c * dx_c + dy_c * dy_c)
        if dc > 1e-12:
            axes.append((dx_c / dc, dy_c / dc))
        max_gap = 0.0
        for nx, ny in axes:
            m1 = min(vx * nx + vy * ny for vx, vy in v1)
            M1 = max(vx * nx + vy * ny for vx, vy in v1)
            m2 = min(vx * nx + vy * ny for vx, vy in v2)
            M2 = max(vx * nx + vy * ny for vx, vy in v2)
            gap = max(m2 - M1, m1 - M2)
            if gap > max_gap:
                max_gap = gap
        return max_gap

    def overlap_penalty(positions):
        """Smooth penalty for overlaps."""
        pen = 0.0
        n = len(positions)
        for i in range(n):
            for j in range(i + 1, n):
                dx = positions[i][0] - positions[j][0]
                dy = positions[i][1] - positions[j][1]
                d = math.sqrt(dx * dx + dy * dy)
                if d < 2.2:
                    # Use smooth penalty based on center distance
                    if d < 2.0:
                        pen += (2.0 - d) ** 2
                    elif d < 2.2:
                        pen += 0.1 * (2.2 - d) ** 2
        return pen

    def objective(params, outer_angle, smooth=False, beta=50.0):
        positions = build_positions(params)
        if smooth:
            side = smooth_outer_side(positions, outer_angle, beta)
        else:
            side = outer_side(positions, outer_angle)
        pen = overlap_penalty(positions)
        return side + 50.0 * pen

    def build_positions(params):
        """
        params: [x0,y0,a0, x1,y1,a1, ..., x10,y10,a10] = 33 params
        """
        positions = []
        for i in range(N):
            positions.append([params[3*i], params[3*i+1], params[3*i+2]])
        return positions

    def nelder_mead(f, x0, tol=1e-9, max_iter=5000, alpha=1.0, gamma=2.0, rho=0.5, sigma=0.5):
        n = len(x0)
        simplex = [list(x0)]
        for i in range(n):
            pt = list(x0)
            pt[i] += 0.1 if pt[i] == 0 else 0.1 * pt[i]
            simplex.append(pt)
        values = [f(pt) for pt in simplex]

        for it in range(max_iter):
            order = sorted(range(n + 1), key=lambda i: values[i])
            simplex = [simplex[i] for i in order]
            values = [values[i] for i in order]
            if values[-1] - values[0] < tol:
                break

            centroid = [sum(simplex[i][j] for i in range(n)) / n for j in range(n)]
            xr = [centroid[j] + alpha * (centroid[j] - simplex[-1][j]) for j in range(n)]
            fr = f(xr)

            if fr < values[0]:
                xe = [centroid[j] + gamma * (centroid[j] - simplex[-1][j]) for j in range(n)]
                fe = f(xe)
                if fe < fr:
                    simplex[-1], values[-1] = xe, fe
                else:
                    simplex[-1], values[-1] = xr, fr
            elif fr < values[-2]:
                simplex[-1], values[-1] = xr, fr
            else:
                xc = [centroid[j] + rho * (simplex[-1][j] - centroid[j]) for j in range(n)]
                fc = f(xc)
                if fc < values[-1]:
                    simplex[-1], values[-1] = xc, fc
                else:
                    for i in range(1, n + 1):
                        simplex[i] = [simplex[0][j] + sigma * (simplex[i][j] - simplex[0][j]) for j in range(n)]
                        values[i] = f(simplex[i])
        best_idx = min(range(n + 1), key=lambda i: values[i])
        return simplex[best_idx], values[best_idx]

    # Initial configuration: honeycomb-like
    # Central hex at origin
    # 6 ring at distance ~sqrt(3)+0.05 at angles 0,60,...,300
    # 4 outer at larger distance
    def make_init(ring_r=None, outer_r=None):
        if ring_r is None:
            ring_r = SQRT3 + 0.05
        if outer_r is None:
            outer_r = 2 * SQRT3 + 0.1
        params = []
        # Central
        params.extend([0.0, 0.0, 0.0])
        # Ring
        for k in range(6):
            ang = math.radians(60 * k)
            params.extend([ring_r * math.cos(ang), ring_r * math.sin(ang), 0.0])
        # 4 outer at 30, 150, 210, 330
        for ang_deg in [30, 150, 210, 330]:
            ang = math.radians(ang_deg)
            params.extend([outer_r * math.cos(ang), outer_r * math.sin(ang), 0.0])
        return params

    best_params = make_init()
    best_val = objective(best_params, 0.0)

    # Stage 1: Smooth Nelder-Mead on 33 params
    if time.time() < deadline - 30:
        f_smooth = lambda p: objective(p, 0.0, smooth=True, beta=30.0)
        res, val = nelder_mead(f_smooth, best_params, tol=1e-8, max_iter=3000)
        if val < best_val:
            best_val = val
            best_params = res

    # Stage 2: Hard Nelder-Mead
    if time.time() < deadline - 20:
        f_hard = lambda p: objective(p, 0.0, smooth=False)
        res, val = nelder_mead(f_hard, best_params, tol=1e-9, max_iter=3000)
        if val < best_val:
            best_val = val
            best_params = res

    # Simulated annealing on 33 params
    if time.time() < deadline - 10:
        cur = list(best_params)
        cur_val = best_val
        temp = 0.5
        cooling = 0.9995
        min_temp = 0.005
        max_steps = 30000
        for s in range(max_steps):
            if time.time() > deadline - 10:
                break
            old = list(cur)
            # Perturb 1-3 random positions (each position = 3 params)
            n_pert = random.randint(1, 3)
            for _ in range(n_pert):
                idx = random.randint(0, N - 1) * 3
                cur[idx] += random.gauss(0, 0.15)
                cur[idx + 1] += random.gauss(0, 0.15)
                cur[idx + 2] += random.gauss(0, 5.0)

            nv = objective(cur, 0.0)
            delta = nv - cur_val
            if delta < 0 or random.random() < math.exp(-delta / temp):
                cur_val = nv
                if cur_val < best_val:
                    best_val = cur_val
                    best_params = list(cur)
            else:
                cur = old
            temp *= cooling

    # Optimize outer angle
    positions = build_positions(best_params)
    best_side = float('inf')
    best_angle = 0.0
    for a in range(0, 60, 1):
        side = outer_side(positions, float(a))
        if side < best_side:
            best_side = side
            best_angle = float(a)

    # Fine-tune angle
    for _ in range(500):
        if time.time() > deadline - 3:
            break
        trial = best_angle + random.gauss(0, 0.5)
        side = outer_side(positions, trial)
        if side < best_side:
            best_side = side
            best_angle = trial

    # Verify no overlap
    pen = overlap_penalty(positions)
    if pen > 1e-4:
        # Try to fix: small SA to reduce overlaps
        for _ in range(5000):
            if time.time() > deadline - 2:
                break
            old = list(best_params)
            idx = random.randint(0, N - 1) * 3
            best_params[idx] += random.gauss(0, 0.1)
            best_params[idx + 1] += random.gauss(0, 0.1)
            positions = build_positions(best_params)
            new_pen = overlap_penalty(positions)
            new_side = outer_side(positions, best_angle)
            if new_pen < pen:
                pen = new_pen
                best_side = new_side
            else:
                best_params = old

        if pen > 1e-3:
            # Fallback to safe spiral
            safe = []
            safe.append([0.0, 0.0, 0.0])
            for i in range(1, N):
                ring = (i - 1) // 6 + 1
                pos_in_ring = (i - 1) % 6
                ang = math.radians(60 * pos_in_ring)
                r = 2.1 * ring
                safe.append([r * math.cos(ang), r * math.sin(ang), 0.0])
            best_side = float('inf')
            best_angle = 0.0
            for a in range(0, 60, 5):
                s = outer_side(safe, float(a))
                if s < best_side:
                    best_side = s
                    best_angle = float(a)
            positions = safe

    return positions, (0.0, 0.0), best_side, best_angle
