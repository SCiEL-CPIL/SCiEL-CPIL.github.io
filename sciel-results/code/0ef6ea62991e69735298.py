"""Pack 11 unit hexagons inside a regular hexagon, minimizing outer side length."""
import numpy as np
import math
import time
import random

N = 11

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s limit

    sqrt3 = math.sqrt(3)
    inradius = sqrt3 / 2.0  # inradius of unit hexagon

    def hexagon_vertices(cx, cy, side, angle_deg):
        angle_rad = math.radians(angle_deg)
        return [(cx + side * math.cos(angle_rad + k * math.pi / 3),
                 cy + side * math.sin(angle_rad + k * math.pi / 3)) for k in range(6)]

    def compute_min_side(inner_positions, outer_angle):
        """Compute minimum outer side length to contain all inner hexagons.
        Outer hexagon centered at origin with given rotation."""
        max_side = 0.0
        angle_rad = math.radians(outer_angle)
        cos_a = math.cos(-angle_rad)
        sin_a = math.sin(-angle_rad)

        normals = [(math.cos(math.radians(30 + k * 60)), math.sin(math.radians(30 + k * 60))) for k in range(6)]

        for ix, iy, iang in inner_positions:
            iv = hexagon_vertices(ix, iy, 1.0, iang)
            for vx, vy in iv:
                rx = vx * cos_a - vy * sin_a
                ry = vx * sin_a + vy * cos_a
                for nx, ny in normals:
                    proj = rx * nx + ry * ny
                    req = proj / inradius
                    if req > max_side:
                        max_side = req
        return max_side

    def sat_overlap(p1, p2):
        """Return penetration depth > 0 if overlapping, 0 if disjoint."""
        dx = p1[0] - p2[0]
        dy = p1[1] - p2[1]
        d = math.sqrt(dx*dx + dy*dy)
        if d >= 2.0:
            return 0.0
        if d < 1e-9:
            return 2.0

        v1 = hexagon_vertices(p1[0], p1[1], 1.0, p1[2])
        v2 = hexagon_vertices(p2[0], p2[1], 1.0, p2[2])

        axes = []
        for verts in (v1, v2):
            for i in range(6):
                x1, y1 = verts[i]
                x2, y2 = verts[(i+1) % 6]
                ex, ey = x2-x1, y2-y1
                L = math.sqrt(ex*ex + ey*ey)
                if L > 1e-12:
                    axes.append((-ey/L, ex/L))
        axes.append((dx/d, dy/d))

        min_gap = float('inf')
        for nx, ny in axes:
            p1min = min(vx*nx+vy*ny for vx,vy in v1)
            p1max = max(vx*nx+vy*ny for vx,vy in v1)
            p2min = min(vx*nx+vy*ny for vx,vy in v2)
            p2max = max(vx*nx+vy*ny for vx,vy in v2)
            gap = max(p2min - p1max, p1min - p2max)
            if gap > 0:
                return 0.0
            min_gap = min(min_gap, -gap)
        return min_gap if min_gap < float('inf') else 0.0

    def smooth_penalty(p1, p2):
        """Smooth overlap penalty for differentiable optimization."""
        dx = p1[0] - p2[0]
        dy = p1[1] - p2[1]
        d = math.sqrt(dx*dx + dy*dy)
        # If far apart, no penalty
        if d >= 2.0:
            return 0.0
        # Use exact SAT for accuracy
        pen = sat_overlap(p1, p2)
        return pen * pen  # quadratic penalty for smoothness

    def objective(params):
        """Objective: outer side + penalty for overlaps.
        params: flat array of length 3*N = [x1,y1,a1, x2,y2,a2, ...]
        Outer hexagon centered at origin, rotation 0 (we optimize rotation separately)."""
        positions = []
        for i in range(N):
            positions.append([params[3*i], params[3*i+1], params[3*i+2]])
        side = compute_min_side(positions, 0.0)
        pen = 0.0
        for i in range(N):
            for j in range(i+1, N):
                pen += smooth_penalty(positions[i], positions[j])
        return side + 100.0 * pen

    def make_honeycomb():
        positions = []
        positions.append([0.0, 0.0, 0.0])
        ring_dist = sqrt3 + 0.1
        for k in range(6):
            a = math.radians(60*k)
            positions.append([ring_dist*math.cos(a), ring_dist*math.sin(a), 0.0])
        gap_dist = 2*ring_dist*math.cos(math.pi/6) + 0.1
        for ad in [30, 150, 210, 330]:
            a = math.radians(ad)
            positions.append([gap_dist*math.cos(a), gap_dist*math.sin(a), 0.0])
        return positions

    def make_ring():
        positions = []
        positions.append([0.0, 0.0, 0.0])
        for k in range(6):
            a = math.radians(60*k)
            positions.append([sqrt3*math.cos(a), sqrt3*math.sin(a), 0.0])
        for k in range(4):
            a = math.radians(30 + 120*k)
            r = 2*sqrt3*math.cos(math.pi/6)
            positions.append([r*math.cos(a), r*math.sin(a), 0.0])
        return positions

    def make_random(scale=3.0):
        positions = []
        positions.append([0.0, 0.0, 0.0])
        for i in range(N-1):
            r = random.uniform(0.5, scale)
            a = random.uniform(0, 2*math.pi)
            positions.append([r*math.cos(a), r*math.sin(a), random.uniform(-30, 30)])
        return positions

    def to_params(positions):
        params = []
        for p in positions:
            params.extend([p[0], p[1], p[2]])
        return np.array(params)

    def from_params(params):
        positions = []
        for i in range(N):
            positions.append([params[3*i], params[3*i+1], params[3*i+2]])
        return positions

    # Try multiple initializations and Nelder-Mead refinement
    inits = [make_honeycomb(), make_ring()]
    for _ in range(4):
        inits.append(make_random(random.uniform(2.0, 4.0)))

    best_positions = None
    best_obj = float('inf')

    for init in inits:
        if time.time() > deadline - 30.0:
            break
        params = to_params(init)
        try:
            from scipy.optimize import minimize
            res = minimize(objective, params, method='Nelder-Mead',
                          options={'maxiter': 3000, 'xatol': 1e-4, 'fatol': 1e-6,
                                   'adaptive': True})
            if res.fun < best_obj:
                best_obj = res.fun
                best_positions = from_params(res.x)
        except Exception:
            pass

    # If Nelder-Mead didn't find anything good, fall back to simple SA
    if best_positions is None or best_obj > 10.0:
        # Quick SA fallback
        current = to_params(make_honeycomb())
        curr_obj = objective(current)
        best_obj = curr_obj
        best_positions = from_params(current)
        temp = 2.0
        for step in range(20000):
            if time.time() > deadline - 25.0:
                break
            idx = random.randint(0, N-1)
            old = current[3*idx:3*idx+3].copy()
            current[3*idx] += random.gauss(0, 0.1)
            current[3*idx+1] += random.gauss(0, 0.1)
            current[3*idx+2] += random.gauss(0, 3.0)
            new_obj = objective(current)
            if new_obj < curr_obj or random.random() < math.exp(-(new_obj - curr_obj)/temp):
                curr_obj = new_obj
                if curr_obj < best_obj:
                    best_obj = curr_obj
                    best_positions = from_params(current)
            else:
                current[3*idx:3*idx+3] = old
            temp *= 0.998

    # Verify and fix overlaps
    positions = best_positions
    # Iteratively separate overlapping pairs
    for it in range(50):
        if time.time() > deadline - 10.0:
            break
        moved = False
        for i in range(N):
            for j in range(i+1, N):
                pen = sat_overlap(positions[i], positions[j])
                if pen > 1e-6:
                    # Push apart
                    dx = positions[i][0] - positions[j][0]
                    dy = positions[i][1] - positions[j][1]
                    d = math.sqrt(dx*dx + dy*dy)
                    if d < 1e-9:
                        dx, dy = 1.0, 0.0
                        d = 1.0
                    push = pen * 0.5 + 0.01
                    positions[i][0] += push * dx/d
                    positions[i][1] += push * dy/d
                    positions[j][0] -= push * dx/d
                    positions[j][1] -= push * dy/d
                    moved = True
        if not moved:
            break

    # Final: optimize outer rotation
    best_side = float('inf')
    best_angle = 0.0
    for ang in range(0, 31, 2):
        s = compute_min_side(positions, float(ang))
        if s < best_side:
            best_side = s
            best_angle = float(ang)
    # Fine-tune angle
    for _ in range(100):
        if time.time() > deadline - 3.0:
            break
        trial = best_angle + random.gauss(0, 0.5)
        s = compute_min_side(positions, trial)
        if s < best_side:
            best_side = s
            best_angle = trial

    # Final overlap check - if still overlapping, use safe fallback
    final_pen = sum(sat_overlap(positions[i], positions[j]) for i in range(N) for j in range(i+1, N))
    if final_pen > 1e-4:
        # Safe spiral
        safe = [[0.0, 0.0, 0.0]]
        for i in range(1, N):
            r = 2.1 * (i // 6 + 1)
            a = math.radians(60 * (i % 6))
            safe.append([r*math.cos(a), r*math.sin(a), 0.0])
        best_side = float('inf')
        best_angle = 0.0
        for ang in range(0, 31, 3):
            s = compute_min_side(safe, float(ang))
            if s < best_side:
                best_side = s
                best_angle = float(ang)
        positions = safe

    return positions, (0.0, 0.0), best_side, best_angle
# EVOLVE-BLOCK-END
