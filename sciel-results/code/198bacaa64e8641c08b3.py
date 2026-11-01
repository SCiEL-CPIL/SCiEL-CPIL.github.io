import math
import time
import random
import numpy as np

N = 12

# ---------- geometry helpers ----------

def _hex_vertices(cx, cy, side, ang_deg):
    a = math.radians(ang_deg)
    ca, sa = math.cos(a), math.sin(a)
    out = []
    for k in range(6):
        t = math.radians(60.0 * k)
        x = side * math.cos(t)
        y = side * math.sin(t)
        out.append((cx + ca * x - sa * y, cy + sa * x + ca * y))
    return out

def _normals(vertices):
    n = len(vertices)
    norms = []
    for i in range(n):
        x1, y1 = vertices[i]
        x2, y2 = vertices[(i + 1) % n]
        ex, ey = x2 - x1, y2 - y1
        L = math.hypot(ex, ey)
        if L < 1e-12:
            continue
        # inward normal for a CCW polygon
        nx, ny = ey / L, -ex / L
        norms.append((nx, ny))
    return norms

def _penetration_depth(A, B, nA, nB):
    # A, B: lists of (x,y); nA,nB: corresponding inward normals
    best = math.inf
    for i, (nx, ny) in enumerate(nA):
        p = A[i]
        dmax = -math.inf
        for q in B:
            d = (q[0] - p[0]) * nx + (q[1] - p[1]) * ny
            if d > dmax:
                dmax = d
        if dmax < 0:
            return 0.0  # separating axis found -> disjoint
        if dmax < best:
            best = dmax
    for i, (nx, ny) in enumerate(nB):
        p = B[i]
        dmax = -math.inf
        for q in A:
            d = (q[0] - p[0]) * nx + (q[1] - p[1]) * ny
            if d > dmax:
                dmax = d
        if dmax < 0:
            return 0.0
        if dmax < best:
            best = dmax
    return best if best < math.inf else 0.0

def _inside_violation(inner_verts, outer_verts, outer_norms):
    # inner_verts inside outer_verts (CCW). Return max penetration (0 if inside).
    # For each outer edge (inward normal), inner vertex must have depth >= 0.
    # violation = max over outer edges of ( -min over inner verts of depth )
    worst = 0.0
    for i, (nx, ny) in enumerate(outer_norms):
        p = outer_verts[i]
        dmin = math.inf
        for q in inner_verts:
            d = (q[0] - p[0]) * nx + (q[1] - p[1]) * ny
            if d < dmin:
                dmin = d
        if dmin < worst:
            worst = dmin
    return -worst if worst < 0 else 0.0

# ---------- main construct ----------

def construct():
    deadline = time.time() + 140.0  # safety margin vs 150s kill
    random.seed(12345)

    # Initial state: a modest outer hexagon containing a hexagonal lattice of 12
    # unit hexagons (side 1). We will shrink it.
    def init_state():
        # hex lattice of 12: center + ring of 6 + next partial ring
        positions = [(0.0, 0.0)]
        # ring 1: 6 neighbors at distance sqrt(3) (hex tiling, side 1)
        d1 = math.sqrt(3.0)
        for k in range(6):
            t = math.radians(60.0 * k)
            positions.append((d1 * math.cos(t), d1 * math.sin(t)))
        # ring 2: 6 more at distance 2*sqrt(3)
        d2 = 2.0 * math.sqrt(3.0)
        for k in range(6):
            t = math.radians(60.0 * k + 30.0)
            positions.append((d2 * math.cos(t), d2 * math.sin(t)))
        # that's 13; drop one to make 12
        positions = positions[:N]
        xs = [p[0] for p in positions]
        ys = [p[1] for p in positions]
        # outer side must cover radius + 1
        r = max(math.hypot(x, y) for x, y in positions) + 1.0
        side = r + 0.5
        return positions, [0.0] * N, side, 0.0, (0.0, 0.0)

    positions, angles, side, oang, ocenter = init_state()

    def evaluate():
        # returns (total_violation, side)
        ox, oy = ocenter
        outer_verts = _hex_vertices(ox, oy, side, oang)
        outer_norms = _normals(outer_verts)
        inner_verts = [_hex_vertices(positions[i][0], positions[i][1], 1.0, angles[i]) for i in range(N)]
        inner_norms = [_normals(iv) for iv in inner_verts]
        total = 0.0
        # inside constraints
        for i in range(N):
            v = _inside_violation(inner_verts[i], outer_verts, outer_norms)
            total += v
        # pairwise disjoint
        for i in range(N):
            for j in range(i + 1, N):
                d = _penetration_depth(inner_verts[i], inner_verts[j], inner_norms[i], inner_norms[j])
                total += d
        return total, side

    best_positions = [p[:] for p in positions]
    best_angles = angles[:]
    best_side = side
    best_oang = oang
    best_ocenter = ocenter[:]
    best_viol, _ = evaluate()

    T = 5.0
    T_min = 1e-6
    cooling = 0.999

    while time.time() < deadline:
        if T < T_min:
            T = T_min
        # pick a move
        r = random.random()
        if r < 0.35:
            i = random.randrange(N)
            dx = random.gauss(0, 0.5)
            dy = random.gauss(0, 0.5)
            positions[i] = (positions[i][0] + dx, positions[i][1] + dy)
        elif r < 0.55:
            i = random.randrange(N)
            angles[i] += random.gauss(0, 5.0)
        elif r < 0.80:
            # shrink outer
            side = max(2.0, side * (1.0 + random.gauss(0, 0.01)))
        else:
            # rotate outer / shift center slightly
            oang += random.gauss(0, 2.0)
            ocenter = (ocenter[0] + random.gauss(0, 0.2), ocenter[1] + random.gauss(0, 0.2))

        viol, _ = evaluate()
        dE = viol - best_viol
        if dE < 0 or random.random() < math.exp(-dE / T):
            best_viol = viol
            if viol <= 1e-6:
                # valid configuration: record and try to shrink more aggressively
                best_positions = [p[:] for p in positions]
                best_angles = angles[:]
                best_side = side
                best_oang = oang
                best_ocenter = ocenter[:]
        else:
            # revert
            if r < 0.35:
                positions[i] = (positions[i][0] - dx, positions[i][1] - dy)
            elif r < 0.55:
                angles[i] -= random.gauss(0, 5.0)  # approximate revert; re-evaluate below
            elif r < 0.80:
                side = max(2.0, side / (1.0 + random.gauss(0, 0.01)))
            else:
                oang -= random.gauss(0, 2.0)
                ocenter = (ocenter[0] - random.gauss(0, 0.2), ocenter[1] - random.gauss(0, 0.2))

        T *= cooling

    # Final: ensure validity by returning the best valid found; if none, return current best
    return (
        [[best_positions[i][0], best_positions[i][1], best_angles[i]] for i in range(N)],
        best_ocenter,
        best_side,
        best_oang,
    )
