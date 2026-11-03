import numpy as np
import time
import math

N = 11
SQ3 = math.sqrt(3.0)

def construct():
    start = time.time()
    deadline = 140.0  # safety margin before 150s kill

    # Unit hexagon vertices (flat orientation: vertex at angle 0)
    unit_hex = np.array([
        [1.0, 0.0],
        [0.5, SQ3/2],
        [-0.5, SQ3/2],
        [-1.0, 0.0],
        [-0.5, -SQ3/2],
        [0.5, -SQ3/2]
    ])

    def hex_verts(x, y, ang_deg):
        a = math.radians(ang_deg)
        c, s = math.cos(a), math.sin(a)
        R = np.array([[c, -s], [s, c]])
        return unit_hex @ R.T + np.array([x, y])

    def min_outer_side(centers, angles, outer_rotation=0.0):
        """Minimal side of a regular hexagon (centered at origin, given rotation)
        that contains all hexagons. Returns inf if any hexagon is degenerate."""
        a = math.radians(outer_rotation)
        c, s = math.cos(a), math.sin(a)
        # 6 outward face normals of the outer hexagon
        normals = np.array([
            [1.0, 0.0],
            [0.5, SQ3/2],
            [-0.5, SQ3/2],
            [-1.0, 0.0],
            [-0.5, -SQ3/2],
            [0.5, -SQ3/2]
        ]) @ np.array([[c, -s], [s, c]]).T
        max_apothem = 0.0
        for k in range(N):
            v = hex_verts(centers[k, 0], centers[k, 1], angles[k])
            # project all vertices onto each normal
            proj = v @ normals.T  # shape (6,6)
            m = proj.max()
            if m > max_apothem:
                max_apothem = m
        return 2.0 * max_apothem / SQ3

    def overlap_penalty(centers, angles, beta=60.0):
        """Smooth penalty for pairwise overlap (vertex vs opposite-edge)."""
        pen = 0.0
        for i in range(N):
            vi = hex_verts(centers[i, 0], centers[i, 1], angles[i])
            for j in range(i + 1, N):
                vj = hex_verts(centers[j, 0], centers[j, 1], angles[j])
                # vertices of i vs edges of j
                for p in vi:
                    for k in range(6):
                        e1 = vj[k]; e2 = vj[(k+1) % 6]
                        edge = e2 - e1
                        L = math.hypot(edge[0], edge[1])
                        if L < 1e-12:
                            continue
                        n = np.array([-edge[1], edge[0]]) / L
                        d = n @ (p - e1)
                        if d < 0:
                            pen += math.exp(-beta * (-d))
                # vertices of j vs edges of i
                for p in vj:
                    for k in range(6):
                        e1 = vi[k]; e2 = vi[(k+1) % 6]
                        edge = e2 - e1
                        L = math.hypot(edge[0], edge[1])
                        if L < 1e-12:
                            continue
                        n = np.array([-edge[1], edge[0]]) / L
                        d = n @ (p - e1)
                        if d < 0:
                            pen += math.exp(-beta * (-d))
        return pen

    def point_in_hex(p, verts):
        """Check point p inside polygon verts (CCW)."""
        for k in range(6):
            e1 = verts[k]; e2 = verts[(k+1) % 6]
            edge = e2 - e1
            n = np.array([-edge[1], edge[0]])
            if n @ (p - e1) < -1e-9:
                return False
        return True

    def valid(centers, angles, tol=1e-9):
        """Exact non-overlap check via vertex containment both ways."""
        for i in range(N):
            vi = hex_verts(centers[i, 0], centers[i, 1], angles[i])
            for j in range(i + 1, N):
                vj = hex_verts(centers[j, 0], centers[j, 1], angles[j])
                for p in vi:
                    if point_in_hex(p, vj):
                        return False
                for p in vj:
                    if point_in_hex(p, vi):
                        return False
        return True

    # ---- Analytical optimal: hexagonal cluster ----
    d = SQ3  # center-to-center distance of adjacent unit hexagons
    centers = np.zeros((N, 2))
    angles = np.zeros(N)
    # center
    centers[0] = [0, 0]
    # ring 1 (6)
    for k in range(6):
        ang = k * math.pi / 3
        centers[1 + k] = [d * math.cos(ang), d * math.sin(ang)]
    # ring 2: 4 hexagons at alternating corners (angles 0, 60, 180, 240)
    ring2_angles = [0, 60, 180, 240]
    for idx, deg in enumerate(ring2_angles):
        ang = math.radians(deg)
        centers[7 + idx] = [2 * d * math.cos(ang), 2 * d * math.sin(ang)]

    side = min_outer_side(centers, angles, 0.0)
    assert valid(centers, angles), "analytical cluster must be valid"
    best_side = side
    best_centers = centers.copy()
    best_angles = angles.copy()

    # ---- Refinement: try to beat the analytical side via local search ----
    def objective(centers, angles):
        return min_outer_side(centers, angles, 0.0) + overlap_penalty(centers, angles)

    def local_search(centers, angles, iters, step0=0.3):
        nonlocal best_side, best_centers, best_angles
        cur_c = centers.copy()
        cur_a = angles.copy()
        cur_obj = objective(cur_c, cur_a)
        step = step0
        for it in range(iters):
            if time.time() > deadline:
                break
            idx = np.random.randint(N)
            new_c = cur_c.copy()
            new_a = cur_a.copy()
            new_c[idx, 0] += np.random.normal(0, step)
            new_c[idx, 1] += np.random.normal(0, step)
            new_a[idx] += np.random.normal(0, step * 25)
            new_obj = objective(new_c, new_a)
            if new_obj < cur_obj:
                cur_c = new_c
                cur_a = new_a
                cur_obj = new_obj
                # exact validity check occasionally
                if it % 20 == 0 and valid(cur_c, cur_a):
                    s = min_outer_side(cur_c, cur_a, 0.0)
                    if s < best_side:
                        best_side = s
                        best_centers = cur_c.copy()
                        best_angles = cur_a.copy()
            step *= 0.9995
        # final exact check
        if valid(cur_c, cur_a):
            s = min_outer_side(cur_c, cur_a, 0.0)
            if s < best_side:
                best_side = s
                best_centers = cur_c.copy()
                best_angles = cur_a.copy()

    # A few refinement attempts from the analytical seed (cheap, fast)
    np.random.seed(7)
    attempts = 0
    while time.time() < deadline - 5 and attempts < 6:
        attempts += 1
        local_search(best_centers.copy(), best_angles.copy(),
                     iters=4000, step0=0.25)

    # ---- Build output ----
    inner = []
    for i in range(N):
        inner.append([float(best_centers[i, 0]), float(best_centers[i, 1]),
                      float(best_angles[i])])
    return inner, (0.0, 0.0), float(best_side), 0.0
