import math
import time
import numpy as np
from scipy.optimize import minimize

N = 11
APOTHEM = math.sqrt(3) / 2.0
START_TIME = time.time()
TIME_LIMIT = 150.0
SAFETY_MARGIN = 5.0
EPS = 1e-9

def construct():
    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
        return [[0.0, 0.0, 0.0]], [0.0, 0.0], 1.0, 0.0

    # Unit regular hexagon vertices (pointy-top, centered at origin)
    # For a hexagon with side length 1, circumradius = 1
    unit_hex_verts = np.array([
        [math.cos(math.pi * k / 3.0), math.sin(math.pi * k / 3.0)]
        for k in range(6)
    ])

    def hexagon_verts(cx, cy, side, angle_deg):
        """Return vertices of a regular hexagon."""
        a = math.radians(angle_deg)
        R = side
        cos_a, sin_a = math.cos(a), math.sin(a)
        rot = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
        return (unit_hex_verts * R) @ rot.T + np.array([cx, cy])

    def smooth_min2(a, b, t):
        """Smooth minimum of two scalars."""
        m = min(a, b)
        d = a - b
        if abs(d) < 1e-10:
            return m
        return m - t * math.log(1.0 + math.exp(-abs(d) / t))

    def pair_non_overlap(verts1, verts2, t=0.1):
        """
        Smooth measure of separation between two hexagons.
        Uses the fact that for two convex polygons, they don't overlap iff
        there exists a separating axis. We approximate the minimum distance
        using vertex-edge checks with smoothing.
        """
        # For efficiency, we'll use a simplified approach:
        # Check the 6 edge normals of each hexagon as potential separating axes
        # plus the axes between corresponding vertices
        min_dist = float('inf')

        # Get edge normals for hex1
        for i in range(6):
            p1 = verts1[i]
            p2 = verts1[(i + 1) % 6]
            ex = p2[0] - p1[0]
            ey = p2[1] - p1[1]
            e_len = math.sqrt(ex * ex + ey * ey)
            if e_len < EPS:
                continue
            nx = -ey / e_len
            ny = ex / e_len

            # Project both hexagons onto this normal
            proj1 = np.array([p[0] * nx + p[1] * ny for p in verts1])
            proj2 = np.array([p[0] * nx + p[1] * ny for p in verts2])

            # Gap between the two projections
            gap = max(proj2) - min(proj1)
            if gap < 0:
                # Overlapping on this axis
                if -gap < min_dist:
                    min_dist = -gap
            else:
                # Separated on this axis
                if gap < min_dist:
                    min_dist = gap

        # Also check the 6 edge normals of hex2
        for i in range(6):
            p1 = verts2[i]
            p2 = verts2[(i + 1) % 6]
            ex = p2[0] - p1[0]
            ey = p2[1] - p1[1]
            e_len = math.sqrt(ex * ex + ey * ey)
            if e_len < EPS:
                continue
            nx = -ey / e_len
            ny = ex / e_len

            proj1 = np.array([p[0] * nx + p[1] * ny for p in verts1])
            proj2 = np.array([p[0] * nx + p[1] * ny for p in verts2])

            gap = max(proj1) - min(proj2)
            if gap < 0:
                if -gap < min_dist:
                    min_dist = -gap
            else:
                if gap < min_dist:
                    min_dist = gap

        # Also check axes between vertex pairs (limited for efficiency)
        for i in range(6):
            for j in range(6):
                dx = verts2[j][0] - verts1[i][0]
                dy = verts2[j][1] - verts1[i][1]
                d_len = math.sqrt(dx * dx + dy * dy)
                if d_len < EPS:
                    continue
                nx = dx / d_len
                ny = dy / d_len

                proj1 = np.array([p[0] * nx + p[1] * ny for p in verts1])
                proj2 = np.array([p[0] * nx + p[1] * ny for p in verts2])

                gap = max(proj2) - min(proj1)
                if gap < 0:
                    if -gap < min_dist:
                        min_dist = -gap
                else:
                    if gap < min_dist:
                        min_dist = gap

        return min_dist

    def total_overlap_penalty(verts_list, t=0.05):
        """Total smooth overlap penalty for all pairs."""
        n = len(verts_list)
        penalty = 0.0
        for i in range(n):
            for j in range(i + 1, n):
                d = pair_non_overlap(verts_list[i], verts_list[j], t)
                # Penalize when d < margin
                margin = 0.0
                if d < margin:
                    penalty += (margin - d) ** 2
        return penalty

    def containment_penalty(verts_list, cx, cy, S, outer_angle_deg):
        """Penalty for inner hex vertices outside outer hexagon."""
        outer_verts = hexagon_verts(cx, cy, S, outer_angle_deg)
        penalty = 0.0
        for verts in verts_list:
            for vx, vy in verts:
                for k in range(6):
                    p1 = outer_verts[k]
                    p2 = outer_verts[(k + 1) % 6]
                    ex = p2[0] - p1[0]
                    ey = p2[1] - p1[1]
                    e_len = math.sqrt(ex * ex + ey * ey)
                    if e_len < EPS:
                        continue
                    nx = -ey / e_len
                    ny = ex / e_len
                    dist = (vx - p1[0]) * nx + (vy - p1[1]) * ny
                    if dist < 0:
                        penalty += (-dist) ** 2
        return penalty

    def compute_min_S(inner_params, cx, cy, outer_angle_deg):
        """Compute minimum S for given inner config and outer center/angle."""
        max_proj = 0.0
        for i in range(N):
            xi = inner_params[3 * i]
            yi = inner_params[3 * i + 1]
            ai = inner_params[3 * i + 2]
            verts = hexagon_verts(xi, yi, 1.0, ai)
            for vx, vy in verts:
                for k in range(6):
                    ang = math.radians(outer_angle_deg) + math.pi / 6.0 + k * math.pi / 3.0
                    nx = math.cos(ang)
                    ny = math.sin(ang)
                    proj = abs((vx - cx) * nx + (vy - cy) * ny)
                    if proj > max_proj:
                        max_proj = proj
        return 2.0 * max_proj / math.sqrt(3.0)

    def full_objective(params):
        """
        params: [x1, y1, a1, ..., xN, yN, aN, cx, cy, outer_angle, S]
        """
        inner_params = params[:3 * N]
        cx = params[3 * N]
        cy = params[3 * N + 1]
        outer_angle = params[3 * N + 2]
        S = params[3 * N + 3]

        if S < 1.0:
            S = 1.0

        # Build vertices
        verts_list = []
        for i in range(N):
            verts_list.append(hexagon_verts(inner_params[3 * i], inner_params[3 * i + 1], 1.0, inner_params[3 * i + 2]))

        # Overlap penalty
        penalty = total_overlap_penalty(verts_list)

        # Containment penalty
        penalty += containment_penalty(verts_list, cx, cy, S, outer_angle)

        # Small cost for large S
        penalty += 0.001 * S ** 2

        return penalty

    def initial_config(seed=0):
        """Generate initial hexagonal grid configuration."""
        params = []
        axial = [(0, 0)]
        ring1 = [(1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1)]
        axial.extend(ring1)
        ring2 = [(2, 0), (2, 1), (1, 2), (0, 2), (-1, 2), (-2, 1),
                 (-2, 0), (-2, -1), (-1, -2), (0, -2), (1, -2), (2, -1)]
        remaining = N - len(axial)
        for ax in ring2[:remaining]:
            axial.append(ax)

        for q, r in axial:
            x = math.sqrt(3) * (q + r / 2.0)
            y = 1.5 * r
            params.extend([x, y, 0.0])

        params.extend([0.0, 0.0, 0.0])
        S_est = compute_min_S(np.array(params[:3 * N]), params[3 * N], params[3 * N + 1], params[3 * N + 2])
        params.append(S_est)
        return np.array(params)

    best_S = float('inf')
    best_params = None

    n_restarts = 4
    for restart in range(n_restarts):
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            break

        if restart == 0:
            params0 = initial_config()
        else:
            params0 = initial_config()
            rng = np.random.RandomState(restart * 98765)
            noise = rng.randn(len(params0)) * 0.15
            noise[-1] = 0
            params0 = params0 + noise
            params0[-1] = max(params0[-1], 1.0)

        try:
            res = minimize(
                full_objective,
                params0,
                method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-12}
            )
            if res.fun < 1e-4:
                S_val = res.x[-1]
                if S_val < best_S:
                    best_S = S_val
                    best_params = res.x
        except Exception:
            pass

    # Fallback
    if best_params is None:
        params0 = initial_config()
        best_params = params0
        best_S = params0[-1]

    # Final: recompute exact S
    inner_final = best_params[:3 * N]
    cx = best_params[3 * N]
    cy = best_params[3 * N + 1]
    outer_angle = best_params[3 * N + 2]

    # Optimize outer params for fixed inner
    def outer_obj(p):
        return compute_min_S(inner_final, p[0], p[1], p[2])

    try:
        res2 = minimize(outer_obj, [cx, cy, outer_angle], method='L-BFGS-B',
                       options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-12})
        if res2.fun < best_S:
            best_S = res2.fun
            cx, cy, outer_angle = res2.x
    except Exception:
        pass

    inner = []
    for i in range(N):
        inner.append([float(best_params[3 * i]), float(best_params[3 * i + 1]), float(best_params[3 * i + 2])])

    return inner, [float(cx), float(cy)], float(best_S), float(outer_angle)
