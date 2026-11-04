import math
import time
import random
import numpy as np
from scipy.optimize import minimize

N = 12
SQRT3 = math.sqrt(3.0)

# EVOLVE-BLOCK-START
def construct():
    t0 = time.time()
    DEADLINE = t0 + 140.0
    random.seed(42)
    np.random.seed(42)

    def get_hex_vertices(cx, cy, side, angle_deg):
        a = math.radians(angle_deg)
        th = a + np.arange(6) * (math.pi / 3)
        return np.column_stack((cx + side * np.cos(th), cy + side * np.sin(th)))

    def sat_disjoint(v1, v2, eps=1e-9):
        e1 = np.roll(v1, -1, axis=0) - v1
        n1 = np.column_stack((-e1[:, 1], e1[:, 0]))
        p1 = v1 @ n1.T; p2 = v2 @ n1.T
        if np.any(np.max(p1, axis=0) < np.min(p2, axis=0) + eps) or np.any(np.max(p2, axis=0) < np.min(p1, axis=0) + eps):
            return True
        e2 = np.roll(v2, -1, axis=0) - v2
        n2 = np.column_stack((-e2[:, 1], e2[:, 0]))
        p1 = v1 @ n2.T; p2 = v2 @ n2.T
        if np.any(np.max(p1, axis=0) < np.min(p2, axis=0) + eps) or np.any(np.max(p2, axis=0) < np.min(p1, axis=0) + eps):
            return True
        return False

    def minkowski_diff_hex(a1, a2):
        """Vertices of the Minkowski difference of two unit hexagons at angles a1, a2 (degrees)."""
        v1 = get_hex_vertices(0, 0, 1.0, a1)
        v2 = get_hex_vertices(0, 0, 1.0, a2)
        # Minkowski difference A - B = A + (-B)
        neg_b = -v2
        # All 36 vertex sums
        pts = np.array([p + q for p in v1 for q in neg_b])
        # Convex hull
        from scipy.spatial import ConvexHull
        try:
            hull = ConvexHull(pts)
            return pts[hull.vertices]
        except Exception:
            return pts

    def min_gap_minkowski(c1, a1, c2, a2, cache={}):
        """Minimum gap between two unit hexagons. Negative = overlap depth."""
        key = (round(a1, 3), round(a2, 3))
        if key not in cache:
            cache[key] = minkowski_diff_hex(a1, a2)
        md = cache[key]
        d = np.array([c1[0] - c2[0], c1[1] - c2[1]])
        # If d is inside md, they overlap. Compute signed distance to md boundary.
        from scipy.spatial import ConvexHull
        try:
            hull = ConvexHull(md)
            # Point-in-polygon test via winding or use distance
            # Use scipy's distance: if point inside, distance is 0; we need signed
            # Check if d is inside the convex hull
            from scipy.spatial import Delaunay
            tri = Delaunay(md)
            inside = tri.find_simplex(d) >= 0
            if inside:
                # Compute penetration depth: max over edges of (dot(n, d) - dot(n, edge_point))
                # Actually for overlap, gap = min over all separating axes of (max_proj_B - min_proj_A)
                # Use SAT-style: the gap along axis n is min_proj_A - max_proj_B (negative if overlap)
                # We want the maximum (least negative) such value = -penetration
                # For Minkowski difference, the gap = - (max over boundary of ... )
                # Simpler: the signed distance from d to md boundary, negative if inside
                # Approximate by projecting onto each edge normal
                min_gap = 0.0  # 0 means touching/inside
                for i in range(len(md)):
                    e = md[(i+1) % len(md)] - md[i]
                    n = np.array([-e[1], e[0]])
                    nn = np.linalg.norm(n)
                    if nn < 1e-12:
                        continue
                    n = n / nn
                    # The half-space: points p with dot(n, p) <= dot(n, md[i]) are inside
                    # d is inside if dot(n, d) <= dot(n, md[i]) for all edges
                    # The "gap" along this axis when d is inside: dot(n, md[i]) - dot(n, d) > 0
                    # The minimum positive such value over all edges = penetration depth
                    val = dot(n, md[i]) - dot(n, d)
                    if val < min_gap:
                        min_gap = val
                # min_gap is the penetration depth (positive). Return negative.
                return -max(0.0, min_gap) if min_gap > 0 else 0.0
            else:
                # Outside: minimum distance from d to md
                from scipy.spatial import distance
                # Distance to convex polygon = min over edges of distance to edge segment
                min_dist = float('inf')
                for i in range(len(md)):
                    a_pt = md[i]
                    b_pt = md[(i+1) % len(md)]
                    ab = b_pt - a_pt
                    t = np.dot(d - a_pt, ab) / (np.dot(ab, ab) + 1e-15)
                    t = max(0.0, min(1.0, t))
                    closest = a_pt + t * ab
                    dist = np.linalg.norm(d - closest)
                    if dist < min_dist:
                        min_dist = dist
                return min_dist
        except Exception:
            return 0.0

    def dot(a, b):
        return a[0]*b[0] + a[1]*b[1]

    def compute_outer_side(centers, angles, ocx, ocy, otheta):
        """Compute the side length of the smallest regular hexagon (center ocx,ocy, rotation otheta)
        containing all inner hexagons. Returns side length."""
        oth = math.radians(otheta)
        # The 6 outward normals of a regular hexagon with rotation otheta
        # A regular hexagon with side s, center (0,0), rotation 0 has vertices at angles 0,60,...,300
        # Its supporting lines (edges) have normals at angles 30, 90, 150, 210, 270, 330 (for rotation 0)
        # For rotation otheta, normals are at otheta + 30 + k*60
        normal_angles = oth + math.pi/6 + np.arange(6) * (math.pi/3)
        nx = np.cos(normal_angles)
        ny = np.sin(normal_angles)
        # For each normal direction, compute the maximum projection of all inner vertices
        max_proj = 0.0
        for i in range(len(centers)):
            cx, cy = centers[i]
            ar = math.radians(angles[i])
            th = ar + np.arange(6) * (math.pi/3)
            vx = cx + np.cos(th)
            vy = cy + np.sin(th)
            dx = vx - ocx
            dy = vy - ocy
            projs = dx[:, np.newaxis] * nx[np.newaxis, :] + dy[:, np.newaxis] * ny[np.newaxis, :]
            mp = np.max(projs)
            if mp > max_proj:
                max_proj = mp
        # For a regular hexagon with side s, the distance from center to each edge (apothem) is s*sqrt(3)/2
        # So max_proj = apothem = s * sqrt(3)/2
        # s = 2 * max_proj / sqrt(3)
        return 2.0 * max_proj / SQRT3

    def calculate_cost(centers, angles, ocx, ocy, otheta, penalty_weight=100.0):
        side = compute_outer_side(centers, angles, ocx, ocy, otheta)
        penalty = 0.0
        verts = [get_hex_vertices(centers[i][0], centers[i][1], 1.0, angles[i]) for i in range(12)]
        for i in range(12):
            for j in range(i+1, 12):
                if not sat_disjoint(verts[i], verts[j]):
                    penalty += penalty_weight
        return side + penalty

    def generate_lattice_config():
        """3x4 hexagonal lattice."""
        d = SQRT3 + 0.01
        u = np.array([d, 0.0])
        v = np.array([d/2.0, d*SQRT3/2.0])
        pts = []
        for i in range(3):
            for j in range(4):
                pts.append(i*u + j*v)
        pts = np.array(pts)
        c = pts.mean(axis=0)
        pts = pts - c
        centers = [np.array(pts[k]) for k in range(12)]
        angles = [30.0]*12
        return centers, angles, 0.0, 0.0, 0.0

    def generate_circular_config():
        centers = [np.array([0.0, 0.0])]
        r1 = SQRT3 + 0.02
        for i in range(6):
            ang = i * math.pi/3
            centers.append(np.array([r1*math.cos(ang), r1*math.sin(ang)]))
        r2 = 2*SQRT3 + 0.02
        for i in range(5):
            ang = i*2*math.pi/5 + math.pi/5
            centers.append(np.array([r2*math.cos(ang), r2*math.sin(ang)]))
        angles = [30.0]*12
        return centers, angles, 0.0, 0.0, 0.0

    def generate_row_config():
        centers = []
        d = SQRT3 + 0.01
        for i in range(6):
            centers.append(np.array([i*d, 0.0]))
        for i in range(6):
            centers.append(np.array([i*d + d/2, d*SQRT3/2]))
        angles = [30.0]*12
        return centers, angles, 0.0, 0.0, 0.0

    def optimize_lattice(shear, scale, rot_deg, otheta):
        """Evaluate a parameterized 3x4 lattice."""
        s = scale
        h = SQRT3/2.0
        e1 = np.array([s, 0.0])
        e2 = np.array([s*(0.5 + shear), s*h])
        pts = []
        for i in range(3):
            for j in range(4):
                pts.append(i*e1 + j*e2)
        pts = np.array(pts)
        c = pts.mean(axis=0)
        pts = pts - c
        theta = math.radians(rot_deg)
        R = np.array([[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]])
        pts = pts @ R.T
        centers = [np.array(pts[k]) for k in range(12)]
        angles = [30.0 + rot_deg]*12
        verts = [get_hex_vertices(c[0], c[1], 1.0, a) for c, a in zip(centers, angles)]
        for i in range(12):
            for j in range(i+1, 12):
                if not sat_disjoint(verts[i], verts[j], eps=1e-6):
                    return 1e6
        return compute_outer_side(centers, angles, 0.0, 0.0, otheta)

    def optimize_lattice_params():
        best_cost = float('inf')
        best_params = (0.0, SQRT3 + 0.01, 0.0, 0.0)
        for shear in np.linspace(-0.3, 0.3, 13):
            for scale in np.linspace(SQRT3 - 0.1, SQRT3 + 0.3, 17):
                for rot_deg in [0.0, 15.0, 30.0, 45.0, 60.0]:
                    for otheta in [0.0, 15.0, 30.0, 45.0, 60.0]:
                        cost = optimize_lattice(shear, scale, rot_deg, otheta)
                        if cost < best_cost:
                            best_cost = cost
                            best_params = (shear, scale, rot_deg, otheta)
                        if time.time() > DEADLINE - 20:
                            break
                    if time.time() > DEADLINE - 20:
                        break
                if time.time() > DEADLINE - 20:
                    break
            if time.time() > DEADLINE - 20:
                break
        # Nelder-Mead refinement
        x0 = np.array(best_params)
        def obj(x):
            if time.time() > DEADLINE - 10:
                return 1e10
            return optimize_lattice(x[0], x[1], x[2], x[3])
        try:
            res = minimize(obj, x0, method='Nelder-Mead', options={'maxiter': 200, 'xatol': 1e-6, 'fatol': 1e-8})
            if res.fun < best_cost:
                best_params = tuple(res.x)
                best_cost = res.fun
        except Exception:
            pass
        # Fine polish
        for scale in [0.01, 0.002, 0.0005]:
            improved = True
            while improved and time.time() < DEADLINE - 5:
                improved = False
                for ds in [-scale, 0, scale]:
                    for dd in [-scale, 0, scale]:
                        for dr in [-scale*10, 0, scale*10]:
                            for do in [-scale*10, 0, scale*10]:
                                nc = optimize_lattice(best_params[0]+ds, best_params[1]+dd, best_params[2]+dr, best_params[3]+do)
                                if nc < best_cost - 1e-12:
                                    best_cost = nc
                                    best_params = (best_params[0]+ds, best_params[1]+dd, best_params[2]+dr, best_params[3]+do)
                                    improved = True
        s = best_params[1]
        h = SQRT3/2.0
        e1 = np.array([s, 0.0])
        e2 = np.array([s*(0.5 + best_params[0]), s*h])
        pts = []
        for i in range(3):
            for j in range(4):
                pts.append(i*e1 + j*e2)
        pts = np.array(pts)
        c = pts.mean(axis=0)
        pts = pts - c
        theta = math.radians(best_params[2])
        R = np.array([[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]])
        pts = pts @ R.T
        centers = [np.array(pts[k]) for k in range(12)]
        angles = [30.0 + best_params[2]]*12
        return centers, angles, 0.0, 0.0, best_params[3], best_cost

    def optimize_sa(centers_init, angles_init, ocx_init, ocy_init, otheta_init, time_budget):
        T = 0.5
        T_min = 0.001
        alpha = 0.98
        steps_per_temp = 20
        cur_c = [c.copy() for c in centers_init]
        cur_a = list(angles_init)
        cur_ocx, cur_ocy, cur_otheta = ocx_init, ocy_init, otheta_init
        cur_cost = calculate_cost(cur_c, cur_a, cur_ocx, cur_ocy, cur_otheta)
        best_c = [c.copy() for c in cur_c]
        best_a = list(cur_a)
        best_ocx, best_ocy, best_otheta = cur_ocx, cur_ocy, cur_otheta
        best_cost = cur_cost
        while T > T_min:
            for _ in range(steps_per_temp):
                if time.time() > DEADLINE - time_budget:
                    return best_c, best_a, best_ocx, best_ocy, best_otheta, best_cost
                new_c = [c.copy() for c in cur_c]
                new_a = list(cur_a)
                new_ocx, new_ocy, new_otheta = cur_ocx, cur_ocy, cur_otheta
                i = random.randint(0, 11)
                step = T * 0.3
                new_c[i][0] += random.gauss(0, step)
                new_c[i][1] += random.gauss(0, step)
                new_a[i] += random.gauss(0, step*5)
                if random.random() < 0.05:
                    new_ocx += random.gauss(0, T*0.1)
                    new_ocy += random.gauss(0, T*0.1)
                    new_otheta += random.gauss(0, T*0.1)
                new_cost = calculate_cost(new_c, new_a, new_ocx, new_ocy, new_otheta)
                delta = new_cost - cur_cost
                if delta < 0 or random.random() < math.exp(-delta/T):
                    cur_c, cur_a = new_c, new_a
                    cur_ocx, cur_ocy, cur_otheta = new_ocx, new_ocy, new_otheta
                    cur_cost = new_cost
                    if cur_cost < best_cost:
                        best_cost = cur_cost
                        best_c = [c.copy() for c in cur_c]
                        best_a = list(cur_a)
                        best_ocx, best_ocy, best_otheta = cur_ocx, cur_ocy, cur_otheta
            T *= alpha
        return best_c, best_a, best_ocx, best_ocy, best_otheta, best_cost

    def optimize_slsqp(centers_init, angles_init, ocx_init, ocy_init, otheta_init, max_iters=80):
        n = len(centers_init)
        x0 = []
        for i in range(n):
            x0.extend([centers_init[i][0], centers_init[i][1], angles_init[i]])
        x0.extend([ocx_init, ocy_init, otheta_init])
        def objective(x):
            if time.time() > DEADLINE - 10:
                return 1e10
            c = [[x[3*i], x[3*i+1]] for i in range(n)]
            a = [x[3*i+2] for i in range(n)]
            ocx, ocy, otheta = x[3*n], x[3*n+1], x[3*n+2]
            return compute_outer_side(c, a, ocx, ocy, otheta)
        constraints = []
        min_dist_sq = (SQRT3 - 0.005)**2
        for i in range(n):
            for j in range(i+1, n):
                def dist_con(x, i=i, j=j):
                    xi, yi = x[3*i], x[3*i+1]
                    xj, yj = x[3*j], x[3*j+1]
                    return (xi-xj)**2 + (yi-yj)**2 - min_dist_sq
                constraints.append({'type': 'ineq', 'fun': dist_con})
        bounds = []
        for i in range(n):
            bounds.extend([(-6, 6), (-6, 6), (0, 60)])
        bounds.extend([(-3, 3), (-3, 3), (0, 60)])
        try:
            result = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constraints,
                             options={'maxiter': max_iters, 'ftol': 1e-12})
            if result.success:
                c = [[result.x[3*i], result.x[3*i+1]] for i in range(n)]
                a = [result.x[3*i+2] for i in range(n)]
                ocx, ocy, otheta = result.x[3*n], result.x[3*n+1], result.x[3*n+2]
                side = compute_outer_side(c, a, ocx, ocy, otheta)
                verts = [get_hex_vertices(c[i][0], c[i][1], 1.0, a[i]) for i in range(n)]
                valid = True
                for i in range(n):
                    for j in range(i+1, n):
                        if not sat_disjoint(verts[i], verts[j]):
                            valid = False
                            break
                    if not valid:
                        break
                if valid:
                    return c, a, ocx, ocy, otheta, side
        except Exception:
            pass
        return None

    def homotopy_continuation(centers_init, angles_init, ocx_init, ocy_init, otheta_init, t_start, t_end, n_steps=50):
        """
        Homotopy continuation: track the solution from a large container (t_start) to minimal (t_end).
        t is a scaling factor on the outer hexagon side length.
        At each step, solve for the configuration that fits in a hexagon of side t.
        """
        cur_c = [c.copy() for c in centers_init]
        cur_a = list(angles_init)
        cur_ocx, cur_ocy, cur_otheta = ocx_init, ocy_init, otheta_init
        for step in range(n_steps):
            if time.time() > DEADLINE - 10:
                break
            t = t_start + (t_end - t_start) * step / n_steps
            # Define the objective: minimize side length, but penalize if side > t
            def obj(x):
                c = [[x[3*i], x[3*i+1]] for i in range(N)]
                a = [x[3*i+2] for i in range(N)]
                ocx, ocy, otheta = x[3*N], x[3*N+1], x[3*N+2]
                side = compute_outer_side(c, a, ocx, ocy, otheta)
                # Penalty if side exceeds t
                penalty = max(0, side - t) * 1000
                return side + penalty
            x0 = []
            for i in range(N):
                x0.extend([cur_c[i][0], cur_c[i][1], cur_a[i]])
            x0.extend([cur_ocx, cur_ocy, cur_otheta])
            constraints = []
            min_dist_sq = (SQRT3 - 0.005)**2
            for i in range(N):
                for j in range(i+1, N):
                    def dist_con(x, i=i, j=j):
                        xi, yi = x[3*i], x[3*i+1]
                        xj, yj = x[3*j], x[3*j+1]
                        return (xi-xj)**2 + (yi-yj)**2 - min_dist_sq
                    constraints.append({'type': 'ineq', 'fun': dist_con})
            bounds = []
            for i in range(N):
                bounds.extend([(-6, 6), (-6, 6), (0, 60)])
            bounds.extend([(-3, 3), (-3, 3), (0, 60)])
            try:
                result = minimize(obj, x0, method='SLSQP', bounds=bounds, constraints=constraints,
                                 options={'maxiter': 40, 'ftol': 1e-10})
                if result.success:
                    cur_c = [[result.x[3*i], result.x[3*i+1]] for i in range(N)]
                    cur_a = [result.x[3*i+2] for i in range(N)]
                    cur_ocx, cur_ocy, cur_otheta = result.x[3*N], result.x[3*N+1], result.x[3*N+2]
            except Exception:
                pass
        side = compute_outer_side(cur_c, cur_a, cur_ocx, cur_ocy, cur_otheta)
        verts = [get_hex_vertices(cur_c[i][0], cur_c[i][1], 1.0, cur_a[i]) for i in range(N)]
        valid = True
        for i in range(N):
            for j in range(i+1, N):
                if not sat_disjoint(verts[i], verts[j]):
                    valid = False
                    break
            if not valid:
                break
        if valid:
            return cur_c, cur_a, cur_ocx, cur_ocy, cur_otheta, side
        return None

    # Main execution
    configs = [
        generate_lattice_config(),
        generate_circular_config(),
        generate_row_config()
    ]
    best_overall_cost = float('inf')
    best_centers = None
    best_angles = None
    best_ocx = 0.0
    best_ocy = 0.0
    best_otheta = 0.0

    # 1. Lattice parameter optimization (fast, structured)
    if time.time() < DEADLINE - 30:
        try:
            lat_c, lat_a, lat_ocx, lat_ocy, lat_otheta, lat_cost = optimize_lattice_params()
            if lat_cost < best_overall_cost:
                best_overall_cost = lat_cost
                best_centers = lat_c
                best_angles = lat_a
                best_ocx = lat_ocx
                best_ocy = lat_ocy
                best_otheta = lat_otheta
        except Exception:
            pass

    # 2. For each initial config, run SA + SLSQP
    for centers_init, angles_init, ocx_init, ocy_init, otheta_init in configs:
        if time.time() > DEADLINE - 30:
            break
        sa_c, sa_a, sa_ocx, sa_ocy, sa_otheta, sa_cost = optimize_sa(
            centers_init, angles_init, ocx_init, ocy_init, otheta_init, 20
        )
        slsqp_result = optimize_slsqp(sa_c, sa_a, sa_ocx, sa_ocy, sa_otheta, max_iters=60)
        if slsqp_result is not None:
            sl_c, sl_a, sl_ocx, sl_ocy, sl_otheta, sl_cost = slsqp_result
            if sl_cost < best_overall_cost:
                best_overall_cost = sl_cost
                best_centers = sl_c
                best_angles = sl_a
                best_ocx = sl_ocx
                best_ocy = sl_ocy
                best_otheta = sl_otheta
        else:
            if sa_cost < best_overall_cost:
                best_overall_cost = sa_cost
                best_centers = sa_c
                best_angles = sa_a
                best_ocx = sa_ocx
                best_ocy = sa_ocy
                best_otheta = sa_otheta

    # 3. Homotopy continuation from a large container to minimal
    if time.time() < DEADLINE - 20 and best_centers is not None:
        try:
            # Start from a large side length and shrink
            large_side = 10.0
            target_side = best_overall_cost * 0.99
            homotopy_result = homotopy_continuation(
                best_centers, best_angles, best_ocx, best_ocy, best_otheta,
                large_side, target_side, n_steps=30
            )
            if homotopy_result is not None:
                hc, ha, hocx, hocy, hotheta, hcost = homotopy_result
                if hcost < best_overall_cost:
                    best_overall_cost = hcost
                    best_centers = hc
                    best_angles = ha
                    best_ocx = hocx
                    best_ocy = hocy
                    best_otheta = hotheta
        except Exception:
            pass

    if best_centers is None:
        best_centers, best_angles, best_ocx, best_ocy, best_otheta = generate_lattice_config()
        best_overall_cost = compute_outer_side(best_centers, best_angles, best_ocx, best_ocy, best_otheta)

    best_side = best_overall_cost
    final_ocx, final_ocy, final_otheta = best_ocx, best_ocy, best_otheta

    # 4. High-precision multi-scale polish of outer hexagon parameters
    scales = [0.5, 0.1, 0.02, 0.005, 0.001]
    for scale in scales:
        if time.time() > DEADLINE - 5:
            break
        improved = True
        while improved and time.time() < DEADLINE - 5:
            improved = False
            for dx in [-scale, 0, scale]:
                for dy in [-scale, 0, scale]:
                    for da_deg in [-scale*20, 0, scale*20]:
                        new_otheta = (final_otheta + da_deg) % 60
                        s = compute_outer_side(best_centers, best_angles, final_ocx + dx, final_ocy + dy, new_otheta)
                        if s < best_side - 1e-12:
                            best_side = s
                            final_ocx += dx
                            final_ocy += dy
                            final_otheta = new_otheta
                            improved = True

    # 5. Final very fine polish
    if time.time() < DEADLINE - 2:
        for _ in range(3):
            improved = True
            while improved and time.time() < DEADLINE - 2:
                improved = False
                for dx in [-0.001, 0, 0.001]:
                    for dy in [-0.001, 0, 0.001]:
                        for da_deg in [-0.01, 0, 0.01]:
                            new_otheta = (final_otheta + da_deg) % 60
                            s = compute_outer_side(best_centers, best_angles, final_ocx + dx, final_ocy + dy, new_otheta)
                            if s < best_side - 1e-14:
                                best_side = s
                                final_ocx += dx
                                final_ocy += dy
                                final_otheta = new_otheta
                                improved = True

    inner = [[float(c[0]), float(c[1]), float(a)] for c, a in zip(best_centers, best_angles)]
    outer_center = (float(final_ocx), float(final_ocy))
    outer_side = float(best_side)
    outer_angle = float(final_otheta)

    return inner, outer_center, outer_side, outer_angle
# EVOLVE-BLOCK-END
