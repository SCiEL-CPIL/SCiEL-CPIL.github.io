import math
import time
import random
import numpy as np
from scipy.optimize import minimize

N = 12
SQRT3 = math.sqrt(3)

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 145.0

    # Precompute unit hexagon vertices (pointy-top, angle 0)
    unit_verts_0 = np.array([[math.cos(math.radians(a)), math.sin(math.radians(a))] for a in range(0, 360, 60)])

    def get_hex_verts(cx, cy, angle_deg):
        angle_rad = math.radians(angle_deg)
        cos_a = math.cos(angle_rad)
        sin_a = math.sin(angle_rad)
        rot = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
        verts = unit_verts_0 @ rot.T
        verts += np.array([cx, cy])
        return verts

    def compute_outer_side(centers, angles, ocx, ocy, otheta):
        """Vectorized computation of outer side length."""
        otheta_rad = math.radians(otheta)
        n = len(centers)
        cx_arr = np.array([c[0] for c in centers])
        cy_arr = np.array([c[1] for c in centers])
        ang_rad = np.radians(np.array(angles))

        vert_angles = ang_rad[:, None] + np.arange(6)[None, :] * math.pi / 3
        vx = cx_arr[:, None] + np.cos(vert_angles)
        vy = cy_arr[:, None] + np.sin(vert_angles)

        vx_flat = vx.flatten()
        vy_flat = vy.flatten()

        normal_angles = otheta_rad + math.pi / 6 + np.arange(6) * math.pi / 3
        nx_arr = np.cos(normal_angles)
        ny_arr = np.sin(normal_angles)

        projs = (vx_flat - ocx)[:, None] * nx_arr[None, :] + (vy_flat - ocy)[:, None] * ny_arr[None, :]
        max_proj = np.max(projs)
        return 2.0 * max_proj / SQRT3

    def sat_gap(poly1, poly2):
        """Exact Minkowski-difference SAT gap. >0 separated, 0 touching, <0 overlap."""
        min_gap = float('inf')
        for i in range(6):
            edge1 = poly1[(i + 1) % 6] - poly1[i]
            nrm = np.array([-edge1[1], edge1[0]])
            norm = np.linalg.norm(nrm)
            if norm < 1e-9:
                continue
            axis = nrm / norm
            proj1 = poly1 @ axis
            proj2 = poly2 @ axis
            gap = max(np.min(proj1) - np.max(proj2), np.min(proj2) - np.max(proj1))
            if gap < 0:
                if gap < min_gap:
                    min_gap = gap
            else:
                return gap
        for i in range(6):
            edge2 = poly2[(i + 1) % 6] - poly2[i]
            nrm = np.array([-edge2[1], edge2[0]])
            norm = np.linalg.norm(nrm)
            if norm < 1e-9:
                continue
            axis = nrm / norm
            proj1 = poly1 @ axis
            proj2 = poly2 @ axis
            gap = max(np.min(proj1) - np.max(proj2), np.min(proj2) - np.max(proj1))
            if gap < 0:
                if gap < min_gap:
                    min_gap = gap
            else:
                return gap
        return min_gap

    def is_valid(centers, angles, tol=1e-6):
        polys = [get_hex_verts(centers[i][0], centers[i][1], angles[i]) for i in range(N)]
        for i in range(N):
            for j in range(i + 1, N):
                if sat_gap(polys[i], polys[j]) < -tol:
                    return False
        return True

    def total_overlap_penalty(centers, angles, weight=50.0):
        polys = [get_hex_verts(centers[i][0], centers[i][1], angles[i]) for i in range(N)]
        pen = 0.0
        for i in range(N):
            for j in range(i + 1, N):
                g = sat_gap(polys[i], polys[j])
                if g < 0:
                    pen += weight * g * g
        return pen

    def objective_vec(params, outer_theta=0.0):
        if time.time() > deadline - 8:
            return 1e9
        centers = [[params[i * 2], params[i * 2 + 1]] for i in range(N)]
        angles = [params[2 * N + i] for i in range(N)]
        side = compute_outer_side(centers, angles, 0.0, 0.0, outer_theta)
        pen = total_overlap_penalty(centers, angles)
        return side + pen

    def make_constraint(i, j, outer_theta):
        def constraint(params):
            if time.time() > deadline - 5:
                return 1.0
            ci = [params[i * 2], params[i * 2 + 1]]
            cj = [params[j * 2], params[j * 2 + 1]]
            ai = params[2 * N + i]
            aj = params[2 * N + j]
            v1 = get_hex_verts(ci[0], ci[1], ai)
            v2 = get_hex_verts(cj[0], cj[1], aj)
            return sat_gap(v1, v2)  # >= 0 means non-overlapping
        return constraint

    def polish_outer_hexagon(centers, angles, ocx, ocy, otheta, max_iters=300):
        """High-precision polish of outer hexagon position and rotation."""
        def outer_obj(p):
            return compute_outer_side(centers, angles, p[0], p[1], p[2])

        cur = compute_outer_side(centers, angles, ocx, ocy, otheta)
        x0 = np.array([ocx, ocy, otheta])
        try:
            res = minimize(outer_obj, x0, method='Nelder-Mead',
                           options={'maxiter': max_iters, 'xatol': 1e-9, 'fatol': 1e-11})
            if res.fun < cur:
                ocx, ocy, otheta = res.x[0], res.x[1], res.x[2]
                cur = res.fun
        except Exception:
            pass

        # Coarse-to-fine grid refinement (homotopy-style continuation)
        best_ocx, best_ocy, best_otheta, best_side = ocx, ocy, otheta, cur
        for scale in [0.05, 0.005, 0.0005, 0.00005]:
            improved = False
            for da in np.linspace(-scale, scale, 7):
                for dx in np.linspace(-scale, scale, 7):
                    for dy in np.linspace(-scale, scale, 7):
                        s = compute_outer_side(centers, angles, best_ocx + dx, best_ocy + dy, best_otheta + da)
                        if s < best_side - 1e-13:
                            best_side = s
                            best_ocx += dx
                            best_ocy += dy
                            best_otheta += da
                            improved = True
            if not improved:
                break
        return best_ocx, best_ocy, best_otheta, best_side

    def solve_slsqp(centers, angles, outer_theta):
        """Hard-constraint SLSQP on positions and angles with exact SAT gap constraints."""
        x0 = []
        for i in range(N):
            x0.append(centers[i][0])
            x0.append(centers[i][1])
        for i in range(N):
            x0.append(angles[i])
        x0 = np.array(x0)

        def obj(p):
            return objective_vec(p, outer_theta)

        cons = []
        for i in range(N):
            for j in range(i + 1, N):
                cons.append({'type': 'ineq', 'fun': make_constraint(i, j, outer_theta)})

        try:
            res = minimize(obj, x0, method='SLSQP', constraints=cons,
                           options={'maxiter': 200, 'ftol': 1e-9})
            if res.success or (not np.isnan(res.fun) and res.fun < objective_vec(x0, outer_theta)):
                new_centers = [[res.x[i * 2], res.x[i * 2 + 1]] for i in range(N)]
                new_angles = [res.x[2 * N + i] for i in range(N)]
                return new_centers, new_angles
        except Exception:
            pass
        return centers, angles

    def optimize_full(centers, angles, outer_theta, use_slsqp=True):
        """Nelder-Mead with soft penalty, then optional SLSQP hard-constraint refinement."""
        x0 = []
        for i in range(N):
            x0.append(centers[i][0])
            x0.append(centers[i][1])
        for i in range(N):
            x0.append(angles[i])
        x0 = np.array(x0)

        try:
            res = minimize(lambda p: objective_vec(p, outer_theta), x0, method='Nelder-Mead',
                           options={'maxiter': 1500, 'xatol': 1e-5, 'fatol': 1e-7})
            if not np.isnan(res.fun):
                centers = [[res.x[i * 2], res.x[i * 2 + 1]] for i in range(N)]
                angles = [res.x[2 * N + i] for i in range(N)]
        except Exception:
            pass

        if use_slsqp and time.time() < deadline - 25:
            centers, angles = solve_slsqp(centers, angles, outer_theta)

        return centers, angles

    # ---- Phase 1: generate initial configurations from honeycomb lattice ----
    def lattice_config(scale, rot_deg, n_extra=5):
        d = SQRT3 * scale
        u = np.array([d, 0.0])
        v = np.array([d / 2.0, d * SQRT3 / 2.0])
        basis = [u, v, v - u, -u, -v, u - v]
        pts = [np.array([0.0, 0.0])]
        ring1 = list(basis)
        for vec in ring1:
            pts.append(vec.copy())
        ring2 = []
        for i in range(6):
            p1 = ring1[i]
            for b in basis:
                p2 = p1 + b
                if np.allclose(p2, np.array([0.0, 0.0])):
                    continue
                if any(np.allclose(p2, p1j) for p1j in ring1):
                    continue
                if any(np.allclose(p2, pc) for pc in ring2):
                    continue
                ring2.append(p2)
        ring2.sort(key=lambda p: np.linalg.norm(p))
        for i in range(n_extra):
            pts.append(ring2[i].copy())
        pts = np.array(pts[:N])
        rot_rad = math.radians(rot_deg)
        rot_mat = np.array([[math.cos(rot_rad), -math.sin(rot_rad)],
                            [math.sin(rot_rad), math.cos(rot_rad)]])
        rotated = pts @ rot_mat.T
        rotated -= np.mean(rotated, axis=0)
        centers = [[float(rotated[i][0]), float(rotated[i][1])] for i in range(N)]
        angles = [float(rot_deg)] * N
        return centers, angles

    # ---- Phase 2: homotopy continuation over outer rotation ----
    # Start from a large, clearly-valid container and shrink, tracking the branch.
    best_centers = None
    best_angles = None
    best_side = float('inf')
    best_ocx = best_ocy = best_otheta = 0.0

    coarse_angles = list(range(0, 360, 15))
    for oa in coarse_angles:
        if time.time() > deadline - 60:
            break
        # Try a few lattice scales as the tracked branch
        for scale, rot in [(1.0, 0), (1.0, 30), (1.05, 0)]:
            centers, angles = lattice_config(scale, rot)
            if not is_valid(centers, angles):
                continue
            centers, angles = optimize_full(centers, angles, oa, use_slsqp=False)
            ocx, ocy, otheta, side = polish_outer_hexagon(centers, angles, 0.0, 0.0, oa, max_iters=200)
            if is_valid(centers, angles) and side < best_side:
                best_side = side
                best_centers = [c[:] for c in centers]
                best_angles = angles[:]
                best_ocx, best_ocy, best_otheta = ocx, ocy, otheta

    # ---- Phase 3: refine the best branch with hard-constraint SLSQP + fine outer search ----
    if best_centers is None:
        # Guaranteed fallback
        centers, angles = lattice_config(1.3, 0)
        best_centers, best_angles = centers, angles
        best_side = compute_outer_side(centers, angles, 0.0, 0.0, 0.0)
        best_ocx = best_ocy = best_otheta = 0.0

    if time.time() < deadline - 30:
        centers, angles = solve_slsqp(best_centers, best_angles, best_otheta)
        if is_valid(centers, angles):
            best_centers, best_angles = centers, angles

    # Fine outer-rotation continuation around the best angle
    for da in np.linspace(-15, 15, 31):
        if time.time() > deadline - 12:
            break
        oa = best_otheta + da
        ocx, ocy, otheta, side = polish_outer_hexagon(best_centers, best_angles, 0.0, 0.0, oa, max_iters=300)
        if is_valid(best_centers, best_angles) and side < best_side - 1e-9:
            best_side = side
            best_ocx, best_ocy, best_otheta = ocx, ocy, otheta

    # Final validity check and fallback
    if not is_valid(best_centers, best_angles):
        centers, angles = lattice_config(1.3, 0)
        best_centers, best_angles = centers, angles
        best_ocx, best_ocy, best_otheta, best_side = polish_outer_hexagon(centers, angles, 0.0, 0.0, 0.0, max_iters=300)

    inner = [[c[0], c[1], a] for c, a in zip(best_centers, best_angles)]
    outer_center = (best_ocx, best_ocy)
    outer_side = best_side
    outer_angle = best_otheta

    return inner, outer_center, outer_side, outer_angle
# EVOLVE-BLOCK-END
