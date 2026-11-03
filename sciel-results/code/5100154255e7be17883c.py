import math
import numpy as np
import time
from scipy.optimize import minimize, minimize_scalar

N = 11
APOTHEM = math.sqrt(3) / 2.0
SQ3 = math.sqrt(3.0)
START_TIME = time.time()
TIME_LIMIT = 150.0
SAFETY_MARGIN = 8.0
EPS = 1e-9


def construct():
    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
        return [[0.0, 0.0, 0.0]], [0.0, 0.0], 1.0, 0.0

    def time_ok():
        return time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN

    # ---------- geometry helpers ----------
    def axial_to_cartesian(axial_list):
        pts = []
        for q, r in axial_list:
            x = SQ3 * (q + r / 2.0)
            y = 1.5 * r
            pts.append((x, y))
        return np.array(pts)

    def hex_support(phi, rot):
        # distance from center to boundary of unit hexagon (circumradius 1)
        # with vertices at angles rot + k*pi/3, in direction phi
        phi_rel = phi - rot
        phi_m = phi_rel % (math.pi / 3)
        delta = phi_m - math.pi / 6
        return APOTHEM / math.cos(delta)

    def hex_pair_min_dist(p1, p2, a1, a2):
        d = np.array(p2) - np.array(p1)
        dist = float(np.linalg.norm(d))
        if dist < 1e-15:
            return 0.0
        theta_d = math.atan2(d[1], d[0])
        s1 = hex_support(theta_d, a1)
        s2 = hex_support(theta_d + math.pi, a2)
        return dist - s1 - s2

    def enclosing_hexagon(pts, beta):
        # Closed-form minimum enclosing regular hexagon of rotation beta.
        # Returns (S, cx, cy, D) or (inf, None, None, None).
        angles = beta + math.pi / 6.0 + np.arange(6) * math.pi / 3.0
        cos_a = np.cos(angles)
        sin_a = np.sin(angles)
        P = np.array(pts)
        proj = P @ np.column_stack([cos_a, sin_a]).T   # (n,6)
        D = 0.0
        best_c = None
        for k in range(6):
            m = 0.5 * (proj[:, k].max() + proj[:, k].min())
            d = 0.5 * (proj[:, k].max() - proj[:, k].min())
            if d > D:
                D = d
                best_c = k
        if best_c is None:
            return float('inf'), None, None, None
        # center: project onto the two adjacent edge normals of the dominating edge
        k1 = (best_c + 1) % 6
        k2 = (best_c + 5) % 6
        a1 = angles[k1]
        a2 = angles[k2]
        A = np.array([[cos_a[k1], sin_a[k1]], [cos_a[k2], sin_a[k2]]])
        b = np.array([proj[:, k1].max(), proj[:, k2].max()])
        try:
            c = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            c = np.array([P[:, 0].mean(), P[:, 1].mean()])
        cx, cy = float(c[0]), float(c[1])
        # verify (max abs projection)
        Pc = P - np.array([cx, cy])
        projc = Pc @ np.column_stack([cos_a, sin_a]).T
        Dv = float(np.max(np.abs(projc)))
        S = (Dv + APOTHEM) / APOTHEM
        return S, cx, cy, Dv

    def exact_S(pts, cx, cy, beta):
        angles = beta + math.pi / 6.0 + np.arange(6) * math.pi / 3.0
        cos_a = np.cos(angles)
        sin_a = np.sin(angles)
        Pc = np.array(pts) - np.array([cx, cy])
        proj = Pc @ np.column_stack([cos_a, sin_a]).T
        D = float(np.max(np.abs(proj)))
        return (D + APOTHEM) / APOTHEM

    def smooth_S(pts, cx, cy, beta, temperature=0.05):
        angles = beta + math.pi / 6.0 + np.arange(6) * math.pi / 3.0
        cos_a = np.cos(angles)
        sin_a = np.sin(angles)
        Pc = np.array(pts) - np.array([cx, cy])
        proj = Pc @ np.column_stack([cos_a, sin_a]).T
        allp = np.concatenate([proj.ravel(), (-proj).ravel()])
        mx = float(np.max(allp))
        if mx > 200:
            return float('inf')
        sh = np.clip((allp - mx) / temperature, -50, 50)
        D = temperature * (np.log(np.sum(np.exp(sh))) + mx / temperature)
        return (D + APOTHEM) / APOTHEM

    def hex_radius(pts):
        cx = float(pts[:, 0].mean())
        cy = float(pts[:, 1].mean())
        best = 0.0
        for k in range(6):
            ang = math.pi / 6.0 + k * math.pi / 3.0
            nx, ny = math.cos(ang), math.sin(ang)
            pr = (pts[:, 0] - cx) * nx + (pts[:, 1] - cy) * ny
            best = max(best, float(np.max(pr)), float(-np.min(pr)))
        return best

    # ---------- candidate configurations ----------
    candidates = [
        [(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(1,1),(-1,2),(-2,1),(2,-1)],
        [(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(2,0),(-2,0),(0,2),(0,-2)],
        [(0,0),(1,0),(2,0),(0,1),(1,1),(2,1),(0,2),(1,2),(2,2),(-1,1),(3,1)],
        [(0,0),(1,0),(2,0),(3,0),(0,1),(1,1),(2,1),(3,1),(1,2),(2,2),(3,2)],
        [(0,0),(1,0),(0,1),(1,1),(2,1),(0,2),(1,2),(2,2),(0,3),(1,3),(2,3)],
        [(0,0),(1,0),(2,0),(0,1),(1,1),(2,1),(0,2),(1,2),(2,2),(0,3),(1,3)],
        [(0,0),(1,0),(-1,0),(0,1),(0,-1),(1,1),(-1,1),(2,0),(-2,0),(0,2),(0,-2)],
        [(0,0),(1,0),(2,0),(3,0),(4,0),(0,1),(1,1),(2,1),(3,1),(4,1),(5,1)],
        [(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(2,1),(-2,2),(-1,3),(1,-2)],
        [(0,0),(1,0),(2,0),(0,1),(1,1),(2,1),(0,2),(1,2),(2,2),(1,3),(2,3)],
        [(0,0),(1,0),(0,1),(1,1),(2,1),(3,1),(0,2),(1,2),(2,2),(3,2),(1,3)],
        [(0,0),(1,0),(2,0),(3,0),(0,1),(1,1),(0,2),(1,2),(0,3),(1,3),(0,4)],
        [(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(2,0),(-2,0),(0,2),(1,2)],
        [(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(2,0),(-2,0),(1,1),(-1,-1)],
        [(0,0),(1,0),(2,0),(1,1),(2,1),(0,2),(1,2),(2,2),(-1,1),(3,1),(0,-1)],
        [(0,0),(1,0),(0,1),(1,1),(2,1),(1,2),(2,2),(0,3),(1,3),(2,3),(3,2)],
        [(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(2,1),(1,2),(-1,2),(0,3)],
        [(0,0),(1,0),(2,0),(3,0),(0,1),(1,1),(2,1),(1,2),(2,2),(3,2)],
        [(0,0),(1,0),(0,1),(1,1),(2,1),(1,2),(2,2),(0,3),(1,3),(2,3),(3,3)],
        [(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(2,0),(-2,0),(1,1),(-1,1)],
        [(0,0),(1,0),(2,0),(3,0),(0,1),(1,1),(2,1),(3,1),(0,2),(1,2),(2,2)],
        [(0,0),(1,0),(2,0),(0,1),(1,1),(2,1),(3,1),(0,2),(1,2),(2,2),(1,3)],
        [(0,0),(1,0),(2,0),(3,0),(1,1),(2,1),(3,1),(0,2),(1,2),(2,2),(3,2),(1,3),(2,3)],
        [(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(2,0),(-2,0),(0,2),(1,-2)],
        [(0,0),(1,0),(0,1),(1,1),(2,1),(0,2),(1,2),(0,3),(1,3),(2,3),(1,4),(2,4)],
        [(0,0),(1,0),(2,0),(0,1),(1,1),(2,1),(3,1),(0,2),(1,2),(2,2),(1,3),(2,3)],
        [(0,0),(1,0),(2,0),(3,0),(0,1),(1,1),(2,1),(3,1),(0,2),(1,2),(2,2)],
        [(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(2,0),(-2,0),(1,1),(-1,-1)],
        [(0,0),(1,0),(0,1),(1,1),(2,1),(1,2),(2,2),(0,3),(1,3),(2,3),(1,4),(2,4)],
        [(0,0),(1,0),(2,0),(0,1),(1,1),(2,1),(0,2),(1,2),(2,2),(0,3),(1,3)],
        [(0,0),(1,0),(2,0),(3,0),(0,1),(1,1),(2,1),(1,2),(2,2),(0,3),(1,3)],
        [(0,0),(1,0),(2,0),(1,1),(2,1),(0,2),(1,2),(2,2),(1,3),(2,3),(0,-1),(3,1)],
        [(0,0),(1,0),(2,0),(3,0),(0,1),(1,1),(2,1),(3,1),(0,2),(1,2),(2,2)],
        [(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(2,0),(-2,0),(0,2),(0,-2)],
        [(0,0),(1,0),(0,1),(1,1),(0,2),(1,2),(2,2),(0,3),(1,3),(0,4)],
        [(0,0),(1,0),(2,0),(0,1),(1,1),(0,2),(1,2),(2,2),(0,3),(1,3),(2,3)],
        [(0,0),(1,0),(2,0),(3,0),(0,1),(1,1),(2,1),(3,1),(0,2),(1,2),(2,2)],
        [(0,0),(1,0),(2,0),(1,1),(2,1),(3,1),(0,2),(1,2),(0,3),(1,3),(2,3)],
        [(0,0),(1,0),(0,1),(1,1),(2,1),(0,2),(1,2),(2,2),(1,3),(2,3),(3,3)],
        [(0,0),(1,0),(2,0),(1,1),(2,1),(0,2),(1,2),(0,3),(1,3),(0,4)],
        [(0,0),(1,0),(0,1),(1,1),(0,2),(1,2),(2,2),(0,3),(1,3),(2,3)],
        [(0,0),(1,0),(2,0),(0,1),(1,1),(2,1),(1,2),(2,2),(0,3),(1,3)],
        [(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(2,1),(-2,2),(0,3),(1,-2)],
        [(0,0),(1,0),(2,0),(0,1),(1,1),(2,1),(1,2),(2,2),(3,2),(0,3),(1,3)],
        [(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(2,0),(-2,0),(1,1),(-1,2)],
        [(0,0),(1,0),(2,0),(1,1),(2,1),(3,1),(0,2),(1,2),(0,3),(1,3),(2,3)],
        [(0,0),(1,0),(0,1),(1,1),(2,1),(0,2),(1,2),(2,2),(1,3),(2,3),(3,3),(0,4)],
        [(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(2,0),(-2,0),(1,2),(-1,-2)],
        [(0,0),(1,0),(2,0),(3,0),(0,1),(1,1),(0,2),(1,2),(2,2),(0,3),(1,3),(2,3)],
        [(0,0),(1,0),(0,1),(1,1),(2,1),(1,2),(2,2),(0,3),(1,3),(2,3),(1,4),(2,4)],
        [(0,0),(1,0),(2,0),(1,1),(2,1),(0,2),(1,2),(0,3),(1,3),(1,4)],
        [(0,0),(1,0),(0,1),(1,1),(0,2),(1,2),(2,2),(0,3),(1,3),(2,3)],
        [(0,0),(1,0),(2,0),(0,1),(1,1),(2,1),(1,2),(2,2),(0,3),(1,3)],
        [(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(2,0),(-2,0),(0,2),(1,2)],
        [(0,0),(1,0),(2,0),(1,1),(2,1),(0,2),(1,2),(0,3),(1,3),(1,4)],
        [(0,0),(1,0),(0,1),(1,1),(0,2),(1,2),(2,2),(0,3),(1,3),(2,3)],
    ]

    # ---------- optimize a fixed point set ----------
    def optimize_shape(pts):
        best_S = float('inf')
        best_beta = 0.0
        best_center = None
        best_free = None

        # 1) coarse grid over beta
        for i in range(61):
            beta = i * (math.pi / 6) / 60
            S, cx, cy, D = enclosing_hexagon(pts, beta)
            if S < best_S:
                best_S, best_beta, best_center = S, beta, (cx, cy)
        if best_center is None:
            return float('inf'), 0.0, None, None

        # 2) refine beta near best and a few others
        for rb in [best_beta, 0.0, math.pi/12, math.pi/6*0.25, math.pi/6*0.5, math.pi/6*0.75, math.pi/6*0.1, math.pi/6*0.4, math.pi/6*0.6, math.pi/6*0.9]:
            if not time_ok():
                break
            lo = max(0.0, rb - 0.25)
            hi = min(math.pi/6, rb + 0.25)
            try:
                r = minimize_scalar(lambda b: enclosing_hexagon(pts, b)[0],
                                    bounds=(lo, hi), method='bounded',
                                    options={'maxiter': 300, 'xatol': 1e-13})
                if r.fun < best_S:
                    S, cx, cy, D = enclosing_hexagon(pts, r.x)
                    best_S, best_beta, best_center = S, r.x, (cx, cy)
            except Exception:
                pass

        # 3) smooth surrogate over (cx, cy, beta)
        for temp in [0.2, 0.1, 0.05, 0.02, 0.01]:
            if not time_ok():
                break
            for sc in [(best_center[0], best_center[1], best_beta),
                       (best_center[0], best_center[1], best_beta + 0.05),
                       (best_center[0], best_center[1], best_beta - 0.05),
                       (best_center[0], best_center[1], 0.0),
                       (best_center[0], best_center[1], math.pi/12)]:
                if not time_ok():
                    break
                def smoo(params, t=temp):
                    cx, cy, beta = params
                    beta = ((beta % (math.pi/6)) + math.pi/6) % (math.pi/6)
                    return smooth_S(pts, cx, cy, beta, temperature=t)
                try:
                    r = minimize(smoo, list(sc), method='Nelder-Mead',
                                 options={'maxiter': 600, 'xatol': 1e-12, 'fatol': 1e-12, 'adaptive': True})
                    if r.fun < 1e10:
                        cx, cy, beta = r.x
                        beta = ((beta % (math.pi/6)) + math.pi/6) % (math.pi/6)
                        Se = exact_S(pts, cx, cy, beta)
                        if Se < best_S:
                            best_S, best_beta, best_center = Se, beta, (float(cx), float(cy))
                except Exception:
                    pass

        # 4) exact high-precision polish
        def obj(params):
            cx, cy, beta = params
            beta = ((beta % (math.pi/6)) + math.pi/6) % (math.pi/6)
            return exact_S(pts, cx, cy, beta)
        bcx, bcy = best_center
        bb = best_beta
        starts = [[bcx, bcy, bb]]
        for dr in [0.05, -0.05, 0.02, -0.02, 0.01, -0.01]:
            for ang in range(0, 360, 30):
                starts.append([bcx + dr*math.cos(math.radians(ang)), bcy + dr*math.sin(math.radians(ang)), bb])
        for db in [-0.03, -0.015, -0.008, -0.004, -0.002, 0.002, 0.004, 0.008, 0.015, 0.03]:
            starts.append([bcx, bcy, bb + db])
        for init in starts:
            if not time_ok():
                break
            try:
                r = minimize(obj, init, method='Nelder-Mead',
                             options={'maxiter': 8000, 'xatol': 1e-15, 'fatol': 1e-15, 'adaptive': True})
                if r.fun < best_S:
                    best_S = r.fun
                    best_center = (float(r.x[0]), float(r.x[1]))
                    best_beta = ((float(r.x[2]) % (math.pi/6)) + math.pi/6) % (math.pi/6)
            except Exception:
                pass

        # 5) ultra-fine
        ultra = [[bcx, bcy, bb], [bcx+0.008, bcy, bb], [bcx-0.008, bcy, bb],
                 [bcx, bcy+0.008, bb], [bcx, bcy-0.008, bb],
                 [bcx, bcy, bb+0.003], [bcx, bcy, bb-0.003],
                 [bcx, bcy, bb+0.001], [bcx, bcy, bb-0.001]]
        for init in ultra:
            if not time_ok():
                break
            try:
                r = minimize(obj, init, method='Nelder-Mead',
                             options={'maxiter': 20000, 'xatol': 1e-17, 'fatol': 1e-17, 'adaptive': True})
                if r.fun < best_S:
                    best_S = r.fun
                    best_center = (float(r.x[0]), float(r.x[1]))
                    best_beta = ((float(r.x[2]) % (math.pi/6)) + math.pi/6) % (math.pi/6)
            except Exception:
                pass

        # 6) free-position relaxation
        if time_ok():
            best_S, best_beta, best_center, best_free = relax_positions(pts, best_S, best_beta, best_center)

        return best_S, best_beta, best_center, best_free

    def relax_positions(pts_init, init_S, init_beta, init_center):
        best_S = init_S
        best_beta = init_beta
        best_center = init_center
        best_free = None
        n = len(pts_init)

        def relaxation_objective(params):
            cx, cy = params[0], params[1]
            beta_r = ((params[2] % (math.pi/6)) + math.pi/6) % (math.pi/6)
            positions = [(params[3+2*i], params[3+2*i+1]) for i in range(n)]
            angles = beta_r + math.pi/6.0 + np.arange(6)*math.pi/3.0
            cos_a = np.cos(angles); sin_a = np.sin(angles)
            P = np.array(positions)
            proj = (P - np.array([cx, cy])) @ np.column_stack([cos_a, sin_a]).T
            S = (float(np.max(np.abs(proj))) + APOTHEM) / APOTHEM
            penalty = 0.0
            for i in range(n):
                for j in range(i+1, n):
                    md = hex_pair_min_dist(positions[i], positions[j], 0.0, 0.0)
                    if md < 0:
                        penalty += md*md
            return S + 100.0*penalty

        start_params = [float(np.mean(pts_init[:,0])), float(np.mean(pts_init[:,1])), 0.0]
        for x, y in pts_init:
            start_params.append(float(x)); start_params.append(float(y))

        for b0 in [0.0, math.pi/12, math.pi/6*0.25, math.pi/6*0.5, math.pi/6*0.75]:
            if not time_ok():
                break
            p0 = list(start_params); p0[2] = b0
            try:
                r = minimize(relaxation_objective, p0, method='L-BFGS-B',
                             options={'maxiter': 1500, 'ftol': 1e-14, 'gtol': 1e-12})
                if r.fun < 1e10:
                    cx_r, cy_r = float(r.x[0]), float(r.x[1])
                    beta_r = ((float(r.x[2]) % (math.pi/6)) + math.pi/6) % (math.pi/6)
                    positions = [(float(r.x[3+2*i]), float(r.x[3+2*i+1])) for i in range(n)]
                    valid = True
                    for i in range(n):
                        for j in range(i+1, n):
                            if hex_pair_min_dist(positions[i], positions[j], 0.0, 0.0) < -1e-6:
                                valid = False; break
                        if not valid: break
                    if valid:
                        Se = exact_S(np.array(positions), cx_r, cy_r, beta_r)
                        if Se < best_S:
                            best_S, best_beta, best_center, best_free = Se, beta_r, (cx_r, cy_r), positions
            except Exception:
                pass

        if best_free is not None and time_ok():
            def polish_obj(params):
                cx, cy = params[0], params[1]
                beta_r = ((params[2] % (math.pi/6)) + math.pi/6) % (math.pi/6)
                positions = [(params[3+2*i], params[3+2*i+1]) for i in range(n)]
                angles = beta_r + math.pi/6.0 + np.arange(6)*math.pi/3.0
                cos_a = np.cos(angles); sin_a = np.sin(angles)
                P = np.array(positions)
                proj = (P - np.array([cx, cy])) @ np.column_stack([cos_a, sin_a]).T
                S = (float(np.max(np.abs(proj))) + APOTHEM) / APOTHEM
                penalty = 0.0
                for i in range(n):
                    for j in range(i+1, n):
                        md = hex_pair_min_dist(positions[i], positions[j], 0.0, 0.0)
                        if md < 0:
                            penalty += md*md
                return S + 50.0*penalty
            init = [best_center[0], best_center[1], best_beta]
            for (x, y) in best_free:
                init.append(x); init.append(y)
            try:
                r = minimize(polish_obj, init, method='Nelder-Mead',
                             options={'maxiter': 5000, 'xatol': 1e-14, 'fatol': 1e-14, 'adaptive': True})
                if r.fun < best_S:
                    cx_r, cy_r = float(r.x[0]), float(r.x[1])
                    beta_r = ((float(r.x[2]) % (math.pi/6)) + math.pi/6) % (math.pi/6)
                    positions = [(float(r.x[3+2*i]), float(r.x[3+2*i+1])) for i in range(n)]
                    valid = True
                    for i in range(n):
                        for j in range(i+1, n):
                            if hex_pair_min_dist(positions[i], positions[j], 0.0, 0.0) < -1e-6:
                                valid = False; break
                        if not valid: break
                    if valid:
                        Se = exact_S(np.array(positions), cx_r, cy_r, beta_r)
                        if Se < best_S:
                            best_S, best_beta, best_center, best_free = Se, beta_r, (cx_r, cy_r), positions
            except Exception:
                pass
        return best_S, best_beta, best_center, best_free

    # ---------- boundary-contact topology search ----------
    def boundary_topology_search(pts):
        # For each assignment of inner hexagons to the 6 outer walls, build a
        # 1-DOF mechanism and minimize S over the joint angle.
        n = len(pts)
        P0 = np.array(pts)
        best = (float('inf'), 0.0, None, None)
        # assign each point to a wall (0..5); to limit combinatorics, only
        # consider assignments where each wall gets ~n/6 points (round-robin + few variants)
        wall_assignments = []
        base = [i % 6 for i in range(n)]
        wall_assignments.append(base)
        # a few perturbations
        for shift in range(6):
            wall_assignments.append([(i + shift) % 6 for i in range(n)])
        for wa in wall_assignments:
            if not time_ok():
                break
            # For a given assignment, the mechanism: each point i is pushed to
            # touch wall wa[i] at distance D along that wall's normal. We solve
            # for (cx, cy, D) given beta, then minimize over beta.
            def S_for_beta(beta):
                angles = beta + math.pi/6.0 + np.arange(6)*math.pi/3.0
                cos_a = np.cos(angles); sin_a = np.sin(angles)
                # We want center c and D such that for each i, the projection
                # of point i onto normal wa[i] equals D (touching), and other
                # projections <= D. Solve least-squares for c using the
                # touching constraints, then take D = max over all projections.
                # Use the touching constraints as equations:
                # (p_i - c) . n_{wa[i]} = D  =>  c.n_{wa[i]} + D = p_i.n_{wa[i]}
                # This is linear in (c, D). Solve least squares.
                A = np.zeros((n, 3)); b = np.zeros(n)
                for i in range(n):
                    k = wa[i]
                    A[i, 0] = cos_a[k]; A[i, 1] = sin_a[k]; A[i, 2] = 1.0
                    b[i] = P0[i,0]*cos_a[k] + P0[i,1]*sin_a[k]
                try:
                    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
                except Exception:
                    return float('inf')
                cx, cy, D = float(sol[0]), float(sol[1]), float(sol[2])
                if D < 0:
                    D = 0.0
                # actual max projection
                proj = (P0 - np.array([cx, cy])) @ np.column_stack([cos_a, sin_a]).T
                Dv = float(np.max(np.abs(proj)))
                return (Dv + APOTHEM) / APOTHEM
            # minimize over beta
            best_b = 0.0; best_S = float('inf')
            for i in range(13):
                beta = i * (math.pi/6) / 12
                S = S_for_beta(beta)
                if S < best_S:
                    best_S, best_b = S, beta
            try:
                r = minimize_scalar(S_for_beta, bounds=(0.0, math.pi/6), method='bounded',
                                    options={'maxiter': 200, 'xatol': 1e-12})
                if r.fun < best_S:
                    best_S, best_b = r.fun, r.x
            except Exception:
                pass
            if best_S < best[0]:
                angles = best_b + math.pi/6.0 + np.arange(6)*math.pi/3.0
                cos_a = np.cos(angles); sin_a = np.sin(angles)
                A = np.zeros((n, 3)); b = np.zeros(n)
                for i in range(n):
                    k = wa[i]
                    A[i,0]=cos_a[k]; A[i,1]=sin_a[k]; A[i,2]=1.0
                    b[i]=P0[i,0]*cos_a[k]+P0[i,1]*sin_a[k]
                try:
                    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
                    cx, cy = float(sol[0]), float(sol[1])
                except Exception:
                    cx, cy = float(P0[:,0].mean()), float(P0[:,1].mean())
                best = (best_S, best_b, (cx, cy), None)
        return best

    # ---------- main loop ----------
    best_global_S = float('inf')
    best_global_pts = None
    best_global_beta = 0.0
    best_global_center = None
    best_global_free = None

    pts_list = [axial_to_cartesian(c) for c in candidates]
    radii = [hex_radius(p) for p in pts_list]
    order = sorted(range(len(pts_list)), key=lambda i: radii[i])

    for idx in order:
        if not time_ok():
            break
        pts = pts_list[idx]
        S, beta, center, free = optimize_shape(pts)
        if S < best_global_S:
            best_global_S, best_global_pts, best_global_beta, best_global_center, best_global_free = S, pts, beta, center, free
        # boundary topology search on this candidate
        if time_ok():
            S2, beta2, center2, _ = boundary_topology_search(pts)
            if S2 < best_global_S:
                best_global_S, best_global_pts, best_global_beta, best_global_center = S2, pts, beta2, center2

    if best_global_center is None or best_global_S == float('inf'):
        best_global_pts = pts_list[0]
        S, cx, cy, D = enclosing_hexagon(best_global_pts, 0.0)
        best_global_S, best_global_center, best_global_beta = S, (cx, cy), 0.0

    cx, cy = best_global_center
    angle_deg = math.degrees(best_global_beta)

    if best_global_free is not None:
        final_positions = best_global_free
    else:
        SCALE = 1.0 + 1e-12
        final_positions = [(cx + (x-cx)*SCALE, cy + (y-cy)*SCALE) for (x, y) in best_global_pts]

    max_proj = 0.0
    for (px, py) in final_positions:
        dx, dy = px-cx, py-cy
        for k in range(6):
            ang = best_global_beta + math.pi/6.0 + k*math.pi/3.0
            nx, ny = math.cos(ang), math.sin(ang)
            proj = abs(dx*nx + dy*ny)
            if proj > max_proj:
                max_proj = proj
    S_scaled = (max_proj + APOTHEM) / APOTHEM

    inner = [[float(x), float(y), float(angle_deg)] for (x, y) in final_positions]
    center = [float(cx), float(cy)]
    side = float(S_scaled)
    rotation = float(angle_deg)
    return inner, center, side, rotation
