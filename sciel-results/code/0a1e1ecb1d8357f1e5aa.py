"""Fully independent hexagon placement with DE + SLSQP polishing."""
import math
import time
import numpy as np
from scipy.optimize import differential_evolution, minimize

N = 12

def _hex_verts(cx, cy, ang, s=1.0):
    a = np.deg2rad(ang)
    t = a + np.arange(6) * np.pi / 3
    return np.column_stack([cx + s * np.cos(t), cy + s * np.sin(t)])

def _pair_dist(v1, v2):
    """Min SAT separation; >=0 disjoint, <0 overlap."""
    best = np.inf
    for i in range(6):
        for src in (v1, v2):
            e = src[(i + 1) % 6] - src[i]
            n = np.array([-e[1], e[0]])
            nn = np.linalg.norm(n)
            if nn < 1e-12:
                continue
            n /= nn
            p1 = v1 @ n
            p2 = v2 @ n
            m1, M1 = p1.min(), p1.max()
            m2, M2 = p2.min(), p2.max()
            if M1 < m2:
                sep = m2 - M1
            elif M2 < m1:
                sep = m1 - M2
            else:
                sep = -min(M1 - m2, M2 - m1)
            if sep < best:
                best = sep
    return best

def _min_side(verts, oa_deg):
    th = np.deg2rad(oa_deg)
    mp = 0.0
    for k in range(6):
        a = th + k * np.pi / 3
        n = np.array([math.cos(a), math.sin(a)])
        p = verts @ n
        mp = max(mp, p.max())
    return 2.0 * mp / math.sqrt(3.0)

def construct():
    t0 = time.time()
    deadline = t0 + 135.0

    def _time_left():
        return deadline - time.time()

    # Parameter layout: [x0..y0, a0, x1..y1, a1, ..., x11..y11, a11, outer_angle]
    # 12*3 + 1 = 37 params
    def _unpack(x):
        pos = np.empty((N, 2))
        ang = np.empty(N)
        for i in range(N):
            pos[i, 0] = x[3 * i]
            pos[i, 1] = x[3 * i + 1]
            ang[i] = x[3 * i + 2]
        oa = x[3 * N]
        return pos, ang, oa

    def _verts_from_params(x):
        pos, ang, _ = _unpack(x)
        return [_hex_verts(pos[i, 0], pos[i, 1], ang[i]) for i in range(N)]

    def _objective(x):
        if time.time() > deadline - 2:
            return 1e10
        pos, ang, oa = _unpack(x)
        verts = [_hex_verts(pos[i, 0], pos[i, 1], ang[i]) for i in range(N)]
        all_v = np.vstack(verts)
        return _min_side(all_v, oa)

    def _overlap_penalty(x, margin=0.0):
        """Sum of squared penetrations (for DE objective)."""
        if time.time() > deadline - 2:
            return 1e10
        pos, ang, _ = _unpack(x)
        verts = [_hex_verts(pos[i, 0], pos[i, 1], ang[i]) for i in range(N)]
        pen = 0.0
        for i in range(N):
            for j in range(i + 1, N):
                d = _pair_dist(verts[i], verts[j])
                if d < margin:
                    pen += (margin - d) ** 2
        return pen

    def _de_objective(x):
        if time.time() > deadline - 2:
            return 1e10
        side = _objective(x)
        pen = _overlap_penalty(x, margin=0.01)
        # Penalize overlaps heavily
        return side + 10.0 * pen

    # Initial honeycomb layout for seeding
    def _honeycomb_seed():
        sx = math.sqrt(3)
        # Ring 0: center
        pts = [(0.0, 0.0)]
        # Ring 1: 6 hexes
        for i in range(6):
            a = np.deg2rad(60 * i)
            pts.append((sx * math.cos(a), sx * math.sin(a)))
        # Ring 2: pick 5 from 12
        ring2 = []
        for q in range(-2, 3):
            for r in range(-2, 3):
                dist = (abs(q) + abs(r) + abs(q + r)) / 2
                if dist == 2:
                    x = sx * (q + r / 2.0)
                    y = 1.5 * r
                    ring2.append((x, y, math.atan2(y, x)))
        ring2.sort(key=lambda t: t[2])
        for i in range(0, 12, 2):
            pts.append((ring2[i][0], ring2[i][1]))
        pts = np.array(pts[:N])
        cx, cy = pts[:, 0].mean(), pts[:, 1].mean()
        pts[:, 0] -= cx
        pts[:, 1] -= cy
        x = np.zeros(3 * N + 1)
        for i in range(N):
            x[3 * i] = pts[i, 0]
            x[3 * i + 1] = pts[i, 1]
            x[3 * i + 2] = 0.0
        x[3 * N] = 0.0
        return x

    def _random_seed():
        x = np.zeros(3 * N + 1)
        for i in range(N):
            x[3 * i] = np.random.uniform(-3, 3)
            x[3 * i + 1] = np.random.uniform(-3, 3)
            x[3 * i + 2] = np.random.uniform(0, 180)
        x[3 * N] = np.random.uniform(0, 180)
        return x

    def _constrain(x):
        """Return list of constraint dicts for SLSQP."""
        pos, ang, _ = _unpack(x)
        verts = [_hex_verts(pos[i, 0], pos[i, 1], ang[i]) for i in range(N)]
        cons = []
        for i in range(N):
            for j in range(i + 1, N):
                d = _pair_dist(verts[i], verts[j])
                cons.append(d)
        return cons

    # Bounds: positions in [-5,5], angles in [0,180], outer angle in [0,180]
    bounds = [(-5, 5), (-5, 5), (0, 180)] * N + [(0, 180)]

    best_x = _honeycomb_seed()
    best_score = _objective(best_x)
    # Check validity of seed
    verts = _verts_from_params(best_x)
    valid = all(_pair_dist(verts[i], verts[j]) >= -1e-6
                for i in range(N) for j in range(i + 1, N))
    if not valid:
        best_score = 1e10

    # Phase 1: Differential Evolution (global search)
    try:
        if _time_left() > 60:
            de_result = differential_evolution(
                _de_objective, bounds,
                seed=42, maxiter=200, tol=1e-4,
                popsize=15, mutation=(0.3, 1.0), recombination=0.8,
                init='sobol', updating='immediate',
                polish=False,
                callback=lambda xk, convergence: (
                    time.time() > deadline - 30
                ),
            )
            if de_result.fun < 1e9:
                # Check if valid (no overlap)
                verts = _verts_from_params(de_result.x)
                is_valid = all(_pair_dist(verts[i], verts[j]) >= -1e-4
                               for i in range(N) for j in range(i + 1, N))
                if is_valid:
                    sc = _objective(de_result.x)
                    if sc < best_score:
                        best_score = sc
                        best_x = de_result.x.copy()
    except Exception:
        pass

    # Phase 2: SLSQP polishing from best DE result
    try:
        if _time_left() > 30:
            # Build SLSQP constraints
            def _cons_fun(x):
                if time.time() > deadline - 2:
                    return np.zeros(N * (N - 1) // 2)
                pos, ang, _ = _unpack(x)
                verts = [_hex_verts(pos[i, 0], pos[i, 1], ang[i]) for i in range(N)]
                c = np.empty(N * (N - 1) // 2)
                idx = 0
                for i in range(N):
                    for j in range(i + 1, N):
                        c[idx] = _pair_dist(verts[i], verts[j])
                        idx += 1
                return c

            slsqp_cons = [{'type': 'ineq', 'fun': _cons_fun}]
            slsqp_result = minimize(
                _objective, best_x, method='SLSQP',
                bounds=bounds, constraints=slsqp_cons,
                options={'maxiter': 300, 'ftol': 1e-8, 'disp': False},
            )
            if slsqp_result.fun < 1e9:
                verts = _verts_from_params(slsqp_result.x)
                is_valid = all(_pair_dist(verts[i], verts[j]) >= -1e-4
                               for i in range(N) for j in range(i + 1, N))
                if is_valid:
                    sc = _objective(slsqp_result.x)
                    if sc < best_score:
                        best_score = sc
                        best_x = slsqp_result.x.copy()
    except Exception:
        pass

    # Phase 3: Multiple SLSQP restarts from perturbed starting points
    try:
        if _time_left() > 15:
            rng = np.random.default_rng(123)
            for restart in range(5):
                if time.time() > deadline - 5:
                    break
                x_start = best_x + rng.normal(0, 0.3, size=len(best_x))
                x_start[3::3] = np.clip(x_start[3::3], -5, 5)  # x coords
                x_start[4::3] = np.clip(x_start[4::3], -5, 5)  # y coords
                x_start[2::3] = x_start[2::3] % 180  # angles
                x_start[3 * N] = x_start[3 * N] % 180

                def _cons_fun_r(x):
                    if time.time() > deadline - 2:
                        return np.zeros(N * (N - 1) // 2)
                    pos, ang, _ = _unpack(x)
                    verts = [_hex_verts(pos[i, 0], pos[i, 1], ang[i]) for i in range(N)]
                    c = np.empty(N * (N - 1) // 2)
                    idx = 0
                    for i in range(N):
                        for j in range(i + 1, N):
                            c[idx] = _pair_dist(verts[i], verts[j])
                            idx += 1
                    return c

                r_cons = [{'type': 'ineq', 'fun': _cons_fun_r}]
                r_result = minimize(
                    _objective, x_start, method='SLSQP',
                    bounds=bounds, constraints=r_cons,
                    options={'maxiter': 150, 'ftol': 1e-8, 'disp': False},
                )
                if r_result.fun < 1e9:
                    verts = _verts_from_params(r_result.x)
                    is_valid = all(_pair_dist(verts[i], verts[j]) >= -1e-4
                                   for i in range(N) for j in range(i + 1, N))
                    if is_valid:
                        sc = _objective(r_result.x)
                        if sc < best_score:
                            best_score = sc
                            best_x = r_result.x.copy()
    except Exception:
        pass

    # Finalize
    pos, ang, oa_deg = _unpack(best_x)

    # Center the configuration
    cx = pos[:, 0].mean()
    cy = pos[:, 1].mean()
    pos[:, 0] -= cx
    pos[:, 1] -= cy

    # Recompute final side
    verts = [_hex_verts(pos[i, 0], pos[i, 1], ang[i]) for i in range(N)]
    all_v = np.vstack(verts)
    final_side = _min_side(all_v, oa_deg)

    # Final validity check
    valid = all(_pair_dist(verts[i], verts[j]) >= -1e-4
                for i in range(N) for j in range(i + 1, N))

    if not valid:
        # Fallback: spread out in a grid
        pos = np.array([[3.0 * (i % 3) - 3.0, 2.6 * (i // 3) - 2.6] for i in range(N)])
        cx_f = pos[:, 0].mean()
        cy_f = pos[:, 1].mean()
        pos[:, 0] -= cx_f
        pos[:, 1] -= cy_f
        ang = np.zeros(N)
        oa_deg = 0.0
        verts = [_hex_verts(pos[i, 0], pos[i, 1], ang[i]) for i in range(N)]
        all_v = np.vstack(verts)
        final_side = _min_side(all_v, oa_deg)

    inner = [[float(pos[i, 0]), float(pos[i, 1]), float(ang[i])] for i in range(N)]
    return inner, (0.0, 0.0), float(final_side), float(oa_deg)
