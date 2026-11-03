import math
import time
import random
import numpy as np
from scipy.optimize import minimize

N = 12
DEADLINE = 115.0

# Precompute unit hexagon data
_HEX_ANGS = np.radians(np.arange(0, 360, 60))
_UV = np.column_stack((np.cos(_HEX_ANGS), np.sin(_HEX_ANGS)))
_EN = np.zeros((6, 2))
for _i in range(6):
    _ex = _UV[(_i + 1) % 6, 0] - _UV[_i, 0]
    _ey = _UV[(_i + 1) % 6, 1] - _UV[_i, 1]
    _nx, _ny = -_ey, _ex
    _nn = math.hypot(_nx, _ny)
    _EN[_i] = [_nx / _nn, _ny / _nn]


def construct():
    t0 = time.time()
    sqrt3 = math.sqrt(3)
    _ux, _uy = _UV[:, 0], _UV[:, 1]
    _enx, _eny = _EN[:, 0], _EN[:, 1]

    def verts_all(pos, ang):
        c = np.cos(ang)
        s = np.sin(ang)
        rx = _ux[None, :] * c[:, None] - _uy[None, :] * s[:, None]
        ry = _ux[None, :] * s[:, None] + _uy[None, :] * c[:, None]
        return np.stack((rx + pos[:, 0:1], ry + pos[:, 1:2]), axis=-1)

    def disjoint(h1, h2):
        dx, dy = h1[0] - h2[0], h1[1] - h2[1]
        if dx * dx + dy * dy > 4.5:
            return True
        c1, s1 = math.cos(h1[2]), math.sin(h1[2])
        c2, s2 = math.cos(h2[2]), math.sin(h2[2])
        v1x = h1[0] + _ux * c1 - _uy * s1
        v1y = h1[1] + _ux * s1 + _uy * c1
        v2x = h2[0] + _ux * c2 - _uy * s2
        v2y = h2[1] + _ux * s2 + _uy * c2
        e1x = _enx * c1 - _eny * s1
        e1y = _enx * s1 + _eny * c1
        e2x = _enx * c2 - _eny * s2
        e2y = _enx * s2 + _eny * c2
        for i in range(6):
            nx, ny = e1x[i], e1y[i]
            p1 = v1x * nx + v1y * ny
            p2 = v2x * nx + v2y * ny
            if p1.max() < p2.min() or p2.max() < p1.min():
                return True
        for i in range(6):
            nx, ny = e2x[i], e2y[i]
            p1 = v1x * nx + v1y * ny
            p2 = v2x * nx + v2y * ny
            if p1.max() < p2.min() or p2.max() < p1.min():
                return True
        return False

    def outer_side(pos, ang_deg):
        ar = np.radians(ang_deg)
        verts = verts_all(pos, ar).reshape(-1, 2)
        ctr = verts.mean(axis=0)
        off = verts - ctr

        def phi_eval(pd):
            p = np.radians(pd)
            na = p + np.pi / 6 + np.arange(6) * np.pi / 3
            nx, ny = np.cos(na), np.sin(na)
            proj = off[:, 0:1] * nx[None, :] + off[:, 1:2] * ny[None, :]
            mp = np.max(np.abs(proj), axis=0)
            return np.max((2.0 / sqrt3) * mp)

        bs, bp = float('inf'), 0.0
        for pd in range(0, 60, 1):
            s = phi_eval(pd)
            if s < bs:
                bs, bp = s, pd
        try:
            lo, hi = max(0.0, bp - 1.0), min(60.0, bp + 1.0)
            r = minimize_scalar(phi_eval, bounds=(lo, hi), method='bounded', options={'xatol': 1e-9})
            if r.success and r.fun < bs:
                bs, bp = r.fun, r.x
        except Exception:
            pass
        return bs, bp, ctr[0], ctr[1]

    def valid(pos, ang):
        ar = np.radians(ang)
        for i in range(N):
            for j in range(i + 1, N):
                if not disjoint((pos[i, 0], pos[i, 1], ar[i]), (pos[j, 0], pos[j, 1], ar[j])):
                    return False
        return True

    # --- Multi-start with diverse initial layouts ---
    a1 = np.array([sqrt3, 0.0])
    a2 = np.array([sqrt3 / 2, 1.5])

    def honeycomb():
        lc = [(0, 0), (1, 0), (-1, 0), (0, 1), (0, -1), (1, -1),
              (-1, 1), (2, -1), (-2, 1), (1, 1), (-1, -1), (2, 0)]
        p = np.zeros((N, 2))
        for i, (ii, jj) in enumerate(lc):
            p[i] = ii * a1 + jj * a2
        return p, np.zeros(N)

    def ring_plus_center():
        p = np.zeros((N, 2))
        ang0 = np.zeros(N)
        p[0] = [0, 0]
        for k in range(1, 12):
            theta = (k - 1) * 2 * math.pi / 11
            r = 2.0 + 0.1 * (k % 3)
            p[k] = [r * math.cos(theta), r * math.sin(theta)]
            ang0[k] = theta * 180 / math.pi
        return p, ang0

    def grid_3x4():
        p = np.zeros((N, 2))
        ang0 = np.zeros(N)
        for i in range(3):
            for j in range(4):
                idx = i * 4 + j
                p[idx] = [j * 1.9, i * 1.8]
                ang0[idx] = 0.0
        return p, ang0

    def spiral():
        p = np.zeros((N, 2))
        ang0 = np.zeros(N)
        for k in range(N):
            theta = k * 0.7
            r = 0.3 * k + 0.5
            p[k] = [r * math.cos(theta), r * math.sin(theta)]
            ang0[k] = k * 15.0
        return p, ang0

    generators = [honeycomb, ring_plus_center, grid_3x4, spiral]

    best_side = float('inf')
    best_pos = None
    best_ang = None
    best_phi = 0.0
    best_cx = 0.0
    best_cy = 0.0

    # Phase 1: SA from multiple starts, shorter each
    sa_budget_each = min(12.0, (DEADLINE - 20.0) / len(generators))

    for gen in generators:
        if time.time() - t0 > DEADLINE - 15.0:
            break
        init_pos, init_ang = gen()
        cur_pos, cur_ang = init_pos.copy(), init_ang.copy()
        cur_side, cur_phi, cur_cx, cur_cy = outer_side(cur_pos, cur_ang)
        if cur_side < best_side:
            best_side, best_pos, best_ang = cur_side, cur_pos.copy(), cur_ang.copy()
            best_phi, best_cx, best_cy = cur_phi, cur_cx, cur_cy

        T, Tmin, cool, step, stepdec = 0.3, 0.0001, 0.99995, 0.25, 0.99998
        sc = 0
        max_steps = 80000
        sa_end = time.time() + sa_budget_each
        while time.time() < sa_end and sc < max_steps:
            sc += 1
            idx = random.randint(0, N - 1)
            np_ = cur_pos.copy()
            na_ = cur_ang.copy()
            np_[idx, 0] += random.gauss(0, step)
            np_[idx, 1] += random.gauss(0, step)
            na_[idx] += random.gauss(0, 4.0)
            ar = np.radians(na_)
            ok = True
            hm = (np_[idx, 0], np_[idx, 1], ar[idx])
            for j in range(N):
                if j == idx:
                    continue
                ho = (np_[j, 0], np_[j, 1], ar[j])
                if not disjoint(hm, ho):
                    ok = False
                    break
            if not ok:
                T = max(Tmin, T * cool)
                step = max(0.004, step * stepdec)
                continue
            ns, np_, nphi, ncx, ncy = outer_side(np_, na_)
            if ns < cur_side or (random.random() < math.exp(-(ns - cur_side) / T) if ns >= cur_side else True):
                cur_pos, cur_ang = np_, na_
                cur_side, cur_phi, cur_cx, cur_cy = ns, nphi, ncx, ncy
                if ns < best_side:
                    best_side, best_pos, best_ang = ns, np_.copy(), na_.copy()
                    best_phi, best_cx, best_cy = nphi, ncx, ncy
            T = max(Tmin, T * cool)
            step = max(0.004, step * stepdec)

    # Phase 2: CMA-ES-like gradient-free local search with adaptive step
    remaining = DEADLINE - (time.time() - t0)
    if remaining > 10.0 and best_pos is not None:
        # Coordinate descent with adaptive step
        cpos, can = best_pos.copy(), best_ang.copy()
        cside, cphi, ccx, ccy = outer_side(cpos, can)
        step = 0.04
        min_step = 0.0005
        cd_end = time.time() + min(remaining * 0.5, 25.0)
        no_improve = 0
        while time.time() < cd_end and step > min_step:
            improved = False
            for idx in range(N):
                for dim in range(3):
                    for d in [1, -1]:
                        np_ = cpos.copy()
                        na_ = can.copy()
                        if dim == 0:
                            np_[idx, 0] += d * step
                        elif dim == 1:
                            np_[idx, 1] += d * step
                        else:
                            na_[idx] += d * step
                        ar = np.radians(na_)
                        ok = True
                        hm = (np_[idx, 0], np_[idx, 1], ar[idx])
                        for j in range(N):
                            if j == idx:
                                continue
                            ho = (np_[j, 0], np_[j, 1], ar[j])
                            if not disjoint(hm, ho):
                                ok = False
                                break
                        if not ok:
                            continue
                        ns, _, _, _, _ = outer_side(np_, na_)
                        if ns < cside - 1e-11:
                            cpos, can = np_, na_
                            cside = ns
                            improved = True
                            break
                    if time.time() >= cd_end:
                        break
                if time.time() >= cd_end:
                    break
            if improved:
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= 2:
                    step *= 0.7
            if cside < best_side:
                best_side, best_pos, best_ang = cside, cpos.copy(), can.copy()
                best_side, best_phi, best_cx, best_cy = outer_side(best_pos, best_ang)

        # Nelder-Mead polish on full parameter vector
        remaining2 = DEADLINE - (time.time() - t0)
        if remaining2 > 8.0:
            def obj(params):
                p = np.zeros((N, 2))
                a = np.zeros(N)
                for i in range(N):
                    p[i, 0], p[i, 1], a[i] = params[3 * i], params[3 * i + 1], params[3 * i + 2]
                ar = np.radians(a)
                for i in range(N):
                    for j in range(i + 1, N):
                        if not disjoint((p[i, 0], p[i, 1], ar[i]), (p[j, 0], p[j, 1], ar[j])):
                            return 1e6
                s, _, _, _ = outer_side(p, a)
                return s

            init = np.zeros(3 * N)
            for i in range(N):
                init[3 * i] = best_pos[i, 0]
                init[3 * i + 1] = best_pos[i, 1]
                init[3 * i + 2] = best_ang[i]
            try:
                r = minimize(obj, init, method='Nelder-Mead',
                             options={'maxiter': 4000, 'xatol': 1e-7, 'fatol': 1e-9})
                if r.fun < best_side:
                    p = np.zeros((N, 2))
                    a = np.zeros(N)
                    for i in range(N):
                        p[i, 0], p[i, 1], a[i] = r.x[3 * i], r.x[3 * i + 1], r.x[3 * i + 2]
                    if valid(p, a):
                        s, ph, cx, cy = outer_side(p, a)
                        if s < best_side:
                            best_side, best_pos, best_ang = s, p, a
                            best_phi, best_cx, best_cy = ph, cx, cy
            except Exception:
                pass

    if best_pos is None or not valid(best_pos, best_ang):
        best_pos, best_ang = honeycomb()
        best_side, best_phi, best_cx, best_cy = outer_side(best_pos, best_ang)

    inner = [[float(best_pos[i, 0]), float(best_pos[i, 1]), float(best_ang[i])] for i in range(N)]
    return inner, (float(best_cx), float(best_cy)), float(best_side), float(best_phi)


from scipy.optimize import minimize_scalar
