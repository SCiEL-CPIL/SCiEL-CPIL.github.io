"""Packing 12 unit hexagons in a regular hexagon – active-set / contact-based formulation."""
import math
import time
import numpy as np
from scipy.optimize import minimize, differential_evolution

N = 12
SQRT3 = math.sqrt(3.0)
TWO_PI = 2.0 * math.pi

def construct():
    t0 = time.time()
    deadline = t0 + 145.0

    # ---------- geometry helpers ----------
    def hex_verts(cx, cy, ang_deg, s=1.0):
        a0 = math.radians(ang_deg)
        th = a0 + np.arange(6) * (math.pi / 3.0)
        return np.column_stack([cx + s * np.cos(th), cy + s * np.sin(th)])

    def hex_edges(verts):
        """Return (6,2) edge direction vectors (unit) and (6,2) outward normals."""
        e = np.diff(verts, axis=0, append=verts[:1])
        L = np.linalg.norm(e, axis=1)
        e_hat = e / L[:, None]
        # outward normal = rotate e_hat by -90 deg (CW)
        n = np.column_stack([e_hat[:, 1], -e_hat[:, 0]])
        return e_hat, n

    def sat_overlap(v1, v2, tol=1e-10):
        """True if two convex hexagons overlap (share interior)."""
        for v in (v1, v2):
            e, n = hex_edges(v)
            for i in range(6):
                axis = n[i]
                p1 = v @ axis
                p2 = v2 @ axis
                if p1.max() < p2.min() - tol or p2.max() < p1.min() - tol:
                    return False
        return True

    def all_disjoint(verts_list, tol=1e-10):
        for i in range(N):
            for j in range(i + 1, N):
                if sat_overlap(verts_list[i], verts_list[j], tol):
                    return False
        return True

    def point_in_hex(px, py, ocx, ocy, oang, side):
        """Check if (px,py) is inside the outer hexagon."""
        v = hex_verts(ocx, ocy, oang, side)
        # use cross-product test for CCW polygon
        for i in range(6):
            ax, ay = v[i]
            bx, by = v[(i + 1) % 6]
            cross = (bx - ax) * (py - ay) - (by - ay) * (px - ax)
            if cross < -1e-9:
                return False
        return True

    def contains_all(verts_list, ocx, ocy, oang, side, margin=1e-6):
        for vl in verts_list:
            for px, py in vl:
                if not point_in_hex(px, py, ocx, ocy, oang, side + margin):
                    return False
        return True

    # ---------- support-function based outer side ----------
    def outer_side_from_verts(verts_list, ocx, ocy, oang):
        """Minimum side of outer hexagon (centered at ocx,ocy, rotated oang) containing all verts."""
        oang_rad = math.radians(oang)
        # 6 outward normals of the outer hexagon
        normals = []
        for k in range(6):
            a = oang_rad + k * math.pi / 3.0 + math.pi / 6.0
            normals.append(np.array([math.cos(a), math.sin(a)]))
        normals = np.array(normals)  # (6,2)
        center = np.array([ocx, ocy])
        max_proj = 0.0
        for vl in verts_list:
            rel = vl - center
            projs = rel @ normals.T  # (6,6)
            max_proj = max(max_proj, projs.max())
        # For a regular hexagon of side s, the apothem = s * sqrt(3)/2
        # max_proj is the max distance from center to any vertex along normals = apothem
        # so s = 2 * max_proj / sqrt(3)
        return 2.0 * max_proj / SQRT3

    def validity(inner, ocx, ocy, oang, side, tol=1e-8):
        verts_list = [hex_verts(x, y, a) for x, y, a in inner]
        if not all_disjoint(verts_list, tol):
            return False
        if not contains_all(verts_list, ocx, ocy, oang, side, margin=1e-5):
            return False
        return True

    # ---------- initial topologies ----------
    def topo_honeycomb():
        """4x3 honeycomb grid of hexagons."""
        centers = []
        for r in range(4):
            for c in range(3):
                x = c * 2.0
                y = r * SQRT3
                if r % 2 == 1:
                    x += 1.0
                centers.append((x, y, 0.0))
        mx = np.mean([c[0] for c in centers])
        my = np.mean([c[1] for c in centers])
        return [(x - mx, y - my, a) for x, y, a in centers], 0.0, 0.0, 0.0

    def topo_ring12():
        """12 hexagons in a ring."""
        data = []
        for i in range(12):
            a = i * math.pi / 6.0
            r = 2.6
            data.append((r * math.cos(a), r * math.sin(a), math.degrees(a) + 90.0))
        return data, 0.0, 0.0, 0.0

    def topo_two_rings():
        """6 inner + 6 outer ring."""
        data = []
        for i in range(6):
            a = i * math.pi / 3.0
            r = 1.9
            data.append((r * math.cos(a), r * math.sin(a), math.degrees(a) + 90.0))
        for i in range(6):
            a = i * math.pi / 3.0 + math.pi / 6.0
            r = 3.3
            data.append((r * math.cos(a), r * math.sin(a), math.degrees(a) + 90.0))
        return data, 0.0, 0.0, 0.0

    def topo_hex_cluster():
        """1 center + 6 ring + 5 outer."""
        data = [(0.0, 0.0, 0.0)]
        for i in range(6):
            a = i * math.pi / 3.0
            data.append((2.0 * math.cos(a), 2.0 * math.sin(a), 0.0))
        for i in range(6):
            if i == 3:
                continue
            a = i * math.pi / 3.0 + math.pi / 6.0
            data.append((3.6 * math.cos(a), 3.6 * math.sin(a), 0.0))
        mx = np.mean([d[0] for d in data])
        my = np.mean([d[1] for d in data])
        data = [(x - mx, y - my, a) for x, y, a in data]
        return data, 0.0, 0.0, 0.0

    def topo_rect():
        data = []
        for r in range(4):
            for c in range(3):
                data.append((c * 2.1 - 2.1, r * 1.75 - 3.5, 0.0))
        return data, 0.0, 0.0, 0.0

    # ---------- objective for inner placement ----------
    def make_objective(ocx, ocy, oang):
        def obj(p):
            inner = [(p[3 * i], p[3 * i + 1], p[3 * i + 2]) for i in range(N)]
            verts_list = [hex_verts(x, y, a) for x, y, a in inner]
            side = outer_side_from_verts(verts_list, ocx, ocy, oang)
            # penalty for overlaps
            pen = 0.0
            for i in range(N):
                for j in range(i + 1, N):
                    if sat_overlap(verts_list[i], verts_list[j]):
                        pen += 5.0
            return side + pen
        return obj

    def make_objective_all(p):
        """Full objective: inner positions + outer center/angle. p = [x0,y0,a0,...,x11,y11,a11, ocx, ocy, oang]."""
        inner = [(p[3 * i], p[3 * i + 1], p[3 * i + 2]) for i in range(N)]
        ocx, ocy, oang = p[3 * N], p[3 * N + 1], p[3 * N + 2]
        verts_list = [hex_verts(x, y, a) for x, y, a in inner]
        side = outer_side_from_verts(verts_list, ocx, ocy, oang)
        pen = 0.0
        for i in range(N):
            for j in range(i + 1, N):
                if sat_overlap(verts_list[i], verts_list[j]):
                    pen += 5.0
        return side + pen

    def p_from_inner(inner):
        p = np.zeros(3 * N + 3)
        for i in range(N):
            p[3 * i] = inner[i][0]
            p[3 * i + 1] = inner[i][1]
            p[3 * i + 2] = inner[i][2]
        return p

    # ---------- optimize inner given outer ----------
    def optimize_inner(inner, ocx, ocy, oang, max_iter=3000):
        x0 = p_from_inner(inner)
        obj = make_objective(ocx, ocy, oang)
        try:
            res = minimize(obj, x0, method='Nelder-Mead',
                           options={'maxiter': max_iter, 'maxfev': max_iter * 3,
                                    'xatol': 1e-5, 'fatol': 1e-6, 'adaptive': True})
            inner_res = [(res.x[3 * i], res.x[3 * i + 1], res.x[3 * i + 2]) for i in range(N)]
            side = outer_side_from_verts([hex_verts(x, y, a) for x, y, a in inner_res], ocx, ocy, oang)
            if validity(inner_res, ocx, ocy, oang, side * 1.001):
                return inner_res, side
            return None
        except Exception:
            return None

    # ---------- optimize outer given inner ----------
    def optimize_outer(inner, ocx, ocy, oang, max_iter=2000):
        def obj(p):
            return outer_side_from_verts([hex_verts(x, y, a) for x, y, a in inner], p[0], p[1], p[2])
        x0 = np.array([ocx, ocy, oang])
        try:
            res = minimize(obj, x0, method='Nelder-Mead',
                           options={'maxiter': max_iter, 'maxfev': max_iter * 2,
                                    'xatol': 1e-6, 'fatol': 1e-7})
            no, ne, na = res.x
            side = outer_side_from_verts([hex_verts(x, y, a) for x, y, a in inner], no, ne, na)
            if validity(inner, no, ne, na, side * 1.001):
                return no, ne, na, side
            return ocx, ocy, oang, outer_side_from_verts([hex_verts(x, y, a) for x, y, a in inner], ocx, ocy, oang)
        except Exception:
            return ocx, ocy, oang, outer_side_from_verts([hex_verts(x, y, a) for x, y, a in inner], ocx, ocy, oang)

    # ---------- full joint optimization ----------
    def optimize_joint(inner, ocx, ocy, oang, max_iter=5000):
        x0 = np.concatenate([p_from_inner(inner), [ocx, ocy, oang]])
        try:
            res = minimize(make_objective_all, x0, method='Nelder-Mead',
                           options={'maxiter': max_iter, 'maxfev': max_iter * 2,
                                    'xatol': 1e-5, 'fatol': 1e-6, 'adaptive': True})
            inner_res = [(res.x[3 * i], res.x[3 * i + 1], res.x[3 * i + 2]) for i in range(N)]
            no, ne, na = res.x[3 * N], res.x[3 * N + 1], res.x[3 * N + 2]
            side = outer_side_from_verts([hex_verts(x, y, a) for x, y, a in inner_res], no, ne, na)
            if validity(inner_res, no, ne, na, side * 1.001):
                return inner_res, no, ne, na, side
            return None
        except Exception:
            return None

    # ---------- main search ----------
    topologies = [topo_honeycomb, topo_ring12, topo_two_rings, topo_hex_cluster, topo_rect]
    best = None  # (side, inner, ocx, ocy, oang)

    for build_fn in topologies:
        if time.time() > deadline - 15:
            break
        inner, ocx, ocy, oang = build_fn()
        verts_list = [hex_verts(x, y, a) for x, y, a in inner]
        side = outer_side_from_verts(verts_list, ocx, ocy, oang)

        # check initial validity
        if all_disjoint(verts_list) and contains_all(verts_list, ocx, ocy, oang, side):
            if best is None or side < best[0]:
                best = (side, inner, ocx, ocy, oang)

        # optimize inner
        if time.time() > deadline - 10:
            break
        res = optimize_inner(inner, ocx, ocy, oang)
        if res is not None:
            inner_res, side_res = res
            if best is None or side_res < best[0]:
                best = (side_res, inner_res, ocx, ocy, oang)

    # If we have a best, polish outer and do joint
    if best is not None:
        side_b, inner_b, ocx_b, ocy_b, oang_b = best
        if time.time() < deadline - 10:
            no, ne, na, side_new = optimize_outer(inner_b, ocx_b, ocy_b, oang_b)
            if side_new < side_b:
                best = (side_new, inner_b, no, ne, na)

        if time.time() < deadline - 15:
            res = optimize_joint(inner_b, best[2], best[3], best[4], max_iter=3000)
            if res is not None:
                inner_r, no, ne, na, side_r = res
                if side_r < best[0]:
                    best = (side_r, inner_r, no, ne, na)

    # Fallback
    if best is None:
        inner, ocx, ocy, oang = topo_honeycomb()
        verts_list = [hex_verts(x, y, a) for x, y, a in inner]
        side = outer_side_from_verts(verts_list, ocx, ocy, oang) * 1.1
        best = (side, inner, ocx, ocy, oang)

    side_final, inner_final, ocx_f, ocy_f, oang_f = best
    inner_out = [[float(x), float(y), float(a)] for x, y, a in inner_final]
    final_side = side_final * 1.005 + 1e-4
    return inner_out, (float(ocx_f), float(ocy_f)), float(final_side), float(oang_f)
