"""Rigid composite: 11 unit hexagons on a hex lattice as one object (6 DOF).
Optimize outer hexagon (center, side, rotation) to tightly enclose the cluster,
then verify no inner overlaps. Score = outer side length (lower is better)."""
import time
import math
import numpy as np
from scipy.optimize import minimize

N = 11
_DEADLINE = time.time() + 130.0  # safety margin under 150s kill


def _hex_verts(cx, cy, s, ang_deg):
    a = math.radians(ang_deg)
    ca, sa = math.cos(a), math.sin(a)
    out = []
    for k in range(6):
        th = math.radians(60.0 * k)
        x = s * math.cos(th)
        y = s * math.sin(th)
        out.append((cx + ca * x - sa * y, cy + sa * x + ca * y))
    return out


def _point_in_hex(px, py, cx, cy, s, ang_deg):
    a = math.radians(ang_deg)
    ca, sa = math.cos(a), math.sin(a)
    dx, dy = px - cx, py - cy
    lx = ca * dx + sa * dy
    ly = -sa * dx + ca * dy
    for k in range(6):
        th = math.radians(60.0 * k)
        nx, ny = math.cos(th), math.sin(th)
        if nx * lx + ny * ly > s + 1e-9:
            return False
    return True


def _hexes_overlap(v1, v2):
    # v1,v2: 6x2 vertex arrays (absolute). True if interiors intersect.
    # Separation: check each edge of v1 against v2 (all points on one side)
    for i in range(6):
        a = v1[i]; b = v1[(i + 1) % 6]
        ex, ey = b[0] - a[0], b[1] - a[1]
        # normal
        nx, ny = -ey, ex
        # distance from a
        da = nx * (a[0] - a[0]) + ny * (a[1] - a[1])  # 0
        # check all v2 vertices on same side (<= da + eps)
        ok = True
        for j in range(6):
            d = nx * (v2[j][0] - a[0]) + ny * (v2[j][1] - a[1])
            if d > da + 1e-9:
                ok = False
                break
        if ok:
            return False
        # symmetric: v1 vs v2 edge
    for i in range(6):
        a = v2[i]; b = v2[(i + 1) % 6]
        ex, ey = b[0] - a[0], b[1] - a[1]
        nx, ny = -ey, ex
        da = 0.0
        ok = True
        for j in range(6):
            d = nx * (v1[j][0] - a[0]) + ny * (v1[j][1] - a[1])
            if d > da + 1e-9:
                ok = False
                break
        if ok:
            return False
    # If no separating axis found, check containment (one inside other)
    if _point_in_hex(v1[0][0], v1[0][1], * _center_of(v2), 1.0, 0.0):
        pass
    return True


def _center_of(v):
    return (v[:, 0].mean(), v[:, 1].mean())


def _hexes_disjoint(v1, v2):
    # Robust: overlap iff a vertex of one is strictly inside the other OR edges cross.
    # For convex hexagons, separation test via all 12 edge normals is exact.
    for i in range(6):
        a = v1[i]; b = v1[(i + 1) % 6]
        ex, ey = b[0] - a[0], b[1] - a[1]
        nx, ny = -ey, ex
        da = nx * (a[0] - a[0]) + ny * (a[1] - a[1])
        for j in range(6):
            d = nx * (v2[j][0] - a[0]) + ny * (v2[j][1] - a[1])
            if d > da + 1e-8:
                break
        else:
            return True  # separating axis found -> disjoint
    for i in range(6):
        a = v2[i]; b = v2[(i + 1) % 6]
        ex, ey = b[0] - a[0], b[1] - a[1]
        nx, ny = -ey, ex
        da = 0.0
        for j in range(6):
            d = nx * (v1[j][0] - a[0]) + ny * (v1[j][1] - a[1])
            if d > da + 1e-8:
                break
        else:
            return True
    return False  # no separating axis -> overlap


def _hex_vertices(cx, cy, s, ang_deg):
    a = math.radians(ang_deg)
    ca, sa = math.cos(a), math.sin(a)
    v = np.empty((6, 2))
    for k in range(6):
        th = math.radians(60.0 * k)
        x = s * math.cos(th); y = s * math.sin(th)
        v[k] = (cx + ca * x - sa * y, cy + sa * x + ca * y)
    return v


def _build_lattice(cx, cy, ang_deg):
    # 11 hexagons on hex lattice: center + 6 neighbors (dist 2) + 6 at dist 2*sqrt3
    # Actually compact cluster: center + 6 around (dist 2) + 6 more (dist 2*sqrt3)
    a = math.radians(ang_deg)
    ca, sa = math.cos(a), math.sin(a)
    centers = [(0.0, 0.0)]
    # ring 1: 6 hexes at distance 2 (tangent) at angles 0,60,...
    for k in range(6):
        th = math.radians(60.0 * k)
        centers.append((2.0 * math.cos(th), 2.0 * math.sin(th)))
    # ring 2: 6 hexes at distance 2*sqrt(3) at angles 30,90,...
    for k in range(6):
        th = math.radians(60.0 * k + 30.0)
        centers.append((2.0 * math.sqrt(3) * math.cos(th), 2.0 * math.sqrt(3) * math.sin(th)))
    pts = []
    for (lx, ly) in centers:
        pts.append((cx + ca * lx - sa * ly, cy + sa * lx + ca * ly))
    return pts


def _inner_vertices(cx, cy, ang_deg):
    return [_hex_vertices(p[0], p[1], 1.0, ang_deg) for p in _build_lattice(cx, cy, ang_deg)]


def _min_side_to_enclose(inner_verts, outer_ang_deg):
    # Given fixed outer rotation, find min side s and center (ox,oy) enclosing all.
    # Outer half-width in direction u (unit) is s * cos(angle between u and nearest normal).
    # For a regular hexagon with circumradius s (side s), support function h(u)=s*cos(phi)
    # where phi = angle between u and the nearest vertex direction. Vertices at 0,60,...
    # h(u) = s * cos(min over k of |angle(u)-60k|) ... actually support = s*cos(delta)
    # where delta = angular distance to nearest vertex. Max delta = 30deg.
    # So h(u) = s * cos(delta(u)) with delta in [0,30deg].
    # We need: for all points p, for all 6 outer normals n_k: n_k.(p - c) <= s*cos(0)=s
    # (normals align with vertex directions? No: face normals are at 30,90,...)
    # Let's use face normals: outer hexagon faces have outward normals at 30,90,...,330.
    # Constraint: max_p n.(p) - min_p n.(p) <= 2*s*cos(0)? For a hexagon, width along normal n
    # (which is a face normal) = 2*s (since face is at distance s from center along that normal).
    # So for each of the 6 face normals: (max n.p - min n.p) <= 2 s  => s >= (max-min)/2.
    # And center c = (max_p n.p + min_p n.p)/2 must be consistent across all 6 normals.
    # That consistency is the tricky part. Instead, optimize s and c directly.
    # We'll just do a 2D+1D search over c and s using the support constraints.
    # Simplest: brute force c on a grid around centroid, then s = max over normals of (max n.p - n.c).
    # For a candidate center c, min enclosing s (for fixed rotation) = max over 6 face normals n_k of
    #   (max_p n_k.p - n_k.c)  ... but also need n_k.c - min_p n_k.p <= s, i.e. s >= max(n_k.c - min, max - n_k.c)
    # So s(c) = max_k max( max_p n_k.p - n_k.c, n_k.c - min_p n_k.p ). Minimize over c.
    # This is convex in c! (max of linear functions). Use scipy.
    # Precompute per-normal max/min over all inner vertices.
    outer_a = math.radians(outer_ang_deg)
    ca, sa = math.cos(outer_a), math.sin(outer_a)
    normals = []
    for k in range(6):
        th = math.radians(60.0 * k + 30.0)  # face normals
        nx, ny = math.cos(th), math.sin(th)
        # rotate into global (outer already rotated by outer_ang? normals defined in outer frame)
        gx = ca * nx - sa * ny
        gy = sa * nx + ca * ny
        normals.append((gx, gy))
    all_pts = np.vstack(inner_verts)  # (36,2)
    proj = []
    for (gx, gy) in normals:
        vals = all_pts[:, 0] * gx + all_pts[:, 1] * gy
        proj.append((vals.max(), vals.min()))
    # s(c) = max_k max( max_k - n_k.c, n_k.c - min_k )
    # = max_k ( |n_k.c - (max_k+min_k)/2| + (max_k-min_k)/2 )
    def s_of_c(c):
        mx = 0.0
        for k in range(6):
            gx, gy = normals[k]
            nc = gx * c[0] + gy * c[1]
            mx_k = (proj[k][0] + proj[k][1]) / 2.0
            half = (proj[k][0] - proj[k][1]) / 2.0
            mx = max(mx, abs(nc - mx_k) + half)
        return mx
    # minimize s_of_c over c (convex). Start at centroid.
    c0 = all_pts.mean(axis=0).tolist()
    res = minimize(lambda c: s_of_c(c), c0, method='Nelder-Mead',
                   options={'maxiter': 500, 'xatol': 1e-4, 'fatol': 1e-6})
    return s_of_c(res.x)


def _check_disjoint(inner_verts):
    for i in range(len(inner_verts)):
        for j in range(i + 1, len(inner_verts)):
            if not _hexes_disjoint(inner_verts[i], inner_verts[j]):
                return False
    return True


def construct():
    best = None
    best_side = float('inf')
    # Search over inner rotation ang and outer rotation outer_ang, and inner center (0,0)
    # The inner cluster centroid can be anywhere; outer center optimized. By translation
    # invariance, fix inner centroid at origin. So inner center = (0,0).
    ang_candidates = list(range(0, 60, 5))
    outer_candidates = list(range(0, 60, 5))
    for ang in ang_candidates:
        if time.time() > _DEADLINE - 20:
            break
        inner_verts = _inner_vertices(0.0, 0.0, ang)
        if not _check_disjoint(inner_verts):
            continue
        for outer_ang in outer_candidates:
            if time.time() > _DEADLINE - 20:
                break
            s = _min_side_to_enclose(inner_verts, outer_ang)
            if s < best_side:
                best_side = s
                best = (ang, outer_ang)
    if best is None:
        # fallback: trivial valid layout
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0
    ang, outer_ang = best
    inner_verts = _inner_vertices(0.0, 0.0, ang)
    # recompute best center and side precisely
    all_pts = np.vstack(inner_verts)
    outer_a = math.radians(outer_ang)
    ca, sa = math.cos(outer_a), math.sin(outer_a)
    normals = []
    for k in range(6):
        th = math.radians(60.0 * k + 30.0)
        gx = ca * math.cos(th) - sa * math.sin(th)
        gy = sa * math.cos(th) + ca * math.sin(th)
        normals.append((gx, gy))
    proj = []
    for (gx, gy) in normals:
        vals = all_pts[:, 0] * gx + all_pts[:, 1] * gy
        proj.append((vals.max(), vals.min()))
    def s_of_c(c):
        mx = 0.0
        for k in range(6):
            gx, gy = normals[k]
            nc = gx * c[0] + gy * c[1]
            mx_k = (proj[k][0] + proj[k][1]) / 2.0
            half = (proj[k][0] - proj[k][1]) / 2.0
            mx = max(mx, abs(nc - mx_k) + half)
        return mx
    c0 = all_pts.mean(axis=0).tolist()
    res = minimize(lambda c: s_of_c(c), c0, method='Nelder-Mead',
                   options={'maxiter': 2000, 'xatol': 1e-6, 'fatol': 1e-8})
    oc = res.x
    s = s_of_c(oc)
    # Build inner list
    pts = _build_lattice(0.0, 0.0, ang)
    inner = [[p[0], p[1], float(ang)] for p in pts]
    return inner, (float(oc[0]), float(oc[1])), float(s), float(outer_ang)
