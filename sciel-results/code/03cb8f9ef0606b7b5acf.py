"""Rewritten: 1D relative-rotation sweep for rigid hexagonal cluster."""
import time
import math
import numpy as np

N = 11

def construct():
    start_time = time.time()
    deadline = start_time + 140.0

    # Build base cluster: 1 center + 6 ring-1 (dist sqrt(3)) + 4 ring-2 (dist 3)
    centers = [(0.0, 0.0)]
    d1 = math.sqrt(3.0)
    for k in range(6):
        a = math.radians(k * 60.0)
        centers.append((d1 * math.cos(a), d1 * math.sin(a)))
    d2 = 3.0
    for k in range(4):
        a = math.radians(30.0 + k * 60.0)
        centers.append((d2 * math.cos(a), d2 * math.sin(a)))

    # Unit hexagon vertex offsets at base orientation 30 deg
    base_a = math.radians(30.0)
    offs = np.array([[math.cos(base_a + k * math.pi / 3.0),
                      math.sin(base_a + k * math.pi / 3.0)] for k in range(6)])

    # All 66 vertices in local coords
    verts_local = []
    for (cx, cy) in centers:
        for (ox, oy) in offs:
            verts_local.append((cx + ox, cy + oy))
    verts_local = np.array(verts_local)  # (66,2)

    # Precompute cos/sin for a dense set of relative rotations
    # The relative rotation delta in [0,60). We'll sweep densely.
    # For a given delta, the cluster is rotated by delta (w.r.t. base), and we set outer
    # rotation such that the relative orientation is delta. Since only delta matters,
    # fix outer_angle = 0 and rotate cluster by delta. Then outer normals at 30+60k.
    # Wait: if outer is at angle 0, its normals are at 0+30+60k = 30,90,...,330.
    # The cluster rotated by delta has vertices rotate_points(verts_local, delta).

    def required_side(delta_deg):
        a = math.radians(delta_deg)
        c, s = math.cos(a), math.sin(a)
        R = np.array([[c, -s], [s, c]])
        verts = verts_local @ R.T
        # outer normals at 30,90,...,330 degrees (outer angle 0)
        na = np.array([math.radians(30.0 + k * 60.0) for k in range(6)])
        normals = np.column_stack([np.cos(na), np.sin(na)])
        proj = verts @ normals.T  # (66,6)
        h = proj.max(axis=0)
        return h.max() / (math.sqrt(3.0) / 2.0)

    # Dense sweep
    n_pts = 36000
    deltas = np.linspace(0.0, 60.0, n_pts, endpoint=False)
    # Vectorize: compute all at once? That's 36000 rotations of 66x2 -> 36000x66x2, too big.
    # Do it in chunks.
    best_side = float('inf')
    best_delta = 0.0
    chunk = 2000
    for start in range(0, n_pts, chunk):
        d_chunk = deltas[start:start+chunk]
        # For each delta, compute required side
        # Build rotation matrices (chunk,2,2)
        a = np.radians(d_chunk)
        c = np.cos(a); s = np.sin(a)
        # verts (66,2). For each delta, verts @ R.T
        # R for delta: [[c,-s],[s,c]], R.T = [[c,s],[-s,c]]
        # verts @ R.T: (66,2) @ (2,2) per delta
        # Do via einsum: result[i, m, n] = sum_k verts[m,k] * R_T[i,k,n]
        # R_T[i] = [[c_i, s_i],[-s_i, c_i]]
        # result[i,m,n] = verts[m,0]*R_T[i,0,n] + verts[m,1]*R_T[i,1,n]
        R_T = np.stack([np.stack([c, s], axis=1), np.stack([-s, c], axis=1)], axis=1)  # (chunk,2,2)
        # result: (chunk, 66, 2)
        result = np.einsum('mn,ikn->imk', verts_local, R_T)
        # normals (6,2)
        na = np.array([math.radians(30.0 + k * 60.0) for k in range(6)])
        normals = np.column_stack([np.cos(na), np.sin(na)])  # (6,2)
        # proj: (chunk,66,6) = result @ normals.T  -> einsum 'imk,kl->iml'
        proj = np.einsum('imk,kl->iml', result, normals.T)
        h = proj.max(axis=1)  # (chunk,6)
        sides = h.max(axis=1) / (math.sqrt(3.0) / 2.0)  # (chunk,)
        idx = np.argmin(sides)
        if sides[idx] < best_side:
            best_side = float(sides[idx])
            best_delta = float(d_chunk[idx])

    # Refine around best_delta with finer grid
    for radius, n_fine in [(2.0, 1000), (0.5, 1000), (0.1, 1000)]:
        d_fine = np.linspace(best_delta - radius, best_delta + radius, n_fine)
        d_fine = np.mod(d_fine, 60.0)
        a = np.radians(d_fine)
        c = np.cos(a); s = np.sin(a)
        R_T = np.stack([np.stack([c, s], axis=1), np.stack([-s, c], axis=1)], axis=1)
        result = np.einsum('mn,ikn->imk', verts_local, R_T)
        na = np.array([math.radians(30.0 + k * 60.0) for k in range(6)])
        normals = np.column_stack([np.cos(na), np.sin(na)])
        proj = np.einsum('imk,kl->iml', result, normals.T)
        h = proj.max(axis=1)
        sides = h.max(axis=1) / (math.sqrt(3.0) / 2.0)
        idx = np.argmin(sides)
        best_side = float(sides[idx])
        best_delta = float(d_fine[idx])
        if time.time() > deadline - 10:
            break

    # Build final inner positions with cluster rotation = best_delta, outer angle = 0
    ca = math.radians(best_delta)
    cos_a, sin_a = math.cos(ca), math.sin(ca)
    inner = []
    for (cx, cy) in centers:
        rx = cx * cos_a - cy * sin_a
        ry = cx * sin_a + cy * cos_a
        inner.append((rx, ry, 30.0 + best_delta))

    eps = 1e-9
    final_side = best_side + eps
    outer_angle = 0.0

    # Validation
    def hex_vertices(cx, cy, side, angle_deg):
        a = math.radians(angle_deg)
        return [(cx + side * math.cos(a + k * math.pi / 3.0),
                 cy + side * math.sin(a + k * math.pi / 3.0)) for k in range(6)]

    def point_in_convex(p, poly):
        for i in range(len(poly)):
            x1, y1 = poly[i]; x2, y2 = poly[(i + 1) % len(poly)]
            if (x2 - x1) * (p[1] - y1) - (y2 - y1) * (p[0] - x1) < -1e-10:
                return False
        return True

    def polygon_area_signed(verts):
        area = 0.0; n = len(verts)
        for i in range(n):
            x1, y1 = verts[i]; x2, y2 = verts[(i + 1) % n]
            area += x1 * y2 - x2 * y1
        return area / 2.0

    def line_intersect(p1, p2, p3, p4):
        x1, y1 = p1; x2, y2 = p2; x3, y3 = p3; x4, y4 = p4
        denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        if abs(denom) < 1e-15:
            return p1
        t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
        return (x1 + t * (x2 - x1), y1 + t * (y2 - y1))

    def overlap_area(v1, v2):
        def clip(subject, cp):
            out = list(subject)
            for i in range(len(cp)):
                if not out: break
                inp = out; out = []; S = inp[-1]
                for E in inp:
                    if point_in_convex(E, cp):
                        if not point_in_convex(S, cp):
                            out.append(line_intersect(S, E, cp[i], cp[(i + 1) % len(cp)]))
                        out.append(E)
                    elif point_in_convex(S, cp):
                        out.append(line_intersect(S, E, cp[i], cp[(i + 1) % len(cp)]))
                    S = E
            return out
        cl = clip(v1, v2)
        return abs(polygon_area_signed(cl)) if len(cl) >= 3 else 0.0

    outer_verts = hex_vertices(0.0, 0.0, final_side, outer_angle)
    valid = True
    for (ix, iy, iang) in inner:
        iv = hex_vertices(ix, iy, 1.0, iang)
        for v in iv:
            if not point_in_convex(v, outer_verts):
                valid = False; break
        if not valid: break

    if valid and time.time() < deadline - 3:
        for i in range(N):
            vi = hex_vertices(inner[i][0], inner[i][1], 1.0, inner[i][2])
            for j in range(i + 1, N):
                vj = hex_vertices(inner[j][0], inner[j][1], 1.0, inner[j][2])
                if overlap_area(vi, vj) > 1e-7:
                    valid = False; break
            if not valid: break

    if not valid:
        # Fallback to seed
        inner_fb = [(x, y, 30.0) for (x, y) in centers]
        max_x = 2 * math.sqrt(3); max_y = 4.0
        side_fb = max(2 * max_x / math.sqrt(3), max_y) + 1e-9
        return [[p[0], p[1], p[2]] for p in inner_fb], (0.0, 0.0), side_fb, 30.0

    return [[p[0], p[1], p[2]] for p in inner], (0.0, 0.0), final_side, outer_angle
