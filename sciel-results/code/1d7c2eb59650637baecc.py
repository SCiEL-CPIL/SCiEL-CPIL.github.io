# EVOLVE-BLOCK-START
import numpy as np
import time

def _hex_vertices(cx, cy, d, angle_deg):
    a = np.radians(angle_deg)
    angles = a + np.radians(60) * np.arange(6)
    return np.column_stack([cx + d * np.cos(angles), cy + d * np.sin(angles)])


def _cluster_centers(pattern):
    """Return (12,2) centers for a compact honeycomb cluster, centered at origin."""
    s3 = np.sqrt(3)
    if pattern == "3432":
        # 4 rows of 3,4,3,2 staggered; spacing 2 in x, s3 in y
        pts = []
        row_counts = [3, 4, 3, 2]
        for r, cnt in enumerate(row_counts):
            y = r * s3
            if cnt == 3:
                xs = [0.0, 2.0, 4.0]
            elif cnt == 4:
                xs = [-1.0, 1.0, 3.0, 5.0]
            elif cnt == 2:
                xs = [1.0, 3.0]
            else:
                xs = []
            for x in xs:
                pts.append([x, y])
        pts = np.array(pts)
    elif pattern == "2343":
        row_counts = [2, 3, 4, 3]
        pts = []
        for r, cnt in enumerate(row_counts):
            y = r * s3
            if cnt == 2:
                xs = [0.0, 2.0]
            elif cnt == 3:
                xs = [-1.0, 1.0, 3.0]
            elif cnt == 4:
                xs = [0.0, 2.0, 4.0, 6.0]
            else:
                xs = []
            for x in xs:
                pts.append([x, y])
        pts = np.array(pts)
    elif pattern == "444":
        pts = []
        for r in range(3):
            y = r * s3
            if r == 0:
                xs = [0.0, 2.0, 4.0, 6.0]
            elif r == 1:
                xs = [-1.0, 1.0, 3.0, 5.0]
            else:
                xs = [0.0, 2.0, 4.0, 6.0]
            for x in xs:
                pts.append([x, y])
        pts = np.array(pts)
    elif pattern == "2334":
        row_counts = [2, 3, 3, 4]
        pts = []
        for r, cnt in enumerate(row_counts):
            y = r * s3
            if cnt == 2:
                xs = [0.0, 2.0]
            elif cnt == 3:
                xs = [-1.0, 1.0, 3.0]
            elif cnt == 4:
                xs = [0.0, 2.0, 4.0, 6.0]
            else:
                xs = []
            for x in xs:
                pts.append([x, y])
        pts = np.array(pts)
    elif pattern == "3333":
        pts = []
        for r in range(4):
            y = r * s3
            if r % 2 == 0:
                xs = [0.0, 2.0, 4.0]
            else:
                xs = [-1.0, 1.0, 3.0]
            for x in xs:
                pts.append([x, y])
        pts = np.array(pts)
    elif pattern == "2433":
        row_counts = [2, 4, 3, 3]
        pts = []
        for r, cnt in enumerate(row_counts):
            y = r * s3
            if cnt == 2:
                xs = [0.0, 2.0]
            elif cnt == 4:
                xs = [-1.0, 1.0, 3.0, 5.0]
            elif cnt == 3:
                xs = [0.0, 2.0, 4.0]
            else:
                xs = []
            for x in xs:
                pts.append([x, y])
        pts = np.array(pts)
    elif pattern == "flower5":
        # center + 6 ring1 + 5 selected ring2
        pts = [[0.0, 0.0]]
        for i in range(6):
            a = np.radians(60 * i)
            pts.append([2.0 * np.cos(a), 2.0 * np.sin(a)])
        # ring2 at distance 4 in the 6 lattice directions + 30deg offsets; pick 5
        ring2 = []
        for i in range(6):
            a = np.radians(60 * i)
            ring2.append([4.0 * np.cos(a), 4.0 * np.sin(a)])
        for i in range(6):
            a = np.radians(60 * i + 30)
            ring2.append([4.0 * np.cos(a), 4.0 * np.sin(a)])
        # pick 5 that are spread out
        for i in [0, 2, 4, 7, 9]:
            pts.append(ring2[i])
        pts = np.array(pts)
    else:
        raise ValueError("unknown pattern")
    pts = pts - pts.mean(axis=0)
    return pts


def _all_inner_vertices(centers, inner_rot):
    """Return (72,2) vertices of 12 unit hexagons at given centers/rotation."""
    unit = _hex_vertices(0.0, 0.0, 1.0, inner_rot)  # (6,2)
    verts = np.tile(unit, (12, 1)) + centers[:, None, :]
    return verts.reshape(72, 2)


def _min_outer_side(centers, inner_rot, outer_rot):
    """
    Analytic minimum enclosing regular hexagon.
    Returns (s, offset) where offset is the outer center (in original coords).
    s = max over 6 outward normals of (max_proj + sqrt(3)/2).
    """
    a = np.radians(outer_rot)
    # outward normals of unit hexagon at angle a: perpendicular to each edge, pointing out
    # vertex i at angle a + 60i. edge i from vertex i to i+1.
    # outward normal of edge i points at angle a + 60i + 30
    n_ang = a + np.radians(60 * np.arange(6) + 30)
    normals = np.column_stack([np.cos(n_ang), np.sin(n_ang)])  # (6,2)

    verts = _all_inner_vertices(centers, inner_rot)  # (72,2)
    projs = verts @ normals.T  # (72,6)
    max_proj = projs.max(axis=0)  # (6,)
    s = (max_proj + np.sqrt(3) / 2.0).max()

    # outer center = (max_proj + sqrt(3)/2) along each normal direction (unit, since s*normal = offset)
    # offset = s * normal_k for the k that achieved max; all consistent in a tight fit
    # Use the mean of the per-edge implied centers: center_k = (max_proj[k] + sqrt(3)/2)*normal_k
    implied = (max_proj + np.sqrt(3) / 2.0)[:, None] * normals  # (6,2)
    offset = implied.mean(axis=0)
    return s, offset


def _validate_packing(inner, outer, s, eps=1e-6):
    if not np.all(np.isfinite(inner)) or not np.all(np.isfinite(outer)) or not np.isfinite(s):
        return False
    if s < np.sqrt(12) - 1e-9:
        return False
    ox, oy, oang = outer
    outer_verts = _hex_vertices(ox, oy, s, oang)
    for i in range(6):
        for j in range(i + 1, 6):
            if np.linalg.norm(outer_verts[i] - outer_verts[j]) < 1e-8:
                return False
    # unit edges
    for i in range(12):
        v = _hex_vertices(inner[i, 0], inner[i, 1], 1.0, inner[i, 2])
        for k in range(6):
            if abs(np.linalg.norm(v[(k + 1) % 6] - v[k]) - 1) > 1e-8:
                return False
    # separation
    for i in range(12):
        vi = _hex_vertices(inner[i, 0], inner[i, 1], 1.0, inner[i, 2])
        for j in range(i + 1, 12):
            vj = _hex_vertices(inner[j, 0], inner[j, 1], 1.0, inner[j, 2])
            normals = []
            for v in (vi, vj):
                for k in range(6):
                    e = v[(k + 1) % 6] - v[k]
                    n = np.array([-e[1], e[0]])
                    n = n / (np.linalg.norm(n) + 1e-15)
                    normals.append(n)
            sep = False
            for n in normals:
                pi = np.dot(vi, n); pj = np.dot(vj, n)
                if pi.max() + eps < pj.min() or pj.max() + eps < pi.min():
                    sep = True
                    break
            if not sep:
                return False
    # containment
    for i in range(12):
        v = _hex_vertices(inner[i, 0], inner[i, 1], 1.0, inner[i, 2])
        for pt in v:
            for k in range(6):
                p1 = outer_verts[k]; p2 = outer_verts[(k + 1) % 6]
                cross = (p2[0] - p1[0]) * (pt[1] - p1[1]) - (p2[1] - p1[1]) * (pt[0] - p1[0])
                if cross < -1e-6:
                    return False
    return True


def hexagon_packing_12():
    start = time.time()
    deadline = start + 600.0

    patterns = ["3432", "2343", "444", "2334", "3333", "2433", "flower5"]
    best_s = np.inf
    best_inner = None
    best_outer = None

    for pattern in patterns:
        if time.time() > deadline - 60:
            break
        base = _cluster_centers(pattern)

        for inner_rot in (0.0, 30.0):
            if time.time() > deadline - 30:
                break
            # coarse sweep
            coarse = np.linspace(0.0, 60.0, 61)
            best_local_s = np.inf
            best_local_rot = None
            for rot in coarse:
                s, off = _min_outer_side(base, inner_rot, rot)
                if s < best_local_s:
                    best_local_s = s
                    best_local_rot = rot
            # refine around best
            if best_local_rot is not None:
                fine = np.linspace(best_local_rot - 5.0, best_local_rot + 5.0, 201)
                for rot in fine:
                    s, off = _min_outer_side(base, inner_rot, rot)
                    if s < best_local_s:
                        best_local_s = s
                        best_local_rot = rot
            # final
            s, off = _min_outer_side(base, inner_rot, best_local_rot)
            centers = base + off
            inner = np.column_stack([centers, np.full(12, inner_rot)])
            outer = np.array([off[0], off[1], best_local_rot])
            if _validate_packing(inner, outer, s) and s < best_s:
                best_s = s
                best_inner = inner.copy()
                best_outer = outer.copy()

    if best_inner is None:
        base = _cluster_centers("3432")
        s, off = _min_outer_side(base, 0.0, 0.0)
        centers = base + off
        best_inner = np.column_stack([centers, np.zeros(12)])
        best_outer = np.array([off[0], off[1], 0.0])
        best_s = s

    return best_inner, best_outer, best_s


# EVOLVE-BLOCK-END
