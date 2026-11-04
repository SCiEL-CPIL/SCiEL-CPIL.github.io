"""Hexagon packing: 11 unit hexagons inside a minimal regular hexagon."""
import math
import time
import numpy as np

N = 11
DEADLINE = 140.0  # safety margin before 150s kill

# EVOLVE-BLOCK-START

def _hex_verts(cx, cy, ang_deg):
    """6 vertices of unit hexagon centered at (cx,cy), rotated ang_deg."""
    a = math.radians(ang_deg)
    return np.array([[cx + math.cos(a + i * math.pi / 3.0),
                      cy + math.sin(a + i * math.pi / 3.0)] for i in range(6)])

def _poly_area_sign(verts):
    """Signed area * 2 of polygon."""
    s = 0.0
    n = len(verts)
    for i in range(n):
        x1, y1 = verts[i]
        x2, y2 = verts[(i + 1) % n]
        s += x1 * y2 - x2 * y1
    return s

def _point_in_convex(pt, poly):
    """Test if point is inside convex polygon (CCW). Returns True if inside or on edge."""
    n = len(poly)
    sign = None
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        cross = (x2 - x1) * (pt[1] - y1) - (y2 - y1) * (pt[0] - x1)
        if abs(cross) < 1e-12:
            # on edge, check if within segment
            dot = (pt[0] - x1) * (x2 - x1) + (pt[1] - y1) * (y2 - y1)
            if -1e-12 <= dot <= (x2-x1)**2 + (y2-y1)**2 + 1e-12:
                return True
            continue
        s = 1 if cross > 0 else -1
        if sign is None:
            sign = s
        elif s != sign:
            return False
    return True

def _minkowski_diff_hex(a_verts, b_verts):
    """
    Minkowski difference A - B for two convex hexagons.
    Returns a convex polygon (up to 12 verts) that is the set of all a - b.
    Two hexagons overlap iff origin is inside (A - B).
    """
    # Compute edges of A and B, merge by angle
    def edges(verts):
        n = len(verts)
        e = []
        for i in range(n):
            dx = verts[(i+1) % n][0] - verts[i][0]
            dy = verts[(i+1) % n][1] - verts[i][1]
            ang = math.atan2(dy, dx)
            e.append((ang, dx, dy))
        e.sort(key=lambda x: x[0])
        return e

    ea = edges(a_verts)
    eb = edges(b_verts)
    # Minkowski sum of A and (-B). -B has edges negated and reversed.
    # Edges of -B: for each edge (ang, dx, dy) of B, the edge of -B is (ang+pi, -dx, -dy)
    eb_neg = []
    for ang, dx, dy in eb:
        new_ang = ang + math.pi
        # normalize to [0, 2pi)
        while new_ang >= 2 * math.pi:
            new_ang -= 2 * math.pi
        while new_ang < 0:
            new_ang += 2 * math.pi
        eb_neg.append((new_ang, -dx, -dy))
    eb_neg.sort(key=lambda x: x[0])

    # Merge
    merged = []
    i = j = 0
    while i < len(ea) and j < len(eb_neg):
        if ea[i][0] <= eb_neg[j][0] + 1e-12:
            merged.append((ea[i][1], ea[i][2]))
            i += 1
        else:
            merged.append((eb_neg[j][1], eb_neg[j][2]))
            j += 1
    while i < len(ea):
        merged.append((ea[i][1], ea[i][2]))
        i += 1
    while j < len(eb_neg):
        merged.append((eb_neg[j][1], eb_neg[j][2]))
        j += 1

    # Build polygon: start at a_verts[0] - b_verts[0], follow merged edges
    start = a_verts[0] - b_verts[0]
    pts = [start]
    cur = start.copy()
    for dx, dy in merged:
        cur = cur + np.array([dx, dy])
        pts.append(cur)
    # Remove last (should equal start)
    if np.linalg.norm(pts[-1] - pts[0]) < 1e-9:
        pts = pts[:-1]
    return np.array(pts)

def _hex_overlap(a_verts, b_verts):
    """Return True if two hexagons overlap (interiors intersect)."""
    md = _minkowski_diff_hex(a_verts, b_verts)
    # Check if origin is inside md
    if len(md) < 3:
        return False
    return _point_in_convex(np.array([0.0, 0.0]), md)

def _min_enclosing_side(centers, angles, outer_ang_deg=0.0):
    """
    Minimal side length of outer hexagon (at given rotation) containing all inner hexagons.
    For a regular hexagon of side S, the width in each of the 6 normal directions is S*sqrt(3).
    We need: for each normal n_k, max_i (v · n_k) - min_i (v · n_k) <= S * sqrt(3).
    So S = max_k (max_i v·n_k - min_i v·n_k) / sqrt(3).
    """
    a = math.radians(outer_ang_deg)
    normals = np.array([[math.cos(a + k * math.pi / 3.0),
                         math.sin(a + k * math.pi / 3.0)] for k in range(6)])
    # For each inner hexagon, project all 6 vertices
    all_proj = []
    for i in range(N):
        verts = _hex_verts(centers[i][0], centers[i][1], angles[i])
        proj = verts @ normals.T  # (6, 6)
        all_proj.append(proj)
    all_proj = np.concatenate(all_proj, axis=0)  # (6N, 6)
    widths = np.max(all_proj, axis=0) - np.min(all_proj, axis=0)  # (6,)
    S = np.max(widths) / math.sqrt(3.0)
    return S

def _total_overlap_penalty(centers, angles):
    """Sum of overlap indicators (1 if overlapping) - used for annealing."""
    polys = [_hex_verts(centers[i][0], centers[i][1], angles[i]) for i in range(N)]
    pen = 0.0
    for i in range(N):
        for j in range(i + 1, N):
            if _hex_overlap(polys[i], polys[j]):
                pen += 1.0
    return pen

def _is_valid(centers, angles, S, outer_ang_deg=0.0, tol=1e-7):
    """Check validity: no overlaps, all inside outer hexagon of side S."""
    polys = [_hex_verts(centers[i][0], centers[i][1], angles[i]) for i in range(N)]
    # Overlap check
    for i in range(N):
        for j in range(i + 1, N):
            if _hex_overlap(polys[i], polys[j]):
                return False
    # Containment: all vertices of inner hexagons inside outer hexagon
    outer_verts = _hex_verts(0.0, 0.0, outer_ang_deg)
    # scale outer to side S: unit hex has side 1, so scale by S
    outer_verts = outer_verts * S
    for i in range(N):
        for v in polys[i]:
            if not _point_in_convex(v, outer_verts):
                return False
    return True

def _seed_flower():
    """Standard flower: center + 6 around + 4 in second ring gaps."""
    d = math.sqrt(3)
    centers = [[0.0, 0.0]]
    angles = [0.0]
    for i in range(6):
        th = i * math.pi / 3.0
        centers.append([d * math.cos(th), d * math.sin(th)])
        angles.append(0.0)
    # 4 second-ring positions at angles 30, 90, 150, 210 deg, distance 2d
    for k in range(4):
        th = math.pi / 6 + k * math.pi / 3.0
        centers.append([2 * d * math.cos(th), 2 * d * math.sin(th)])
        angles.append(0.0)
    centers = np.array(centers)
    angles = np.array(angles)
    S = _min_enclosing_side(centers, angles) * 1.01
    return centers, angles, S

def construct():
    t0 = time.time()

    # Start from seed
    centers, angles, S = _seed_flower()
    best_S = S
    best_centers = centers.copy()
    best_angles = angles.copy()

    # Verify seed
    if _is_valid(centers, angles, S):
        pass
    else:
        S = _min_enclosing_side(centers, angles) * 1.05
        best_S = S
        best_centers = centers.copy()
        best_angles = angles.copy()

    # Simulated annealing on positions and angles
    # Objective: S + big_penalty * overlap_count
    def objective(c, a):
        S_val = _min_enclosing_side(c, a)
        pen = _total_overlap_penalty(c, a)
        return S_val + 50.0 * pen

    def elapsed():
        return time.time() - t0

    # Annealing parameters
    T = 0.5
    T_min = 0.001
    alpha = 0.997
    step_pos = 0.3
    step_ang = 10.0

    cur_c = best_centers.copy()
    cur_a = best_angles.copy()
    cur_obj = objective(cur_c, cur_a)

    # Precompute current polys for speed in overlap
    # We'll recompute each iteration (N=11, 55 pairs, each Minkowski is cheap)

    while elapsed() < DEADLINE - 5.0:
        # Propose a move: perturb one hexagon's center and/or angle
        idx = int(np.random.randint(N))
        dc = np.array([np.random.normal(0, step_pos), np.random.normal(0, step_pos)])
        da = np.random.normal(0, step_ang)

        new_c = cur_c.copy()
        new_a = cur_a.copy()
        new_c[idx] = cur_c[idx] + dc
        new_a[idx] = np.mod(cur_a[idx] + da, 180.0)

        new_obj = objective(new_c, new_a)
        delta = new_obj - cur_obj

        if delta < 0 or np.random.rand() < math.exp(-delta / max(T, 1e-9)):
            cur_c = new_c
            cur_a = new_a
            cur_obj = new_obj

            # If no overlaps and S improved, update best
            if cur_obj < best_S + 1e-9:
                # check validity
                S_val = _min_enclosing_side(cur_c, cur_a)
                pen = _total_overlap_penalty(cur_c, cur_a)
                if pen == 0 and S_val < best_S:
                    best_S = S_val
                    best_centers = cur_c.copy()
                    best_angles = cur_a.copy()

        T = max(T * alpha, T_min)

        # Occasionally restart from best if stuck
        if int(np.random.rand() * 100) == 0 and cur_obj > best_S + 0.1:
            cur_c = best_centers + np.random.normal(0, 0.2, size=(N, 2))
            cur_a = np.mod(best_angles + np.random.normal(0, 5, size=N), 180.0)
            cur_obj = objective(cur_c, cur_a)

    # Local refinement: gradient-free Nelder-Mead on the best
    if elapsed() < DEADLINE - 2.0:
        from scipy.optimize import minimize
        def nm_obj(params):
            if elapsed() > DEADLINE - 1.0:
                return 1e10
            c = params[:2*N].reshape(N, 2)
            a = params[2*N:]
            S_val = _min_enclosing_side(c, a)
            pen = _total_overlap_penalty(c, a)
            return S_val + 100.0 * pen

        x0 = best_centers.flatten().tolist() + best_angles.tolist()
        try:
            res = minimize(nm_obj, x0, method='Nelder-Mead',
                           options={'xatol': 1e-6, 'fatol': 1e-6, 'maxiter': 2000, 'maxfev': 4000})
            if res.fun < best_S:
                c = res.x[:2*N].reshape(N, 2)
                a = res.x[2*N:]
                S_val = _min_enclosing_side(c, a)
                pen = _total_overlap_penalty(c, a)
                if pen == 0 and S_val < best_S:
                    best_S = S_val
                    best_centers = c
                    best_angles = a
        except Exception:
            pass

    # Final: recompute S exactly and add tiny margin for numerical safety
    final_S = _min_enclosing_side(best_centers, best_angles)
    # Check validity
    if not _is_valid(best_centers, best_angles, final_S * 1.001):
        final_S = final_S * 1.01
    else:
        final_S = final_S * 1.0005  # tiny margin

    # Format output
    inner = []
    for i in range(N):
        inner.append([float(best_centers[i][0]), float(best_centers[i][1]), float(best_angles[i])])

    return inner, (0.0, 0.0), float(final_S), 0.0

# EVOLVE-BLOCK-END
