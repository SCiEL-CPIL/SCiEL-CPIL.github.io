"""Optimized hexagon packing: minimize outer hexagon side length for N unit hexagons.

Structural change: the outer hexagon is ALWAYS analytically optimal for the current
cluster. The optimizer only moves the 3N inner-hexagon parameters (cx, cy, angle).
This removes 3 dimensions from the search and makes the objective smoother.

Uses:
- Fast support-function computation for the minimal enclosing regular hexagon.
- Multiple diverse seeds including hexagonal lattice, 3-4-3, and perturbed variants.
- L-BFGS-B on the reduced 3N-dim problem with smooth overlap penalty.
- Iterative local search with random restarts to escape local minima.
- Final analytical outer-hexagon tightening.
"""
import numpy as np
import time
from scipy.optimize import minimize

N = 11

# EVOLVE-BLOCK-START

def hexagon_vertices(cx, cy, s, angle_deg):
    """Return 6 vertices of a regular hexagon with side s, center (cx,cy), rotation angle_deg."""
    a = np.radians(angle_deg)
    angles = a + np.pi / 3 * np.arange(6)
    return np.column_stack([cx + s * np.cos(angles), cy + s * np.sin(angles)])

def polygons_overlap(v1, v2, tol=1e-9):
    """Check if two convex polygons overlap using SAT."""
    axes = []
    for v in (v1, v2):
        n = len(v)
        for i in range(n):
            edge = v[(i + 1) % n] - v[i]
            normal = np.array([-edge[1], edge[0]])
            norm = np.linalg.norm(normal)
            if norm > 1e-12:
                axes.append(normal / norm)

    for axis in axes:
        proj1 = np.dot(v1, axis)
        proj2 = np.dot(v2, axis)
        min1, max1 = proj1.min(), proj1.max()
        min2, max2 = proj2.min(), proj2.max()
        if max1 < min2 - tol or max2 < min1 - tol:
            return False, axis
    return True, None

def convex_hull(points):
    """Andrew's monotone chain. Returns hull vertices (counter-clockwise, no last=first repeat)."""
    pts = np.unique(np.round(points, 12), axis=0)
    if len(pts) <= 2:
        return pts
    pts = pts[np.lexsort((pts[:, 1], pts[:, 0]))]

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper = []
    for p in pts[::-1]:
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return np.vstack([lower[:-1], upper[:-1]])

def hexagon_distance(p1, a1, p2, a2):
    """Compute minimum distance between two unit hexagons (0 if overlapping)."""
    d = np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)
    if d > 4.0:
        return d - 2.0
    v1 = hexagon_vertices(p1[0], p1[1], 1.0, a1)
    v2 = hexagon_vertices(p2[0], p2[1], 1.0, a2)
    overlap, axis = polygons_overlap(v1, v2, tol=1e-10)
    if not overlap:
        min_dist = float('inf')
        for i in range(6):
            for j in range(6):
                d_ij = np.linalg.norm(v1[i] - v2[j])
                if d_ij < min_dist:
                    min_dist = d_ij
        return min_dist
    if axis is not None:
        proj1 = np.dot(v1, axis)
        proj2 = np.dot(v2, axis)
        min1, max1 = proj1.min(), proj1.max()
        min2, max2 = proj2.min(), proj2.max()
        overlap_depth = min(max1, max2) - max(min1, min2)
        return -overlap_depth
    return -0.5

def compute_min_outer_fast(inner_hexagons):
    """Fast analytical computation of the minimal enclosing regular hexagon.
    
    For a fixed cluster, the minimal outer hexagon side length S is determined by
    the maximum over all 6 face-normal directions of the projection width of the
    cluster's convex hull. The optimal outer angle minimizes this max-width.
    
    Returns (S, ox, oy, outer_angle_deg).
    """
    # Gather all vertices
    all_verts = []
    for cx, cy, ang in inner_hexagons:
        all_verts.append(hexagon_vertices(cx, cy, 1.0, ang))
    all_verts = np.vstack(all_verts)

    hull = convex_hull(all_verts)
    
    # For a regular hexagon of side S, apothem r = S*sqrt(3)/2, so S = 2*r/sqrt(3).
    # The 6 face normals are at angles oa + 30 + 60k degrees.
    # Required apothem for normal n: max projection of hull onto n (with center at hull centroid).
    # But we can shift center. The width along normal n is proj_max - proj_min.
    # The apothem needed is width/2. So S = max_k (width_k) / sqrt(3).
    # We need to find the oa that minimizes max_k width_k.
    
    # Sample angles
    best_S = float('inf')
    best_oa = 0.0
    
    for oa_deg in np.arange(0.0, 60.0, 1.0):
        oa = np.radians(oa_deg)
        max_w = 0.0
        for k in range(6):
            theta = oa + np.radians(30 + 60 * k)
            nrm = np.array([np.cos(theta), np.sin(theta)])
            proj = np.dot(hull, nrm)
            w = proj.max() - proj.min()
            if w > max_w:
                max_w = w
        S_candidate = max_w / np.sqrt(3.0)
        if S_candidate < best_S - 1e-12:
            best_S = S_candidate
            best_oa = oa

    # Ternary search refinement
    lo = best_oa - np.radians(1.0)
    hi = best_oa + np.radians(1.0)
    for _ in range(25):
        m1 = lo + (hi - lo) / 3.0
        m2 = hi - (hi - lo) / 3.0
        def max_width(oa):
            mw = 0.0
            for k in range(6):
                theta = oa + np.radians(30 + 60 * k)
                nrm = np.array([np.cos(theta), np.sin(theta)])
                proj = np.dot(hull, nrm)
                w = proj.max() - proj.min()
                if w > mw:
                    mw = w
            return mw
        w1, w2 = max_width(m1), max_width(m2)
        if w1 < w2:
            hi = m2
        else:
            lo = m1
    best_oa = (lo + hi) / 2.0

    # Final S
    max_w = 0.0
    for k in range(6):
        theta = best_oa + np.radians(30 + 60 * k)
        nrm = np.array([np.cos(theta), np.sin(theta)])
        proj = np.dot(hull, nrm)
        w = proj.max() - proj.min()
        if w > max_w:
            max_w = w
    best_S = max_w / np.sqrt(3.0)

    # Center: use centroid of hull, then refine by subgradient
    c = hull.mean(axis=0)
    for _ in range(40):
        worst_val = -float('inf')
        worst_nrm = None
        for k in range(6):
            theta = best_oa + np.radians(30 + 60 * k)
            nrm = np.array([np.cos(theta), np.sin(theta)])
            proj = np.dot(hull - c, nrm)
            val = proj.max()
            if val > worst_val:
                worst_val = val
                worst_nrm = nrm
        c = c + 0.02 * worst_nrm

    # Recompute final S with refined center
    max_w = 0.0
    for k in range(6):
        theta = best_oa + np.radians(30 + 60 * k)
        nrm = np.array([np.cos(theta), np.sin(theta)])
        proj = np.dot(hull - c, nrm)
        w = proj.max() - proj.min()
        if w > max_w:
            max_w = w
    best_S = max_w / np.sqrt(3.0)

    return best_S, float(c[0]), float(c[1]), float(np.degrees(best_oa) % 360)

def compute_outer_side_length(inner_hexagons, outer_center, outer_angle):
    """Compute outer hexagon side length that contains all inner hexagons (for reporting)."""
    ox, oy = outer_center
    oa = np.radians(outer_angle)
    normals = []
    for k in range(6):
        theta = oa + np.radians(30 + 60 * k)
        normals.append(np.array([np.cos(theta), np.sin(theta)]))
    min_S = 0.0
    oc = np.array([ox, oy])
    for cx, cy, ang in inner_hexagons:
        verts = hexagon_vertices(cx, cy, 1.0, ang)
        for v in verts:
            for nrm in normals:
                d = np.dot(v - oc, nrm)
                if d > 0:
                    S_needed = 2.0 * d / np.sqrt(3.0)
                    if S_needed > min_S:
                        min_S = S_needed
    return min_S

def check_validity(inner_hexagons, outer_center, outer_side, outer_angle):
    """Check if the configuration is valid: no overlaps, all inside outer hexagon."""
    ox, oy = outer_center

    for i in range(len(inner_hexagons)):
        for j in range(i + 1, len(inner_hexagons)):
            cx1, cy1, a1 = inner_hexagons[i]
            cx2, cy2, a2 = inner_hexagons[j]
            v1 = hexagon_vertices(cx1, cy1, 1.0, a1)
            v2 = hexagon_vertices(cx2, cy2, 1.0, a2)
            overlap, _ = polygons_overlap(v1, v2, tol=1e-8)
            if overlap:
                return False

    outer_verts = hexagon_vertices(ox, oy, outer_side, outer_angle)

    for cx, cy, ang in inner_hexagons:
        verts = hexagon_vertices(cx, cy, 1.0, ang)
        for v in verts:
            px, py = v
            inside = True
            for i in range(6):
                v1 = outer_verts[i]
                v2 = outer_verts[(i + 1) % 6]
                edge = v2 - v1
                normal = np.array([-edge[1], edge[0]])
                d1 = np.dot(normal, np.array([px, py]) - v1)
                d2 = np.dot(normal, outer_verts[(i + 2) % 6] - v1)
                if d1 * d2 < 0:
                    inside = False
                    break
            if not inside:
                return False

    return True

def objective_reduced(params, deadline):
    """Objective: minimal outer side length + overlap penalty.
    Only 3N parameters (inner hexagons). Outer hexagon is analytically optimal."""
    if time.time() > deadline:
        return float('inf')

    inner = []
    for i in range(N):
        cx = params[3 * i]
        cy = params[3 * i + 1]
        ang = np.degrees(params[3 * i + 2]) % 360
        inner.append((cx, cy, ang))

    S, _, _, _ = compute_min_outer_fast(inner)

    penalty = 0.0
    for i in range(N):
        for j in range(i + 1, N):
            cx1, cy1, a1 = inner[i]
            cx2, cy2, a2 = inner[j]
            d = np.sqrt((cx1 - cx2) ** 2 + (cy1 - cy2) ** 2)
            if d < 2.5:
                dist = hexagon_distance(np.array([cx1, cy1]), a1, np.array([cx2, cy2]), a2)
                if dist < 0:
                    penalty += dist ** 2 * 500.0
                else:
                    penalty += max(0.0, 0.3 - dist) ** 2 * 50.0

    return S + penalty

def _build_seeds():
    """Build diverse initial conditions as 3N-dim parameter vectors."""
    seeds = []
    s3 = np.sqrt(3)

    def make_p3n(positions, angles=None):
        p = np.zeros(3 * N)
        if angles is None:
            angles = [0.0] * N
        for i, (px, py) in enumerate(positions):
            p[3 * i] = px
            p[3 * i + 1] = py
            p[3 * i + 2] = np.radians(angles[i])
        return p

    # Seed 1: Standard hexagonal lattice (1 + 6 + 4)
    hex_pos = [(0.0, 0.0)]
    for k in range(6):
        a = np.radians(60 * k)
        hex_pos.append((s3 * np.cos(a), s3 * np.sin(a)))
    for k in range(6):
        a = np.radians(60 * k)
        hex_pos.append((2 * s3 * np.cos(a), 2 * s3 * np.sin(a)))
        if len(hex_pos) >= N:
            break
    hex_pos = hex_pos[:N]
    seeds.append(make_p3n(hex_pos))

    # Seed 2: Perturbed lattice
    np.random.seed(42)
    seeds.append(make_p3n(hex_pos) + np.random.randn(3 * N) * 0.15)

    # Seed 3: Rotated inner hexagons
    seeds.append(make_p3n(hex_pos, angles=[30.0] * N))

    # Seed 4: 3-4-3 arrangement
    try:
        row_y = [0.0, s3, 2 * s3]
        cluster = []
        for m in range(-1, 2):
            cluster.append((m * 2.0, row_y[2]))
        for m in range(-1, 3):
            cluster.append((m * 2.0 + 1.0, row_y[1]))
        for m in range(-1, 2):
            cluster.append((m * 2.0, row_y[0]))
        cluster = cluster[:N]
        while len(cluster) < N:
            cluster.append((0.0, 3 * s3))
        cx_m = np.mean([c[0] for c in cluster])
        cy_m = np.mean([c[1] for c in cluster])
        cluster = [(c[0] - cx_m, c[1] - cy_m) for c in cluster]
        seeds.append(make_p3n(cluster))
    except Exception:
        pass

    # Seed 5: Tighter lattice
    tight = [(x * 0.97, y * 0.97) for x, y in hex_pos]
    seeds.append(make_p3n(tight))

    # Seed 6: Defect - move one hexagon
    dp = hex_pos.copy()
    dp[10] = (1.5, 1.5)
    seeds.append(make_p3n(dp))

    # Seed 7: Random
    np.random.seed(123)
    rp = [(x + np.random.randn() * 0.4, y + np.random.randn() * 0.4) for x, y in hex_pos]
    seeds.append(make_p3n(rp))

    # Seed 8: Linear
    lp = [(i * 2.01 - 10.05, 0.0) for i in range(N)]
    seeds.append(make_p3n(lp))

    # Seed 9: Hexagonal lattice with 15-degree rotation of all
    seeds.append(make_p3n(hex_pos, angles=[15.0] * N))

    # Seed 10: Alternate rotation pattern
    alts = [0.0, 30.0] * 6
    seeds.append(make_p3n(hex_pos, angles=alts[:N]))

    return seeds

def construct():
    """Main entry point: find optimal packing of N unit hexagons in minimum outer hexagon."""
    start_time = time.time()
    deadline = start_time + 145.0

    seeds = _build_seeds()

    best_S = float('inf')
    best_params = None
    best_result = None

    def _make_result(params):
        inner = []
        for i in range(N):
            cx = params[3 * i]
            cy = params[3 * i + 1]
            ang = np.degrees(params[3 * i + 2]) % 360
            inner.append([cx, cy, ang])
        S, ox, oy, oa = compute_min_outer_fast(inner)
        return inner, (ox, oy), S, oa

    # Phase 1: Evaluate seeds analytically
    for seed in seeds:
        if time.time() > deadline - 60:
            break
        try:
            inner = []
            for i in range(N):
                cx = seed[3 * i]
                cy = seed[3 * i + 1]
                ang = np.degrees(seed[3 * i + 2]) % 360
                inner.append((cx, cy, ang))
            S, ox, oy, oa = compute_min_outer_fast(inner)
            if check_validity(inner, (ox, oy), S, oa) and S < best_S:
                best_S = S
                best_params = seed.copy()
                best_result = (inner, (ox, oy), S, oa)
        except Exception:
            pass

    # Phase 2: L-BFGS-B from each seed (reduced 3N-dim problem)
    for si, seed in enumerate(seeds):
        if time.time() > deadline - 30:
            break
        try:
            res = minimize(objective_reduced, seed.copy(), args=(deadline,),
                           method='L-BFGS-B',
                           options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10})
            if res.fun < best_S:
                inner, (ox, oy), S, oa = _make_result(res.x)
                if check_validity(inner, (ox, oy), S, oa) and S < best_S:
                    best_S = S
                    best_params = res.x.copy()
                    best_result = (inner, (ox, oy), S, oa)
        except Exception:
            pass

    # Fallback
    if best_result is None:
        s3 = np.sqrt(3)
        hex_pos = [(0.0, 0.0)]
        for k in range(6):
            a = np.radians(60 * k)
            hex_pos.append((s3 * np.cos(a), s3 * np.sin(a)))
        for k in range(6):
            a = np.radians(60 * k)
            hex_pos.append((2 * s3 * np.cos(a), 2 * s3 * np.sin(a)))
            if len(hex_pos) >= N:
                break
        hex_pos = hex_pos[:N]
        inner = [[px, py, 0.0] for px, py in hex_pos]
        S, ox, oy, oa = compute_min_outer_fast(inner)
        best_result = (inner, (ox, oy), S, oa)
        best_S = S
        best_params = np.zeros(3 * N)
        for i, (px, py) in enumerate(hex_pos):
            best_params[3*i] = px
            best_params[3*i+1] = py

    # Phase 3: Iterative local search with random restarts
    if time.time() < deadline - 15:
        np.random.seed(456)
        for restart in range(8):
            if time.time() > deadline - 15:
                break
            try:
                # Start from best + random perturbation
                scale = 0.3 * (1.0 - restart / 10.0) + 0.1
                perturbed = best_params + np.random.randn(3 * N) * scale
                # Keep angles in reasonable range
                perturbed[2::3] = np.radians((np.degrees(perturbed[2::3]) % 360))

                res = minimize(objective_reduced, perturbed, args=(deadline,),
                               method='L-BFGS-B',
                               options={'maxiter': 150, 'ftol': 1e-12, 'gtol': 1e-10})
                if res.fun < best_S:
                    inner, (ox, oy), S, oa = _make_result(res.x)
                    if check_validity(inner, (ox, oy), S, oa) and S < best_S:
                        best_S = S
                        best_params = res.x.copy()
                        best_result = (inner, (ox, oy), S, oa)
            except Exception:
                pass

    # Phase 4: Final polish
    if time.time() < deadline - 3:
        try:
            res = minimize(objective_reduced, best_params.copy(), args=(deadline,),
                           method='L-BFGS-B',
                           options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-12})
            if res.fun < best_S:
                inner, (ox, oy), S, oa = _make_result(res.x)
                if check_validity(inner, (ox, oy), S, oa) and S < best_S:
                    best_S = S
                    best_params = res.x.copy()
                    best_result = (inner, (ox, oy), S, oa)
        except Exception:
            pass

    # Final analytical outer tightening
    if time.time() < deadline - 1:
        try:
            inner_fixed = best_result[0]
            S_an, ox_an, oy_an, oa_an = compute_min_outer_fast(inner_fixed)
            if check_validity(inner_fixed, (ox_an, oy_an), S_an, oa_an) and S_an < best_S:
                best_S = S_an
                best_result = (inner_fixed, (ox_an, oy_an), S_an, oa_an)
        except Exception:
            pass

    return best_result

# EVOLVE-BLOCK-END
