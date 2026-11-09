"""Direct construction of 12 unit hexagons inside a regular hexagon, minimizing outer side."""
N = 12

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time
import math

def _hex_verts(cx, cy, side, angle_deg):
    """Vertices of a regular hexagon centered at (cx, cy) with given side and rotation."""
    a = np.radians(angle_deg)
    angles = a + np.radians(60 * np.arange(6))
    return np.column_stack([
        cx + side * np.cos(angles),
        cy + side * np.sin(angles)
    ])

def _hex_verts_all(params):
    """Compute vertices for all N inner hexagons from a 3N-length parameter vector."""
    verts = []
    for i in range(N):
        cx = params[3*i]
        cy = params[3*i+1]
        ang = params[3*i+2]
        verts.append(_hex_verts(cx, cy, 1.0, ang))
    return verts

def _outer_half_widths(outer_center, outer_side, outer_angle):
    """Return 6 (edge_normal, offset) for the outer hexagon: n·p >= offset means inside."""
    a = np.radians(outer_angle)
    verts = np.array([
        [outer_center[0] + outer_side * np.cos(a + np.radians(60 * i)),
         outer_center[1] + outer_side * np.sin(a + np.radians(60 * i))]
        for i in range(6)
    ])
    normals = []
    offsets = []
    for i in range(6):
        v1 = verts[i]
        v2 = verts[(i + 1) % 6]
        edge = v2 - v1
        L = np.linalg.norm(edge)
        if L < 1e-12:
            continue
        n = np.array([-edge[1], edge[0]]) / L
        c = np.array(outer_center, dtype=float)
        if np.dot(n, c - v1) < 0:
            n = -n
        offset = np.dot(n, v1)
        normals.append(n)
        offsets.append(offset)
    normals = np.array(normals)
    offsets = np.array(offsets)
    return normals, offsets

def _containment_penalty_vec(inner_verts_list, normals, offsets):
    """Sum over all inner vertices of max(0, offset - n·p) across the 6 outer edges."""
    total = 0.0
    for iv in inner_verts_list:
        M = iv @ normals.T - offsets
        viol = np.maximum(0.0, -M).max(axis=1)
        total += float(viol.sum())
    return total

def _sat_overlap_depth(h1, h2):
    """Minimum penetration depth between two convex polygons using SAT. Returns 0 if disjoint."""
    min_depth = float('inf')
    axes = []
    for poly in (h1, h2):
        n = len(poly)
        for i in range(n):
            p1 = poly[i]
            p2 = poly[(i + 1) % n]
            edge = p2 - p1
            norm = np.linalg.norm(edge)
            if norm < 1e-10:
                continue
            normal = np.array([-edge[1], edge[0]]) / norm
            axes.append(normal)

    for axis in axes:
        proj1 = h1 @ axis
        proj2 = h2 @ axis
        min1, max1 = proj1.min(), proj1.max()
        min2, max2 = proj2.min(), proj2.max()
        if max1 <= min2 or max2 <= min1:
            return 0.0
        overlap = min(max1, max2) - max(min1, min2)
        if overlap < min_depth:
            min_depth = overlap
    return min_depth

def _compute_penalty(params, outer_side, outer_center, outer_angle):
    """Compute all inner hex vertices and return containment + disjointness penalties."""
    inner_verts_list = _hex_verts_all(params)

    normals, offsets = _outer_half_widths(outer_center, outer_side, outer_angle)
    cont_pen = _containment_penalty_vec(inner_verts_list, normals, offsets)

    disj_pen = 0.0
    for i in range(N):
        for j in range(i + 1, N):
            depth = _sat_overlap_depth(inner_verts_list[i], inner_verts_list[j])
            if depth > 0:
                disj_pen += depth

    return cont_pen + disj_pen

def _compute_penalty_smooth(params, outer_side, outer_center, outer_angle, beta=50.0):
    """Smooth penalty using log-sum-exp for containment and soft-min for overlap."""
    inner_verts_list = _hex_verts_all(params)

    normals, offsets = _outer_half_widths(outer_center, outer_side, outer_angle)
    
    # Smooth containment: use soft-max instead of hard max
    total_cont = 0.0
    for iv in inner_verts_list:
        M = iv @ normals.T - offsets  # negative means violation
        viol = -M  # positive means violation
        # soft-max via log-sum-exp
        max_viol = viol.max()
        if max_viol > 0:
            total_cont += float(np.log(np.exp(beta * (viol - max_viol)).sum()) / beta + max_viol)
    
    # Smooth disjointness: use soft-min for overlap depth
    total_disj = 0.0
    for i in range(N):
        for j in range(i + 1, N):
            h1, h2 = inner_verts_list[i], inner_verts_list[j]
            min_overlap = float('inf')
            axes = []
            for poly in (h1, h2):
                n = len(poly)
                for k in range(n):
                    p1 = poly[k]
                    p2 = poly[(k + 1) % n]
                    edge = p2 - p1
                    norm = np.linalg.norm(edge)
                    if norm < 1e-10:
                        continue
                    normal = np.array([-edge[1], edge[0]]) / norm
                    axes.append(normal)
            
            overlaps = []
            for axis in axes:
                proj1 = h1 @ axis
                proj2 = h2 @ axis
                min1, max1 = proj1.min(), proj1.max()
                min2, max2 = proj2.min(), proj2.max()
                if max1 <= min2 or max2 <= min1:
                    overlaps.append(0.0)
                else:
                    overlaps.append(min(max1, max2) - max(min1, min2))
            
            if overlaps:
                # Use minimum overlap; if all positive, they overlap
                min_ov = min(overlaps)
                if min_ov > 0:
                    # Soft penalty: smooth the min using log-sum-exp on negatives
                    neg_ov = -np.array(overlaps)
                    max_neg = neg_ov.max()
                    if max_neg < 0:  # all overlaps positive
                        total_disj += float(-(np.log(np.exp(beta * (neg_ov - max_neg)).sum()) / beta + max_neg))
    
    return total_cont + total_disj



def construct():
    start_time = time.time()
    deadline = start_time + 135.0

    def build_hexagonal_lattice(nrows, ncols):
        positions = []
        for r in range(nrows):
            for c in range(ncols):
                x = c * 2.0 + (1.0 if r % 2 == 1 else 0.0)
                y = r * np.sqrt(3)
                positions.append((x, y))
        return positions

    def build_ring_config():
        positions = [(0.0, 0.0)]
        for i in range(6):
            ang = np.radians(60 * i)
            positions.append((2.0 * np.cos(ang), 2.0 * np.sin(ang)))
        for i in range(6):
            ang = np.radians(60 * i + 30)
            positions.append((3.464 * np.cos(ang), 3.464 * np.sin(ang)))
        return positions[:12]

    configs = []

    pos_4x3 = build_hexagonal_lattice(4, 3)
    if len(pos_4x3) == N:
        arr = np.array(pos_4x3)
        centroid = arr.mean(axis=0)
        configs.append((arr - centroid).tolist())

    pos_3x4 = build_hexagonal_lattice(3, 4)
    if len(pos_3x4) == N:
        arr = np.array(pos_3x4)
        centroid = arr.mean(axis=0)
        configs.append((arr - centroid).tolist())

    pos_2x6 = build_hexagonal_lattice(2, 6)
    if len(pos_2x6) == N:
        arr = np.array(pos_2x6)
        centroid = arr.mean(axis=0)
        configs.append((arr - centroid).tolist())

    ring_pos = build_ring_config()
    if len(ring_pos) == N:
        arr = np.array(ring_pos)
        centroid = arr.mean(axis=0)
        configs.append((arr - centroid).tolist())

    pos_6x2 = build_hexagonal_lattice(6, 2)
    if len(pos_6x2) == N:
        arr = np.array(pos_6x2)
        centroid = arr.mean(axis=0)
        configs.append((arr - centroid).tolist())

    line_pos = [(float(i - 5.5), 0.0) for i in range(12)]
    configs.append(line_pos)

    # Voronoi-cell-inspired initialization: place hexagons at centers of 12 Voronoi cells
    # of a regular hexagon. We approximate this by using a 3x4 lattice rotated to align
    # with the outer hexagon's symmetry, plus a ring configuration.
    # The key insight: by starting from configurations that respect the outer hexagon's
    # 6-fold symmetry, we give the optimizer a better landscape to navigate.

    # Add symmetry-respecting configurations
    for rot_angle in [0.0, 30.0]:
        for config in configs[:]:
            rot = np.radians(rot_angle)
            c = np.cos(rot)
            s = np.sin(rot)
            rotated = []
            for (x, y) in config:
                rx = x * c - y * s
                ry = x * s + y * c
                rotated.append((rx, ry))
            configs.append(rotated)

    best_score = float('inf')
    best_result = None

    # Phase 1: Quick L-BFGS-B search with multiple outer angles and sides
    outer_angles = [0.0, 30.0]
    outer_sides = [5.0, 4.5, 4.2, 4.0, 3.8, 3.6, 3.5, 3.4, 3.3, 3.2, 3.1]

    for outer_angle in outer_angles:
        for outer_side in outer_sides:
            if time.time() > deadline - 30.0:
                break

            outer_center = (0.0, 0.0)

            for config in configs:
                if time.time() > deadline - 25.0:
                    break

                init_params = []
                for (x, y) in config:
                    init_params.extend([float(x), float(y), 0.0])
                init_params = np.array(init_params)

                pen = _compute_penalty(init_params, outer_side, outer_center, outer_angle)

                if pen < 1e-6:
                    if outer_side < best_score:
                        best_score = outer_side
                        best_result = (init_params.copy(), outer_center, outer_side, outer_angle)
                    continue

                if time.time() > deadline - 25.0:
                    continue

                def objective(p):
                    return _compute_penalty(p, outer_side, outer_center, outer_angle)

                try:
                    result = minimize(
                        objective,
                        init_params,
                        method='L-BFGS-B',
                        options={'maxiter': 80, 'maxfun': 2500, 'ftol': 1e-14, 'gtol': 1e-10}
                    )
                    opt_params = result.x
                    opt_val = result.fun
                except Exception:
                    continue

                if opt_val < 1e-6:
                    if outer_side < best_score:
                        best_score = outer_side
                        best_result = (opt_params, outer_center, outer_side, outer_angle)
                    else:
                        # Try smooth penalty for better local optimization
                        if time.time() < deadline - 25.0:
                            def smooth_obj(p):
                                return _compute_penalty_smooth(p, outer_side, outer_center, outer_angle, beta=100.0)
                            try:
                                smooth_result = minimize(
                                    smooth_obj,
                                    opt_params,
                                    method='L-BFGS-B',
                                    options={'maxiter': 100, 'maxfun': 3000, 'ftol': 1e-14, 'gtol': 1e-11}
                                )
                                smooth_params = smooth_result.x
                                hard_pen = _compute_penalty(smooth_params, outer_side, outer_center, outer_angle)
                                if hard_pen < 1e-6:
                                    if outer_side < best_score:
                                        best_score = outer_side
                                        best_result = (smooth_params, outer_center, outer_side, outer_angle)
                                    else:
                                        opt_params = smooth_params
                            except Exception:
                                pass

    # Phase 2: Binary search for minimal side length with more iterations
    if best_result is not None and time.time() < deadline - 5.0:
        opt_params, outer_center, outer_side, outer_angle = best_result

        low = max(2.8, outer_side - 0.5)
        high = outer_side

        for _ in range(12):
            if time.time() > deadline - 5.0:
                break
            mid = (low + high) / 2.0
            if high - low < 1e-5:
                break

            # First try with hard penalty
            try:
                result = minimize(
                    lambda p, s=mid: _compute_penalty(p, s, outer_center, outer_angle),
                    opt_params,
                    method='L-BFGS-B',
                    options={'maxiter': 100, 'maxfun': 3000, 'ftol': 1e-15, 'gtol': 1e-12}
                )
                if result.fun < 1e-8:
                    high = mid
                    opt_params = result.x
                    outer_side = mid
                else:
                    # Try smooth penalty as backup
                    if time.time() < deadline - 5.0:
                        try:
                            smooth_result = minimize(
                                lambda p, s=mid: _compute_penalty_smooth(p, s, outer_center, outer_angle, beta=200.0),
                                opt_params,
                                method='L-BFGS-B',
                                options={'maxiter': 150, 'maxfun': 4000, 'ftol': 1e-14, 'gtol': 1e-11}
                            )
                            smooth_params = smooth_result.x
                            hard_pen = _compute_penalty(smooth_params, mid, outer_center, outer_angle)
                            if hard_pen < 1e-8:
                                high = mid
                                opt_params = smooth_params
                                outer_side = mid
                            else:
                                low = mid
                        except Exception:
                            low = mid
                    else:
                        low = mid
            except Exception:
                low = mid

        # Final high-precision polish at the found side length
        if time.time() < deadline - 3.0:
            # Try both hard and smooth penalties for final polish
            best_polish_params = opt_params
            best_polish_pen = _compute_penalty(opt_params, outer_side, outer_center, outer_angle)
            
            try:
                result = minimize(
                    lambda p: _compute_penalty(p, outer_side, outer_center, outer_angle),
                    opt_params,
                    method='L-BFGS-B',
                    options={'maxiter': 200, 'maxfun': 5000, 'ftol': 1e-15, 'gtol': 1e-13}
                )
                if result.fun < best_polish_pen and result.fun < 1e-8:
                    best_polish_params = result.x
                    best_polish_pen = result.fun
            except Exception:
                pass

            if time.time() < deadline - 3.0:
                try:
                    smooth_result = minimize(
                        lambda p: _compute_penalty_smooth(p, outer_side, outer_center, outer_angle, beta=500.0),
                        opt_params,
                        method='L-BFGS-B',
                        options={'maxiter': 200, 'maxfun': 5000, 'ftol': 1e-14, 'gtol': 1e-12}
                    )
                    smooth_params = smooth_result.x
                    hard_pen = _compute_penalty(smooth_params, outer_side, outer_center, outer_angle)
                    if hard_pen < best_polish_pen and hard_pen < 1e-8:
                        best_polish_params = smooth_params
                        best_polish_pen = hard_pen
                except Exception:
                    pass

            opt_params = best_polish_params

        best_result = (opt_params, outer_center, outer_side, outer_angle)

    # Fallback
    if best_result is None:
        inner = []
        pos = build_hexagonal_lattice(3, 4)
        arr = np.array(pos)
        centroid = arr.mean(axis=0)
        for (x, y) in (arr - centroid):
            inner.append([float(x), float(y), 0.0])
        return inner, (0.0, 0.0), 5.5, 0.0

    opt_params, outer_center, outer_side, outer_angle = best_result

    inner = []
    for i in range(N):
        x = float(opt_params[3 * i])
        y = float(opt_params[3 * i + 1])
        ang = float(opt_params[3 * i + 2])
        inner.append([x, y, ang])

    # Final validation
    inner_verts_list = []
    for i in range(N):
        cx, cy, ang = opt_params[3 * i], opt_params[3 * i + 1], opt_params[3 * i + 2]
        inner_verts_list.append(_hex_verts(cx, cy, 1.0, ang))
    normals, offsets = _outer_half_widths(outer_center, outer_side, outer_angle)
    cont_pen = _containment_penalty_vec(inner_verts_list, normals, offsets)
    disj_pen = 0.0
    for i in range(N):
        for j in range(i + 1, N):
            d = _sat_overlap_depth(inner_verts_list[i], inner_verts_list[j])
            if d > 0:
                disj_pen += d

    if cont_pen > 1e-4 or disj_pen > 1e-4:
        inner = []
        pos = build_hexagonal_lattice(3, 4)
        arr = np.array(pos)
        centroid = arr.mean(axis=0)
        for (x, y) in (arr - centroid):
            inner.append([float(x), float(y), 0.0])
        return inner, (0.0, 0.0), 5.5, 0.0

    return inner, (float(outer_center[0]), float(outer_center[1])), float(outer_side), float(outer_angle)

# EVOLVE-BLOCK-END
