import numpy as np
import time
import scipy.optimize as opt
from scipy.spatial import ConvexHull

N = 13

def construct():
    start_time = time.time()
    time_limit = 145.0  # Safety margin for 150s kill

    # Pre-compute all triples (i, j, k) with i < j < k
    triple_i, triple_j, triple_k = [], [], []
    for i in range(N):
        for j in range(i + 1, N):
            for k in range(j + 1, N):
                triple_i.append(i)
                triple_j.append(j)
                triple_k.append(k)
    triple_i = np.array(triple_i, dtype=np.int32)
    triple_j = np.array(triple_j, dtype=np.int32)
    triple_k = np.array(triple_k, dtype=np.int32)
    n_triples = len(triple_i)

    def compute_score_fast(points):
        """
        Compute min triangle area / convex hull area.
        points: (N, 2) array
        Returns (score, area_hull)
        """
        try:
            hull = ConvexHull(points)
            area_hull = hull.volume
        except Exception:
            return 0.0, 0.0

        if area_hull < 1e-12:
            return 0.0, 0.0

        x = points[:, 0]
        y = points[:, 1]

        xi = x[triple_i]
        xj = x[triple_j]
        xk = x[triple_k]
        yi = y[triple_i]
        yj = y[triple_j]
        yk = y[triple_k]

        cross = np.abs((xj - xi) * (yk - yi) - (yj - yi) * (xk - xi))
        areas = 0.5 * cross
        min_area = np.min(areas)

        score = min_area / area_hull
        return score, area_hull

    def _regular_polygon(n, radius, center=(0.0, 0.0), angle_offset=0.0):
        if n <= 1:
            if n == 1:
                return np.array([[center[0], center[1]]])
            return np.array([]).reshape(0, 2)
        angles = np.linspace(angle_offset, 2 * np.pi + angle_offset, n, endpoint=False)
        xs = center[0] + radius * np.cos(angles)
        ys = center[1] + radius * np.sin(angles)
        return np.column_stack([xs, ys])

    # ------------------------------------------------------------------
    # New structural approach: symmetric layered parameterization
    # Points are organized as:
    #   - n_hull vertices of a regular convex polygon (outer layer)
    #   - n_inner points arranged in concentric rings on symmetry axes
    # Parameters: hull_radius, inner ring radii, angular offsets
    # This enforces convexity and high symmetry, avoiding local minima traps
    # ------------------------------------------------------------------

    def make_symmetric_config(n_hull, inner_ring_sizes, hull_radius, inner_radii,
                              hull_offset=0.0, inner_offsets=None, center=(0.0, 0.0)):
        """
        Create a symmetric configuration with n_hull polygon vertices
        and interior points in concentric rings.
        
        inner_ring_sizes: list of counts for each ring (sum must equal N - n_hull)
        inner_radii: list of radii for each ring
        inner_offsets: list of angular offsets for each ring (optional)
        """
        if n_hull < 3:
            return None
        if sum(inner_ring_sizes) != N - n_hull:
            return None
        if len(inner_radii) != len(inner_ring_sizes):
            return None
            # Return None
        if inner_offsets is None:
            inner_offsets = [0.0] * len(inner_ring_sizes)
        if len(inner_offsets) != len(inner_ring_sizes):
            return None

        pts = _regular_polygon(n_hull, hull_radius, center=center, angle_offset=hull_offset)

        for ring_size, radius, offset in zip(inner_ring_sizes, inner_radii, inner_offsets):
            if ring_size > 0:
                ring_pts = _regular_polygon(ring_size, radius, center=center, angle_offset=offset)
                pts = np.vstack([pts, ring_pts])

        if pts.shape[0] != N:
            return None

        return pts

    def make_two_layer_config(n_hull, hull_radius, inner_radius,
                              hull_offset=0.0, inner_offset=0.0):
        """
        Simple two-layer: n_hull polygon vertices + (N-n_hull) inner polygon vertices.
        """
        n_inner = N - n_hull
        if n_hull < 3 or n_inner < 0:
            return None
        if n_inner == 0:
            return _regular_polygon(n_hull, hull_radius, angle_offset=hull_offset)

        hull_pts = _regular_polygon(n_hull, hull_radius, angle_offset=hull_offset)
        inner_pts = _regular_polygon(n_inner, inner_radius, angle_offset=inner_offset)
        return np.vstack([hull_pts, inner_pts])

    def make_three_layer_config(n_hull, r1, r2, r3,
                                hull_offset=0.0, off1=0.0, off2=0.0, off3=0.0):
        """
        Three layers: hull polygon + two inner rings.
        n_hull + n1 + n2 = N where n1, n2 are ring sizes.
        """
        n_inner = N - n_hull
        if n_hull < 3 or n_inner < 2:
            return None
        # Split inner points into two rings
        n1 = max(1, n_inner // 2)
        n2 = n_inner - n1

        hull_pts = _regular_polygon(n_hull, r1, angle_offset=hull_offset)
        ring1 = _regular_polygon(n1, r2, angle_offset=off1)
        ring2 = _regular_polygon(n2, r3, angle_offset=off2)
        return np.vstack([hull_pts, ring1, ring2])

    # ------------------------------------------------------------------
    # Phase 1: Global grid search over symmetric parameterizations
    # ------------------------------------------------------------------
    best_points = None
    best_score = -float('inf')

    def evaluate_and_update(points):
        nonlocal best_points, best_score
        if points is None or points.shape != (N, 2):
            return
        s, _ = compute_score_fast(points)
        if s > best_score:
            best_score = s
            best_points = points.copy()

    # Strategy A: Two-layer configurations (hull polygon + inner polygon)
    # Grid over: n_hull, hull_radius, inner_radius, offsets
    hull_radii = np.linspace(0.5, 1.5, 12)
    inner_radii = np.linspace(0.0, 0.6, 13)
    offset_vals = np.linspace(0, np.pi / 2, 7)

    for n_hull in range(3, N + 1):
        n_inner = N - n_hull
        if n_inner < 0:
            continue

        for hr in hull_radii:
            if time.time() - start_time > time_limit - 30:
                break
            if n_inner == 0:
                for ho in offset_vals:
                    cand = make_two_layer_config(n_hull, hr, 0.0, hull_offset=ho)
                    evaluate_and_update(cand)
            else:
                for ir in inner_radii:
                    if time.time() - start_time > time_limit - 30:
                        break
                    for ho in offset_vals:
                        for io in offset_vals:
                            if time.time() - start_time > time_limit - 30:
                                break
                            cand = make_two_layer_config(n_hull, hr, ir,
                                                         hull_offset=ho, inner_offset=io)
                            evaluate_and_update(cand)

    # Strategy B: Three-layer configurations
    # Grid over: n_hull, r1 (hull), r2 (inner ring 1), r3 (inner ring 2), offsets
    r1_vals = np.linspace(0.6, 1.4, 8)
    r2_vals = np.linspace(0.1, 0.5, 7)
    r3_vals = np.linspace(0.2, 0.7, 7)
    off_vals = np.linspace(0, np.pi / 3, 5)

    for n_hull in range(4, N + 1):
        n_inner = N - n_hull
        if n_inner < 2:
            continue

        for r1 in r1_vals:
            if time.time() - start_time > time_limit - 25:
                break
            for r2 in r2_vals:
                for r3 in r3_vals:
                    if time.time() - start_time > time_limit - 25:
                        break
                    # Only consider r2 < r1 and r3 < r1 for convexity
                    if r2 >= r1 or r3 >= r1:
                        continue
                    for ho in off_vals:
                        for o1 in off_vals:
                            for o2 in off_vals:
                                if time.time() - start_time > time_limit - 25:
                                    break
                                cand = make_three_layer_config(n_hull, r1, r2, r3,
                                                               hull_offset=ho,
                                                               off1=o1, off2=o2)
                                evaluate_and_update(cand)

    # Strategy C: Asymmetric inner points on symmetry axes of hexagon
    # Place hull as regular polygon, inner points along 6 symmetry axes
    # at various radial distances
    if time.time() - start_time < time_limit - 20:
        n_hull_options = [6, 7, 8]
        for n_hull in n_hull_options:
            n_inner = N - n_hull
            if n_inner < 0 or n_inner > 6:
                continue

            hr_vals = np.linspace(0.7, 1.3, 6)
            # Inner points distributed on axes: for hexagon, 6 axes at 0, 30, 60, ... degrees
            # We can place points at various radii along these axes

            for hr in hr_vals:
                if time.time() - start_time > time_limit - 20:
                    break
                # Generate inner points on symmetry axes
                # For a regular n_hull-gon, symmetry axes are at angles k*pi/n_hull
                # Place n_inner points: distribute among axes
                if n_inner <= 6:
                    # One point per axis for the first n_inner axes
                    axis_angles = np.arange(n_inner) * (np.pi / n_hull)
                    # Try different radial patterns
                    for pattern_idx in range(5):
                        if time.time() - start_time > time_limit - 20:
                            break
                        # Different radial distributions
                        if pattern_idx == 0:
                            radii = np.full(n_inner, 0.3)
                        elif pattern_idx == 1:
                            radii = np.linspace(0.1, 0.5, n_inner)
                        elif pattern_idx == 2:
                            radii = np.linspace(0.5, 0.1, n_inner)
                        elif pattern_idx == 3:
                            radii = np.full(n_inner, 0.2)
                        else:
                            radii = np.full(n_inner, 0.4)

                        inner_pts = np.column_stack([
                            radii * np.cos(axis_angles),
                            radii * np.sin(axis_angles)
                        ])
                        hull_pts = _regular_polygon(n_hull, hr)
                        cand = np.vstack([hull_pts, inner_pts])
                        evaluate_and_update(cand)

    # ------------------------------------------------------------------
    # Phase 2: Local refinement using Nelder-Mead on the best candidate
    # ------------------------------------------------------------------
    if best_points is not None and time.time() - start_time < time_limit - 15:
        def hard_obj(x):
            pts = x.reshape(-1, 2)
            s, _ = compute_score_fast(pts)
            return -s

        try:
            res = opt.minimize(
                hard_obj,
                best_points.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 800, 'adaptive': True,
                         'xatol': 1e-12, 'fatol': 1e-14}
            )
            nm_pts = res.x.reshape(-1, 2)
            nm_score, _ = compute_score_fast(nm_pts)
            if nm_score > best_score:
                best_score = nm_score
                best_points = nm_pts.copy()
        except Exception:
            pass

        # Second Nelder-Mead pass with tighter tolerance
        if time.time() - start_time < time_limit - 5:
            try:
                res = opt.minimize(
                    hard_obj,
                    best_points.flatten(),
                    method='Nelder-Mead',
                    options={'maxiter': 500, 'adaptive': True,
                             'xatol': 1e-13, 'fatol': 1e-15}
                )
                final_pts = res.x.reshape(-1, 2)
                final_score, _ = compute_score_fast(final_pts)
                if final_score > best_score:
                    best_score = final_score
                    best_points = final_pts.copy()
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Phase 3: Try smooth surrogate optimization from best point
    # ------------------------------------------------------------------
    if best_points is not None and time.time() - start_time < time_limit - 3:
        beta = 150.0

        def soft_obj(x):
            pts = x.reshape(-1, 2)
            try:
                hull = ConvexHull(pts)
                area_hull = hull.volume
            except Exception:
                return 1e6
            if area_hull < 1e-12:
                return 1e6

            xi = pts[triple_i, 0]
            xj = pts[triple_j, 0]
            xk = pts[triple_k, 0]
            yi = pts[triple_i, 1]
            yj = pts[triple_j, 1]
            yk = pts[triple_k, 1]

            cross = np.abs((xj - xi) * (yk - yi) - (yj - yi) * (xk - xi))
            areas = 0.5 * cross

            neg_beta = -beta * areas
            max_val = np.max(neg_beta)
            lse = max_val + np.log(np.sum(np.exp(neg_beta - max_val)))
            soft_min = -lse / beta
            return -(soft_min / area_hull)

        try:
            res = opt.minimize(
                soft_obj,
                best_points.flatten(),
                method='L-BFGS-B',
                options={'maxiter': 80, 'ftol': 1e-14, 'gtol': 1e-12}
            )
            lb_pts = res.x.reshape(-1, 2)
            lb_score, _ = compute_score_fast(lb_pts)
            if lb_score > best_score:
                best_score = lb_score
                best_points = lb_pts.copy()
        except Exception:
            pass

    if best_points is None:
        best_points = _regular_polygon(13, radius=1.0)

    return best_points
