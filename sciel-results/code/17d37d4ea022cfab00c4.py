"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

N = 13

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull


def _min_triangle_area(points):
    """Compute the minimum triangle area over all triples of points."""
    n = len(points)
    min_area = float('inf')
    for i in range(n - 2):
        for j in range(i + 1, n - 1):
            for k in range(j + 1, n):
                v1 = points[j] - points[i]
                v2 = points[k] - points[i]
                area = 0.5 * abs(v1[0] * v2[1] - v1[1] * v2[0])
                if area < min_area:
                    min_area = area
    return min_area


def _convex_hull_area(points):
    """Compute the area of the convex hull of the given points."""
    try:
        hull = ConvexHull(points)
        return hull.volume
    except Exception:
        if len(points) < 3:
            return 0.0
        c = np.mean(points, axis=0)
        angles = np.arctan2(points[:, 1] - c[1], points[:, 0] - c[0])
        order = np.argsort(angles)
        pts = points[order]
        n = len(pts)
        area = 0.0
        for i in range(n):
            j = (i + 1) % n
            area += pts[i][0] * pts[j][1]
            area -= pts[j][0] * pts[i][1]
        return abs(area) * 0.5


def _score(points):
    """Score = minimum triangle area / convex hull area."""
    hull_area = _convex_hull_area(points)
    if hull_area < 1e-10:
        return 0.0
    min_area = _min_triangle_area(points)
    return min_area / hull_area


def _build_from_shells(params):
    """
    Build 13 points from concentric regular polygon shells.
    params: list of (count, radius, phase) for each shell.
    Returns (n, 2) array.
    """
    pts = []
    for count, radius, phase in params:
        if count <= 0:
            continue
        angles = phase + 2 * np.pi * np.arange(count) / count
        x = radius * np.cos(angles)
        y = radius * np.sin(angles)
        pts.append(np.column_stack((x, y)))
    if not pts:
        return np.zeros((0, 2))
    return np.vstack(pts)


def _score_from_params(params):
    """Score the configuration defined by shell parameters."""
    pts = _build_from_shells(params)
    if len(pts) < 3:
        return 0.0
    return _score(pts)


def _grid_search_two_shell(r1_vals, r2_vals, phase_vals, counts):
    """
    Grid search over two-shell configurations.
    counts: list of (n1, n2) where n1 + n2 = 13
    """
    best_score = 0.0
    best_params = None
    for n1, n2 in counts:
        if n1 + n2 != N:
            continue
        for r1 in r1_vals:
            for r2 in r2_vals:
                for ph1 in phase_vals:
                    for ph2 in phase_vals:
                        params = [(n1, r1, ph1), (n2, r2, ph2)]
                        s = _score_from_params(params)
                        if s > best_score:
                            best_score = s
                            best_params = params
    return best_score, best_params


def _grid_search_three_shell(r1_vals, r2_vals, r3_vals, phase_vals, counts):
    """
    Grid search over three-shell configurations.
    counts: list of (n1, n2, n3) where n1+n2+n3 = 13
    """
    best_score = 0.0
    best_params = None
    for n1, n2, n3 in counts:
        if n1 + n2 + n3 != N:
            continue
        for r1 in r1_vals:
            for r2 in r2_vals:
                for r3 in r3_vals:
                    for ph1 in phase_vals:
                        for ph2 in phase_vals:
                            for ph3 in phase_vals:
                                params = [(n1, r1, ph1), (n2, r2, ph2), (n3, r3, ph3)]
                                s = _score_from_params(params)
                                if s > best_score:
                                    best_score = s
                                    best_params = params
    return best_score, best_params


def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit

    best_score = 0.0
    best_points = None

    def _update_best(params):
        nonlocal best_score, best_points
        pts = _build_from_shells(params)
        if len(pts) != N:
            return
        s = _score(pts)
        if s > best_score:
            best_score = s
            best_points = pts.copy()

    # --- Phase 1: Dense grid search over two-shell configs ---
    # Common partitions of 13 into 2: (1,12), (2,11), (3,10), (4,9), (5,8), (6,7)
    two_shell_counts = [(1, 12), (2, 11), (3, 10), (4, 9), (5, 8), (6, 7)]

    # Radii: inner shell smaller, outer shell larger
    # Try a range of radius ratios
    r1_vals = np.linspace(0.1, 0.8, 12)  # inner radius
    r2_vals = np.linspace(0.9, 1.5, 12)  # outer radius
    phase_vals = np.linspace(0, 2 * np.pi, 13, endpoint=False)

    for n1, n2 in two_shell_counts:
        if time.time() > deadline:
            break
        for r1 in r1_vals:
            for r2 in r2_vals:
                if r1 >= r2:
                    continue
                for ph1 in phase_vals:
                    for ph2 in phase_vals:
                        if time.time() > deadline:
                            break
                        params = [(n1, r1, ph1), (n2, r2, ph2)]
                        _update_best(params)

    # --- Phase 2: Three-shell grid search (coarser) ---
    three_shell_counts = [(1, 2, 10), (1, 3, 9), (1, 4, 8), (1, 5, 7),
                          (2, 3, 8), (2, 4, 7), (2, 5, 6), (3, 4, 6), (3, 5, 5),
                          (1, 6, 6), (2, 2, 9), (2, 6, 5), (3, 3, 7), (4, 4, 5)]

    r1_vals_3 = np.linspace(0.1, 0.5, 6)
    r2_vals_3 = np.linspace(0.5, 0.9, 6)
    r3_vals_3 = np.linspace(0.9, 1.4, 6)
    phase_vals_3 = np.linspace(0, 2 * np.pi, 7, endpoint=False)

    for n1, n2, n3 in three_shell_counts:
        if time.time() > deadline:
            break
        for r1 in r1_vals_3:
            for r2 in r2_vals_3:
                for r3 in r3_vals_3:
                    if not (r1 < r2 < r3):
                        continue
                    for ph1 in phase_vals_3:
                        for ph2 in phase_vals_3:
                            for ph3 in phase_vals_3:
                                if time.time() > deadline:
                                    break
                                params = [(n1, r1, ph1), (n2, r2, ph2), (n3, r3, ph3)]
                                _update_best(params)

    # --- Phase 3: Local refinement using Nelder-Mead on the best config ---
    if best_points is not None and time.time() < deadline:
        # Try to refine by perturbing the best points slightly
        rng = np.random.default_rng(123)

        def objective(x):
            pts = x.reshape(-1, 2)
            hull_area = _convex_hull_area(pts)
            if hull_area < 1e-10:
                return 1e10
            min_area = _min_triangle_area(pts)
            return -min_area / hull_area

        # Refine the best point found
        for restart in range(8):
            if time.time() > deadline:
                break
            x0 = best_points.copy()
            if restart > 0:
                x0 = x0 + rng.normal(0, 0.02, size=x0.shape)
            res = minimize(objective, x0.flatten(), method='Nelder-Mead',
                           options={'maxiter': 200, 'xatol': 1e-9, 'fatol': 1e-11})
            candidate = res.x.reshape(-1, 2)
            s = _score(candidate)
            if s > best_score:
                best_score = s
                best_points = candidate.copy()

    # --- Phase 4: Try a few more shell configurations with finer phase resolution ---
    if time.time() < deadline:
        # Focus on the best partition found, with finer phase grid
        # Re-try top partitions with finer phases
        fine_phase_vals = np.linspace(0, 2 * np.pi, 25, endpoint=False)
        for n1, n2 in [(3, 10), (4, 9), (5, 8), (6, 7)]:
            if time.time() > deadline:
                break
            for r1 in np.linspace(0.2, 0.7, 8):
                for r2 in np.linspace(1.0, 1.4, 8):
                    if r1 >= r2:
                        continue
                    for ph1 in fine_phase_vals:
                        for ph2 in fine_phase_vals:
                            if time.time() > deadline:
                                break
                            params = [(n1, r1, ph1), (n2, r2, ph2)]
                            _update_best(params)

    if best_points is None:
        # Fallback: regular 13-gon
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        best_points = np.column_stack((np.cos(angles), np.sin(angles)))

    return best_points
# EVOLVE-BLOCK-END
