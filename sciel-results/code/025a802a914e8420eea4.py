"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 13

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize, differential_evolution
from scipy.spatial import ConvexHull

DEADLINE = 150.0
START_TIME = time.time()
SAFETY_MARGIN = 5.0


def _all_triangle_areas(pts):
    """Compute all triangle areas among triples of points."""
    n = len(pts)
    min_area = float('inf')
    areas = []
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                v1 = pts[j] - pts[i]
                v2 = pts[k] - pts[i]
                cross = v1[0] * v2[1] - v1[1] * v2[0]
                area = 0.5 * abs(cross)
                areas.append(area)
                if area < min_area:
                    min_area = area
    return np.array(areas), min_area


def _soft_min_triangle_area(pts, beta):
    """
    Soft-min of triangle areas using log-sum-exp for smoothness.
    As beta -> inf, this approaches the true minimum.
    """
    areas, _ = _all_triangle_areas(pts)
    if len(areas) == 0:
        return 0.0
    # Soft min: -log(sum(exp(-beta * a))) / beta
    # Numerically stable version
    min_a = np.min(areas)
    # Shift for numerical stability
    shifted = areas - min_a
    # exp(-beta * (a - min_a)) = exp(-beta * shifted)
    log_sum = np.log(np.sum(np.exp(-beta * shifted)))
    soft_min = min_a + log_sum / beta
    return soft_min


def _convex_hull_area(pts):
    """Compute convex hull area."""
    try:
        hull = ConvexHull(pts)
        return hull.volume
    except Exception:
        return 0.0


def _score(pts):
    """Score = min triangle area / convex hull area."""
    hull_area = _convex_hull_area(pts)
    if hull_area <= 1e-9:
        return 0.0
    _, min_ta = _all_triangle_areas(pts)
    return min_ta / hull_area


def _soft_score(pts, beta):
    """Smooth score using soft-min triangle area."""
    hull_area = _convex_hull_area(pts)
    if hull_area <= 1e-9:
        return 0.0
    soft_min_ta = _soft_min_triangle_area(pts, beta)
    return soft_min_ta / hull_area


def _neg_soft_score(pts, beta):
    """Negative soft score for minimization."""
    return -_soft_score(pts, beta)


def _regular_polygon(n, radius=1.0):
    """Generate a regular n-gon centered at origin."""
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return radius * np.column_stack([np.cos(angles), np.sin(angles)])


def _optimize_with_annealing(pts, max_iter=200, xatol=1e-8, fatol=1e-10):
    """
    Optimize using L-BFGS-B with beta annealing on the soft-min surrogate.
    Beta annealing: start with low beta (smooth), increase to high beta (sharp).
    """
    def _flatten(p):
        return p.flatten()

    def _unflatten(x):
        return x.reshape(N, 2)

    best_pts = pts.copy()
    best_score = _score(best_pts)

    # Beta annealing schedule
    betas = [10, 20, 50, 100, 200]

    current_pts = pts.copy()
    for beta in betas:
        if time.time() - START_TIME > DEADLINE - SAFETY_MARGIN:
            break

        def objective(x):
            points = x.reshape(N, 2)
            return -_soft_score(points, beta)

        try:
            result = minimize(
                objective,
                _flatten(current_pts),
                method='L-BFGS-B',
                options={'maxiter': max_iter, 'ftol': fatol, 'gtol': xatol}
            )
            if result.fun < 0:
                candidate = _unflatten(result.x)
                cand_score = _score(candidate)
                if cand_score > best_score:
                    best_pts = candidate
                    best_score = cand_score
                    current_pts = candidate
        except Exception:
            pass

    return best_pts, best_score


def _try_nelder_mead(pts, max_iter=300, xatol=1e-8, fatol=1e-10):
    """Try Nelder-Mead optimization as a fallback."""
    def _flatten(p):
        return p.flatten()

    def _unflatten(x):
        return x.reshape(N, 2)

    def _neg_score_flat(x):
        points = x.reshape(N, 2)
        return -_score(points)

    try:
        result = minimize(
            _neg_score_flat,
            _flatten(pts),
            method='Nelder-Mead',
            options={'maxiter': max_iter, 'xatol': xatol, 'fatol': fatol, 'adaptive': True}
        )
        if result.fun < 0:
            candidate = _unflatten(result.x)
            cand_score = _score(candidate)
            return candidate, cand_score
    except Exception:
        pass
    return None, 0.0


def construct():
    """Construct 13 points maximizing min triangle area / convex hull area."""

    # 1. Start with regular polygon
    pts0 = _regular_polygon(N)
    best_pts = pts0.copy()
    best_score = _score(pts0)

    # 2. Optimize from regular polygon using beta-annealed L-BFGS-B
    elapsed = time.time() - START_TIME
    remaining = DEADLINE - SAFETY_MARGIN - elapsed

    if remaining > 10:
        candidate, cand_score = _optimize_with_annealing(best_pts, max_iter=100)
        if cand_score > best_score:
            best_pts = candidate
            best_score = cand_score

    # 3. Differential Evolution Global Search
    elapsed = time.time() - START_TIME
    remaining = DEADLINE - SAFETY_MARGIN - elapsed

    if remaining > 30:
        bounds = [(-1, 1)] * (2 * N)

        def objective_de(x):
            if time.time() - START_TIME > DEADLINE - SAFETY_MARGIN:
                return 0.0
            points = x.reshape(-1, 2)
            return -_score(points)

        try:
            res_de = differential_evolution(
                objective_de,
                bounds,
                maxiter=30,
                popsize=15,
                tol=1e-5,
                seed=42,
                polish=False
            )

            if res_de.fun < -best_score:
                candidate = res_de.x.reshape(-1, 2)
                cand_score = _score(candidate)
                if cand_score > best_score:
                    best_score = cand_score
                    best_pts = candidate
        except Exception:
            pass

        # Local refinement from DE result using annealed L-BFGS-B
        if best_score > 0:
            elapsed = time.time() - START_TIME
            remaining = DEADLINE - SAFETY_MARGIN - elapsed
            if remaining > 15:
                candidate, cand_score = _optimize_with_annealing(best_pts, max_iter=100)
                if cand_score > best_score:
                    best_pts = candidate
                    best_score = cand_score

    # 4. Multi-start with diverse initializations
    elapsed = time.time() - START_TIME
    remaining = DEADLINE - SAFETY_MARGIN - elapsed

    np.random.seed(42)

    while remaining > 20:
        strategy = int(time.time() * 1000) % 4

        if strategy == 0:
            pts_init = best_pts + np.random.normal(0, 0.05, size=(N, 2))
        elif strategy == 1:
            pts_init = best_pts + np.random.normal(0, 0.15, size=(N, 2))
        elif strategy == 2:
            r = 0.5 + np.random.random() * 1.0
            theta = np.random.random() * 2 * np.pi
            pts_init = _regular_polygon(N, radius=r)
            rot = np.array([[np.cos(theta), -np.sin(theta)],
                            [np.sin(theta), np.cos(theta)]])
            pts_init = pts_init @ rot.T
        else:
            angles = np.random.uniform(0, 2 * np.pi, N)
            radii = np.sqrt(np.random.uniform(0.1, 1.0, N))
            pts_init = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])

        # Use annealed L-BFGS-B for refinement
        candidate, cand_score = _optimize_with_annealing(pts_init, max_iter=80)
        if cand_score > best_score:
            best_pts = candidate
            best_score = cand_score

        elapsed = time.time() - START_TIME
        remaining = DEADLINE - SAFETY_MARGIN - elapsed

    # 5. Final fine-tuning with Nelder-Mead for precision
    elapsed = time.time() - START_TIME
    remaining = DEADLINE - SAFETY_MARGIN - elapsed

    if remaining > 10:
        for _ in range(2):
            if time.time() - START_TIME > DEADLINE - SAFETY_MARGIN:
                break
            candidate, cand_score = _try_nelder_mead(best_pts, max_iter=200, xatol=1e-10, fatol=1e-12)
            if candidate is not None and cand_score > best_score:
                best_pts = candidate
                best_score = cand_score

    # Final check to ensure valid output
    if _score(best_pts) <= 0:
        best_pts = _regular_polygon(N)

    return best_pts
# EVOLVE-BLOCK-END
