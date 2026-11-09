# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
from itertools import combinations


def heilbronn_convex13() -> np.ndarray:
    """
    Construct an arrangement of 13 points on or inside a convex region in order to maximize the area of the
    smallest triangle formed by these points.
    
    Returns:
        points: np.ndarray of shape (13,2) with the x,y coordinates of the points.
    """
    n = 13
    start_time = time.time()
    time_limit = 580  # Leave 50s margin from 630s

    # Pre-compute all 286 triangle index triples
    tri_indices = list(combinations(range(n), 3))
    tri_arr = np.array(tri_indices)
    i_idx = tri_arr[:, 0]
    j_idx = tri_arr[:, 1]
    k_idx = tri_arr[:, 2]

    def compute_all_areas(points):
        """Compute areas of all 286 triangles. Returns array of shape (286,)."""
        pi = points[i_idx]
        pj = points[j_idx]
        pk = points[k_idx]
        areas = 0.5 * np.abs(pi[:, 0] * (pj[:, 1] - pk[:, 1]) +
                              pj[:, 0] * (pk[:, 1] - pi[:, 1]) +
                              pk[:, 0] * (pi[:, 1] - pj[:, 1]))
        return areas

    def compute_hull_area(points):
        """Compute convex hull area."""
        try:
            hull = ConvexHull(points)
            return hull.volume
        except Exception:
            return 0.0

    def compute_score(points):
        """Compute A_min / A_hull."""
        points = np.asarray(points)
        if not np.all(np.isfinite(points)):
            return -1.0
        areas = compute_all_areas(points)
        min_area = np.min(areas)
        if min_area < 1e-12:
            return -1.0
        hull_area = compute_hull_area(points)
        if hull_area < 1e-12:
            return -1.0
        return min_area / hull_area

    def soft_min_score(points, beta=100.0):
        """
        Smooth surrogate: use log-sum-exp for soft-min of triangle areas,
        and approximate hull area.
        Returns a smooth objective (higher is better).
        """
        points = np.asarray(points)
        if not np.all(np.isfinite(points)):
            return -1.0
        areas = compute_all_areas(points)
        # Soft-min via log-sum-exp
        max_a = np.max(areas)
        if max_a < 1e-12:
            return -1.0
        log_sum_exp = max_a + np.log(np.sum(np.exp(-areas / beta)))
        soft_min = -beta * log_sum_exp + beta * max_a  # This isn't quite right
        # Actually: soft_min = -beta * log(sum(exp(-areas/beta)))
        soft_min = -beta * np.log(np.sum(np.exp(-areas / beta)))
        
        hull_area = compute_hull_area(points)
        if hull_area < 1e-12:
            return -1.0
        return soft_min / hull_area

    def neg_objective_smooth(flat, beta=100.0):
        """Negative smooth objective for minimization."""
        points = flat.reshape(n, 2)
        score = soft_min_score(points, beta)
        if score < 0:
            return 1.0
        return -score

    def neg_objective_exact(flat):
        """Negative exact objective for minimization."""
        points = flat.reshape(n, 2)
        score = compute_score(points)
        if score < 0:
            return 1.0
        return -score

    def make_initialization(method='polygon', seed=0):
        rng = np.random.default_rng(seed)
        if method == 'polygon':
            # Regular 13-gon
            angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
            points = np.column_stack([np.cos(angles), np.sin(angles)])
            # Small perturbation
            points += rng.random((n, 2)) * 0.02
        elif method == 'hex':
            # Hexagonal lattice arrangement
            pts = []
            pts.append([0, 0])
            # First ring: 6 points at distance 1
            for k in range(6):
                angle = k * np.pi / 3
                pts.append([np.cos(angle), np.sin(angle)])
            # Second ring: 6 points at distance 2, offset
            for k in range(6):
                angle = k * np.pi / 3 + np.pi / 6
                pts.append([2 * np.cos(angle), 2 * np.sin(angle)])
            points = np.array(pts, dtype=float)
            points -= points.mean(axis=0)
            scale = np.max(np.abs(points))
            if scale > 0:
                points /= scale
            points += rng.random((n, 2)) * 0.02
        elif method == 'grid':
            # 4x4 grid minus one corner, then add center
            pts = []
            for i in range(4):
                for j in range(4):
                    if len(pts) < n - 1:
                        pts.append([j, i])
            # Add center
            pts.append([1.5, 1.5])
            points = np.array(pts, dtype=float)
            points -= points.mean(axis=0)
            scale = np.max(np.abs(points))
            if scale > 0:
                points /= scale
            points += rng.random((n, 2)) * 0.02
        else:
            points = rng.random((n, 2))
            points -= points.mean(axis=0)
        return points

    best_score = -1.0
    best_points = None

    # Phase 1: Try multiple initializations with smooth objective
    initializations = [
        ('polygon', 0), ('polygon', 1), ('polygon', 2),
        ('hex', 0), ('hex', 1), ('hex', 2),
        ('grid', 0), ('grid', 1),
        ('random', 0), ('random', 1),
    ]

    for method, seed in initializations:
        if time.time() - start_time > time_limit - 60:
            break

        init_points = make_initialization(method, seed)
        init_flat = init_points.flatten()

        # Use smooth objective with L-BFGS-B
        try:
            result = minimize(neg_objective_smooth, init_flat, method='L-BFGS-B',
                             args=(80.0,),
                             options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-8})
        except Exception:
            result = None

        if result is not None and result.x is not None:
            final_points = result.x.reshape(n, 2)
            final_score = compute_score(final_points)
            if final_score > best_score:
                best_score = final_score
                best_points = final_points.copy()

        # Also evaluate initial point
        init_score = compute_score(init_points)
        if init_score > best_score:
            best_score = init_score
            best_points = init_points.copy()

    # Phase 2: Polish with exact objective using Nelder-Mead
    if best_points is not None:
        if time.time() - start_time < time_limit - 30:
            try:
                result = minimize(neg_objective_exact, best_points.flatten(),
                                 method='Nelder-Mead',
                                 options={'maxiter': 5000, 'xatol': 1e-9, 'fatol': 1e-11, 'adaptive': True})
                if result.x is not None:
                    final_points = result.x.reshape(n, 2)
                    final_score = compute_score(final_points)
                    if final_score > best_score:
                        best_score = final_score
                        best_points = final_points.copy()
            except Exception:
                pass

    # Phase 3: Try a few more polish passes with different strategies
    if best_points is not None and time.time() - start_time < time_limit - 20:
        # Try perturbing the best solution and re-optimizing
        rng = np.random.default_rng(42)
        for attempt in range(3):
            if time.time() - start_time > time_limit - 10:
                break
            perturbed = best_points + rng.normal(0, 0.01, (n, 2))
            try:
                result = minimize(neg_objective_smooth, perturbed.flatten(), method='L-BFGS-B',
                                 args=(50.0,),
                                 options={'maxiter': 300, 'ftol': 1e-10, 'gtol': 1e-8})
                if result.x is not None:
                    final_points = result.x.reshape(n, 2)
                    final_score = compute_score(final_points)
                    if final_score > best_score:
                        best_score = final_score
                        best_points = final_points.copy()
            except Exception:
                pass

    # Final polish with very tight tolerances
    if best_points is not None and time.time() - start_time < time_limit - 5:
        try:
            result = minimize(neg_objective_exact, best_points.flatten(),
                             method='Nelder-Mead',
                             options={'maxiter': 3000, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True})
            if result.x is not None:
                final_points = result.x.reshape(n, 2)
                final_score = compute_score(final_points)
                if final_score > best_score:
                    best_score = final_score
                    best_points = final_points.copy()
        except Exception:
            pass

    # Ensure valid output
    if best_points is None or not np.all(np.isfinite(best_points)):
        angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
        best_points = np.column_stack([np.cos(angles), np.sin(angles)])

    return best_points


# EVOLVE-BLOCK-END
