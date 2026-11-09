# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import ConvexHull
from scipy.optimize import minimize


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

    def triangle_area(p1, p2, p3):
        return 0.5 * abs(p1[0]*(p2[1]-p3[1]) + p2[0]*(p3[1]-p1[1]) + p3[0]*(p1[1]-p2[1]))

    def compute_score(points):
        """Compute A_min / A_hull"""
        points = np.asarray(points)
        # Check for degeneracy
        if not np.all(np.isfinite(points)):
            return -1.0
        # Compute all triangle areas
        min_area = np.inf
        for i in range(n):
            for j in range(i+1, n):
                for k in range(j+1, n):
                    a = triangle_area(points[i], points[j], points[k])
                    if a < min_area:
                        min_area = a
        if min_area < 1e-12:
            return -1.0
        # Compute hull area
        try:
            hull = ConvexHull(points)
            hull_area = hull.volume
        except Exception:
            return -1.0
        if hull_area < 1e-12:
            return -1.0
        return min_area / hull_area

    def neg_objective(flat):
        points = flat.reshape(n, 2)
        score = compute_score(points)
        if score < 0:
            return 1.0  # Large penalty
        return -score

    def make_initialization(method='circle', seed=0):
        rng = np.random.default_rng(seed)
        if method == 'circle':
            angles = np.linspace(0, 2*np.pi, n, endpoint=False) + rng.random() * 0.1
            points = np.column_stack([np.cos(angles), np.sin(angles)])
            # Add small random perturbation
            points += rng.random((n, 2)) * 0.05
        elif method == 'grid':
            # Arrange in a roughly grid pattern
            cols = 4
            rows = 4  # 13 points: 4+4+4+1
            pts = []
            for i in range(rows):
                for j in range(cols):
                    if len(pts) < n:
                        pts.append([j, i])
            points = np.array(pts, dtype=float)
            # Center and scale
            points -= points.mean(axis=0)
            points /= np.max(np.abs(points))
            points += rng.random((n, 2)) * 0.05
        elif method == 'hex':
            # Hexagonal-like arrangement
            pts = []
            # Center
            pts.append([0, 0])
            # First ring (6 points)
            for k in range(6):
                angle = k * np.pi / 3
                pts.append([np.cos(angle), np.sin(angle)])
            # Second ring partial (6 more points to reach 13)
            for k in range(6):
                angle = k * np.pi / 3 + np.pi / 6
                pts.append([2*np.cos(angle), 2*np.sin(angle)])
            points = np.array(pts, dtype=float)
            points -= points.mean(axis=0)
            points /= np.max(np.abs(points))
            points += rng.random((n, 2)) * 0.05
        else:
            points = rng.random((n, 2))
        return points

    best_score = -1.0
    best_points = None

    # Try multiple initializations and optimization methods
    initializations = [
        ('circle', 0), ('circle', 1), ('circle', 2),
        ('grid', 0), ('grid', 1),
        ('hex', 0), ('hex', 1),
        ('random', 0), ('random', 1), ('random', 2),
    ]

    for method, seed in initializations:
        if time.time() - start_time > time_limit:
            break

        init_points = make_initialization(method, seed)
        init_flat = init_points.flatten()

        # Nelder-Mead optimization
        try:
            result = minimize(neg_objective, init_flat, method='Nelder-Mead',
                             options={'maxiter': 5000, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True})
        except Exception:
            continue

        if time.time() - start_time > time_limit:
            break

        final_points = result.x.reshape(n, 2)
        final_score = compute_score(final_points)

        if final_score > best_score:
            best_score = final_score
            best_points = final_points.copy()

        # Also try the initial point's score
        init_score = compute_score(init_points)
        if init_score > best_score:
            best_score = init_score
            best_points = init_points.copy()

    # If no valid result, fallback to circle
    if best_points is None:
        angles = np.linspace(0, 2*np.pi, n, endpoint=False)
        best_points = np.column_stack([np.cos(angles), np.sin(angles)])
    else:
        # Final polish: run one more Nelder-Mead from the best point
        if time.time() - start_time < time_limit - 60:
            try:
                result = minimize(neg_objective, best_points.flatten(), method='Nelder-Mead',
                                 options={'maxiter': 10000, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True})
                final_points = result.x.reshape(n, 2)
                final_score = compute_score(final_points)
                if final_score > best_score:
                    best_score = final_score
                    best_points = final_points.copy()
            except Exception:
                pass

    # Ensure valid output
    if not np.all(np.isfinite(best_points)):
        angles = np.linspace(0, 2*np.pi, n, endpoint=False)
        best_points = np.column_stack([np.cos(angles), np.sin(angles)])

    return best_points


# EVOLVE-BLOCK-END
