# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial.distance import pdist
from scipy.optimize import minimize

def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (14,3) containing the (x,y,z) coordinates of the 14 points.
    """
    n = 14
    d = 3
    start_time = time.time()
    deadline = start_time + 620.0  # Safety margin before 630s kill

    best_score = -1.0
    best_points = None

    def _compute_metric(points: np.ndarray) -> float:
        """Compute (d_min / d_max)^2 for the given points."""
        if points.shape[0] < 2:
            return 0.0
        dists = pdist(points)
        if len(dists) == 0:
            return 0.0
        d_min = np.min(dists)
        d_max = np.max(dists)
        if d_max == 0 or not np.isfinite(d_min) or not np.isfinite(d_max):
            return 0.0
        if d_min <= 0:
            return 0.0
        ratio = d_min / d_max
        return ratio * ratio

    def _try_candidate(pts):
        nonlocal best_score, best_points
        if not np.all(np.isfinite(pts)):
            return
        pts = pts.reshape(n, d)
        score = _compute_metric(pts)
        if score > best_score:
            best_score = score
            best_points = pts.copy()

    # Objective function for the optimizer
    # We pass raw coordinates, but we project them to the unit sphere inside.
    # This effectively optimizes the directions.
    def objective(x):
        pts = x.reshape(n, d)
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        norms[norms < 1e-10] = 1.0
        pts_on_sphere = pts / norms
        metric = _compute_metric(pts_on_sphere)
        return -metric

    # Initializations
    np.random.seed(42)

    # 1. Icosahedron-based initialization (12 vertices) + 2 random
    phi = (1 + 5**0.5) / 2
    ico = np.array([
        [0, 1, phi], [0, 1, -phi], [0, -1, phi], [0, -1, -phi],
        [1, phi, 0], [1, -phi, 0], [-1, phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ])
    ico_norms = np.linalg.norm(ico, axis=1, keepdims=True)
    ico = ico / ico_norms

    # 2. Fibonacci Sphere (uniform distribution)
    def fibonacci_sphere(n):
        points = []
        golden_ratio = (1 + np.sqrt(5)) / 2
        for i in range(n):
            z = 1 - (i + 0.5) * 2 / n
            radius = np.sqrt(1 - z*z)
            theta = 2 * np.pi * i / golden_ratio
            x = np.cos(theta) * radius
            y = np.sin(theta) * radius
            points.append([x, y, z])
        return np.array(points)

    fib_pts = fibonacci_sphere(n)

    # 3. Random points
    random_pts = np.random.randn(n, d)
    random_pts /= np.linalg.norm(random_pts, axis=1, keepdims=True)

    # Create a list of starting points
    start_points = [
        np.vstack([ico, np.random.randn(2, d)]),
        fib_pts,
        random_pts
    ]

    # Add more random restarts
    for _ in range(15):
        rp = np.random.randn(n, d)
        rp /= np.linalg.norm(rp, axis=1, keepdims=True)
        start_points.append(rp)

    # Optimization Loop
    for i, init_pts in enumerate(start_points):
        if time.time() > deadline - 10:
            break

        x0 = init_pts.flatten()

        try:
            # Nelder-Mead
            res = minimize(objective, x0, method='Nelder-Mead',
                           options={'maxiter': 3000, 'maxfev': 10000, 'xatol': 1e-8, 'fatol': 1e-10})
            candidate = res.x.reshape(n, d)
            _try_candidate(candidate)

            # If we have time, try a second method or a perturbed start
            if time.time() < deadline - 20:
                # Perturb the best found so far or the initial
                if best_points is not None:
                    x0_perturbed = best_points.flatten() + np.random.randn(len(x0)) * 0.05
                else:
                    x0_perturbed = x0 + np.random.randn(len(x0)) * 0.1
                res2 = minimize(objective, x0_perturbed, method='Nelder-Mead',
                                options={'maxiter': 2000, 'maxfev': 5000, 'xatol': 1e-8, 'fatol': 1e-10})
                candidate2 = res2.x.reshape(n, d)
                _try_candidate(candidate2)

        except Exception:
            pass

    # Final Fallback
    if best_points is None:
        best_points = fib_pts

    # Ensure the returned points are finite and have the correct shape
    best_points = best_points.reshape(n, d)
    if not np.all(np.isfinite(best_points)):
        best_points = fib_pts

    return best_points

# EVOLVE-BLOCK-END
