# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist
from scipy.optimize import minimize
import time

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions to maximize (d_min/d_max)^2.
    Uses hexagonal lattice initialization with multi-start Nelder-Mead refinement.
    """
    n = 16
    start_time = time.time()
    deadline = start_time + 585  # Safety margin for 600s limit

    def time_exceeded():
        return time.time() > deadline

    def objective(params):
        """Minimize -R where R = (d_min/d_max)^2"""
        points = params.reshape((n, 2))
        dists = pdist(points, 'euclidean')
        d_min = np.min(dists)
        d_max = np.max(dists)
        if d_max < 1e-12:
            return 0.0
        ratio = d_min / d_max
        return -(ratio ** 2)

    def evaluate(points):
        dists = pdist(points, 'euclidean')
        d_min = np.min(dists)
        d_max = np.max(dists)
        if d_max < 1e-12:
            return 0.0
        return (d_min / d_max) ** 2

    best_points = None
    best_score = -1.0

    def update_best(points):
        nonlocal best_score, best_points
        score = evaluate(points)
        if score > best_score:
            best_score = score
            best_points = points.copy()

    # Generate hexagonal lattice points
    def generate_hex_grid(radius, spacing=1.0):
        points = []
        for q in range(-radius, radius + 1):
            r_min = max(-radius, -q - radius)
            r_max = min(radius, -q + radius)
            for r in range(r_min, r_max + 1):
                x = spacing * (q + 0.5 * r)
                y = spacing * (np.sqrt(3) / 2 * r)
                points.append([x, y])
        return np.array(points)

    # Create diverse starting configurations
    starts = []

    # 1. Hexagonal lattice: pick 16 closest to center from a larger grid
    for radius in [3, 4]:
        hex_pts = generate_hex_grid(radius)
        hex_pts -= np.mean(hex_pts, axis=0)
        # Normalize to unit scale
        norm = np.max(np.linalg.norm(hex_pts, axis=1))
        if norm > 1e-10:
            hex_pts = hex_pts / norm
        dists_center = np.linalg.norm(hex_pts, axis=1)
        sorted_idx = np.argsort(dists_center)
        starts.append(hex_pts[sorted_idx[:16]])

    # 2. Circular arrangement
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    circle = np.column_stack((np.cos(angles), np.sin(angles)))
    starts.append(circle)

    # 3. Random starts with fixed seeds for reproducibility
    for seed in [1, 7, 13, 42, 99]:
        np.random.seed(seed)
        pts = np.random.uniform(-1, 1, (n, 2))
        norm = np.max(np.linalg.norm(pts, axis=1))
        if norm > 1e-10:
            pts = pts / norm
        starts.append(pts)

    # 4. Square grid
    x = np.linspace(-1, 1, 4)
    y = np.linspace(-1, 1, 4)
    xx, yy = np.meshgrid(x, y)
    grid = np.column_stack((xx.ravel(), yy.ravel()))
    starts.append(grid)

    # 5. Two concentric circles (8 points each)
    angles1 = np.linspace(0, 2 * np.pi, 8, endpoint=False)
    angles2 = np.linspace(0, 2 * np.pi, 8, endpoint=False) + np.pi / 8
    r1, r2 = 0.5, 1.0
    pts1 = np.column_stack((r1 * np.cos(angles1), r1 * np.sin(angles1)))
    pts2 = np.column_stack((r2 * np.cos(angles2), r2 * np.sin(angles2)))
    starts.append(np.vstack([pts1, pts2]))

    # Local optimization from each start
    for i, start in enumerate(starts):
        if time_exceeded():
            break

        def callback(xk):
            return time_exceeded()

        try:
            # Nelder-Mead with tight tolerances
            res = minimize(
                objective,
                start.ravel(),
                method='Nelder-Mead',
                options={
                    'maxiter': 5000,
                    'xatol': 1e-14,
                    'fatol': 1e-16,
                    'adaptive': True
                },
                callback=callback
            )
            if not time_exceeded():
                update_best(res.x.reshape((n, 2)))
        except Exception:
            pass

    # Additional refinement: try small perturbations of the best solution
    if not time_exceeded() and best_points is not None:
        for trial in range(20):
            if time_exceeded():
                break
            np.random.seed(1000 + trial)
            noise_scale = 0.05 / (trial + 1)
            noise = np.random.randn(n, 2) * noise_scale
            perturbed = best_points + noise

            norm = np.max(np.linalg.norm(perturbed, axis=1))
            if norm > 1e-10:
                perturbed = perturbed / norm

            def callback2(xk):
                return time_exceeded()

            try:
                res = minimize(
                    objective,
                    perturbed.ravel(),
                    method='Nelder-Mead',
                    options={
                        'maxiter': 3000,
                        'xatol': 1e-14,
                        'fatol': 1e-16,
                        'adaptive': True
                    },
                    callback=callback2
                )
                if not time_exceeded():
                    update_best(res.x.reshape((n, 2)))
            except Exception:
                pass

    # Final polish: very tight optimization on the best
    if not time_exceeded() and best_points is not None:
        def callback3(xk):
            return time_exceeded()

        try:
            res = minimize(
                objective,
                best_points.ravel(),
                method='Nelder-Mead',
                options={
                    'maxiter': 10000,
                    'xatol': 1e-16,
                    'fatol': 1e-18,
                    'adaptive': True
                },
                callback=callback3
            )
            if not time_exceeded():
                update_best(res.x.reshape((n, 2)))
        except Exception:
            pass

    # Fallback
    if best_points is None:
        np.random.seed(0)
        best_points = np.random.randn(n, 2)

    return best_points

# EVOLVE-BLOCK-END
