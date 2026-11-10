# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist
from scipy.optimize import minimize
import time


def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.

    """
    n = 16
    d = 2
    deadline = time.time() + 620  # safety margin before 630s kill

    def score(points):
        if points.shape != (n, d):
            return 0.0
        dists = pdist(points)
        d_min = np.min(dists)
        d_max = np.max(dists)
        if d_max == 0:
            return 0.0
        return (d_min / d_max) ** 2

    def objective(flat):
        pts = flat.reshape(n, d)
        return -score(pts)

    # Initial configurations
    def make_circular(n, r=1.0):
        angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
        return np.column_stack([r * np.cos(angles), r * np.sin(angles)])

    def make_grid(n):
        # 4x4 grid
        size = int(np.ceil(np.sqrt(n)))
        pts = []
        for i in range(size):
            for j in range(size):
                if len(pts) < n:
                    pts.append([i - size / 2, j - size / 2])
        return np.array(pts[:n])

    def make_hexagonal(n):
        # Hexagonal packing
        pts = [[0, 0]]
        ring = 1
        while len(pts) < n:
            for i in range(6):
                angle = i * np.pi / 3
                for j in range(1, ring + 1):
                    if len(pts) < n:
                        r = j * 0.866
                        pts.append([r * np.cos(angle), r * np.sin(angle)])
            ring += 1
        return np.array(pts[:n])

    def make_random(seed):
        np.random.seed(seed)
        return np.random.randn(n, d)

    # Try multiple initializations
    initializations = [
        make_circular(n),
        make_grid(n),
        make_hexagonal(n),
        make_random(42),
        make_random(123),
        make_random(456),
    ]

    # Scale each to have max distance ~1 for better conditioning
    for i, init in enumerate(initializations):
        dists = pdist(init)
        d_max = np.max(dists)
        if d_max > 0:
            initializations[i] = init / d_max

    best_points = None
    best_score = -np.inf

    for init in initializations:
        if time.time() > deadline - 60:
            break

        # Try multiple optimization methods
        for method in ['L-BFGS-B', 'Nelder-Mead', 'Powell']:
            if time.time() > deadline - 30:
                break

            try:
                result = minimize(
                    objective,
                    init.flatten(),
                    method=method,
                    options={'maxiter': 5000, 'maxfev': 5000}
                )

                if result.success or result.fun < best_score:
                    pts = result.x.reshape(n, d)
                    s = score(pts)
                    if s > best_score:
                        best_score = s
                        best_points = pts.copy()

            except Exception:
                pass

    # If no optimization succeeded, use the best initial
    if best_points is None:
        best_points = initializations[0]

    return best_points


# EVOLVE-BLOCK-END
