# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial.distance import pdist
from scipy.optimize import minimize


def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the (x,y) coordinates.

    """
    n = 16
    d = 2
    start_time = time.time()
    deadline = start_time + 590  # Safety margin before 630s kill

    def soft_min(dists, t=50.0):
        """Differentiable approximation of the minimum distance using log-sum-exp."""
        a = -dists / t
        a_max = np.max(a)
        return -t * (a_max + np.log(np.sum(np.exp(a - a_max)) / len(dists)))

    def get_score(pts):
        """Compute the actual score R = (d_min/d_max)^2"""
        dists = pdist(pts)
        d_min = np.min(dists)
        d_max = np.max(dists)
        if d_max == 0:
            return 0.0
        return (d_min / d_max) ** 2

    def objective(x):
        """Minimize negative soft-min of pairwise distances, with max distance normalized."""
        pts = x.reshape(n, d)
        dists = pdist(pts)
        d_max = np.max(dists)
        if d_max < 1e-12:
            return 0.0
        # Normalize so max distance = 1, then maximize min distance
        # We minimize -soft_min(dists / d_max)
        norm_dists = dists / d_max
        return -soft_min(norm_dists, t=50.0)

    def hex_lattice_init(seed=0):
        """Create a hexagonal lattice-based initialization for 16 points."""
        np.random.seed(seed)
        # Hexagonal packing: place points in a hex grid pattern
        # For 16 points, use a 4x4 grid with hexagonal offsets
        points = np.zeros((n, d))
        # Use a hexagonal arrangement
        # Rows of hex grid
        row_spacing = 1.0
        col_spacing = 1.0
        # Place points in a roughly hexagonal pattern
        # 4 rows of 4 points each, with alternating row offsets
        idx = 0
        for row in range(4):
            for col in range(4):
                x = col * col_spacing + (row % 2) * 0.5 * col_spacing
                y = row * row_spacing * np.sqrt(3) / 2
                points[idx, 0] = x
                points[idx, 1] = y
                idx += 1
        # Add small random perturbation to break symmetry
        points += np.random.randn(n, d) * 0.01
        return points

    def circle_init(seed=0):
        """Create a circular initialization for 16 points."""
        np.random.seed(seed)
        angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
        points = np.column_stack([np.cos(angles), np.sin(angles)])
        points += np.random.randn(n, d) * 0.01
        return points

    def random_init(seed=0):
        """Create a random initialization."""
        np.random.seed(seed)
        points = np.random.randn(n, d)
        return points

    # Try multiple initializations and keep the best result
    best_points = None
    best_score = -1.0

    initializers = [
        hex_lattice_init,
        circle_init,
        random_init,
    ]

    for init_idx, init_func in enumerate(initializers):
        if time.time() > deadline - 60:
            break

        for seed in range(3):
            if time.time() > deadline - 60:
                break

            x0 = init_func(seed=seed + init_idx * 10)
            # Flatten to 1D for scipy
            x0_flat = x0.flatten()

            try:
                result = minimize(
                    objective,
                    x0_flat,
                    method='L-BFGS-B',
                    options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10}
                )
            except Exception:
                continue

            if result.success or result.fun < 0:
                pts = result.x.reshape(n, d)
                score = get_score(pts)
                if score > best_score:
                    best_score = score
                    best_points = pts

    # If no optimization succeeded, fall back to best initial
    if best_points is None:
        # Use hex lattice as fallback
        best_points = hex_lattice_init(seed=0)
        best_score = get_score(best_points)

    # Final polish: try a few more L-BFGS-B iterations with tighter tolerances
    if time.time() < deadline - 30 and best_points is not None:
        x0_flat = best_points.flatten()
        try:
            result = minimize(
                objective,
                x0_flat,
                method='L-BFGS-B',
                options={'maxiter': 1000, 'ftol': 1e-14, 'gtol': 1e-12}
            )
            if result.success:
                pts = result.x.reshape(n, d)
                score = get_score(pts)
                if score > best_score:
                    best_score = score
                    best_points = pts
        except Exception:
            pass

    # Ensure valid output
    if best_points is None or best_points.shape != (n, d) or not np.all(np.isfinite(best_points)):
        best_points = hex_lattice_init(seed=0)

    return best_points


# EVOLVE-BLOCK-END
