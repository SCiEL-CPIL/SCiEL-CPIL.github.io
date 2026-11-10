# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial.distance import pdist
from scipy.optimize import minimize
from scipy.optimize import differential_evolution


def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.

    """

    n = 16
    d = 2
    start_time = time.time()
    deadline = start_time + 590  # Safety margin before 630s kill

    def soft_min(dists, t=50.0):
        """Differentiable approximation of the minimum distance."""
        # Use log-sum-exp trick for numerical stability
        a = -dists / t
        a_max = np.max(a)
        return -t * (a_max + np.log(np.sum(np.exp(a - a_max)) / len(dists)))

    def get_score(pts):
        """Compute the actual score R = (d_min/d_max)^2"""
        dists = pdist(pts)
        d_min = np.min(dists)
        d_max = np.max(dists)
        if d_max < 1e-12:
            return 0.0
        return (d_min / d_max) ** 2

    def objective(x):
        """Smooth objective to minimize (negative of soft-min of normalized distances)."""
        pts = x.reshape(n, d)
        dists = pdist(pts)
        d_max = np.max(dists)
        if d_max < 1e-12:
            return 0.0
        dists_norm = dists / d_max
        d_min_soft = soft_min(dists_norm, t=50.0)
        return -d_min_soft

    # Generate diverse initializations
    def hex_lattice_init():
        """Hexagonal lattice arrangement."""
        pts = np.zeros((n, d))
        side = 1.0
        dx = side
        dy = side * np.sqrt(3) / 2
        row = 0
        col = 0
        for i in range(n):
            if col % 2 == 1:
                pts[i, 1] = dy * (row + 0.5)
            else:
                pts[i, 1] = dy * row
            pts[i, 0] = dx * col
            col += 1
            if col >= 4:
                col = 0
                row += 1
        # Center the points
        pts -= np.mean(pts, axis=0)
        return pts

    def grid_init():
        """4x4 grid."""
        x = np.arange(4)
        y = np.arange(4)
        X, Y = np.meshgrid(x, y)
        pts = np.column_stack([X.flatten(), Y.flatten()])
        pts -= np.mean(pts, axis=0)
        return pts

    def random_init(seed):
        """Random initialization."""
        rng = np.random.RandomState(seed)
        pts = rng.randn(n, d)
        pts -= np.mean(pts, axis=0)
        return pts

    def circle_init():
        """Points on a circle."""
        angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
        pts = np.column_stack([np.cos(angles), np.sin(angles)])
        return pts

    initializations = [
        hex_lattice_init(),
        grid_init(),
        random_init(42),
        random_init(123),
        random_init(456),
        circle_init(),
    ]

    best_points = None
    best_score = -np.inf

    # Try each initialization with L-BFGS-B
    for init_idx, init_pts in enumerate(initializations):
        if time.time() > deadline:
            break

        x0 = init_pts.flatten()
        try:
            result = minimize(
                objective,
                x0,
                method='L-BFGS-B',
                options={'maxiter': 1000, 'ftol': 1e-15, 'gtol': 1e-12}
            )
            pts = result.x.reshape(n, d)
            score = get_score(pts)

            if score > best_score:
                best_score = score
                best_points = pts.copy()
        except Exception as e:
            continue

        # Nelder-Mead refinement on the best result so far
        if time.time() > deadline:
            break
        if best_points is not None:
            try:
                result_nm = minimize(
                    objective,
                    best_points.flatten(),
                    method='Nelder-Mead',
                    options={'maxiter': 2000, 'xatol': 1e-12, 'fatol': 1e-12}
                )
                pts_nm = result_nm.x.reshape(n, d)
                score_nm = get_score(pts_nm)
                if score_nm > best_score:
                    best_score = score_nm
                    best_points = pts_nm.copy()
            except Exception:
                pass

    # If we still have time, try a few more random restarts
    if time.time() < deadline - 60 and best_points is not None:
        for seed in range(789, 819):
            if time.time() > deadline:
                break
            pts = random_init(seed)
            try:
                result = minimize(
                    objective,
                    pts.flatten(),
                    method='L-BFGS-B',
                    options={'maxiter': 500, 'ftol': 1e-15, 'gtol': 1e-12}
                )
                pts_opt = result.x.reshape(n, d)
                score = get_score(pts_opt)
                if score > best_score:
                    best_score = score
                    best_points = pts_opt.copy()
            except Exception:
                continue

    # Final high-precision polish
    if time.time() < deadline - 10 and best_points is not None:
        try:
            result_final = minimize(
                objective,
                best_points.flatten(),
                method='L-BFGS-B',
                options={'maxiter': 2000, 'ftol': 1e-18, 'gtol': 1e-15}
            )
            pts_final = result_final.x.reshape(n, d)
            score_final = get_score(pts_final)
            if score_final > best_score:
                best_score = score_final
                best_points = pts_final.copy()
        except Exception:
            pass

    if best_points is None:
        # Fallback: return the hex lattice
        best_points = hex_lattice_init()

    return best_points


# EVOLVE-BLOCK-END
