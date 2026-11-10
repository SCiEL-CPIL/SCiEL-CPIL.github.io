# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist
from scipy.optimize import minimize


def _compute_ratio(points: np.ndarray) -> float:
    """Compute R = (d_min / d_max)^2 for the given points."""
    d = pdist(points)
    d_min = np.min(d)
    d_max = np.max(d)
    if d_max == 0:
        return 0.0
    return (d_min / d_max) ** 2


def _solve_with_lower_bound(lb: float) -> tuple:
    """
    Try to find 16 points such that all pairwise distances >= lb.
    We minimize the maximum pairwise distance subject to min pairwise distance >= lb.
    Returns (success, points, d_max).
    """
    n = 16
    d = 2

    # Use a symmetric initialization: points on a circle or grid
    # Try a few different initializations
    best_success = False
    best_points = None
    best_dmax = np.inf

    # Initialization 1: regular polygon / circle
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    x0 = np.column_stack([np.cos(angles), np.sin(angles)])

    # Initialization 2: grid-like
    grid_x = np.array([1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4], dtype=float)
    grid_y = np.array([1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4], dtype=float)
    x0_grid = np.column_stack([grid_x, grid_y])

    # Initialization 3: random
    np.random.seed(123)
    x0_rand = np.random.randn(n, 2) * 2.0

    initializations = [x0, x0_grid, x0_rand]

    for x0 in initializations:
        # Objective: minimize the maximum pairwise distance
        # Constraint: all pairwise distances >= lb

        def objective(x):
            pts = x.reshape(n, 2)
            dists = pdist(pts)
            return np.max(dists)

        def constraint_min_dist(x):
            pts = x.reshape(n, 2)
            dists = pdist(pts)
            return np.min(dists) - lb

        # Use SLSQP with constraint
        cons = [{'type': 'ineq', 'fun': constraint_min_dist}]

        try:
            res = minimize(
                objective,
                x0.flatten(),
                method='SLSQP',
                constraints=cons,
                options={'maxiter': 500, 'ftol': 1e-10}
            )

            if res.success:
                pts = res.x.reshape(n, 2)
                d_max = np.max(pdist(pts))
                d_min = np.min(pdist(pts))
                if d_min >= lb - 1e-8:  # constraint satisfied
                    if not best_success or d_max < best_dmax:
                        best_success = True
                        best_points = pts
                        best_dmax = d_max
        except Exception:
            pass

    return best_success, best_points, best_dmax


def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.
    """
    n = 16
    d = 2

    # Bisection search on the lower bound lb
    # We want to maximize (d_min/d_max)^2
    # Approach: for a given lb, find the minimal d_max such that all pairwise distances >= lb
    # Then R = (lb / d_max)^2
    # We bisect on lb to find the optimal packing

    # First, get an initial feasible solution to bound the search
    # Start with a reasonable range
    lb_low = 0.0
    lb_high = 5.0  # upper bound on possible min distance

    # Find the maximum feasible lb
    # Binary search: find the largest lb such that we can pack 16 points with min dist >= lb
    # and then compute the ratio

    best_points = None
    best_ratio = -1.0

    # Coarse bisection to find feasible range
    # First check if lb_high is feasible
    success, pts, dmax = _solve_with_lower_bound(lb_high)
    if success:
        # Try higher
        for _ in range(10):
            lb_high *= 2
            success, pts, dmax = _solve_with_lower_bound(lb_high)
            if not success:
                break
    else:
        # lb_high is not feasible, lower it
        for _ in range(20):
            lb_high /= 2
            success, pts, dmax = _solve_with_lower_bound(lb_high)
            if success:
                break

    if not success:
        # Fallback to random
        np.random.seed(42)
        return np.random.randn(n, d)

    # Now bisect between lb_low (0, always feasible) and lb_high (feasible)
    # We want to maximize R = (lb / d_max(lb))^2
    # This is not monotonic in lb necessarily, but typically increasing until a point

    # Do a finer bisection search
    best_lb = 0.0
    best_pts = None
    best_dmax = np.inf

    # Sample a range of lb values
    num_samples = 30
    lb_values = np.linspace(0.01, lb_high, num_samples)

    for lb in lb_values:
        success, pts, dmax = _solve_with_lower_bound(lb)
        if success and dmax < best_dmax:
            # Compute ratio
            if dmax > 0:
                ratio = (lb / dmax) ** 2
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_lb = lb
                    best_pts = pts
                    best_dmax = dmax

    # Refine around the best lb found
    if best_pts is not None:
        # Narrow search around best_lb
        lb_range = lb_high / num_samples
        lb_fine = np.linspace(max(0.01, best_lb - lb_range), min(lb_high, best_lb + lb_range), 20)
        for lb in lb_fine:
            success, pts, dmax = _solve_with_lower_bound(lb)
            if success and dmax > 0:
                ratio = (lb / dmax) ** 2
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_lb = lb
                    best_pts = pts
                    best_dmax = dmax

    if best_pts is not None:
        return best_pts
    else:
        # Fallback
        np.random.seed(42)
        return np.random.randn(n, d)


# EVOLVE-BLOCK-END
