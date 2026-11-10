import numpy as np
import time
from scipy.spatial.distance import pdist

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.
    """
    start_time = time.time()
    TIME_LIMIT = 600.0
    SAFETY_MARGIN = 15.0

    def get_score(pts):
        """Calculate the true metric R = (d_min / d_max)^2. Higher is better."""
        d = pdist(pts)
        if len(d) == 0:
            return 0.0
        d_min = np.min(d)
        d_max = np.max(d)
        if d_min <= 1e-12 or d_max <= 1e-12:
            return 0.0
        return (d_min / d_max) ** 2

    def get_cost(pts):
        """Calculate (d_max / d_min)^2. Lower is better."""
        d = pdist(pts)
        if len(d) == 0:
            return np.inf
        d_min = np.min(d)
        d_max = np.max(d)
        if d_min <= 1e-12 or d_max <= 1e-12:
            return np.inf
        return (d_max / d_min) ** 2

    def hexagonal_lattice_init(n_points=16):
        """Generate a hexagonal lattice subset centered at origin."""
        i_range = np.arange(-3, 4)
        j_range = np.arange(-3, 4)
        lattice_points = []
        for i in i_range:
            for j in j_range:
                x = i + 0.5 * j
                y = (np.sqrt(3) / 2) * j
                lattice_points.append([x, y])
        lattice_points = np.array(lattice_points)
        dists = np.linalg.norm(lattice_points, axis=1)
        idx = np.argsort(dists)
        hex_pts = lattice_points[idx[:n_points]]
        hex_pts = hex_pts - np.mean(hex_pts, axis=0)
        return hex_pts

    def grid_init(n=4):
        """Generate a square grid centered at origin."""
        g = np.array([(i, j) for i in range(n) for j in range(n)], dtype=float)
        g = g - np.mean(g, axis=0)
        return g

    def random_init(scale=1.0):
        """Generate random points centered at origin."""
        return np.random.randn(16, 2) * scale

    def two_ring_init():
        """Generate two concentric rings of 8 points each."""
        theta1 = np.linspace(0, 2 * np.pi, 8, endpoint=False)
        theta2 = theta1 + np.pi / 8
        r1, r2 = 1.0, 1.7
        pts = np.vstack([
            np.column_stack([r1 * np.cos(theta1), r1 * np.sin(theta1)]),
            np.column_stack([r2 * np.cos(theta2), r2 * np.sin(theta2)])
        ])
        return pts - np.mean(pts, axis=0)

    def three_ring_init():
        """Generate three rings: 4, 6, 6 points."""
        t1 = np.linspace(0, 2 * np.pi, 4, endpoint=False)
        t2 = np.linspace(0, 2 * np.pi, 6, endpoint=False)
        t3 = t2 + np.pi / 6
        pts = np.vstack([
            np.column_stack([np.cos(t1), np.sin(t1)]),
            np.column_stack([1.5 * np.cos(t2), 1.5 * np.sin(t2)]),
            np.column_stack([2.0 * np.cos(t3), 2.0 * np.sin(t3)])
        ])
        return pts - np.mean(pts, axis=0)

    # Generate diverse initial candidates
    rng_init = np.random.RandomState(123)
    candidates = [
        grid_init(4),
        hexagonal_lattice_init(16),
        np.column_stack([np.cos(np.linspace(0, 2 * np.pi, 16, endpoint=False)),
                         np.sin(np.linspace(0, 2 * np.pi, 16, endpoint=False))]),
        two_ring_init(),
        three_ring_init(),
        np.vstack([
            np.column_stack([np.cos(np.linspace(0, 2 * np.pi, 8, endpoint=False)),
                             np.sin(np.linspace(0, 2 * np.pi, 8, endpoint=False))]),
            np.column_stack([1.8 * np.cos(np.linspace(0, 2 * np.pi, 8, endpoint=False) + np.pi / 8),
                             1.8 * np.sin(np.linspace(0, 2 * np.pi, 8, endpoint=False) + np.pi / 8)])
        ]),
        rng_init.randn(16, 2),
        rng_init.randn(16, 2) * 0.5,
        rng_init.randn(16, 2) * 1.5,
    ]

    best_score = np.inf
    best_points = None

    # Evaluate initial candidates
    for c in candidates:
        pts = np.asarray(c).reshape(16, 2)
        if not np.all(np.isfinite(pts)):
            continue
        score = get_cost(pts)
        if np.isfinite(score) and score < best_score:
            best_score = score
            best_points = pts.copy()

    # Simulated Annealing with fast cost evaluation
    rng = np.random.RandomState(42)
    T0 = 0.5
    T_end = 1e-5
    cooling_rate = 0.9998
    max_iter_per_run = 300000

    for c in candidates:
        if time.time() - start_time > TIME_LIMIT - SAFETY_MARGIN:
            break

        current_pts = np.asarray(c).reshape(16, 2).copy()
        current_cost = get_cost(current_pts)

        if not np.isfinite(current_cost):
            continue

        T = T0
        no_improve = 0
        max_no_improve = 40000

        for iteration in range(max_iter_per_run):
            if time.time() - start_time > TIME_LIMIT - SAFETY_MARGIN:
                break

            # Generate perturbation
            n_moves = rng.randint(1, 4)
            indices = rng.choice(16, size=n_moves, replace=False)
            step_size = T * 0.3
            new_pts = current_pts.copy()
            new_pts[indices] += rng.randn(n_moves, 2) * step_size

            new_cost = get_cost(new_pts)

            if np.isfinite(new_cost):
                if new_cost < current_cost:
                    current_pts = new_pts
                    current_cost = new_cost
                    no_improve = 0
                    if current_cost < best_score:
                        best_score = current_cost
                        best_points = current_pts.copy()
                else:
                    delta = new_cost - current_cost
                    if delta < 50.0 and rng.rand() < np.exp(-delta / max(T, 1e-10)):
                        current_pts = new_pts
                        current_cost = new_cost
                    else:
                        no_improve += 1

            T *= cooling_rate
            if T < T_end:
                break

            if no_improve > max_no_improve:
                break

    # Also run a dedicated SA from the best point found
    if best_points is not None and time.time() - start_time < TIME_LIMIT - SAFETY_MARGIN:
        current_pts = best_points.copy()
        current_cost = get_cost(current_pts)

        if np.isfinite(current_cost):
            T = 0.3
            no_improve = 0
            for iteration in range(200000):
                if time.time() - start_time > TIME_LIMIT - SAFETY_MARGIN:
                    break

                n_moves = rng.randint(1, 3)
                indices = rng.choice(16, size=n_moves, replace=False)
                step_size = T * 0.2
                new_pts = current_pts.copy()
                new_pts[indices] += rng.randn(n_moves, 2) * step_size

                new_cost = get_cost(new_pts)

                if np.isfinite(new_cost):
                    if new_cost < current_cost:
                        current_pts = new_pts
                        current_cost = new_cost
                        no_improve = 0
                        if current_cost < best_score:
                            best_score = current_cost
                            best_points = current_pts.copy()
                    else:
                        delta = new_cost - current_cost
                        if delta < 30.0 and rng.rand() < np.exp(-delta / max(T, 1e-10)):
                            current_pts = new_pts
                            current_cost = new_cost
                        else:
                            no_improve += 1

                T *= 0.9999
                if T < 1e-5:
                    break
                if no_improve > 30000:
                    break

    # Final local refinement using Nelder-Mead on best solution
    if best_points is not None and time.time() - start_time < TIME_LIMIT - SAFETY_MARGIN:
        from scipy.optimize import minimize

        def objective(params):
            pts = params.reshape(16, 2)
            return get_cost(pts)

        res = minimize(objective, best_points.flatten(), method='Nelder-Mead',
                       options={'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True})

        if hasattr(res, 'fun') and np.isfinite(res.fun):
            if res.fun < best_score:
                best_score = res.fun
                best_points = res.x.reshape(16, 2)

    # Second polish pass with tighter tolerance
    if best_points is not None and time.time() - start_time < TIME_LIMIT - SAFETY_MARGIN:
        from scipy.optimize import minimize

        def objective(params):
            pts = params.reshape(16, 2)
            return get_cost(pts)

        res = minimize(objective, best_points.flatten(), method='Nelder-Mead',
                       options={'maxiter': 10000, 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True})

        if hasattr(res, 'fun') and np.isfinite(res.fun):
            if res.fun < best_score:
                best_score = res.fun
                best_points = res.x.reshape(16, 2)

    # Ensure valid output
    if best_points is None:
        best_points = grid_init(4)

    final_pts = best_points.reshape(16, 2)
    if not np.all(np.isfinite(final_pts)):
        final_pts = grid_init(4)

    return final_pts
