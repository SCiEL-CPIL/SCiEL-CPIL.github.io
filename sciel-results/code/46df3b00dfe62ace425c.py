# EVOLVE-BLOCK-START
import numpy as np
import time
import scipy.spatial.distance
import scipy.optimize


def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the (x,y) points.

    """
    n = 16
    d = 2
    start_time = time.time()
    deadline = start_time + 570  # 30s safety margin

    def get_distances(points):
        return scipy.spatial.distance.pdist(points)

    def compute_score(points):
        dists = get_distances(points)
        d_min = np.min(dists)
        d_max = np.max(dists)
        if d_max == 0:
            return 0.0
        return (d_min / d_max) ** 2

    def objective_neg(points_flat):
        points = points_flat.reshape(n, d)
        return -compute_score(points)

    def objective_neg_log(points_flat):
        """
        Smooth surrogate: maximize log(d_min) - log(d_max)
        Use soft-min and soft-max for smoothness
        """
        points = points_flat.reshape(n, d)
        dists = get_distances(points)
        if np.any(dists <= 1e-12):
            return 1e6
        # soft-min: -log-sum-exp(-dists / tau)
        tau = 0.1
        log_dmin = -np.log(np.sum(np.exp(-dists / tau))) * tau
        log_dmax = np.log(np.sum(np.exp(dists / tau))) * tau
        return -(log_dmin - log_dmax)

    # Hexagonal lattice initial guess
    def hexagonal_initial(n_points):
        points = np.zeros((n_points, 2))
        # Place points in a hexagonal pattern
        # For 16 points, use a roughly square arrangement with hexagonal spacing
        # 4x4 grid with alternating rows offset
        spacing = 1.0
        for i in range(4):
            for j in range(4):
                idx = i * 4 + j
                x = j * spacing + (i % 2) * spacing * 0.5
                y = i * spacing * np.sqrt(3) / 2
                points[idx] = [x, y]
        # Center the configuration
        points -= np.mean(points, axis=0)
        # Normalize to unit scale
        dists = scipy.spatial.distance.pdist(points)
        d_max = np.max(dists)
        if d_max > 0:
            points /= d_max
        return points

    # Random initial guesses
    def random_initial(n_points, seed=None):
        if seed is not None:
            np.random.seed(seed)
        else:
            np.random.seed(int(time.time() * 1000) % 2**31)
        points = np.random.randn(n_points, d)
        # Center
        points -= np.mean(points, axis=0)
        # Normalize
        dists = scipy.spatial.distance.pdist(points)
        d_max = np.max(dists)
        if d_max > 0:
            points /= d_max
        return points

    best_points = None
    best_score = -1.0

    # Try multiple initializations
    initializations = []

    # Hexagonal lattice
    initializations.append(hexagonal_initial(n))

    # Random initializations with different seeds
    for seed in [42, 123, 456, 789, 101, 202, 303, 404]:
        initializations.append(random_initial(n, seed=seed))

    # Also try a circular arrangement
    def circular_initial(n_points):
        angles = np.linspace(0, 2 * np.pi, n_points, endpoint=False)
        points = np.column_stack([np.cos(angles), np.sin(angles)])
        return points

    initializations.append(circular_initial(n))

    # Simulated Annealing + Nelder-Mead refinement
    for init_idx, init_points in enumerate(initializations):
        if time.time() > deadline:
            break

        current_points = init_points.copy()
        current_score = compute_score(current_points)

        if current_score > best_score:
            best_score = current_score
            best_points = current_points.copy()

        # Simulated Annealing phase
        T = 0.1  # Initial temperature
        T_min = 1e-6
        cooling_rate = 0.995
        steps_per_temp = 50
        step_size = 0.1

        sa_points = current_points.copy()
        sa_score = current_score

        for temp in np.geomspace(T, T_min, num=200):
            if time.time() > deadline:
                break

            for _ in range(steps_per_temp):
                if time.time() > deadline:
                    break

                # Random perturbation
                perturbation = np.random.randn(n, d) * step_size
                new_points = sa_points + perturbation

                # Center and normalize
                new_points -= np.mean(new_points, axis=0)
                dists = scipy.spatial.distance.pdist(new_points)
                d_max = np.max(dists)
                if d_max > 0:
                    new_points /= d_max
                else:
                    new_points = sa_points
                    continue

                new_score = compute_score(new_points)

                if new_score > sa_score or np.random.random() < np.exp((new_score - sa_score) / temp):
                    sa_points = new_points
                    sa_score = new_score

            # Update best
            if sa_score > best_score:
                best_score = sa_score
                best_points = sa_points.copy()

        # Nelder-Mead refinement on the best point found by SA
        if time.time() < deadline:
            try:
                result = scipy.optimize.minimize(
                    objective_neg,
                    sa_points.flatten(),
                    method='Nelder-Mead',
                    options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-10}
                )
                refined_points = result.x.reshape(n, d)
                # Center and normalize
                refined_points -= np.mean(refined_points, axis=0)
                dists = scipy.spatial.distance.pdist(refined_points)
                d_max = np.max(dists)
                if d_max > 0:
                    refined_points /= d_max
                refined_score = compute_score(refined_points)

                if refined_score > best_score:
                    best_score = refined_score
                    best_points = refined_points.copy()
            except Exception:
                pass

    # Final high-precision polish pass using soft-min/soft-max surrogate
    if time.time() < deadline - 30 and best_points is not None:
        try:
            result = scipy.optimize.minimize(
                objective_neg_log,
                best_points.flatten(),
                method='L-BFGS-B',
                options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            polished_points = result.x.reshape(n, d)
            # Center and normalize
            polished_points -= np.mean(polished_points, axis=0)
            dists = scipy.spatial.distance.pdist(polished_points)
            d_max = np.max(dists)
            if d_max > 0:
                polished_points /= d_max
            polished_score = compute_score(polished_points)

            if polished_score > best_score:
                best_score = polished_score
                best_points = polished_points.copy()
        except Exception:
            pass

    # One more Nelder-Mead pass on the best result
    if time.time() < deadline - 10 and best_points is not None:
        try:
            result = scipy.optimize.minimize(
                objective_neg,
                best_points.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 1000, 'xatol': 1e-10, 'fatol': 1e-12}
            )
            final_points = result.x.reshape(n, d)
            # Center and normalize
            final_points -= np.mean(final_points, axis=0)
            dists = scipy.spatial.distance.pdist(final_points)
            d_max = np.max(dists)
            if d_max > 0:
                final_points /= d_max
            final_score = compute_score(final_points)

            if final_score > best_score:
                best_score = final_score
                best_points = final_points.copy()
        except Exception:
            pass

    if best_points is None:
        # Fallback to random
        np.random.seed(42)
        best_points = np.random.randn(n, d)

    return best_points


# EVOLVE-BLOCK-END
