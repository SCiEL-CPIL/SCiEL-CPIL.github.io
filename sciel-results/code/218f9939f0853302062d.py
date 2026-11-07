"""Adaptive covariance-free search with rotation adaptation and Nelder-Mead polish for BBOB f12."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Covariance-free adaptive search with rotation-based anisotropy and NM polish."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 275.0  # 25s safety margin before 300s limit
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        # Track global best
        best_x = np.zeros(dim)
        best_f = np.inf
        total_calls = 0

        def check_budget():
            nonlocal total_calls
            if total_calls >= budget or time.time() > deadline:
                return False
            return True

        def record(x, f_val):
            nonlocal best_x, best_f, total_calls
            total_calls += 1
            if f_val < best_f:
                best_f = f_val
                best_x = x.copy()
            return f_val

        # Initialize with center
        center = np.zeros(dim)
        f_center = func(center)
        record(center, f_center)

        # Phase 1: Adaptive search with rotation matrix
        # Maintain a rotation matrix R that adapts to align with successful directions
        R = np.eye(dim)
        sigma = 2.0  # Initial step size
        success_history = []  # Store (direction, improvement) pairs
        max_history = 20
        sigma_min = 0.01
        sigma_max = 5.0

        # Number of steps in adaptive phase (use most of budget)
        adaptive_budget = int(budget * 0.85)

        for step in range(adaptive_budget):
            if not check_budget():
                break

            # Generate candidate in rotated space
            z = rng.standard_normal(dim)
            x_candidate = best_x + sigma * R @ z

            # Clip to bounds
            x_candidate = np.clip(x_candidate, -5, 5)

            f_candidate = func(x_candidate)
            record(x_candidate, f_candidate)

            if f_candidate < best_f + 1e-10:  # Success (including equal)
                # Update rotation to align with successful direction
                direction = x_candidate - best_x
                norm_dir = np.linalg.norm(direction)
                if norm_dir > 1e-10:
                    direction = direction / norm_dir
                    success_history.append((direction, best_f - f_candidate))
                    if len(success_history) > max_history:
                        success_history.pop(0)

                    # Update rotation: blend current R with rank-1 update toward direction
                    # Simple heuristic: project R's columns toward the success direction
                    weight = 0.3
                    for i in range(dim):
                        # Compute alignment of R[:, i] with direction
                        alignment = np.dot(R[:, i], direction)
                        if alignment < 0.5:  # Not well aligned
                            R[:, i] = (1 - weight) * R[:, i] + weight * direction
                            # Re-orthogonalize (simplified: just normalize)
                            norm_R = np.linalg.norm(R[:, i])
                            if norm_R > 1e-10:
                                R[:, i] /= norm_R

                # Increase sigma on success
                sigma = min(sigma * 1.05, sigma_max)
            else:
                # Failure: decrease sigma
                sigma = max(sigma * 0.95, sigma_min)

                # Occasionally randomize rotation to escape local minima
                if step % 50 == 0 and len(success_history) < 5:
                    # Partial randomization
                    angle = rng.uniform(-0.5, 0.5)
                    # Simple 2D rotation in a random plane
                    i, j = rng.choice(dim, size=2, replace=False)
                    c, s = np.cos(angle), np.sin(angle)
                    R[:, i] = c * R[:, i] - s * R[:, j]
                    R[:, j] = s * R[:, i] + c * R[:, j]
                    # Re-orthogonalize
                    for k in range(dim):
                        norm_R = np.linalg.norm(R[:, k])
                        if norm_R > 1e-10:
                            R[:, k] /= norm_R

        # Phase 2: Nelder-Mead polish
        if check_budget():
            from scipy.optimize import minimize

            # Set up NM with reasonable initial simplex
            x0 = best_x.copy()
            # Initial simplex size
            simplex_size = max(sigma * 0.5, 0.1)

            def nm_objective(x):
                nonlocal total_calls
                if total_calls >= budget or time.time() > deadline:
                    return best_f
                x_clipped = np.clip(x, -5, 5)
                val = func(x_clipped)
                record(x_clipped, val)
                return val

            try:
                res = minimize(
                    nm_objective,
                    x0,
                    method='Nelder-Mead',
                    options={
                        'maxiter': 200,
                        'fatol': 1e-12,
                        'xatol': 1e-10,
                        'initial_simplex': np.array([x0 + simplex_size * e for e in
                                                      [np.zeros(dim)] +
                                                      [np.eye(dim)[i] for i in range(dim)]])
                    }
                )
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = np.clip(res.x, -5, 5).copy()
            except Exception:
                pass

        return best_x, best_f
# EVOLVE-BLOCK-END
