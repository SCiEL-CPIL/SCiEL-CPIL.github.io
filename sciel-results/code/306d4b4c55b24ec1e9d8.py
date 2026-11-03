import numpy as np
import time
from scipy.optimize import minimize

N = 26

def construct():
    """
    Packs 26 circles in a unit square to maximize sum of radii.
    Uses a hybrid approach:
    1. Vectorized Simulated Annealing (SA) with penalty method for global exploration.
    2. Smooth objective function using soft-min for radii calculation.
    3. Gradient-based polish (L-BFGS-B) on the smooth objective to refine local optima.
    """
    start_time = time.time()
    deadline = start_time + 115.0  # 5s safety margin before 120s limit

    def is_valid(x, y, r, tol=1e-8):
        """Check if the configuration is valid (no overlaps, within bounds)."""
        # Boundary check
        if np.any(x - r < -tol) or np.any(x + r > 1 + tol):
            return False
        if np.any(y - r < -tol) or np.any(y + r > 1 + tol):
            return False
        # Overlap check
        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        d = np.sqrt(dx**2 + dy**2)
        r_sum = r[:, None] + r[None, :]
        mask = ~np.eye(N, dtype=bool)
        if np.any((d < r_sum - tol) & mask):
            return False
        return True

    def compute_radii_from_centers(centers, margin=0.0):
        """
        Given circle centers, compute maximum non-overlapping radii.
        Uses the conservative bound: r_i <= min_j dist(i,j)/2 and boundary distance.
        This is fast and provides a valid lower bound on the optimal radii for fixed centers.
        """
        centers = np.asarray(centers)
        n = len(centers)
        
        # Boundary distances
        d_left = centers[:, 0]
        d_right = 1.0 - centers[:, 0]
        d_bottom = centers[:, 1]
        d_top = 1.0 - centers[:, 1]
        radii = np.minimum(np.minimum(d_left, d_right), np.minimum(d_bottom, d_top))
        
        # Pairwise distances
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dist = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dist, np.inf)
        min_pair_dist = np.min(dist, axis=1)
        pair_constraint = min_pair_dist / 2.0
        radii = np.minimum(radii, pair_constraint)
        
        if margin > 0:
            radii = np.maximum(0.0, radii - margin)
        
        return radii

    def soft_min(values, beta=100.0):
        """Smooth approximation of min."""
        values = np.asarray(values, dtype=np.float64)
        m = np.min(values)
        if m < -1000:
            return m
        shifted = np.clip(beta * (values - m), -700, 700)
        return m - np.log(np.sum(np.exp(shifted))) / beta

    def smooth_objective(centers_flat, beta=50.0):
        """
        Objective function for gradient-based optimization.
        Maximizes the sum of soft-min radii.
        Returns negative sum for minimization.
        """
        centers = centers_flat.reshape(-1, 2)
        n = len(centers)
        
        d_left = centers[:, 0]
        d_right = 1.0 - centers[:, 0]
        d_bottom = centers[:, 1]
        d_top = 1.0 - centers[:, 1]
        
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dist = np.sqrt(np.sum(diff**2, axis=2))
        pair_constraints = dist / 2.0
        np.fill_diagonal(pair_constraints, np.inf)
        
        total_radii = 0.0
        for i in range(n):
            constraints = np.concatenate([
                [d_left[i], d_right[i], d_bottom[i], d_top[i]],
                pair_constraints[i]
            ])
            total_radii += soft_min(constraints, beta)
            
        return -total_radii

    def smooth_gradient(centers_flat, beta=50.0):
        """
        Numerical gradient of the smooth objective.
        """
        centers = centers_flat.reshape(-1, 2)
        n = len(centers)
        grad = np.zeros_like(centers_flat)
        h = 1e-6
        
        for i in range(2 * n):
            perturbed = centers_flat.copy()
            perturbed[i] += h
            f_perturbed = smooth_objective(perturbed, beta)
            grad[i] = (f_perturbed - smooth_objective(centers_flat, beta)) / h
            
        return grad

    def polish(centers, max_time=5.0):
        """
        Refine a configuration using L-BFGS-B on the smooth objective.
        Returns the polished centers and radii.
        """
        if time.time() > deadline - 2.0:
            return centers, compute_radii_from_centers(centers)
        
        # Initial radii based on centers
        r_init = compute_radii_from_centers(centers)
        
        # Use the centers as the optimization variable
        # We want to maximize sum of radii, so we minimize -sum(soft_min_radii)
        # Note: The soft_min radii depend on the centers.
        
        res = minimize(
            smooth_objective,
            centers.flatten(),
            jac=smooth_gradient,
            method='L-BFGS-B',
            options={'maxiter': 100, 'maxfun': 10000, 'disp': False}
        )
        
        if res.success or res.status in [1, 2, 3, 4, 5, 6]:
            polished_centers = res.x.reshape(-1, 2)
            # Ensure centers are within [0, 1]
            polished_centers = np.clip(polished_centers, 0, 1)
            polished_radii = compute_radii_from_centers(polished_centers)
            return polished_centers, polished_radii
        
        return centers, r_init

    best_sum = 0.0
    best_x = None
    best_y = None
    best_r = None

    # SA Parameters
    T_start = 1.0
    T_end = 1e-4
    alpha = 0.9995
    num_restarts = 50 # Increased restarts due to faster SA and polish

    for restart in range(num_restarts):
        if time.time() > deadline:
            break

        # Initialize
        x = np.random.uniform(0.1, 0.9, N)
        y = np.random.uniform(0.1, 0.9, N)
        r = np.full(N, 0.01)

        # Project to valid bounds initially
        x = np.clip(x, r, 1 - r)
        y = np.clip(y, r, 1 - r)

        current_score, _ = 0.0, False
        T = T_start

        restart_best_sum = 0
        restart_best_x = x.copy()
        restart_best_y = y.copy()
        restart_best_r = r.copy()

        # SA Loop
        steps_per_restart = 10000
        for step in range(steps_per_restart):
            if time.time() > deadline:
                break

            # Generate Move
            i = np.random.randint(N)
            move_type = np.random.random()

            x_new = x.copy()
            y_new = y.copy()
            r_new = r.copy()

            if move_type < 0.7:
                # Move position
                step_size = 0.05 * (T + 0.1)
                x_new[i] += np.random.normal(0, step_size)
                y_new[i] += np.random.normal(0, step_size)
            else:
                # Change radius
                if np.random.random() < 0.6:
                    r_new[i] += np.random.uniform(0, 0.02)
                else:
                    r_new[i] -= np.random.uniform(0, 0.02)
                r_new[i] = max(0.001, r_new[i])
                r_new[i] = min(r_new[i], 0.5)

            # Evaluate new state
            # We use a penalty method for SA, but for tracking best, we use hard validity.
            # For SA acceptance, we need a score.
            # Let's compute the "soft" score: sum(r) - penalty
            dx = x_new[:, None] - x_new[None, :]
            dy = y_new[:, None] - y_new[None, :]
            d = np.sqrt(dx**2 + dy**2)
            r_sum = r_new[:, None] + r_new[None, :]
            mask = ~np.eye(N, dtype=bool)
            overlap = np.where(mask, np.maximum(0, r_sum - d), 0)
            penalty = np.sum(overlap**2)
            
            bound_pen_x = np.sum(np.maximum(0, r_new - x_new)**2) + np.sum(np.maximum(0, x_new - (1 - r_new))**2)
            bound_pen_y = np.sum(np.maximum(0, r_new - y_new)**2) + np.sum(np.maximum(0, y_new - (1 - r_new))**2)
            
            total_penalty = penalty + bound_pen_x + bound_pen_y
            new_score = np.sum(r_new) - 10.0 * total_penalty

            delta = new_score - current_score

            # Metropolis Acceptance
            if delta > 0 or np.random.random() < np.exp(delta / T):
                x = x_new
                y = y_new
                r = r_new
                current_score = new_score

                # Check if valid and better
                if abs(current_score - np.sum(r)) < 1e-9: # Valid state
                    if np.sum(r) > restart_best_sum:
                        restart_best_sum = np.sum(r)
                        restart_best_x = x.copy()
                        restart_best_y = y.copy()
                        restart_best_r = r.copy()

            # Cool
            T *= alpha
            if T < T_end:
                T = T_end

        # End of SA restart
        if restart_best_sum > best_sum:
            best_sum = restart_best_sum
            best_x = restart_best_x
            best_y = restart_best_y
            best_r = restart_best_r

        # Polish the best found in this restart
        if time.time() < deadline - 10.0:
            centers = np.column_stack((restart_best_x, restart_best_y))
            polished_centers, polished_radii = polish(centers)
            polished_sum = np.sum(polished_radii)
            
            if polished_sum > best_sum:
                # Validate the polished result
                px, py = polished_centers[:, 0], polished_centers[:, 1]
                if is_valid(px, py, polished_radii):
                    best_sum = polished_sum
                    best_x = px
                    best_y = py
                    best_r = polished_radii
                else:
                    # If invalid, try to shrink radii to make it valid
                    scale = 1.0
                    while not is_valid(px, py, polished_radii * scale) and scale > 0.1:
                        scale *= 0.95
                    if is_valid(px, py, polished_radii * scale):
                        polished_radii *= scale
                        polished_sum = np.sum(polished_radii)
                        if polished_sum > best_sum:
                            best_sum = polished_sum
                            best_x = px
                            best_y = py
                            best_r = polished_radii

    # Final Validation and Formatting
    if best_x is None:
        # Fallback to seed
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        return np.array(cells[:N])

    # Ensure final validity
    if not is_valid(best_x, best_y, best_r):
        scale = 1.0
        while not is_valid(best_x, best_y, best_r * scale) and scale > 0.1:
            scale *= 0.95
        best_r *= scale

    result = np.column_stack((best_x, best_y, best_r))
    return result
