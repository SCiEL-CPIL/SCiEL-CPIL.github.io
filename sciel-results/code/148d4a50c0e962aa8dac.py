"""Ridge-aware optimizer using Nelder-Mead with randomized initialization and adaptive polishing."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Nelder-Mead optimizer with strict budget control and adaptive refinement."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin
        dim = self.dim
        budget = self.budget

        rng = np.random.default_rng(42)

        def clip(v):
            return np.clip(np.asarray(v, dtype=float), -5.0, 5.0)

        def time_ok():
            return time.time() < deadline

        best_x = None
        best_f = np.inf
        calls_used = 0

        # Nelder-Mead parameters
        alpha = 1.0
        gamma = 2.0
        rho = 0.5
        sigma = 0.5
        max_iter = 200

        def run_nelder_mead(center, initial_step, random_init=False):
            nonlocal best_x, best_f, calls_used

            # Initialize simplex
            simplex = []
            simplex_f = []

            # First point is center
            x0 = clip(center)
            f0 = func(x0)
            calls_used += 1
            if f0 < best_f:
                best_f = f0
                best_x = x0.copy()
            simplex.append(x0)
            simplex_f.append(f0)

            if calls_used >= budget - 5:
                return

            # Other points
            if random_init:
                # Generate random orthogonal-ish directions
                for i in range(dim):
                    if calls_used >= budget - 5:
                        return
                    direction = rng.normal(0, 1, dim)
                    norm = np.linalg.norm(direction)
                    if norm > 1e-10:
                        direction /= norm
                    x_i = center + initial_step * direction
                    x_i = clip(x_i)
                    f_i = func(x_i)
                    calls_used += 1
                    if f_i < best_f:
                        best_f = f_i
                        best_x = x_i.copy()
                    simplex.append(x_i)
                    simplex_f.append(f_i)
            else:
                # Axis-aligned
                for i in range(dim):
                    if calls_used >= budget - 5:
                        return
                    x_i = center.copy()
                    x_i[i] += initial_step
                    x_i = clip(x_i)
                    f_i = func(x_i)
                    calls_used += 1
                    if f_i < best_f:
                        best_f = f_i
                        best_x = x_i.copy()
                    simplex.append(x_i)
                    simplex_f.append(f_i)

            # Main NM loop
            for _ in range(max_iter):
                if not time_ok() or calls_used >= budget - 5:
                    break

                # Sort simplex by fitness
                order = np.argsort(simplex_f)
                simplex = [simplex[i] for i in order]
                simplex_f = [simplex_f[i] for i in order]

                # Check for convergence
                if max(simplex_f) - min(simplex_f) < 1e-12:
                    break

                # Centroid of all but worst
                centroid = np.mean(simplex[:-1], axis=0)
                worst = simplex[-1]
                worst_f = simplex_f[-1]

                # Reflection
                x_r = centroid + alpha * (centroid - worst)
                x_r = clip(x_r)
                f_r = func(x_r)
                calls_used += 1
                if f_r < best_f:
                    best_f = f_r
                    best_x = x_r.copy()

                if calls_used >= budget - 5:
                    break

                if f_r < simplex_f[0]:
                    # Expansion
                    x_e = centroid + gamma * (x_r - centroid)
                    x_e = clip(x_e)
                    f_e = func(x_e)
                    calls_used += 1
                    if f_e < best_f:
                        best_f = f_e
                        best_x = x_e.copy()

                    if calls_used >= budget - 5:
                        break

                    if f_e < f_r:
                        simplex[-1] = x_e
                        simplex_f[-1] = f_e
                    else:
                        simplex[-1] = x_r
                        simplex_f[-1] = f_r
                elif f_r < simplex_f[-2]:
                    # Accept reflection
                    simplex[-1] = x_r
                    simplex_f[-1] = f_r
                else:
                    # Contraction
                    if f_r < worst_f:
                        # Outer contraction
                        x_c = centroid + rho * (x_r - centroid)
                        x_c = clip(x_c)
                        f_c = func(x_c)
                        calls_used += 1
                        if f_c < best_f:
                            best_f = f_c
                            best_x = x_c.copy()

                        if calls_used >= budget - 5:
                            break

                        if f_c < f_r:
                            simplex[-1] = x_c
                            simplex_f[-1] = f_c
                        else:
                            # Shrink
                            for i in range(1, len(simplex)):
                                if calls_used >= budget - 5:
                                    return
                                simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                                simplex[i] = clip(simplex[i])
                                f_s = func(simplex[i])
                                calls_used += 1
                                if f_s < best_f:
                                    best_f = f_s
                                    best_x = simplex[i].copy()
                                simplex_f[i] = f_s
                    else:
                        # Inner contraction
                        x_c = centroid - rho * (centroid - worst)
                        x_c = clip(x_c)
                        f_c = func(x_c)
                        calls_used += 1
                        if f_c < best_f:
                            best_f = f_c
                            best_x = x_c.copy()

                        if calls_used >= budget - 5:
                            break

                        if f_c < worst_f:
                            simplex[-1] = x_c
                            simplex_f[-1] = f_c
                        else:
                            # Shrink
                            for i in range(1, len(simplex)):
                                if calls_used >= budget - 5:
                                    return
                                simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                                simplex[i] = clip(simplex[i])
                                f_s = func(simplex[i])
                                calls_used += 1
                                if f_s < best_f:
                                    best_f = f_s
                                    best_x = simplex[i].copy()
                                simplex_f[i] = f_s

        def estimate_ridge_direction(x_center, h=0.1):
            """Estimate the ridge direction using gradient approximation."""
            nonlocal calls_used
            if calls_used >= budget - 10:
                return None
            
            f_center = func(x_center)
            calls_used += 1
            
            # Estimate gradient using central differences
            grad = np.zeros(dim)
            for i in range(dim):
                if calls_used >= budget - 10:
                    return None
                x_plus = x_center.copy()
                x_plus[i] += h
                x_minus = x_center.copy()
                x_minus[i] -= h
                x_plus = clip(x_plus)
                x_minus = clip(x_minus)
                f_plus = func(x_plus)
                calls_used += 1
                if calls_used >= budget - 10:
                    return None
                f_minus = func(x_minus)
                calls_used += 1
                grad[i] = (f_plus - f_minus) / (2 * h)
            
            norm = np.linalg.norm(grad)
            if norm < 1e-10:
                return None
            return grad / norm

        def ridge_aware_polish(center, steps=100):
            """Adaptive random search with ridge-aware direction bias."""
            nonlocal best_x, best_f, calls_used

            x = clip(center).copy()
            f_curr = func(x)
            calls_used += 1
            if f_curr < best_f:
                best_f = f_curr
                best_x = x.copy()

            # Adaptive step size with momentum
            sigma = 0.3
            sigma_min = 1e-5
            sigma_max = 1.5
            velocity = np.zeros(dim)
            momentum = 0.5
            ridge_dir = None
            ridge_strength = 0.0

            success_count = 0
            total_count = 0
            window_size = 8

            for step_idx in range(steps):
                if not time_ok() or calls_used >= budget - 2:
                    break

                # Periodically re-estimate ridge direction
                if step_idx % 15 == 0 and step_idx > 0:
                    grad_dir = estimate_ridge_direction(x)
                    if grad_dir is not None:
                        # Ridge direction is approximately perpendicular to gradient
                        # For a sharp ridge, the function changes slowly along the ridge
                        # and quickly in perpendicular directions
                        # We want to move along directions where the function is flat
                        # Use the null space of the gradient
                        if dim == 1:
                            ridge_dir = None
                        else:
                            # Project random direction onto orthogonal complement of gradient
                            rand_dir = rng.standard_normal(dim)
                            rand_dir -= np.dot(rand_dir, grad_dir) * grad_dir
                            rand_norm = np.linalg.norm(rand_dir)
                            if rand_norm > 1e-10:
                                ridge_dir = rand_dir / rand_norm
                                ridge_strength = 0.4
                            else:
                                ridge_dir = None
                                ridge_strength = 0.0

                # Generate candidate with adaptive sigma and momentum
                noise = rng.standard_normal(dim)
                
                # Bias towards ridge direction if available
                if ridge_dir is not None and ridge_strength > 0:
                    noise = (1 - ridge_strength) * noise + ridge_strength * ridge_dir * np.linalg.norm(noise)
                
                if np.linalg.norm(velocity) > 1e-8:
                    dir_unit = velocity / np.linalg.norm(velocity)
                    noise = 0.6 * noise + 0.4 * dir_unit * np.linalg.norm(noise)

                candidate = x + sigma * noise
                candidate = clip(candidate)
                f_new = func(candidate)
                calls_used += 1

                if f_new < f_curr:
                    # Improvement
                    velocity = momentum * velocity + (1 - momentum) * (candidate - x)
                    x = candidate
                    f_curr = f_new
                    sigma = min(sigma * 1.1, sigma_max)
                    success_count += 1

                    if f_new < best_f:
                        best_f = f_new
                        best_x = x.copy()
                else:
                    # No improvement
                    velocity = momentum * velocity
                    sigma = max(sigma * 0.9, sigma_min)

                total_count += 1

                # Periodically adjust sigma based on success rate
                if total_count % window_size == 0:
                    success_rate = success_count / total_count
                    if success_rate > 0.6:
                        sigma = min(sigma * 1.05, sigma_max)
                    elif success_rate < 0.2:
                        sigma = max(sigma * 0.8, sigma_min)
                    success_count = 0
                    total_count = 0

                # Occasionally try a larger jump to escape local optima
                if step_idx % 25 == 24 and sigma < 1.0:
                    big_noise = rng.standard_normal(dim) * 0.8
                    if ridge_dir is not None:
                        big_noise += 0.5 * ridge_dir * np.linalg.norm(big_noise)
                    big_candidate = clip(x + big_noise)
                    if calls_used < budget - 2:
                        big_fy = func(big_candidate)
                        calls_used += 1
                        if big_fy < f_curr:
                            x = big_candidate
                            f_curr = big_fy
                            sigma = min(sigma * 1.2, sigma_max)
                            if big_fy < best_f:
                                best_f = big_fy
                                best_x = x.copy()

        def fine_polish(center, max_steps=50):
            """Very fine local search with decreasing step sizes."""
            nonlocal best_x, best_f, calls_used
            
            x = clip(center).copy()
            f_curr = func(x)
            calls_used += 1
            if f_curr < best_f:
                best_f = f_curr
                best_x = x.copy()
            
            step_sizes = [0.05, 0.02, 0.01, 0.005, 0.002, 0.001, 0.0005, 0.0002, 0.0001]
            
            for step in step_sizes:
                for _ in range(max(1, max_steps // len(step_sizes))):
                    if not time_ok() or calls_used >= budget - 2:
                        return
                    direction = rng.normal(0, 1, dim)
                    norm = np.linalg.norm(direction)
                    if norm > 1e-10:
                        direction /= norm
                    x_new = clip(x + step * direction)
                    f_new = func(x_new)
                    calls_used += 1
                    if f_new < f_curr:
                        f_curr = f_new
                        x = x_new.copy()
                        if f_new < best_f:
                            best_f = f_new
                            best_x = x_new.copy()
                    else:
                        # Try opposite direction
                        x_new2 = clip(x - step * direction)
                        if calls_used < budget - 2:
                            f_new2 = func(x_new2)
                            calls_used += 1
                            if f_new2 < f_curr:
                                f_curr = f_new2
                                x = x_new2.copy()
                                if f_new2 < best_f:
                                    best_f = f_new2
                                    best_x = x_new2.copy()

        # Phase 0: Quick diverse initialization for early progress
        init_points = [
            np.zeros(dim),
            np.array([3.0] * dim),
            np.array([-3.0] * dim),
            np.array([2.0, -2.0, 2.0, -2.0, 2.0]),
            np.array([-2.0, 2.0, -2.0, 2.0, -2.0]),
        ]
        for pt in init_points:
            if calls_used >= budget - 5:
                break
            x_test = clip(pt)
            f_test = func(x_test)
            calls_used += 1
            if f_test < best_f:
                best_f = f_test
                best_x = x_test.copy()

        # Phase 1: Coarse search with randomized initialization from best found
        center = best_x if best_x is not None else np.zeros(dim)
        run_nelder_mead(center, 2.5, random_init=True)

        # Phase 2: Finer search / Restart from best
        if calls_used < budget - 60 and best_x is not None:
            # Restart from best with smaller step
            run_nelder_mead(best_x, 1.0, random_init=False)

        # Phase 3: Another restart with random init for exploration
        if calls_used < budget - 50 and best_x is not None:
            perturbed_center = best_x + rng.normal(0, 0.4, dim)
            run_nelder_mead(perturbed_center, 0.4, random_init=True)

        # Phase 4: Ridge-aware adaptive polish
        if calls_used < budget - 15 and best_x is not None:
            ridge_aware_polish(best_x, steps=100)

        # Phase 5: Final fine polish
        if calls_used < budget - 5 and best_x is not None:
            fine_polish(best_x, max_steps=40)

        if best_x is None:
            # Fallback
            best_x = np.zeros(dim)
            best_f = func(best_x)
            calls_used += 1

        return best_x, best_f
# EVOLVE-BLOCK-END
