import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 290

        best_x = None
        best_f = float('inf')
        calls = 0

        def eval_func(x):
            nonlocal best_x, best_f, calls
            x = np.clip(x, -5, 5)
            f = func(x)
            calls += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        rng = np.random.default_rng(42)

        # Phase 1: Quick population exploration to get initial best
        pop_size = 8
        population = [np.zeros(dim)]
        for _ in range(pop_size - 1):
            population.append(rng.uniform(-4, 4, dim))

        pop_f = []
        for x in population:
            if calls >= budget or time.time() > deadline:
                break
            pop_f.append(eval_func(x))

        if calls >= budget or time.time() > deadline:
            return best_x, best_f

        # Phase 2: Estimate rotation by probing
        # For Rastrigin rotated, f(x) = sum(10*(1-cos(2*pi*x_i_rotated)) + x_i_rotated^2)
        # where x_rotated = R @ (x - shift)
        # We can estimate R by finding points that minimize each coordinate
        # Strategy: use the known structure - probe along axes to detect rotation

        # Better approach: generate lattice candidates directly
        # The minimum of Rastrigin in rotated coords is at 2*pi*k for integer k
        # So in original coords: x = shift + R^T @ (2*pi*k)
        # We don't know shift or R, but we can estimate them

        # Estimate shift: the function is symmetric, so try to find the center
        # Use the initial population to get a rough center
        center_est = np.mean(population, axis=0)

        # Estimate rotation: probe along each axis from center
        # For a rotated Rastrigin, moving along original axis d changes rotated coords
        # We can estimate R columns by seeing which direction decreases f most

        # Actually, let's use a different approach:
        # Generate candidates at integer lattice points in rotated space
        # For each dimension, try k in {-1, 0, 1} (since range is [-5,5] and 2*pi ~ 6.28)
        # This gives 3^5 = 243 candidates, which is manageable

        # But we need to know R and shift. Let's estimate them.

        # Estimate shift by minimizing a quadratic approximation
        # Or simpler: use the fact that the minimum is in [-4,4]^5

        # Let's try a direct approach: generate grid candidates in original space
        # that are likely to be near rotated-space minima

        # Phase 2: Lattice-based candidate generation
        # Generate candidates at points that are likely minima
        # For Rastrigin, minima occur at x_rot = 2*pi*k
        # So x = shift + R^T @ (2*pi*k)

        # Estimate R and shift using a few probes
        # Probe: evaluate at center and at center +/- small perturbations
        # to estimate the local gradient and Hessian

        # Actually, let's use a simpler heuristic:
        # The rotation is orthogonal, so R^T = R^-1
        # We can estimate R by finding the directions of steepest descent

        # For now, let's generate candidates on a grid in the original space
        # that covers the likely minimum region

        # Phase 2: Generate lattice candidates
        # Try integer multiples of 2*pi in rotated space, mapped back
        # Since we don't know R, let's try a different approach:
        # Evaluate at points that are "lattice-like" in the original space

        # Better: use the population to estimate where the minimum might be
        # Then generate fine grid around the best point

        # Phase 2: Fine grid search around the best point
        # The minimum of Rastrigin is at integer multiples of 2*pi in rotated coords
        # In original coords, this is a rotated grid
        # Let's generate candidates on a fine grid

        # Use the best point from Phase 1 as center
        if best_x is not None:
            center = best_x.copy()
        else:
            center = np.zeros(dim)

        # Generate candidates on a grid around center
        # Step size: 2*pi / 4 = pi/2 ~ 1.57, but let's use smaller
        # Actually, for Rastrigin, the period is 2*pi, so let's use step = pi
        # But we want to catch the minima, so use step = 2*pi / n for some n

        # Let's try step = 1.0 (since 2*pi ~ 6.28, step=1 gives ~6 points per period)
        # Range: center +/- 3*step in each dimension
        step = 1.0
        n_points_per_dim = 7  # -3 to +3 in steps of 1

        # Generate grid
        grid_vals = np.linspace(center - 3*step, center + 3*step, n_points_per_dim)
        grid_candidates = []
        for vals in np.ndindex(*([n_points_per_dim]*dim)):
            x = np.array([grid_vals[d] for d in range(dim)])
            x = np.clip(x, -5, 5)
            grid_candidates.append(x)

        # Sort by distance to center (closer first)
        grid_candidates.sort(key=lambda x: np.sum((x - center)**2))

        # Evaluate top candidates
        max_grid_evals = min(100, budget - calls)
        for x in grid_candidates[:max_grid_evals]:
            if calls >= budget or time.time() > deadline:
                break
            eval_func(x)

        if calls >= budget or time.time() > deadline:
            return best_x, best_f

        # Phase 3: (1+1)-ES exploitation from the best point
        if best_x is not None and calls < budget and time.time() < deadline:
            x = best_x.copy()
            fx = best_f
            sigma_es = 0.5

            window = 10
            recent = [False] * window
            idx = 0

            max_es_calls = min(150, budget - calls)
            for _ in range(max_es_calls):
                if calls >= budget or time.time() > deadline:
                    break

                y = np.clip(x + sigma_es * rng.standard_normal(dim), -5, 5)
                fy = eval_func(y)

                if fy < fx:
                    x, fx = y, fy
                    recent[idx % window] = True
                else:
                    recent[idx % window] = False
                idx += 1

                if idx >= window:
                    sr = sum(recent) / window
                    if sr > 0.4:
                        sigma_es *= 1.15
                    elif sr < 0.2:
                        sigma_es *= 0.85
                    else:
                        sigma_es *= 1.0

                sigma_es = np.clip(sigma_es, 0.05, 3.0)

        # Phase 4: Coordinate-wise refinement
        if best_x is not None and calls < budget and time.time() < deadline:
            x_local = best_x.copy()
            f_local = best_f

            step = 0.3
            for _ in range(40):
                if calls >= budget or time.time() > deadline:
                    break
                improved = False
                for d in range(dim):
                    if calls >= budget or time.time() > deadline:
                        break
                    for sign in [1, -1]:
                        if calls >= budget or time.time() > deadline:
                            break
                        y = x_local.copy()
                        y[d] += sign * step
                        y = np.clip(y, -5, 5)
                        fy = eval_func(y)
                        if fy < f_local:
                            x_local, f_local = y, fy
                            improved = True
                            break
                if not improved:
                    step *= 0.5
                    if step < 0.01:
                        break

            if f_local < best_f:
                best_f = f_local
                best_x = x_local.copy()

        return best_x, best_f
