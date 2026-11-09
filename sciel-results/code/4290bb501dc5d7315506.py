"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Coarse-to-fine Halton grid search combined with adaptive (1+1)-ES refinement."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 280  # Safety margin before 300s limit

        rng = np.random.default_rng(42)

        # Global best tracking
        best_x = np.zeros(dim)
        best_f = func(best_x)
        calls = 1

        # Halton sequence generator
        def halton_sequence(index, base):
            q = 1.0
            r = 0.0
            while index > 0:
                q /= base
                r += q * (index % base)
                index //= base
            return r

        primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]

        def generate_halton_points(n, dim):
            """Generate n Halton points in [0,1]^dim."""
            points = np.zeros((n, dim))
            for i in range(1, n + 1):
                for j in range(dim):
                    base = primes[j % len(primes)]
                    points[i - 1, j] = halton_sequence(i, base)
            return points

        # Phase 1: Coarse-to-fine Halton grid search
        # Start with a coarse grid, then refine around the best point found
        coarse_counts = [15, 30, 50, 80]  # Number of Halton points at each level
        current_best_x = best_x.copy()
        current_best_f = best_f

        for level, n_points in enumerate(coarse_counts):
            if calls >= budget:
                break
            if time.time() > deadline:
                break

            # Generate Halton points
            halton_pts = generate_halton_points(n_points, dim)
            
            # At coarse levels, sample over the full domain
            # At finer levels, sample around the current best with decreasing radius
            if level == 0:
                # Full domain sampling
                points = halton_pts * 10.0 - 5.0
            else:
                # Refine around current best with decreasing radius
                radius = 4.0 / (level + 1)  # 4, 2, 1.33, 1.0
                points = current_best_x + (halton_pts * 2.0 - 1.0) * radius
                points = np.clip(points, -5, 5)

            for pt in points:
                if calls >= budget:
                    break
                if time.time() > deadline:
                    break
                pt = np.clip(pt, -5, 5)
                f = func(pt)
                calls += 1
                if f < current_best_f:
                    current_best_f = f
                    current_best_x = pt.copy()
                    if f < best_f:
                        best_f = f
                        best_x = pt.copy()

        # Phase 2: Adaptive (1+1)-ES refinement from the best point found
        x = best_x.copy()
        fx = best_f
        sigma = 0.5  # Start with a moderate step size for refinement

        # Track success rate for 1/5th rule adaptation
        success_window = []
        window_size = 15

        # Stagnation tracking
        stagnation = 0
        no_improve_count = 0

        # Adaptive parameters
        sigma_min = 0.001
        sigma_max = 2.0

        # Budget allocation: use last 20% for final polish
        polish_start = int(budget * 0.80)

        while calls < budget:
            if time.time() > deadline:
                break

            # Determine if we're in polish phase
            in_polish = calls >= polish_start
            if in_polish:
                sigma = min(sigma, 0.05)

            # Generate offspring
            y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
            fy = func(y)
            calls += 1

            # Track success
            success = fy < fx
            success_window.append(success)
            if len(success_window) > window_size:
                success_window.pop(0)

            # 1/5th success rule for sigma adaptation
            if len(success_window) >= window_size:
                success_rate = sum(success_window) / len(success_window)
                if success_rate > 0.3:
                    sigma = min(sigma * 1.25, sigma_max)
                elif success_rate < 0.1:
                    sigma = max(sigma * 0.75, sigma_min)

            # Update current solution
            if success:
                x, fx = y, fy
                stagnation = 0
            else:
                stagnation += 1

            # Update global best
            if fy < best_f:
                best_f = fy
                best_x = y.copy()
                no_improve_count = 0
            else:
                no_improve_count += 1

            # Adaptive restart conditions
            progress_ratio = calls / budget
            stagnation_limit = int(30 * (1.0 - 0.5 * progress_ratio))
            no_improve_limit = int(50 * (1.0 - 0.5 * progress_ratio))

            if in_polish:
                stagnation_limit = 10
                no_improve_limit = 20

            # Restart if stagnated
            if stagnation >= stagnation_limit:
                # Try a local perturbation around the best point
                local_sigma = max(sigma * 2.0, 0.3)
                x = np.clip(best_x + local_sigma * rng.standard_normal(dim), -5, 5)
                sigma = local_sigma

                fx = func(x)
                calls += 1
                stagnation = 0
                no_improve_count = 0
                success_window = []

                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()

                if calls >= budget:
                    break

            # Aggressive restart if no global improvement for a long time
            if no_improve_count >= no_improve_limit:
                # Try a few random restarts and pick the best
                best_restart_x = x
                best_restart_f = fx
                n_restarts = 3
                for _ in range(n_restarts):
                    if calls >= budget:
                        break
                    cand = rng.uniform(-5, 5, dim)
                    cand_f = func(cand)
                    calls += 1
                    if cand_f < best_restart_f:
                        best_restart_f = cand_f
                        best_restart_x = cand
                    if cand_f < best_f:
                        best_f = cand_f
                        best_x = cand.copy()

                x = best_restart_x
                fx = best_restart_f
                sigma = 1.0
                stagnation = 0
                no_improve_count = 0
                success_window = []

                if calls >= budget:
                    break

        return best_x, best_f
# EVOLVE-BLOCK-END
