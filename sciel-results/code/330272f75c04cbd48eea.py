"""Optimizer using Ridge-Parallel Evolutionary Strategy for BBOB f13 (Sharp Ridge).

Strategy:
1. Maintain a population of candidate points.
2. After each generation, estimate the ridge axis from the covariance of
   recent evaluations (points with similar fitness lie along the ridge).
3. Apply anisotropic mutations: heavily damped perpendicular to the ridge,
   amplified along it.
4. Periodically re-seed with random points to escape local minima.
5. Use a final local refinement phase with small steps.
"""
# EVOLVE-BLOCK-START
import time
import numpy as np


class Optimizer:
    """
    Ridge-Parallel Evolutionary Strategy for Sharp Ridge functions.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        t0 = time.time()
        deadline = t0 + 285.0

        def elapsed_ok():
            return time.time() < deadline

        rng = np.random.default_rng(42)

        def clip(v):
            return np.clip(v, -5.0, 5.0)

        best_x = None
        best_f = np.inf
        evaluated = 0

        def eval_point(x):
            nonlocal evaluated, best_x, best_f
            x = np.asarray(x, dtype=float)
            x = clip(x)
            f = func(x)
            evaluated += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        # Population-based Ridge-Parallel ES
        pop_size = 20
        # History buffer for ridge estimation
        history_size = 40

        def ridge_parallel_es(start_x, budget_alloc, seed_offset=0):
            nonlocal evaluated, best_x, best_f

            if evaluated >= budget or not elapsed_ok():
                return

            rng_local = np.random.default_rng(42 + seed_offset)

            # Initialize population around start
            population = []
            for i in range(min(pop_size, budget_alloc)):
                if evaluated >= budget or not elapsed_ok():
                    break
                x_candidate = clip(start_x + rng_local.normal(0, 1.0, size=dim))
                f_val = eval_point(x_candidate)
                population.append((x_candidate, f_val))

            if len(population) < 3:
                return

            # Sort by fitness
            population.sort(key=lambda x: x[1])

            # Ridge axis estimation from recent similar-fitness points
            def estimate_ridge_axis(pop, n_recent=10):
                """Estimate the ridge direction from points with similar fitness."""
                if len(pop) < n_recent:
                    n_recent = len(pop)
                # Take the best n_recent points
                recent = [p[0] for p in pop[:n_recent]]
                points = np.array(recent)
                mean = points.mean(axis=0)
                centered = points - mean
                # Covariance matrix
                cov = (centered.T @ centered) / max(1, n_recent - 1)
                # Eigenvector with largest eigenvalue = ridge direction
                try:
                    eigvals, eigvecs = np.linalg.eigh(cov)
                    # Largest eigenvalue corresponds to ridge direction
                    ridge_dir = eigvecs[:, -1]
                    # Ensure consistent sign
                    if ridge_dir[0] < 0:
                        ridge_dir = -ridge_dir
                    return ridge_dir / (np.linalg.norm(ridge_dir) + 1e-15)
                except np.linalg.LinAlgError:
                    return np.ones(dim) / np.sqrt(dim)

            # Anisotropic mutation
            def anisotropic_mutation(x, ridge_dir, sigma_along, sigma_perp):
                """Apply mutation heavily damped perpendicular to ridge, amplified along it."""
                # Project x onto ridge direction and perpendicular subspace
                proj_along = np.dot(x, ridge_dir) * ridge_dir
                proj_perp = x - proj_along

                # Sample mutations
                noise_along = rng_local.normal(0, sigma_along)
                noise_perp = rng_local.normal(0, sigma_perp, size=dim)

                # Reconstruct mutated point
                new_x = proj_along + noise_along * ridge_dir + proj_perp + noise_perp
                return clip(new_x)

            # Adaptive sigma parameters
            sigma_along = 1.0
            sigma_perp = 0.3
            sigma_decay = 0.95
            sigma_min = 0.01

            # Main loop
            while evaluated < budget and elapsed_ok():
                if evaluated + pop_size > budget:
                    # Evaluate only remaining
                    pop_size_remaining = budget - evaluated
                else:
                    pop_size_remaining = pop_size

                if pop_size_remaining < 2:
                    break

                # Estimate ridge axis from current population
                ridge_dir = estimate_ridge_axis(population)

                # Generate new population
                new_population = []
                # Keep the best individual (elitism)
                new_population.append((population[0][0].copy(), population[0][1]))

                for i in range(1, pop_size_remaining):
                    if evaluated >= budget or not elapsed_ok():
                        break
                    # Select parent: weighted by rank
                    parent_idx = int(rng_local.pareto(2, 1) - 1)
                    if parent_idx >= len(population):
                        parent_idx = 0
                    parent_x = population[parent_idx][0]

                    # Anisotropic mutation
                    child_x = anisotropic_mutation(parent_x, ridge_dir, sigma_along, sigma_perp)
                    f_val = eval_point(child_x)
                    new_population.append((child_x, f_val))

                if len(new_population) < 2:
                    break

                # Sort by fitness
                new_population.sort(key=lambda x: x[1])

                # Adapt sigmas based on improvement
                old_best = population[0][1]
                new_best = new_population[0][1]
                if new_best < old_best:
                    # Good progress: slightly increase exploration
                    sigma_along = min(sigma_along * 1.05, 2.0)
                    sigma_perp = min(sigma_perp * 1.02, 0.5)
                else:
                    # No improvement: reduce step size
                    sigma_along *= sigma_decay
                    sigma_perp *= sigma_decay

                # Ensure minimum sigma
                sigma_along = max(sigma_along, sigma_min)
                sigma_perp = max(sigma_perp, sigma_min)

                population = new_population

                # Periodic re-seeding to escape local minima
                if len(population) < pop_size // 2 and evaluated < budget - pop_size and elapsed_ok():
                    # Inject some random individuals
                    for _ in range(pop_size // 4):
                        if evaluated >= budget or not elapsed_ok():
                            break
                        x_random = clip(rng_local.uniform(-4.0, 4.0, size=dim))
                        f_val = eval_point(x_random)
                        population.append((x_random, f_val))
                    population.sort(key=lambda x: x[1])
                    # Keep only top pop_size
                    if len(population) > pop_size:
                        population = population[:pop_size]

            # Local refinement
            if evaluated < budget and elapsed_ok() and best_x is not None:
                local_sigma = 0.05
                for _ in range(min(30, budget - evaluated)):
                    if evaluated >= budget or not elapsed_ok():
                        break
                    x_candidate = clip(best_x + local_sigma * rng_local.normal(0, 1, size=dim))
                    f_val = eval_point(x_candidate)
                    local_sigma *= 0.9

        # Multi-start strategy with ridge-aware initialization
        n_starts = 4
        budget_per_start = budget // n_starts

        # First start: center
        start1 = np.zeros(dim)
        ridge_parallel_es(start1, budget_per_start, seed_offset=0)

        # Second start: random
        start2 = rng.uniform(-3.0, 3.0, size=dim)
        ridge_parallel_es(start2, budget_per_start, seed_offset=100)

        # Third start: near best found
        if best_x is not None:
            start3 = best_x + rng.normal(0, 0.5, size=dim)
        else:
            start3 = rng.uniform(-3.0, 3.0, size=dim)
        ridge_parallel_es(start3, budget_per_start, seed_offset=200)

        # Fourth start: use remaining budget, near best
        remaining = budget - evaluated
        if remaining > 10 and elapsed_ok():
            if best_x is not None:
                start4 = best_x + rng.normal(0, 0.3, size=dim)
            else:
                start4 = rng.uniform(-2.0, 2.0, size=dim)
            ridge_parallel_es(start4, remaining, seed_offset=300)

        return best_x, best_f
# EVOLVE-BLOCK-END
