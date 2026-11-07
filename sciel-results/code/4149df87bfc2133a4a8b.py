"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Population-based ES with directional memory for Rosenbrock."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 300s limit

        rng = np.random.default_rng(42)

        # Initialize population
        pop_size = 10
        pop = np.array([rng.uniform(-5, 5, dim) for _ in range(pop_size)])
        fvals = np.array([func(x) for x in pop])

        # Track best
        best_idx = np.argmin(fvals)
        best_x = pop[best_idx].copy()
        best_f = fvals[best_idx]

        # Adaptive sigma per dimension (diagonal covariance)
        sigma = np.ones(dim) * 2.0

        # Directional memory: track recent successful directions
        dir_history = []
        max_history = 5

        # Success rate tracking for sigma adaptation
        success_count = 0
        total_evals = pop_size

        calls_made = pop_size

        while calls_made < budget:
            # Check time
            if time.time() > deadline:
                break

            # Generate offspring
            offspring = np.zeros_like(pop)
            offspring_fvals = np.zeros(pop_size)

            for i in range(pop_size):
                if time.time() > deadline:
                    break

                # Generate mutation
                if dir_history and rng.random() < 0.3:
                    # Use directional memory: bias toward recent success direction
                    dir_vec = dir_history[-1]
                    # Normalize and scale
                    norm = np.linalg.norm(dir_vec)
                    if norm > 1e-10:
                        dir_unit = dir_vec / norm
                        # Mix random and directional
                        noise = sigma * rng.standard_normal(dim)
                        offspring[i] = np.clip(pop[i] + 0.5 * noise + 0.5 * sigma * dir_unit * rng.uniform(0.5, 1.5), -5, 5)
                    else:
                        offspring[i] = np.clip(pop[i] + sigma * rng.standard_normal(dim), -5, 5)
                else:
                    offspring[i] = np.clip(pop[i] + sigma * rng.standard_normal(dim), -5, 5)

                offspring_fvals[i] = func(offspring[i])
                calls_made += 1

            # Evaluate all offspring (if we broke early, fill with best)
            for i in range(calls_made - pop_size, pop_size):
                if i >= 0 and i < pop_size:
                    pass  # already evaluated

            # Actually, let's redo this more carefully
            # Let me restructure

            # Combine parent and offspring, select best pop_size
            combined = np.vstack([pop, offspring])
            combined_fvals = np.concatenate([fvals, offspring_fvals])

            # Sort by fitness
            sorted_idx = np.argsort(combined_fvals)
            pop = combined[sorted_idx[:pop_size]]
            fvals = combined_fvals[sorted_idx[:pop_size]]

            # Update best
            if fvals[0] < best_f:
                old_best = best_x.copy()
                best_x = pop[0].copy()
                best_f = fvals[0]

                # Record successful direction
                dir_vec = best_x - old_best
                dir_history.append(dir_vec)
                if len(dir_history) > max_history:
                    dir_history.pop(0)

            # Adapt sigma based on success rate
            # Count how many offspring improved their parent
            n_success = 0
            for i in range(pop_size):
                if offspring_fvals[i] < fvals[i]:
                    n_success += 1

            success_rate = n_success / pop_size
            total_evals += pop_size

            # Adapt sigma: increase if success rate high, decrease if low
            if success_rate > 0.5:
                sigma *= 1.2
            elif success_rate < 0.2:
                sigma *= 0.8
            else:
                sigma *= 0.95

            # Clamp sigma
            sigma = np.clip(sigma, 0.01, 10.0)

            # Periodically re-center sigma adaptation
            if total_evals % 100 < pop_size:
                # Decay sigma toward a baseline
                sigma = 0.9 * sigma + 0.1 * 1.0

        return best_x, best_f
# EVOLVE-BLOCK-END
