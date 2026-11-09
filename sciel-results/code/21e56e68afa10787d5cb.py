"""Bootstrap portfolio optimizer for BBOB f23 (Katsuura) on [-5,5]^5."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Multi-strategy optimizer for Katsuura:
    1) Latin Hypercube sampling for fast initial basin discovery.
    2) Population-based CMA-ES-like search with multiple directions.
    3) Deadline guard to stay within wall-clock limits.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self._deadline_margin = 30.0  # seconds to spare before hard limit

    @staticmethod
    def _latin_hypercube(n, dim, rng):
        """Generate n points in [-5,5]^dim using Latin Hypercube sampling."""
        points = np.zeros((n, dim))
        for d in range(dim):
            intervals = np.linspace(-5, 5, n + 1)
            lower = intervals[:-1]
            upper = intervals[1:]
            perm = rng.permutation(n)
            points[:, d] = lower[perm] + rng.uniform(0, 1, n) * (upper[perm] - lower[perm])
        return points

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0
        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        # Phase 1: Latin Hypercube sampling for fast initial discovery
        n_lhs = min(30, budget // 10)
        lhs_points = self._latin_hypercube(n_lhs, dim, rng)
        best_x = None
        best_f = np.inf
        lhs_values = []

        for i in range(n_lhs):
            x = lhs_points[i]
            fx = func(x)
            lhs_values.append(fx)
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        # Phase 2: Population-based CMA-ES-like search
        pop_size = 10
        x = best_x.copy()
        sigma = 0.5
        calls_made = n_lhs

        # Initialize population around best point
        population = [x.copy() for _ in range(pop_size)]
        pop_values = [best_f for _ in range(pop_size)]

        # Add some diversity to initial population
        for i in range(1, pop_size):
            noise_scale = 0.5 * (1 + i / pop_size)
            population[i] = np.clip(x + noise_scale * rng.standard_normal(dim), -5, 5)
            pop_values[i] = func(population[i])
            calls_made += 1
            if pop_values[i] < best_f:
                best_f = pop_values[i]
                best_x = population[i].copy()

        # CMA-ES-like evolution
        no_improve = 0
        stagnation_limit = 20
        restart_count = 0
        max_restarts = 6

        while calls_made < budget:
            if time.time() > deadline:
                break

            # Generate offspring for each parent
            offspring = []
            offspring_values = []

            for i in range(pop_size):
                if calls_made >= budget:
                    break
                # Adaptive sigma based on population diversity
                local_sigma = sigma * (1.0 + 0.1 * i / pop_size)
                child = np.clip(population[i] + local_sigma * rng.standard_normal(dim), -5, 5)
                fval = func(child)
                calls_made += 1
                offspring.append(child)
                offspring_values.append(fval)

                if fval < best_f:
                    best_f = fval
                    best_x = child.copy()
                    no_improve = 0

            # Selection: keep best pop_size individuals from population + offspring
            combined = [(pop_values[i], population[i]) for i in range(pop_size)]
            combined += [(offspring_values[i], offspring[i]) for i in range(len(offspring))]
            combined.sort(key=lambda t: t[0])

            population = [combined[i][1].copy() for i in range(min(pop_size, len(combined)))]
            pop_values = [combined[i][0] for i in range(min(pop_size, len(combined)))]

            # Update sigma based on improvement
            if no_improve == 0:
                sigma = min(sigma * 1.05, 2.0)
            else:
                no_improve += 1
                sigma = max(sigma * 0.95, 0.01)

            # Restart if stagnated
            if no_improve >= stagnation_limit and restart_count < max_restarts:
                restart_count += 1
                noise_scale = 1.0 + 0.5 * restart_count
                x = np.clip(best_x + noise_scale * rng.standard_normal(dim), -5, 5)
                fx = func(x)
                calls_made += 1
                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()

                # Reinitialize population around new best
                population = [best_x.copy() for _ in range(pop_size)]
                pop_values = [best_f for _ in range(pop_size)]
                for i in range(1, pop_size):
                    if calls_made >= budget:
                        break
                    noise_scale2 = 0.5 * (1 + i / pop_size)
                    population[i] = np.clip(best_x + noise_scale2 * rng.standard_normal(dim), -5, 5)
                    pop_values[i] = func(population[i])
                    calls_made += 1
                    if pop_values[i] < best_f:
                        best_f = pop_values[i]
                        best_x = population[i].copy()

                no_improve = 0
                sigma = 0.8
                stagnation_limit = max(15, stagnation_limit - 3)

            # Final polishing phase
            if no_improve > 80 and restart_count >= max_restarts:
                sigma = 0.05

        return best_x, best_f
# EVOLVE-BLOCK-END
