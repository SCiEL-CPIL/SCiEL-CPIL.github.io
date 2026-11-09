"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """(mu+lambda)-ES with adaptive step size and restarts for Schwefel f20."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 280  # safety margin before 300s limit

        rng = np.random.default_rng(42)

        # Schwefel function has a known global minimum structure;
        # start near promising regions
        # The shifted/rotated version means we need diverse starts
        best_x = None
        best_f = np.inf

        # Phase 1: Diverse initial sampling (10% of budget)
        n_init = max(10, budget // 10)
        for i in range(n_init):
            if time.time() > deadline:
                break
            # Mix of uniform and biased sampling
            if i < dim:
                # Try near origin (Schwefel's local structure)
                x = rng.uniform(-2, 2, dim)
            elif i < 2 * dim:
                # Try near typical Schwefel minimum region
                x = rng.uniform(-4, 4, dim)
            else:
                x = rng.uniform(-5, 5, dim)
            fx = func(x)
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        if best_x is None:
            best_x = np.zeros(dim)
            best_f = func(best_x)

        # Phase 2: (mu+lambda)-ES with adaptive sigma
        mu = max(2, dim)
        lam = max(4, 2 * dim)
        sigma = 1.0
        best_x_es = best_x.copy()
        best_f_es = best_f

        # Population
        pop = [best_x_es.copy()]
        pop_f = [best_f_es]

        calls_made = n_init
        stagnation_count = 0
        improvement_threshold = 1e-10

        while calls_made < budget:
            if time.time() > deadline:
                break

            # Generate offspring
            offspring = []
            offspring_f = []
            for j in range(lam):
                if time.time() > deadline:
                    break
                # Select parent from current population
                parent_idx = rng.integers(0, len(pop))
                parent = pop[parent_idx]
                # Mutate
                child = np.clip(parent + sigma * rng.standard_normal(dim), -5, 5)
                f_child = func(child)
                calls_made += 1
                offspring.append(child)
                offspring_f.append(f_child)

            if not offspring:
                break

            # Combine parents and offspring
            combined_x = pop + offspring
            combined_f = pop_f + offspring_f

            # Sort by fitness
            indices = np.argsort(combined_f)
            combined_x = [combined_x[i] for i in indices]
            combined_f = [combined_f[i] for i in indices]

            # Keep best mu as new population
            pop = combined_x[:mu]
            pop_f = combined_f[:mu]

            # Update best
            if combined_f[0] < best_f_es:
                if combined_f[0] < best_f_es - improvement_threshold:
                    stagnation_count = 0
                else:
                    stagnation_count += 1
                best_f_es = combined_f[0]
                best_x_es = combined_x[0].copy()
                if best_f_es < best_f:
                    best_f = best_f_es
                    best_x = best_x_es.copy()
            else:
                stagnation_count += 1

            # Adapt sigma based on success rate
            # Count how many offspring beat the worst parent
            n_improved = sum(1 for f in offspring_f if f < pop_f[-1])
            success_rate = n_improved / max(1, len(offspring))

            if success_rate > 0.3:
                sigma *= 1.1
            elif success_rate < 0.1:
                sigma *= 0.9
            else:
                sigma *= 1.0

            sigma = np.clip(sigma, 0.01, 10.0)

            # Restart if stagnant
            if stagnation_count > 50 and calls_made < budget - 20:
                if time.time() > deadline:
                    break
                # Local restart around best point with larger step
                for _ in range(min(10, budget - calls_made)):
                    if time.time() > deadline:
                        break
                    x_restart = np.clip(best_x_es + sigma * 2 * rng.standard_normal(dim), -5, 5)
                    f_restart = func(x_restart)
                    calls_made += 1
                    if f_restart < best_f:
                        best_f = f_restart
                        best_x = x_restart.copy()
                        best_f_es = f_restart
                        best_x_es = x_restart.copy()
                stagnation_count = 0
                sigma = np.clip(sigma * 0.5, 0.01, 10.0)

            # If we've used most of budget, do fine-tuning
            if calls_made > budget * 0.8:
                # Switch to (1+1)-ES for fine tuning
                while calls_made < budget:
                    if time.time() > deadline:
                        break
                    y = np.clip(best_x_es + sigma * 0.3 * rng.standard_normal(dim), -5, 5)
                    fy = func(y)
                    calls_made += 1
                    if fy < best_f_es:
                        best_x_es = y
                        best_f_es = fy
                        if fy < best_f:
                            best_f = fy
                            best_x = y.copy()
                        sigma *= 1.05
                    else:
                        sigma *= 0.95
                    sigma = np.clip(sigma, 0.001, 10.0)
                break

        return best_x, best_f
# EVOLVE-BLOCK-END
