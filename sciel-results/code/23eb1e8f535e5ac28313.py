"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-start (mu+lambda)-ES with simulated annealing restarts."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # 40s safety margin before 330s limit
        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        # Phase 1: Initial diverse sampling
        n_init = min(10, budget)
        population = []
        for _ in range(n_init):
            x = rng.uniform(-5, 5, dim)
            fx = func(x)
            population.append((fx, x))

        population.sort(key=lambda t: t[0])
        best_fx, best_x = population[0]
        calls_made = n_init

        # Phase 2: (mu+lambda)-ES with restarts
        mu = 3       # number of parents
        lam = 10     # number of offspring per generation
        sigma = 1.5  # initial step size
        sigma_min = 1e-4
        sigma_max = 5.0
        restart_interval = 15  # restart every 15 generations
        gen = 0

        while calls_made < budget:
            # Time guard
            if time.time() > deadline:
                break

            gen += 1
            # Get parents (best mu)
            parents = population[:mu]

            # Generate offspring
            offspring = []
            for i in range(lam):
                if calls_made >= budget:
                    break
                # Pick a parent (tournament-like: weighted by rank)
                pi = min(i // (lam // mu), mu - 1) if lam >= mu else 0
                parent_fx, parent_x = parents[pi]
                # Adaptive sigma based on progress
                step = sigma * rng.standard_normal(dim)
                y = np.clip(parent_x + step, -5, 5)
                fy = func(y)
                calls_made += 1
                offspring.append((fy, y))

            # Combine parents and offspring
            population = population + offspring
            population.sort(key=lambda t: t[0])
            # Keep only best (mu + lam)
            population = population[:mu + lam]

            # Update best
            if population[0][0] < best_fx:
                best_fx, best_x = population[0]

            # Adaptive sigma: success rule
            # If best offspring beat best parent, increase sigma
            parent_best = parents[0][0]
            offspring_best = min(f for f, _ in offspring) if offspring else parent_best
            if offspring_best < parent_best:
                sigma *= 1.3
            else:
                sigma *= 0.85
            sigma = np.clip(sigma, sigma_min, sigma_max)

            # Restart: inject random points to escape local minima
            if gen % restart_interval == 0 and calls_made < budget:
                n_restarts = min(3, budget - calls_made)
                for _ in range(n_restarts):
                    if calls_made >= budget:
                        break
                    x = rng.uniform(-5, 5, dim)
                    fx = func(x)
                    calls_made += 1
                    population.append((fx, x))
                population.sort(key=lambda t: t[0])
                population = population[:mu + lam]
                sigma = 2.0  # reset step size on restart

        return best_x, best_fx
# EVOLVE-BLOCK-END
