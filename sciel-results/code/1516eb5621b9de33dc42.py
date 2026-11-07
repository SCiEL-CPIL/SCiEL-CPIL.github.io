"""Rewritten optimizer: population-based DE with LHS init, adaptive F, restarts, and NM polish."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """DE/rand/1 with adaptive mutation, LHS initialization, stagnation restarts, and NM polish."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # safety margin before 300s limit
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        def lhs(n, d):
            """Latin Hypercube Sampling."""
            pts = np.zeros((n, d))
            for i in range(n):
                for j in range(d):
                    u = rng.random()
                    pts[i, j] = -5.0 + 10.0 * (i + u) / n
            return pts

        # Initial population via LHS
        pop_size = min(30, budget // 10)
        pop = lhs(pop_size, dim)
        fvals = np.array([func(p) for p in pop])
        best_idx = np.argmin(fvals)
        best_x = pop[best_idx].copy()
        best_f = fvals[best_idx]
        calls_made = pop_size

        # Track for stagnation
        no_improve_count = 0
        stagnation_threshold = 50  # evaluations without improvement

        # Adaptive mutation factor
        F = 0.8
        CR = 0.5

        # Main loop
        while calls_made < budget:
            # Check deadline
            if time.time() > deadline:
                break

            # Adaptive F: increase if no recent improvement
            if no_improve_count > 20:
                F = min(0.95, F + 0.05)
            elif no_improve_count > 5:
                F = max(0.4, F - 0.02)

            # Generate offspring
            trial = np.zeros((pop_size, dim))
            trial_fvals = np.full(pop_size, np.inf)
            
            for i in range(pop_size):
                if calls_made >= budget or time.time() > deadline:
                    break

                # DE/rand/1
                a, b, c = rng.choice(pop_size, 3, replace=False)
                while a == i:
                    a = rng.choice(pop_size)
                while b == i:
                    b = rng.choice(pop_size)
                while c == i:
                    c = rng.choice(pop_size)

                v = pop[a] + F * (pop[b] - pop[c])
                v = np.clip(v, -5, 5)

                # Crossover
                j_rand = rng.choice(dim)
                for j in range(dim):
                    if rng.random() < CR or j == j_rand:
                        trial[i, j] = v[j]
                    else:
                        trial[i, j] = pop[i, j]

                trial_fvals[i] = func(trial[i])
                calls_made += 1

            # Selection
            improved = False
            for i in range(pop_size):
                if trial_fvals[i] < fvals[i]:
                    pop[i] = trial[i]
                    fvals[i] = trial_fvals[i]
                    if trial_fvals[i] < best_f:
                        best_f = trial_fvals[i]
                        best_x = trial[i].copy()
                        improved = True

            if improved:
                no_improve_count = 0
            else:
                no_improve_count += 1

            # Stagnation restart: perturb population
            if no_improve_count >= stagnation_threshold:
                no_improve_count = 0
                # Keep best, reinitialize rest
                pop[0] = best_x
                fvals[0] = best_f
                new_pop = lhs(pop_size - 1, dim)
                # Perturb around best
                for i in range(pop_size - 1):
                    new_pop[i] = np.clip(best_x + rng.normal(0, 1.0, dim), -5, 5)
                pop[1:] = new_pop
                fvals[1:] = np.array([func(p) for p in new_pop])
                calls_made += (pop_size - 1)
                # Update best if any new point is better
                for i in range(1, pop_size):
                    if fvals[i] < best_f:
                        best_f = fvals[i]
                        best_x = pop[i].copy()

        # Final polish with Nelder-Mead
        if calls_made < budget - 10 and time.time() < deadline:
            try:
                result = minimize(
                    func, best_x, method='Nelder-Mead',
                    options={'maxiter': min(budget - calls_made, 200),
                             'xatol': 1e-6, 'fatol': 1e-8}
                )
                if result.fun < best_f:
                    best_f = result.fun
                    best_x = result.x
            except Exception:
                pass

        return best_x, best_f
# EVOLVE-BLOCK-END
