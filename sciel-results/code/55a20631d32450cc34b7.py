"""Optimizer for BBOB-style f4 (Büche-Rastrigin) function."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Hybrid: (1+1)-ES with 1/5 success rule and restarts, plus DE for exploration."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0

        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        # Track global best
        global_best_x = None
        global_best_fx = np.inf
        calls_made = 0

        def try_update(x, fx):
            nonlocal global_best_x, global_best_fx
            if fx < global_best_fx:
                global_best_fx = fx
                global_best_x = x.copy()
                return True
            return False

        # Phase 1: (1+1)-ES with 1/5 success rule
        # Start at a random point
        x = rng.uniform(-4, 4, size=dim)
        fx = func(x)
        calls_made += 1
        try_update(x, fx)

        # Step size
        sigma = 1.0
        # 1/5 success rule parameters
        success_threshold = 0.2
        step_up = 1.2
        step_down = 1.0 / 1.2

        # Track successes over a window
        window_size = 10
        successes = 0
        steps_in_window = 0

        # Restart logic
        no_improve_count = 0
        max_no_improve = 40
        es_budget = int(budget * 0.3)  # Use 30% of budget for ES

        while calls_made < es_budget:
            if time.time() > deadline:
                break

            # Generate candidate
            direction = rng.standard_normal(dim)
            candidate = x + sigma * direction
            candidate = np.clip(candidate, -5, 5)

            candidate_fx = func(candidate)
            calls_made += 1

            # Check success
            if candidate_fx <= fx:
                x = candidate
                fx = candidate_fx
                successes += 1
                try_update(x, fx)
                no_improve_count = 0
            else:
                no_improve_count += 1

            steps_in_window += 1

            # Update step size based on 1/5 rule
            if steps_in_window >= window_size:
                success_rate = successes / steps_in_window
                if success_rate > success_threshold:
                    sigma *= step_up
                elif success_rate < success_threshold:
                    sigma *= step_down
                successes = 0
                steps_in_window = 0

            # Stagnation check: restart if no improvement for max_no_improve steps
            if no_improve_count >= max_no_improve:
                # Try a few random restarts, biased toward current best
                for _ in range(4):
                    if calls_made >= es_budget:
                        break
                    if global_best_x is not None and rng.random() < 0.5:
                        x_restart = global_best_x + 2.0 * rng.standard_normal(dim)
                    else:
                        x_restart = rng.uniform(-4, 4, size=dim)
                    x_restart = np.clip(x_restart, -5, 5)
                    x_restart_fx = func(x_restart)
                    calls_made += 1
                    if try_update(x_restart, x_restart_fx):
                        x = x_restart
                        fx = x_restart_fx
                        sigma = 1.5
                        no_improve_count = 0
                        break
                # Reset counters
                no_improve_count = 0
                successes = 0
                steps_in_window = 0

        # Phase 2: DE for exploration with remaining budget
        de_budget = budget - calls_made
        if de_budget > 0 and global_best_x is not None:
            pop_size = min(25, max(12, de_budget // 4))
            
            # Initialize population around current best and random points
            population = np.zeros((pop_size, dim))
            fitness = np.zeros(pop_size)

            # First individual is the current best
            population[0] = global_best_x
            fitness[0] = global_best_fx

            # Fill rest with mix of local perturbations and global random
            for i in range(1, pop_size):
                if calls_made >= budget:
                    break
                r = rng.random()
                if r < 0.4:
                    # Local perturbation around best
                    population[i] = global_best_x + 1.5 * rng.standard_normal(dim)
                elif r < 0.7:
                    # Medium exploration
                    population[i] = rng.uniform(-3, 3, size=dim)
                else:
                    # Global random
                    population[i] = rng.uniform(-4, 4, size=dim)
                population[i] = np.clip(population[i], -5, 5)
                fitness[i] = func(population[i])
                calls_made += 1
                try_update(population[i], fitness[i])

            # DE/rand/1/bin iterations with adaptive parameters
            no_improve_count = 0
            max_no_improve = 25
            current_F = 0.8
            current_CR = 0.7
            best_F = 0.8
            best_CR = 0.7
            gen = 0
            max_gen = 100

            while calls_made < budget and gen < max_gen:
                if time.time() > deadline:
                    break

                # Check for stagnation
                if no_improve_count >= max_no_improve:
                    # Diversify: replace worst half with random
                    half = pop_size // 2
                    order = np.argsort(fitness)
                    for i in range(half, pop_size):
                        if calls_made >= budget:
                            break
                        if rng.random() < 0.4:
                            population[i] = global_best_x + 2.0 * rng.standard_normal(dim)
                        elif rng.random() < 0.5:
                            population[i] = rng.uniform(-3, 3, size=dim)
                        else:
                            population[i] = rng.uniform(-4, 4, size=dim)
                        population[i] = np.clip(population[i], -5, 5)
                        fitness[i] = func(population[i])
                        calls_made += 1
                        try_update(population[i], fitness[i])

                    no_improve_count = 0
                    # Reset to best parameters
                    current_F = best_F
                    current_CR = best_CR
                    max_no_improve = 25
                    gen = 0

                # DE/rand/1/bin iteration
                for i in range(pop_size):
                    if calls_made >= budget or time.time() > deadline:
                        break

                    # Select three distinct random individuals
                    while True:
                        r1 = rng.integers(0, pop_size)
                        r2 = rng.integers(0, pop_size)
                        r3 = rng.integers(0, pop_size)
                        if r1 != i and r2 != i and r3 != i and r1 != r2 and r2 != r3 and r1 != r3:
                            break

                    # Create mutant vector
                    v = population[r1] + current_F * (population[r2] - population[r3])
                    v = np.clip(v, -5, 5)

                    # Binary crossover
                    trial = population[i].copy()
                    j_rand = rng.integers(0, dim)
                    for j in range(dim):
                        if rng.random() < current_CR or j == j_rand:
                            trial[j] = v[j]

                    # Evaluate trial
                    trial_fx = func(trial)
                    calls_made += 1

                    # Selection
                    if trial_fx <= fitness[i]:
                        population[i] = trial
                        fitness[i] = trial_fx
                        if try_update(trial, trial_fx):
                            no_improve_count = 0
                            # Track best parameters
                            best_F = current_F
                            best_CR = current_CR
                        else:
                            no_improve_count += 1
                    else:
                        no_improve_count += 1

                # Adapt F and CR at generation boundary
                gen += 1
                if gen % 3 == 0:
                    # L-SHADE style: perturb around best
                    current_F = np.clip(best_F + 0.15 * rng.standard_normal(), 0.5, 1.2)
                    current_CR = np.clip(best_CR + 0.1 * rng.standard_normal(), 0.4, 1.0)

        # Phase 3: Final local refinement using Scipy Nelder-Mead
        if global_best_x is not None and calls_made < budget and time.time() < deadline:
            remaining_budget = min(budget - calls_made, 50)  # Cap NM budget

            def wrapped_func(x):
                nonlocal calls_made
                if calls_made >= budget or time.time() > deadline:
                    return 1e10
                calls_made += 1
                return func(np.clip(x, -5, 5))

            try:
                result = minimize(wrapped_func, global_best_x, method='Nelder-Mead',
                                  options={'maxiter': remaining_budget, 'maxfev': remaining_budget,
                                           'xatol': 1e-10, 'fatol': 1e-10,
                                           'initial_simplex': 0.5, 'adaptive': True})
                if result.fun < global_best_fx:
                    global_best_fx = result.fun
                    global_best_x = result.x.copy()
            except Exception:
                pass

        return global_best_x, global_best_fx
# EVOLVE-BLOCK-END
