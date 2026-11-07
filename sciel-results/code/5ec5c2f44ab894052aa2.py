"""Rewritten optimizer for BBOB f14 (Different powers)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Differential Evolution with adaptive parameters, multi-start, and deadline guard."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        deadline = time.time() + 280
        rng = np.random.default_rng(12345)
        d = self.dim
        max_calls = self.budget

        best_x = None
        best_f = float('inf')
        calls_used = 0

        def try_eval(x):
            nonlocal best_x, best_f, calls_used
            if calls_used >= max_calls:
                return None
            fx = func(x)
            calls_used += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # Phase 1: Diverse initialization
        # Start with a set of diverse points
        init_points = [np.zeros(d)]
        # Add axis-aligned extreme points
        for i in range(d):
            for val in [-4, 4]:
                pt = np.zeros(d)
                pt[i] = val
                init_points.append(pt)
        # Add random points
        for _ in range(8):
            init_points.append(rng.uniform(-5, 5, d))

        for p in init_points:
            p_clipped = np.clip(p, -5, 5)
            fx = try_eval(p_clipped)
            if fx is None:
                break
            if fx < best_f:
                best_f = fx
                best_x = p_clipped.copy()

        if calls_used >= max_calls or time.time() > deadline - 15:
            return best_x, best_f

        # Phase 2: Differential Evolution
        # Population size
        pop_size = 12
        # Initialize population around the best point found so far
        # But also keep some diversity
        pop = np.zeros((pop_size, d))
        pop[0] = best_x
        for i in range(1, pop_size):
            # Mix of local and global exploration
            if i < pop_size // 2:
                # Local: around best
                pop[i] = np.clip(best_x + 1.0 * rng.standard_normal(d), -5, 5)
            else:
                # Global: random
                pop[i] = rng.uniform(-5, 5, d)

        # Evaluate initial population
        pop_f = np.full(pop_size, float('inf'))
        for i in range(pop_size):
            fx = try_eval(pop[i])
            if fx is None:
                break
            pop_f[i] = fx

        # DE parameters
        F = 0.8  # scaling factor
        CR = 0.9  # crossover rate

        # Track stagnation
        stagnation = 0
        max_stagnation = 15

        while calls_used < max_calls:
            if time.time() > deadline - 5:
                break

            # Find the best in population
            best_idx = np.argmin(pop_f)
            current_best_f = pop_f[best_idx]

            improved_this_gen = False

            # For each individual, generate a trial
            for i in range(pop_size):
                if calls_used >= max_calls:
                    break
                if time.time() > deadline - 3:
                    break

                # Select 3 distinct random indices different from i
                candidates = [j for j in range(pop_size) if j != i]
                a, b, c = rng.choice(candidates, size=3, replace=False)

                # Differential variation
                mutant = pop[a] + F * (pop[b] - pop[c])
                mutant = np.clip(mutant, -5, 5)

                # Crossover: binomial
                j_rand = rng.integers(0, d)
                trial = pop[i].copy()
                for j in range(d):
                    if rng.random() < CR or j == j_rand:
                        trial[j] = mutant[j]
                trial = np.clip(trial, -5, 5)

                # Evaluate trial
                f_trial = try_eval(trial)
                if f_trial is None:
                    break

                # Selection
                if f_trial < pop_f[i]:
                    pop[i] = trial
                    pop_f[i] = f_trial
                    improved_this_gen = True

            # Update global best
            if improved_this_gen:
                stagnation = 0
            else:
                stagnation += 1

            # If stagnating, restart with new random individuals
            if stagnation >= max_stagnation:
                # Replace worst half of population with new random points
                n_replace = pop_size // 2
                for k in range(n_replace):
                    idx = np.argmax(pop_f)  # worst
                    new_pt = rng.uniform(-5, 5, d)
                    pop[idx] = new_pt
                    f_new = try_eval(new_pt)
                    if f_new is None:
                        break
                    pop_f[idx] = f_new
                stagnation = 0
                max_stagnation = 10  # be more aggressive after restart

            # Adapt F and CR based on progress
            if improved_this_gen:
                F = max(0.5, F * 0.95)  # decrease F slowly on success
            else:
                F = min(1.2, F * 1.05)  # increase F on failure

        # Phase 3: Polish with local search around best
        if calls_used < max_calls and time.time() < deadline - 2:
            x = best_x.copy()
            fx = best_f
            sigma = 0.05
            polish_calls = min(30, max_calls - calls_used)
            for _ in range(polish_calls):
                if time.time() > deadline - 1:
                    break
                y = np.clip(x + sigma * rng.standard_normal(d), -5, 5)
                fy = try_eval(y)
                if fy is None:
                    break
                if fy < fx:
                    x, fx = y, fy
                    sigma *= 0.9
                else:
                    sigma *= 0.8

        return best_x, best_f
# EVOLVE-BLOCK-END
