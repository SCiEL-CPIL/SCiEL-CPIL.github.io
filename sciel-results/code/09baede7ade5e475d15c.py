"""Hierarchical multi-scale stochastic tunneling with adaptive step-size and DE mutation."""
# EVOLVE-BLOCK-START
import time
import numpy as np


class Optimizer:
    """Hierarchical multi-scale stochastic tunneling with DE mutation and CD/ES polish."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 290.0
        n = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        calls = 0
        best_x = None
        best_f = np.inf

        def time_ok():
            return time.time() < deadline

        def evaluate(x):
            nonlocal calls, best_x, best_f
            if calls >= budget or not time_ok():
                return None
            x = np.clip(np.asarray(x, dtype=float), -5.0, 5.0)
            try:
                fx = float(func(x))
            except Exception:
                fx = best_f
            calls += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # Phase 1: Diverse multi-scale initial sampling (20 points)
        n_initial = 20
        candidates = []
        for i in range(n_initial):
            if calls >= budget or not time_ok():
                break
            if i == 0:
                x = np.zeros(n)
            elif i == 1:
                x = rng.uniform(-4, 4, n)
            elif i == 2:
                x = rng.uniform(-2, 2, n)
            elif i == 3:
                x = rng.uniform(-5, 5, n)
            elif i == 4:
                x = rng.uniform(-3, 3, n)
            elif i == 5:
                x = rng.uniform(-1, 1, n)
            elif i == 6:
                x = rng.uniform(-4.5, 4.5, n)
            elif i == 7:
                x = rng.uniform(-2.5, 2.5, n)
            elif i == 8:
                x = rng.uniform(-3.5, 3.5, n)
            elif i == 9:
                x = rng.uniform(-1.5, 1.5, n)
            else:
                # Remaining: mix of scales
                scale = 5.0 * (1.0 - i / (n_initial + 1))
                scale = max(scale, 0.5)
                x = rng.uniform(-scale, scale, n)
            fx = evaluate(x)
            if fx is not None:
                candidates.append((fx, x.copy()))

        if not candidates:
            x = rng.uniform(-5, 5, n)
            evaluate(x)
            return best_x, best_f

        candidates.sort(key=lambda t: t[0])

        # Phase 2: Hierarchical stochastic tunneling with DE mutation
        # Maintain a population of top candidates, adapt step sizes hierarchically
        pop_size = min(8, len(candidates))
        population = [candidates[i][1].copy() for i in range(pop_size)]
        pop_f = [candidates[i][0] for i in range(pop_size)]

        # Hierarchical step sizes: start large, shrink on success, grow on failure
        base_sigma = 2.0
        min_sigma = 1e-6
        max_sigma = 4.0
        sigma = base_sigma
        stagnation_count = 0
        max_stagnation = 15

        # DE mutation parameters
        F = 0.8  # scaling factor
        CR = 0.5  # crossover rate

        remaining = budget - calls
        # Allocate ~70% to tunneling, 15% to CD, 15% to ES
        tunneling_budget = int(remaining * 0.70)
        cd_budget = int(remaining * 0.15)
        es_budget = remaining - tunneling_budget - cd_budget

        # Stochastic tunneling loop
        for it in range(tunneling_budget):
            if calls >= budget or not time_ok():
                break

            # Select parent: best with some diversity
            if rng.random() < 0.6:
                parent_idx = 0  # best
            else:
                parent_idx = rng.integers(0, pop_size)

            parent_x = population[parent_idx].copy()
            parent_f = pop_f[parent_idx]

            # DE mutation: v = x_r1 + F * (x_r2 - x_r3)
            r1, r2, r3 = rng.choice(pop_size, size=3, replace=False)
            v = population[r1] + F * (population[r2] - population[r3])
            
            # Add Gaussian noise with adaptive sigma (tunneling component)
            v = v + sigma * rng.standard_normal(n)
            v = np.clip(v, -5, 5)

            # Crossover: binomial
            trial = parent_x.copy()
            jrand = rng.integers(0, n)
            for j in range(n):
                if rng.random() < CR or j == jrand:
                    trial[j] = v[j]
            trial = np.clip(trial, -5, 5)

            f_trial = evaluate(trial)
            if f_trial is None:
                break

            # Selection: replace parent if trial is better
            if f_trial < parent_f:
                population[parent_idx] = trial
                pop_f[parent_idx] = f_trial
                sigma = min(sigma * 1.1, max_sigma)  # grow step on success
                stagnation_count = 0
            else:
                sigma = max(sigma * 0.9, min_sigma)  # shrink step on failure
                stagnation_count += 1

            # Tunneling: if stuck, make a large jump from best
            if stagnation_count >= max_stagnation:
                jump_x = best_x + 3.0 * rng.standard_normal(n)
                jump_x = np.clip(jump_x, -5, 5)
                f_jump = evaluate(jump_x)
                if f_jump is not None and f_jump < pop_f[-1]:
                    population[-1] = jump_x
                    pop_f[-1] = f_jump
                # Also try a random restart
                restart_x = rng.uniform(-5, 5, n)
                f_restart = evaluate(restart_x)
                if f_restart is not None and f_restart < pop_f[-1]:
                    population[-1] = restart_x
                    pop_f[-1] = f_restart
                stagnation_count = 0
                sigma = base_sigma  # reset sigma

            # Keep population sorted by fitness
            order = np.argsort(pop_f)
            population = [population[i].copy() for i in order]
            pop_f = [pop_f[i] for i in order]

        # Phase 3: Coordinate descent polish on best point
        if best_x is not None and calls < budget and time_ok():
            x = best_x.copy()
            f = best_f
            step = 0.5
            for cd_iter in range(cd_budget):
                if calls >= budget or not time_ok():
                    break
                improved = False
                for j in range(n):
                    if calls >= budget or not time_ok():
                        break
                    xp = x.copy()
                    xp[j] = min(xp[j] + step, 5.0)
                    fp = evaluate(xp)
                    if fp is None:
                        break
                    if fp < f:
                        x, f = xp, fp
                        improved = True
                    else:
                        xn = x.copy()
                        xn[j] = max(xn[j] - step, -5.0)
                        fn = evaluate(xn)
                        if fn is None:
                            break
                        if fn < f:
                            x, f = xn, fn
                            improved = True
                if not improved:
                    step *= 0.5
                    if step < 1e-6:
                        break

        # Phase 4: (1+1)-ES polish on best point
        if best_x is not None and calls < budget and time_ok():
            x = best_x.copy()
            f = best_f
            sigma = 0.2
            min_sigma = 1e-8
            for _ in range(es_budget):
                if not time_ok():
                    break
                y = np.clip(x + sigma * rng.standard_normal(n), -5, 5)
                fy = evaluate(y)
                if fy is None:
                    break
                if fy < f:
                    x, f = y, fy
                    sigma = min(sigma * 1.2, 1.5)
                else:
                    sigma = max(sigma * 0.8, min_sigma)

        if best_x is None:
            best_x = np.zeros(n)
        return best_x, best_f
# EVOLVE-BLOCK-END
