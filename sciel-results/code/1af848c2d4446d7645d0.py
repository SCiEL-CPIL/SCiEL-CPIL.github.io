"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Differential Evolution (DE) with adaptive parameters and Nelder-Mead polish."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin before 330s kill
        rng = np.random.default_rng(12345)
        n = self.dim
        budget = self.budget

        def check_deadline():
            return time.time() >= deadline

        calls_used = 0

        def do_call(x):
            nonlocal calls_used
            if calls_used >= budget or check_deadline():
                return np.inf
            fx = func(np.clip(x, -5, 5))
            calls_used += 1
            return fx

        # --- Phase 1: Differential Evolution ---
        # DE maintains a population of NP candidates
        NP = 20  # population size
        # Allocate ~70% of budget to DE
        de_budget = int(budget * 0.7)
        max_generations = de_budget // NP

        # Initialize population: mix of random points and points around origin
        pop = np.zeros((NP, n))
        # First few points: structured initialization
        pop[0] = 0.0
        pop[1] = rng.uniform(-3, 3, n)
        pop[2] = rng.uniform(-4, 4, n)
        pop[3] = rng.uniform(-2, 2, n)
        for i in range(4, NP):
            pop[i] = rng.uniform(-5, 5, n)

        # Evaluate initial population
        pop_f = np.array([do_call(x) for x in pop])
        best_idx = np.argmin(pop_f)
        best_x = pop[best_idx].copy()
        best_f = pop_f[best_idx]

        # DE parameters
        F = 0.8  # scaling factor
        CR = 0.5  # crossover rate
        # Adaptive parameters
        F_min, F_max = 0.4, 1.0
        CR_min, CR_max = 0.1, 0.9

        gen = 0
        while gen < max_generations and calls_used < de_budget and not check_deadline():
            gen += 1
            # Adaptive F and CR based on generation progress
            progress = gen / max(1, max_generations)
            F = F_max - (F_max - F_min) * progress
            CR = CR_min + (CR_max - CR_min) * progress

            # Sort population by fitness
            sorted_idx = np.argsort(pop_f)
            sorted_pop = pop[sorted_idx]
            sorted_f = pop_f[sorted_idx]

            # Update best
            if sorted_f[0] < best_f:
                best_f = sorted_f[0]
                best_x = sorted_pop[0].copy()

            # Generate offspring for each member
            new_pop = pop.copy()
            new_f = pop_f.copy()

            for i in range(NP):
                if calls_used >= de_budget or check_deadline():
                    break

                # Select three distinct random individuals different from i
                candidates = list(range(NP))
                candidates.remove(i)
                r1, r2, r3 = rng.choice(candidates, size=3, replace=False)

                # Differential variation
                v = pop[i] + F * (pop[r1] - pop[r2])

                # Binomial crossover
                j_rand = rng.integers(n)
                trial = pop[i].copy()
                for j in range(n):
                    if rng.random() < CR or j == j_rand:
                        trial[j] = v[j]
                trial = np.clip(trial, -5, 5)

                # Evaluate trial
                trial_f = do_call(trial)

                # Selection
                if trial_f <= pop_f[i]:
                    new_pop[i] = trial
                    new_f[i] = trial_f

            pop = new_pop
            pop_f = new_f

        # --- Phase 2: Local polish with Nelder-Mead ---
        remaining_budget = budget - calls_used
        if remaining_budget > 30 and not check_deadline():
            # Polish from best and a few nearby points
            polish_starts = [best_x.copy()]
            for scale in [0.1, 0.3, 0.5]:
                px = best_x + scale * rng.standard_normal(n)
                px = np.clip(px, -5, 5)
                polish_starts.append(px)

            for ps in polish_starts:
                if check_deadline() or calls_used >= budget - 10:
                    break

                remaining = budget - calls_used
                if remaining < 20:
                    break

                def objective(x):
                    nonlocal calls_used
                    if check_deadline() or calls_used >= budget:
                        raise ValueError("budget exceeded")
                    val = func(np.clip(x, -5, 5))
                    calls_used += 1
                    return val

                try:
                    res = minimize(
                        objective,
                        ps,
                        method='Nelder-Mead',
                        options={
                            'maxiter': min(remaining - 5, 200),
                            'xatol': 1e-10,
                            'fatol': 1e-12,
                            'adaptive': True
                        }
                    )
                    if res.fun < best_f:
                        best_f = res.fun
                        best_x = np.clip(res.x, -5, 5)
                except Exception:
                    pass

        # Ensure at least one call was made
        if calls_used == 0:
            x = np.zeros(n)
            fx = do_call(x)
            best_x = x
            best_f = fx

        return best_x, best_f
# EVOLVE-BLOCK-END
