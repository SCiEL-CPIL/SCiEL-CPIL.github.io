"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize


class Optimizer:
    """Population-based Differential Evolution (DE) for Rosenbrock rotated."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        n_evals = 0
        max_evals = budget

        # 1. Initialization: Stratified sampling
        pop_size = 20
        pop = np.zeros((pop_size, dim))
        for d in range(dim):
            edges = np.linspace(-5, 5, pop_size + 1)
            for i in range(pop_size):
                pop[i, d] = rng.uniform(edges[i], edges[i + 1])

        fvals = np.empty(pop_size)
        for i in range(pop_size):
            fvals[i] = func(pop[i])
            n_evals += 1
            if n_evals >= max_evals:
                break

        best_idx = np.argmin(fvals)
        best_x = pop[best_idx].copy()
        best_f = fvals[best_idx]

        # 2. Differential Evolution Phase (80% of budget)
        de_budget = int(0.8 * max_evals)
        
        F = 0.7
        CR = 0.5

        while n_evals < de_budget:
            best_idx = np.argmin(fvals)
            best_x = pop[best_idx].copy()
            best_f = fvals[best_idx]

            for i in range(pop_size):
                if n_evals >= de_budget:
                    break

                # Mixed strategies: 50% DE/rand/1, 50% DE/current-to-best/1
                if rng.random() < 0.5:
                    # DE/rand/1
                    r1, r2 = rng.choice(pop_size, size=2, replace=False)
                    while r1 == i or r2 == i:
                        r1, r2 = rng.choice(pop_size, size=2, replace=False)
                    v = pop[r1] + F * (pop[r2] - pop[i])
                else:
                    # DE/current-to-best/1
                    r1, r2 = rng.choice(pop_size, size=2, replace=False)
                    while r1 == i or r2 == i:
                        r1, r2 = rng.choice(pop_size, size=2, replace=False)
                    v = pop[i] + F * (best_x - pop[i]) + F * (pop[r1] - pop[r2])

                # Binomial crossover
                trial = pop[i].copy()
                j_rand = rng.integers(0, dim)
                for j in range(dim):
                    if rng.random() < CR or j == j_rand:
                        trial[j] = v[j]

                trial = np.clip(trial, -5, 5)
                f_trial = func(trial)
                n_evals += 1

                if f_trial < fvals[i]:
                    pop[i] = trial
                    fvals[i] = f_trial

                if f_trial < best_f:
                    best_f = f_trial
                    best_x = trial.copy()

            # Decay F to encourage finer search
            progress = n_evals / de_budget
            F = 0.7 - 0.3 * progress
            CR = 0.5 + 0.2 * progress

        # 3. Nelder-Mead Refinement Phase (Remaining 20% of budget)
        if n_evals < max_evals:
            remaining_budget = max_evals - n_evals
            nm_eval_count = [0]
            
            def nm_func(x):
                if nm_eval_count[0] >= remaining_budget:
                    return best_f
                x_clipped = np.clip(x, -5, 5)
                val = func(x_clipped)
                nm_eval_count[0] += 1
                return val

            try:
                res = minimize(
                    nm_func, 
                    best_x, 
                    method='Nelder-Mead', 
                    options={
                        'maxiter': remaining_budget, 
                        'xatol': 1e-6, 
                        'fatol': 1e-6,
                        'adaptive': True
                    }
                )
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = np.clip(res.x, -5, 5)
            except Exception:
                pass

        return best_x, best_f
# EVOLVE-BLOCK-END
