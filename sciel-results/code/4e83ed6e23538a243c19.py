"""BBOB f22 optimizer: Adaptive Hyperparameter Annealing with single population."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """AHA optimizer: single population with adaptive step-size annealing."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 295.0
        n = self.dim
        budget = self.budget
        calls = 0
        best_x = np.zeros(n)
        best_f = np.inf
        rng = np.random.default_rng(42)

        def eval_point(x):
            nonlocal calls, best_f, best_x
            if calls >= budget:
                return np.inf
            if time.time() > deadline:
                return np.inf
            calls += 1
            x = np.clip(x, -5.0, 5.0)
            f = float(func(x))
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        # Phase 1: LHS spread (10 points) to get a diverse initial population
        n_lhs = 10
        lhs = np.zeros((n_lhs, n))
        for i in range(n):
            perm = rng.permutation(n_lhs)
            lhs[:, i] = -5.0 + 10.0 * (perm + 0.5) / n_lhs

        pop_x = []
        pop_f = []
        for i in range(n_lhs):
            f = eval_point(lhs[i])
            if np.isinf(f):
                break
            pop_x.append(lhs[i].copy())
            pop_f.append(f)

        if not pop_x:
            return best_x, best_f

        pop_x = np.array(pop_x)
        pop_f = np.array(pop_f)

        # Phase 2: Adaptive Hyperparameter Annealing
        # Single population, adaptive mutation scale
        pop_size = 10
        sigma = 1.5  # initial mutation scale
        min_sigma = 1e-4
        max_sigma = 4.0
        improvement_window = 10
        best_f_history = [best_f]

        while calls < budget and time.time() < deadline:
            # Sort population
            order = np.argsort(pop_f)
            pop_x = pop_x[order]
            pop_f = pop_f[order]

            # Update best
            if pop_f[0] < best_f:
                best_f = pop_f[0]
                best_x = pop_x[0].copy()
            best_f_history.append(best_f)

            # Adaptive sigma: reduce if improving, increase if stuck
            if len(best_f_history) > improvement_window:
                recent_improvement = best_f_history[-improvement_window - 1] - best_f
                if recent_improvement < 1e-6:
                    # Stuck: increase sigma to escape
                    sigma = min(max_sigma, sigma * 1.5)
                elif recent_improvement > 1.0:
                    # Good progress: decrease sigma to refine
                    sigma = max(min_sigma, sigma * 0.8)
                else:
                    # Moderate: slight decrease
                    sigma = max(min_sigma, sigma * 0.95)

            # Generate offspring: mutate each parent
            new_x = []
            new_f = []
            for i in range(min(pop_size, len(pop_x))):
                # Mutation: gaussian with adaptive sigma
                mutant = pop_x[i] + sigma * rng.standard_normal(n)
                mutant = np.clip(mutant, -5.0, 5.0)
                f = eval_point(mutant)
                if np.isinf(f):
                    break
                new_x.append(mutant)
                new_f.append(f)

            if not new_x:
                break

            # Selection: keep best pop_size from parents + offspring
            all_x = np.vstack([pop_x, np.array(new_x)])
            all_f = np.concatenate([pop_f, np.array(new_f)])
            sel_order = np.argsort(all_f)
            pop_x = all_x[sel_order[:pop_size]]
            pop_f = all_f[sel_order[:pop_size]]

            # Convergence check
            if np.max(pop_f) - np.min(pop_f) < 1e-12:
                break
            if sigma < min_sigma * 1.1:
                break

        # Phase 3: Local polish with small perturbations around best
        if calls < budget and time.time() < deadline and best_f < np.inf:
            polish_sigma = max(0.05, sigma * 0.3)
            for _ in range(min(30, budget - calls)):
                if time.time() > deadline:
                    break
                candidate = best_x + polish_sigma * rng.standard_normal(n)
                candidate = np.clip(candidate, -5.0, 5.0)
                f = eval_point(candidate)
                if np.isinf(f):
                    break

        return best_x, best_f

# EVOLVE-BLOCK-END
