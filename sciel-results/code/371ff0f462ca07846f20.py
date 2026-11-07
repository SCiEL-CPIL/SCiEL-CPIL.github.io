"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time


class Optimizer:
    """Multi-start L-BFGS-B + CMA-ES style polish for Discus (f11)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 270.0  # safety margin before 330s kill

        rng = np.random.default_rng(42)

        # State
        best_x = None
        best_f = np.inf
        call_count = [0]

        def evaluate(x):
            nonlocal best_x, best_f
            if call_count[0] >= budget:
                return best_f if np.isfinite(best_f) else 0.0
            call_count[0] += 1
            x = np.clip(x, -5.0, 5.0)
            fx = float(func(x))
            if not np.isfinite(fx):
                fx = 1e10
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # Phase 1: Initial diverse sampling
        n_initial = min(40, budget)
        for i in range(n_initial):
            if time.time() > deadline or call_count[0] >= budget:
                break
            if i < dim:
                # Positive coordinate axis
                x = np.zeros(dim)
                x[i] = 3.0
            elif i < 2 * dim:
                # Negative coordinate axis
                x = np.zeros(dim)
                x[i - dim] = -3.0
            elif i < 3 * dim:
                # Diagonal pairs
                d = (i - 2 * dim) % dim
                x = np.zeros(dim)
                x[d] = 2.5
                x[(d + 1) % dim] = 2.5
            elif i < 4 * dim:
                # Opposite diagonal pairs
                d = (i - 3 * dim) % dim
                x = np.zeros(dim)
                x[d] = -2.5
                x[(d + 1) % dim] = -2.5
            else:
                # Random uniform
                x = rng.uniform(-4.0, 4.0, dim)
            evaluate(x)

        if time.time() > deadline or call_count[0] >= budget:
            return best_x, best_f

        # Phase 2: Multi-start L-BFGS-B with adaptive restarts
        # Build diverse start points from best evaluations
        starts = []
        if best_x is not None:
            starts.append(best_x.copy())
        # Center
        starts.append(np.zeros(dim))
        # Perturbations around best
        if best_x is not None:
            for _ in range(5):
                pert = best_x + rng.normal(0, 0.5, dim)
                starts.append(np.clip(pert, -5, 5))
        # Random starts
        for _ in range(10):
            starts.append(rng.uniform(-4.5, 4.5, dim))

        # Shuffle for diversity
        rng.shuffle(starts)

        # Budget allocation: decay with emphasis on early starts
        n_starts = len(starts)
        remaining = budget - call_count[0]
        if remaining <= 0:
            return best_x, best_f

        # Allocate budget: first few get more
        budgets = []
        rem = remaining
        for i in range(n_starts):
            if i == 0:
                b = min(rem, max(30, remaining // 4))
            elif i < 3:
                b = min(rem, max(15, remaining // 10))
            elif i < 8:
                b = min(rem, max(8, remaining // 20))
            else:
                b = min(rem, max(5, remaining // 40))
            budgets.append(b)
            rem -= b
            if rem <= 0:
                break

        for i, x0 in enumerate(starts):
            if time.time() > deadline or call_count[0] >= budget:
                break
            if i >= len(budgets):
                break

            x0 = np.clip(x0, -5.0, 5.0)
            f0 = evaluate(x0)
            if call_count[0] >= budget:
                break

            # L-BFGS-B from this start
            try:
                res = minimize(
                    evaluate,
                    x0,
                    method='L-BFGS-B',
                    bounds=[(-5.0, 5.0)] * dim,
                    options={
                        'maxiter': 80,
                        'maxls': 50,
                        'gtol': 1e-8,
                        'ftol': 1e-12,
                        'maxfun': max(5, budgets[i]),
                    }
                )
            except Exception:
                pass

            if time.time() > deadline or call_count[0] >= budget:
                break

        # Phase 3: CMA-ES style evolution strategy polish
        if time.time() < deadline and call_count[0] < budget and best_x is not None:
            x = best_x.copy()
            fx = best_f
            sigma = 0.5
            # Population-based ES
            pop_size = max(4, dim + 2)
            
            while call_count[0] < budget and time.time() < deadline:
                # Generate population
                for _ in range(pop_size):
                    if call_count[0] >= budget or time.time() > deadline:
                        break
                    y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                    fy = evaluate(y)
                    if fy < fx:
                        x, fx = y, fy
                        sigma *= 1.1
                    else:
                        sigma *= 0.95
                    if sigma < 0.005:
                        break
                if sigma < 0.005:
                    break

        # Phase 4: Final Nelder-Mead refinement
        if time.time() < deadline and call_count[0] < budget and best_x is not None:
            nm_budget = max(20, min(80, budget - call_count[0]))
            try:
                res_nm = minimize(
                    evaluate,
                    best_x.copy(),
                    method='Nelder-Mead',
                    options={
                        'maxiter': 100,
                        'maxfev': nm_budget,
                        'xatol': 1e-8,
                        'fatol': 1e-10,
                        'adaptive': True,
                    }
                )
            except Exception:
                pass

        if best_x is None:
            best_x = np.zeros(dim)
            best_f = evaluate(best_x)

        return best_x, best_f
# EVOLVE-BLOCK-END
