"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time


class Optimizer:
    """Multi-start L-BFGS-B + Nelder-Mead + ES polish for Discus (f11)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 275.0  # safety margin before 330s kill

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
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # Phase 1: Initial diverse sampling (Latin hypercube + structured)
        n_initial = min(30, budget)
        for i in range(n_initial):
            if time.time() > deadline or call_count[0] >= budget:
                break
            if i < dim:
                # Coordinate axis exploration
                x = np.zeros(dim)
                x[i] = 4.0 * (1 if i % 2 == 0 else -1)
            elif i < 2 * dim:
                # Negative coordinate axis
                x = np.zeros(dim)
                x[i - dim] = -4.0 * (1 if (i - dim) % 2 == 0 else -1)
            elif i < 3 * dim:
                # Diagonal
                d = (i - 2 * dim) % dim
                x = np.zeros(dim)
                x[d] = 3.5
                x[(d + 1) % dim] = 3.5
            else:
                x = rng.uniform(-4.5, 4.5, dim)
            evaluate(x)

        if time.time() > deadline or call_count[0] >= budget:
            return best_x, best_f

        # Phase 2: Multi-start L-BFGS-B
        # Build diverse start points
        starts = []
        # Best found so far
        if best_x is not None:
            starts.append(best_x.copy())
        # Center
        starts.append(np.zeros(dim))
        # Corners
        for c in [[-5.0] * dim, [5.0] * dim]:
            starts.append(np.array(c))
        # Mid-edge points
        for i in range(dim):
            pt = np.zeros(dim)
            pt[i] = -5.0
            starts.append(pt.copy())
            pt[i] = 5.0
            starts.append(pt.copy())
        # Random
        for _ in range(5):
            starts.append(rng.uniform(-4.5, 4.5, dim))

        # Shuffle for diversity
        rng.shuffle(starts)

        # Allocate budget: first start gets more
        n_starts = len(starts)
        remaining = budget - call_count[0]
        if remaining <= 0:
            return best_x, best_f

        # Budget allocation: decay
        budgets = []
        rem = remaining
        for i in range(n_starts):
            if i == 0:
                b = min(rem, max(20, remaining // 3))
            elif i < n_starts // 2:
                b = min(rem, max(10, remaining // 8))
            else:
                b = min(rem, max(5, remaining // 16))
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
                        'maxiter': 50,
                        'maxls': 30,
                        'gtol': 1e-6,
                        'ftol': 1e-10,
                        'maxfun': max(5, budgets[i]),
                    }
                )
            except Exception:
                pass

            if time.time() > deadline or call_count[0] >= budget:
                break

            # Nelder-Mead refinement from best found so far
            if best_x is not None and call_count[0] < budget:
                nm_budget = max(10, min(30, budget - call_count[0]))
                try:
                    res_nm = minimize(
                        evaluate,
                        best_x.copy(),
                        method='Nelder-Mead',
                        options={
                            'maxiter': 50,
                            'maxfev': nm_budget,
                            'xatol': 1e-6,
                            'fatol': 1e-8,
                            'adaptive': True,
                        }
                    )
                except Exception:
                    pass

        # Phase 3: Final (1+1)-ES polish
        if time.time() < deadline and call_count[0] < budget and best_x is not None:
            x = best_x.copy()
            fx = best_f
            sigma = 0.3
            while call_count[0] < budget and time.time() < deadline:
                y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                fy = evaluate(y)
                if fy < fx:
                    x, fx = y, fy
                    sigma *= 1.25
                else:
                    sigma *= 0.8
                    if sigma < 0.01:
                        break

        if best_x is None:
            best_x = np.zeros(dim)
            best_f = evaluate(best_x)

        return best_x, best_f
# EVOLVE-BLOCK-END
