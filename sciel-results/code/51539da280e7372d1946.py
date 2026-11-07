import numpy as np
import time
from scipy.optimize import minimize
from scipy.stats import qmc

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        # Use 295s to be safe against system overhead/IO (limit is 300s, evaluator kills at 330s)
        deadline = start_time + 295.0

        best_x = None
        best_f = np.inf
        calls_used = 0
        candidates = []

        # Use a fixed seed for reproducibility within the instance
        rng = np.random.default_rng(42)

        def check_budget():
            nonlocal calls_used
            if calls_used >= self.budget:
                return False
            if time.time() > deadline:
                return False
            return True

        def safe_call(x):
            nonlocal calls_used, best_f, best_x, candidates
            if not check_budget():
                return None
            x = np.clip(x, -5.0, 5.0)
            try:
                fx = float(func(x))
            except Exception:
                fx = np.inf
            calls_used += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            candidates.append((fx, x.copy()))
            return fx

        # Phase 1: Global Exploration
        # 60 points: 40 Sobol + 20 Random
        # Sobol provides better low-discrepancy coverage for global search.
        n_sobol = 40
        n_random = 20

        sampler = qmc.Sobol(d=self.dim, scramble=True, seed=42)
        sobol_points = sampler.random(n_sobol)
        X_sobol = sobol_points * 10.0 - 5.0

        X_random = rng.uniform(-5.0, 5.0, size=(n_random, self.dim))

        X_initial = np.vstack([X_sobol, X_random])

        for x in X_initial:
            if not check_budget():
                break
            safe_call(x)

        if best_x is None:
            # Fallback
            x = np.zeros(self.dim)
            safe_call(x)
            if best_x is None:
                return np.zeros(self.dim), 0.0

        # Phase 2: Local Refinement with L-BFGS-B
        # Sort candidates by function value
        candidates.sort(key=lambda t: t[0])

        # Select top 5 unique-ish points to avoid redundant local searches
        top_candidates = []
        for fx, x in candidates:
            if len(top_candidates) >= 5:
                break
            is_unique = True
            for _, x_prev in top_candidates:
                if np.linalg.norm(x - x_prev) < 0.3:
                    is_unique = False
                    break
            if is_unique:
                top_candidates.append((fx, x))

        bounds = [(-5.0, 5.0)] * self.dim

        # Run L-BFGS-B from top candidates
        # L-BFGS-B is effective for smooth functions like Discus (f11).
        # We limit iterations to manage budget.

        for fx0, x0 in top_candidates:
            if not check_budget() or time.time() > deadline - 15.0:
                break

            def objective(x):
                fx = safe_call(x)
                if fx is None:
                    raise Exception("Budget exceeded")
                return fx

            try:
                res = minimize(
                    objective,
                    x0,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={
                        'maxiter': 50,
                        'ftol': 1e-14,
                        'gtol': 1e-10,
                        'maxls': 50  # Limit line search steps to prevent excessive calls
                    }
                )
            except Exception:
                pass

        # Phase 3: Derivative-Free Refinement (Powell / COBYLA)
        # Powell's method is derivative-free and can handle box constraints via reflection
        # or by relying on the function's behavior at boundaries.
        # We use COBYLA as a robust derivative-free alternative that handles constraints
        # explicitly, or Powell if available.
        # Let's try Powell first as it is often very efficient for smooth functions.
        # Note: Powell in scipy does not natively support bounds, so we must be careful.
        # However, for box-constrained problems, if the function is well-behaved,
        # Powell might stay inside or we can clip.
        # Actually, COBYLA handles inequality constraints well.
        # Let's use COBYLA for the final polish to ensure we stay within bounds and
        # to provide a derivative-free refinement that might escape shallow L-BFGS-B traps.

        if check_budget() and time.time() < deadline - 5.0:
            # Define inequality constraints for COBYLA: -5 <= x_i <= 5
            # COBYLA constraints are of the form c(x) >= 0
            def constraints(x):
                cons = []
                for i in range(self.dim):
                    cons.append(x[i] + 5.0)  # x_i >= -5
                    cons.append(5.0 - x[i])  # x_i <= 5
                return cons

            def objective_co(x):
                fx = safe_call(x)
                if fx is None:
                    raise Exception("Budget exceeded")
                return fx

            try:
                res = minimize(
                    objective_co,
                    best_x,
                    method='COBYLA',
                    constraints=constraints,
                    options={
                        'maxiter': 100,
                        'rhobeg': 0.5,  # Initial step size
                        'tol': 1e-12
                    }
                )
            except Exception:
                pass

        # Phase 4: Random Restart if budget remains
        # If we have plenty of budget left, try a few more random starts
        # This can help escape local minima if the function is multimodal
        if check_budget() and time.time() < deadline - 10.0:
            for _ in range(10):
                if not check_budget():
                    break
                x = rng.uniform(-5.0, 5.0, size=self.dim)
                fx = safe_call(x)
                if fx is None:
                    break
                if fx < best_f + 1e-6:
                    # If this random point is good, try to refine it
                    def objective_restart(x):
                        fx = safe_call(x)
                        if fx is None:
                            raise Exception("Budget exceeded")
                        return fx
                    try:
                        res = minimize(
                            objective_restart,
                            x,
                            method='L-BFGS-B',
                            bounds=bounds,
                            options={
                                'maxiter': 20,
                                'ftol': 1e-12,
                                'gtol': 1e-8,
                                'maxls': 20
                            }
                        )
                    except Exception:
                        pass

        if best_x is None:
            best_x = np.zeros(self.dim)
            if check_budget():
                safe_call(best_x)

        return best_x, best_f
