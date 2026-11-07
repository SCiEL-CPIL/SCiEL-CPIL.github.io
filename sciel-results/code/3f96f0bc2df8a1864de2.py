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

        # Phase 3: Derivative-Free Refinement using Powell's method with reflection
        # Powell's method is derivative-free and efficient for smooth functions.
        # We implement a custom Powell with reflection to handle box constraints.
        
        if check_budget() and time.time() < deadline - 5.0:
            def powell_reflect(x, direction, step):
                """Move along direction with reflection at boundaries."""
                new_x = x + step * direction
                # Reflect at boundaries
                for i in range(self.dim):
                    if new_x[i] < -5.0:
                        new_x[i] = -5.0 + (-5.0 - new_x[i])
                    elif new_x[i] > 5.0:
                        new_x[i] = 5.0 - (new_x[i] - 5.0)
                return np.clip(new_x, -5.0, 5.0)

            def powell_method(x0, max_iter=50):
                """Custom Powell's method with reflection for box constraints."""
                x = x0.copy()
                fx = safe_call(x)
                if fx is None:
                    return x, fx

                # Initial directions: standard basis vectors
                directions = [np.eye(self.dim)[i] for i in range(self.dim)]

                for iteration in range(max_iter):
                    if not check_budget() or time.time() > deadline - 5.0:
                        break

                    x_old = x.copy()
                    f_old = fx

                    for i in range(self.dim):
                        if not check_budget():
                            break

                        direction = directions[i]
                        step = 0.5

                        # Try positive step
                        x_new = powell_reflect(x, direction, step)
                        fx_new = safe_call(x_new)
                        if fx_new is None:
                            return x, fx

                        if fx_new < fx:
                            # Move in positive direction
                            x = x_new
                            fx = fx_new
                            # Try to extend in same direction
                            while check_budget() and time.time() < deadline - 5.0:
                                x_next = powell_reflect(x, direction, step)
                                fx_next = safe_call(x_next)
                                if fx_next is None:
                                    return x, fx
                                if fx_next < fx:
                                    x = x_next
                                    fx = fx_next
                                else:
                                    break
                        else:
                            # Try negative step
                            x_new = powell_reflect(x, direction, -step)
                            fx_new = safe_call(x_new)
                            if fx_new is None:
                                return x, fx
                            if fx_new < fx:
                                x = x_new
                                fx = fx_new
                                # Try to extend in negative direction
                                while check_budget() and time.time() < deadline - 5.0:
                                    x_next = powell_reflect(x, direction, -step)
                                    fx_next = safe_call(x_next)
                                    if fx_next is None:
                                        return x, fx
                                    if fx_next < fx:
                                        x = x_next
                                        fx = fx_next
                                    else:
                                        break

                    # Check if we made progress
                    if fx >= f_old - 1e-15:
                        break

                    # Update directions: replace the direction with the largest improvement
                    # and add a new direction from x_old to x
                    new_direction = x - x_old
                    norm_new = np.linalg.norm(new_direction)
                    if norm_new > 1e-10:
                        new_direction = new_direction / norm_new
                        # Replace the last direction with the new one
                        directions[-1] = new_direction

                return x, fx

            try:
                x_powell, f_powell = powell_method(best_x.copy(), max_iter=30)
                if f_powell is not None and f_powell < best_f:
                    best_x = x_powell
                    best_f = f_powell
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
