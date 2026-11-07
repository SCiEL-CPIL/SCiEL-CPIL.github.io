"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """Nelder-Mead simplex optimization with adaptive restarts and local polishing."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        # Track best solution found overall
        best_x = None
        best_f = np.inf
        calls_made = 0

        def evaluate(x):
            nonlocal calls_made, best_x, best_f
            x = np.clip(x, -5.0, 5.0)
            fx = func(x)
            calls_made += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # Nelder-Mead parameters
        alpha = 1.0    # reflection
        gamma = 2.0    # expansion
        rho = 0.5      # contraction
        sigma_nm = 0.5 # shrink

        # Strategy: Multiple phases with adaptive step sizes
        # Phase 1: Global exploration with larger steps
        # Phase 2: Mid-range refinement
        # Phase 3: Fine local search

        # Budget allocation
        phase1_budget = int(budget * 0.4)   # 40% for global exploration
        phase2_budget = int(budget * 0.35)  # 35% for mid-range refinement
        phase3_budget = budget - phase1_budget - phase2_budget  # 25% for fine search

        # Phase 1: Global exploration
        if best_x is None:
            # Try several random starting points
            for attempt in range(3):
                if calls_made >= phase1_budget:
                    break
                center = rng.uniform(-4.5, 4.5, dim)
                self._run_nm(func, evaluate, center, 1.5, phase1_budget, alpha, gamma, rho, sigma_nm, dim)
        
        # Phase 2: Mid-range refinement around best point
        if best_x is not None and calls_made < budget:
            for attempt in range(2):
                if calls_made >= phase1_budget + phase2_budget:
                    break
                center = best_x + rng.normal(0, 0.3, dim)
                center = np.clip(center, -5.0, 5.0)
                self._run_nm(func, evaluate, center, 0.8, phase1_budget + phase2_budget, alpha, gamma, rho, sigma_nm, dim)

        # Phase 3: Fine local search
        if best_x is not None and calls_made < budget:
            # First do Nelder-Mead with very small steps
            self._run_nm(func, evaluate, best_x, 0.3, budget, alpha, gamma, rho, sigma_nm, dim)
            
            # Then coordinate descent for final polish
            if calls_made < budget:
                self._coordinate_descent(func, evaluate, budget)

        return best_x, best_f

    def _run_nm(self, func, evaluate, center, delta, max_calls, alpha, gamma, rho, sigma_nm, dim):
        """Run Nelder-Mead from a given center with given step size."""
        nonlocal calls_made, best_f, best_x
        
        # Create simplex
        simplex = np.zeros((dim + 1, dim))
        simplex[0] = center.copy()
        for i in range(dim):
            simplex[i + 1] = center.copy()
            simplex[i + 1][i] += delta

        # Evaluate initial simplex
        fx_values = np.zeros(dim + 1)
        for i in range(dim + 1):
            if calls_made >= max_calls:
                break
            fx_values[i] = evaluate(simplex[i])

        if calls_made >= max_calls:
            return

        # Nelder-Mead iterations
        no_improve_count = 0
        last_best_f = best_f
        stagnation_limit = 15

        while calls_made < max_calls:
            # Sort simplex by function value
            order = np.argsort(fx_values)
            simplex = simplex[order]
            fx_values = fx_values[order]

            # Check for convergence
            f_range = fx_values[-1] - fx_values[0]
            if f_range < 1e-12:
                break

            # Compute centroid of all but worst point
            centroid = np.mean(simplex[:-1], axis=0)

            # Reflection
            x_r = centroid + alpha * (centroid - simplex[-1])
            x_r = np.clip(x_r, -5.0, 5.0)
            f_r = evaluate(x_r)

            if calls_made >= max_calls:
                break

            if f_r < fx_values[0]:
                # Expansion
                x_e = centroid + gamma * (x_r - centroid)
                x_e = np.clip(x_e, -5.0, 5.0)
                f_e = evaluate(x_e)

                if calls_made >= max_calls:
                    break

                if f_e < f_r:
                    simplex[-1] = x_e
                    fx_values[-1] = f_e
                else:
                    simplex[-1] = x_r
                    fx_values[-1] = f_r
            elif f_r < fx_values[-2]:
                # Simple reflection
                simplex[-1] = x_r
                fx_values[-1] = f_r
            else:
                # Contraction
                if f_r < fx_values[-1]:
                    # Outer contraction
                    x_c = centroid + rho * (x_r - centroid)
                else:
                    # Inner contraction
                    x_c = centroid - rho * (x_r - centroid)
                x_c = np.clip(x_c, -5.0, 5.0)
                f_c = evaluate(x_c)

                if calls_made >= max_calls:
                    break

                if f_c < min(f_r, fx_values[-1]):
                    simplex[-1] = x_c
                    fx_values[-1] = f_c
                else:
                    # Shrink
                    for i in range(1, dim + 1):
                        simplex[i] = simplex[0] + sigma_nm * (simplex[i] - simplex[0])
                        if calls_made < max_calls:
                            fx_values[i] = evaluate(simplex[i])

                    if calls_made >= max_calls:
                        break

            # Check for improvement
            if best_f < last_best_f - 1e-10:
                no_improve_count = 0
                last_best_f = best_f
            else:
                no_improve_count += 1

            if no_improve_count >= stagnation_limit:
                break

    def _coordinate_descent(self, func, evaluate, max_calls):
        """Coordinate descent for final polishing."""
        nonlocal best_x, best_f, calls_made
        
        if best_x is None:
            return
            
        dim = self.dim
        step = 0.2
        min_step = 1e-6
        
        while calls_made < max_calls and step >= min_step:
            improved = False
            for i in range(dim):
                if calls_made >= max_calls:
                    break
                # Try positive step
                x_try = best_x.copy()
                x_try[i] += step
                x_try = np.clip(x_try, -5.0, 5.0)
                f_try = evaluate(x_try)
                if f_try < best_f:
                    best_x = x_try
                    best_f = f_try
                    improved = True

                if calls_made >= max_calls:
                    break

                # Try negative step
                x_try = best_x.copy()
                x_try[i] -= step
                x_try = np.clip(x_try, -5.0, 5.0)
                f_try = evaluate(x_try)
                if f_try < best_f:
                    best_x = x_try
                    best_f = f_try
                    improved = True

            if not improved:
                step *= 0.5
# EVOLVE-BLOCK-END
