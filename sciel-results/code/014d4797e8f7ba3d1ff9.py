"""Rewritten optimizer for BBOB f12 (Bent cigar) with robust multi-strategy search."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Robust optimizer for bent cigar function using Hooke-Jeeves with restarts."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 300s limit
        budget = self.budget
        dim = self.dim

        # Track best solution
        best_x = np.array([0.0] * dim)
        best_f = np.inf
        call_count = 0

        def safe_call(x):
            nonlocal call_count
            if call_count >= budget:
                return None
            try:
                val = func(x)
                call_count += 1
                if val < best_f:
                    best_f = val
                    best_x = x.copy()
                return val
            except Exception:
                call_count += 1
                return None

        # Strategy 1: Hooke-Jeeves pattern search with adaptive steps
        def hooke_jeeves(x0, remaining_budget, deadline):
            nonlocal best_x, best_f, call_count
            if call_count >= budget or time.time() > deadline:
                return best_x, best_f

            x = x0.copy()
            fx = safe_call(x)
            if fx is None:
                return best_x, best_f

            # Initial step sizes
            steps = np.ones(dim) * 1.0
            # Adaptive parameters
            step_increase = 2.0
            step_decrease = 0.5
            max_steps_per_iter = 3

            for _ in range(min(remaining_budget, budget - call_count)):
                if time.time() > deadline:
                    break

                improved = False
                for _coord_pass in range(max_steps_per_iter):
                    if time.time() > deadline:
                        break
                    for i in range(dim):
                        if call_count >= budget:
                            break
                        # Try positive step
                        x_new = x.copy()
                        x_new[i] = np.clip(x_new[i] + steps[i], -5, 5)
                        f_new = safe_call(x_new)
                        if f_new is not None and f_new < fx:
                            x = x_new
                            fx = f_new
                            improved = True
                            steps[i] *= step_increase
                        else:
                            # Try negative step
                            x_new = x.copy()
                            x_new[i] = np.clip(x_new[i] - steps[i], -5, 5)
                            f_new = safe_call(x_new)
                            if f_new is not None and f_new < fx:
                                x = x_new
                                fx = f_new
                                improved = True
                                steps[i] *= step_increase
                            else:
                                steps[i] *= step_decrease

                    if not improved:
                        break

                if not improved:
                    # Reduce all steps
                    steps *= step_decrease
                    if np.all(steps < 1e-6):
                        break

            return best_x, best_f

        # Strategy 2: Nelder-Mead simplex method
        def nelder_mead(x0, remaining_budget, deadline):
            nonlocal best_x, best_f, call_count
            if call_count >= budget or time.time() > deadline:
                return best_x, best_f

            # Initialize simplex
            simplex = []
            fvals = []

            x0 = x0.copy()
            f0 = safe_call(x0)
            if f0 is None:
                return best_x, best_f
            simplex.append(x0.copy())
            fvals.append(f0)

            for i in range(dim):
                xi = x0.copy()
                xi[i] += 0.5
                xi = np.clip(xi, -5, 5)
                fi = safe_call(xi)
                if fi is None:
                    break
                simplex.append(xi)
                fvals.append(fi)

            if len(simplex) < dim + 1:
                return best_x, best_f

            simplex = np.array(simplex)
            fvals = np.array(fvals)

            # Nelder-Mead parameters
            alpha = 1.0
            gamma = 2.0
            rho = 0.5
            sigma = 0.5
            max_iter = min(remaining_budget, budget - call_count)

            for _ in range(max_iter):
                if time.time() > deadline or call_count >= budget:
                    break

                # Sort by function value
                order = np.argsort(fvals)
                simplex = simplex[order]
                fvals = fvals[order]

                centroid = np.mean(simplex[:-1], axis=0)

                # Reflection
                xr = centroid + alpha * (centroid - simplex[-1])
                xr = np.clip(xr, -5, 5)
                fr = safe_call(xr)
                if fr is None:
                    break

                if fvals[0] <= fr < fvals[-2]:
                    simplex[-1] = xr
                    fvals[-1] = fr
                elif fr < fvals[0]:
                    # Expansion
                    xe = centroid + gamma * (xr - centroid)
                    xe = np.clip(xe, -5, 5)
                    fe = safe_call(xe)
                    if fe is not None:
                        if fe < fr:
                            simplex[-1] = xe
                            fvals[-1] = fe
                        else:
                            simplex[-1] = xr
                            fvals[-1] = fr
                    else:
                        simplex[-1] = xr
                        fvals[-1] = fr
                else:
                    # Contraction
                    if fr < fvals[-1]:
                        # Outside contraction
                        xc = centroid + rho * (simplex[-1] - centroid)
                    else:
                        # Inside contraction
                        xc = centroid - rho * (centroid - simplex[-1])
                    xc = np.clip(xc, -5, 5)
                    fc = safe_call(xc)
                    if fc is None:
                        break
                    if fc <= min(fr, fvals[-1]):
                        simplex[-1] = xc
                        fvals[-1] = fc
                    else:
                        # Shrink
                        for i in range(1, dim + 1):
                            simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], -5, 5)
                            fi = safe_call(simplex[i])
                            if fi is None:
                                return best_x, best_f
                            fvals[i] = fi

            return best_x, best_f

        # Strategy 3: Coordinate descent with adaptive steps
        def coordinate_descent(x0, remaining_budget, deadline):
            nonlocal best_x, best_f, call_count
            if call_count >= budget or time.time() > deadline:
                return best_x, best_f

            x = x0.copy()
            fx = safe_call(x)
            if fx is None:
                return best_x, best_f

            steps = np.ones(dim) * 1.0
            step_increase = 1.5
            step_decrease = 0.5
            max_passes = min(remaining_budget, budget - call_count)

            for _ in range(max_passes):
                if time.time() > deadline or call_count >= budget:
                    break

                improved = False
                for i in range(dim):
                    if call_count >= budget:
                        break
                    # Try positive
                    x_new = x.copy()
                    x_new[i] = np.clip(x_new[i] + steps[i], -5, 5)
                    f_new = safe_call(x_new)
                    if f_new is not None and f_new < fx:
                        x = x_new
                        fx = f_new
                        improved = True
                        steps[i] *= step_increase
                    else:
                        # Try negative
                        x_new = x.copy()
                        x_new[i] = np.clip(x_new[i] - steps[i], -5, 5)
                        f_new = safe_call(x_new)
                        if f_new is not None and f_new < fx:
                            x = x_new
                            fx = f_new
                            improved = True
                            steps[i] *= step_increase
                        else:
                            steps[i] *= step_decrease

                if not improved:
                    if np.all(steps < 1e-6):
                        break
                    steps *= step_decrease

            return best_x, best_f

        # Multi-restart strategy
        rng = np.random.default_rng(42)
        num_restarts = min(5, max(1, budget // 100))
        budget_per_restart = max(50, budget // num_restarts)

        for restart in range(num_restarts):
            if time.time() > deadline:
                break

            # Generate starting point
            if restart == 0:
                x0 = np.zeros(dim)
            else:
                x0 = rng.uniform(-4, 4, dim)

            remaining = budget - call_count
            if remaining < 20:
                break

            # Choose strategy based on restart number
            if restart % 3 == 0:
                hooke_jeeves(x0, min(remaining, budget_per_restart), deadline)
            elif restart % 3 == 1:
                nelder_mead(x0, min(remaining, budget_per_restart), deadline)
            else:
                coordinate_descent(x0, min(remaining, budget_per_restart), deadline)

            # If we have budget left, do a final Hooke-Jeeves polish
            if call_count < budget and time.time() < deadline:
                hooke_jeeves(best_x, budget - call_count, deadline)

        return best_x, best_f
# EVOLVE-BLOCK-END
