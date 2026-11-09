import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285.0  # safety margin before 300s limit
        budget = self.budget
        dim = self.dim

        best_x = None
        best_f = float('inf')
        calls_made = 0

        def eval_func(x):
            nonlocal calls_made, best_f, best_x
            if calls_made >= budget:
                return None
            x = np.asarray(x, dtype=np.float64)
            x = np.clip(x, -5.0, 5.0)
            calls_made += 1
            try:
                fx = float(func(x))
            except Exception:
                fx = 1e10
            if not np.isfinite(fx):
                fx = 1e10
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        def halton_sequence(n, dim, base_start=2):
            """Generate Halton sequence points in [0,1]^dim."""
            points = np.zeros((n, dim))
            for d in range(dim):
                base = base_start + d
                for i in range(n):
                    f = 1.0
                    r = 0.0
                    idx = i + 1
                    while idx > 0:
                        idx, rem = divmod(idx, base)
                        f /= base
                        r += rem * f
                    points[i, d] = r
            return points

        def nelder_mead(x0, max_evals, seed=42):
            """Nelder-Mead simplex method for local refinement."""
            nonlocal best_x, best_f, calls_made
            n = dim
            rng = np.random.default_rng(seed)

            # Build initial simplex
            simplex = np.zeros((n + 1, n))
            simplex[0] = x0
            for i in range(n):
                simplex[i + 1] = x0.copy()
                simplex[i + 1, i] += 0.25 * max(0.1, abs(x0[i]))

            # Evaluate simplex
            fvals = np.zeros(n + 1)
            for i in range(n + 1):
                fx = eval_func(simplex[i])
                if fx is None:
                    return
                fvals[i] = fx

            # Nelder-Mead parameters
            alpha = 1.0   # reflection
            gamma = 2.0   # expansion
            rho = 0.5     # contraction
            sigma = 0.5   # shrink

            for iteration in range(max_evals):
                if time.time() > deadline:
                    break
                if calls_made >= budget - 2:
                    break

                # Sort by fitness
                idx = np.argsort(fvals)
                simplex = simplex[idx]
                fvals = fvals[idx]

                # Centroid of all but worst
                centroid = np.mean(simplex[:-1], axis=0)

                # Reflection
                x_reflect = centroid + alpha * (centroid - simplex[-1])
                x_reflect = np.clip(x_reflect, -5.0, 5.0)
                f_reflect = eval_func(x_reflect)
                if f_reflect is None:
                    break

                if f_reflect < fvals[0]:
                    # Expansion
                    x_expand = centroid + gamma * (x_reflect - centroid)
                    x_expand = np.clip(x_expand, -5.0, 5.0)
                    f_expand = eval_func(x_expand)
                    if f_expand is None:
                        break
                    if f_expand < f_reflect:
                        simplex[-1] = x_expand
                        fvals[-1] = f_expand
                    else:
                        simplex[-1] = x_reflect
                        fvals[-1] = f_reflect
                elif f_reflect < fvals[-2]:
                    # Accept reflection
                    simplex[-1] = x_reflect
                    fvals[-1] = f_reflect
                else:
                    # Contraction
                    if f_reflect < fvals[-1]:
                        # Outside contraction
                        x_contract = centroid + rho * (x_reflect - centroid)
                    else:
                        # Inside contraction
                        x_contract = centroid - rho * (x_reflect - centroid)
                    x_contract = np.clip(x_contract, -5.0, 5.0)
                    f_contract = eval_func(x_contract)
                    if f_contract is None:
                        break
                    if f_contract < min(f_reflect, fvals[-1]):
                        simplex[-1] = x_contract
                        fvals[-1] = f_contract
                    else:
                        # Shrink
                        for i in range(1, n + 1):
                            simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                            fx = eval_func(simplex[i])
                            if fx is None:
                                return
                            fvals[i] = fx
                        break

                # Check convergence
                if np.max(fvals) - np.min(fvals) < 1e-12 * (1 + np.abs(fvals[0])):
                    break

            return simplex[0], fvals[0]

        def es_trajectory(x0, sigma, max_evals, seed=42):
            """(1+1)-ES trajectory with adaptive step size."""
            nonlocal best_x, best_f, calls_made
            rng = np.random.default_rng(seed)
            n = dim
            x = np.asarray(x0, dtype=np.float64).copy()
            fx = eval_func(x)
            if fx is None:
                return x, fx

            success_count = 0
            total_steps = 0

            for step in range(max_evals):
                if time.time() > deadline:
                    break
                if calls_made >= budget - 2:
                    break

                # Generate candidate
                y = rng.standard_normal(n)
                x_trial = x + sigma * y
                x_trial = np.clip(x_trial, -5.0, 5.0)
                f_trial = eval_func(x_trial)
                if f_trial is None:
                    break

                total_steps += 1

                if f_trial < fx:
                    # Success: increase sigma slightly
                    x = x_trial
                    fx = f_trial
                    sigma *= 1.05
                    success_count += 1
                else:
                    # Failure: decrease sigma
                    sigma *= 0.95

                # Adapt sigma based on recent success rate
                if total_steps >= 20:
                    recent_success = success_count / total_steps
                    target_success = 1 / (n + 1)  # ~1/6 for n=5
                    if recent_success > target_success * 1.5:
                        sigma *= 1.1
                    elif recent_success < target_success * 0.5:
                        sigma *= 0.9

                sigma = np.clip(sigma, 1e-6, 5.0)

            return x, fx

        # Phase 1: Halton space-filling for systematic early exploration
        n_halton = min(80, max(40, budget // 6))

        # Add center point first
        x0 = np.zeros(dim)
        fx0 = eval_func(x0)
        if fx0 is None:
            return best_x, best_f

        # Halton points mapped to [-5, 5]^dim
        halton_pts = halton_sequence(n_halton - 1, dim)
        for i in range(n_halton - 1):
            if time.time() > deadline:
                break
            x = halton_pts[i] * 10.0 - 5.0
            fx = eval_func(x)
            if fx is None:
                break

        # Phase 2: Local refinement using Nelder-Mead and (1+1)-ES
        if best_x is not None:
            remaining = budget - calls_made
            if remaining > 20:
                # Use Nelder-Mead for initial local refinement
                nm_budget = min(remaining // 2, 100)
                if nm_budget > 10:
                    nelder_mead(best_x, nm_budget, seed=42)
                
                # Use (1+1)-ES for further refinement
                remaining = budget - calls_made
                if remaining > 10:
                    es_trajectory(best_x, 0.1, remaining, seed=123)

        return best_x, best_f
