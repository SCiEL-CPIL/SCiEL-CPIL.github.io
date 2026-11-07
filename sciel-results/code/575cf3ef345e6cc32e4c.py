"""Optimizer for BBOB f19 (Griewank-Rosenbrock) in 5D."""
import numpy as np
import time


class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 300s kill
        dim = self.dim
        budget = self.budget
        lo, hi = -5.0, 5.0
        rng = np.random.default_rng(42)

        best_x = np.full(dim, 0.0)
        best_f = np.inf
        call_count = 0

        def safe_call(x):
            nonlocal call_count
            if call_count >= budget:
                return np.inf
            x = np.clip(np.asarray(x, dtype=float), lo, hi)
            if len(x) != dim:
                x = np.resize(x, dim)
            try:
                fx = float(func(x))
            except Exception:
                fx = np.inf
            call_count += 1
            return fx

        # Phase 1: Diverse random sampling to find good basins
        # Spend ~30% of budget on exploration
        n_explore = max(20, budget // 4)
        for i in range(n_explore):
            if time.time() > deadline:
                break
            x = rng.uniform(lo, hi, dim)
            fx = safe_call(x)
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        # Phase 2: Nelder-Mead local refinement from best point
        # Use remaining budget for NM iterations
        remaining = budget - call_count
        if remaining > 10:
            best_x, best_f = self._nelder_mead(
                func, best_x, best_f, dim, remaining,
                lo, hi, safe_call, deadline
            )

        # Phase 3: Additional restarts from perturbed best
        remaining = budget - call_count
        if remaining > 20 and time.time() < deadline:
            n_restarts = min(5, remaining // 30)
            for _ in range(n_restarts):
                if time.time() > deadline or call_count >= budget:
                    break
                # Perturb best point
                sigma = 1.0
                x_pert = best_x + sigma * rng.standard_normal(dim)
                x_pert = np.clip(x_pert, lo, hi)
                fx = safe_call(x_pert)
                if fx < best_f:
                    best_f = fx
                    best_x = x_pert.copy()

                # Run NM from this point with limited budget
                rem = budget - call_count
                if rem > 15:
                    x_new, f_new = self._nelder_mead(
                        func, x_pert, fx, dim, min(rem, 40),
                        lo, hi, safe_call, deadline
                    )
                    if f_new < best_f:
                        best_f = f_new
                        best_x = x_new.copy()

        return best_x, best_f

    def _nelder_mead(self, func, x0, f0, dim, max_evals, lo, hi, safe_call, deadline):
        """Nelder-Mead simplex method with time guard."""
        # Initialize simplex
        simplex = [x0.copy()]
        fvals = [f0]

        # Create other simplex points
        for i in range(dim):
            xi = x0.copy()
            step = 0.5 if x0[i] == 0 else 0.05 * abs(x0[i])
            xi[i] += step
            xi = np.clip(xi, lo, hi)
            fi = safe_call(xi)
            if time.time() > deadline or len(fvals) >= max_evals:
                break
            simplex.append(xi)
            fvals.append(fi)

        if len(simplex) < 2:
            return x0, f0

        alpha = 1.0  # reflection
        gamma = 2.0  # expansion
        rho = 0.5   # contraction
        sigma = 0.5 # shrink

        max_iter = max_evals
        for _ in range(max_iter):
            if time.time() > deadline or len(fvals) > max_evals:
                break

            # Sort by function value
            order = np.argsort(fvals)
            simplex = [simplex[i] for i in order]
            fvals = [fvals[i] for i in order]

            # Centroid of all but worst
            centroid = np.mean(simplex[:-1], axis=0)

            # Reflection
            x_ref = centroid + alpha * (centroid - simplex[-1])
            x_ref = np.clip(x_ref, lo, hi)
            f_ref = safe_call(x_ref)

            if f_ref < fvals[0]:
                # Try expansion
                x_exp = centroid + gamma * (x_ref - centroid)
                x_exp = np.clip(x_exp, lo, hi)
                f_exp = safe_call(x_exp)
                if f_exp < f_ref:
                    simplex[-1] = x_exp
                    fvals[-1] = f_exp
                else:
                    simplex[-1] = x_ref
                    fvals[-1] = f_ref
            elif f_ref < fvals[-2]:
                simplex[-1] = x_ref
                fvals[-1] = f_ref
            else:
                # Contraction
                if f_ref < fvals[-1]:
                    x_con = centroid + rho * (x_ref - centroid)
                    f_con = safe_call(x_con)
                else:
                    x_con = centroid - rho * (simplex[-1] - centroid)
                    f_con = safe_call(x_con)

                if f_con < fvals[-1]:
                    simplex[-1] = np.clip(x_con, lo, hi)
                    fvals[-1] = f_con
                else:
                    # Shrink
                    for i in range(1, len(simplex)):
                        simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                        simplex[i] = np.clip(simplex[i], lo, hi)
                        fvals[i] = safe_call(simplex[i])

            # Early termination if simplex is very small
            spread = max(fvals) - min(fvals)
            if spread < 1e-12:
                break

        # Return best point
        best_idx = np.argmin(fvals)
        return simplex[best_idx].copy(), fvals[best_idx]
