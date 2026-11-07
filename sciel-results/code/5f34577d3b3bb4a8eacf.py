"""Improved optimizer for BBOB f19 (Griewank-Rosenbrock)."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Deterministic multi-start with stratified sampling, BFGS, and Nelder-Mead polish."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # safety margin before 330s kill

        best_x = None
        best_f = np.inf
        total_calls = 0
        dim = self.dim
        budget = self.budget

        def safe_call(x):
            nonlocal total_calls
            if total_calls >= budget:
                return None
            try:
                val = func(x)
                total_calls += 1
                if np.isfinite(val):
                    return float(val)
                return None
            except Exception:
                total_calls += 1
                return None

        def lhd_sample(n_samples, dim, rng):
            """Latin Hypercube Design for stratified sampling."""
            samples = np.zeros((n_samples, dim))
            for d in range(dim):
                strata = np.linspace(0, 1, n_samples + 1)
                for i in range(n_samples):
                    samples[i, d] = rng.uniform(strata[i], strata[i + 1])
            for d in range(dim):
                perm = rng.permutation(n_samples)
                samples[:, d] = samples[perm, d]
            samples = samples * 10 - 5
            return samples

        def bfgs_local_search(x0, max_evals=50, tight=False):
            """BFGS local search from x0, using budgeted function calls."""
            nonlocal best_x, best_f, total_calls

            def objective(x):
                nonlocal total_calls
                if total_calls >= budget or time.time() > deadline:
                    return np.inf
                x_clipped = np.clip(x, -5, 5)
                val = safe_call(x_clipped)
                if val is None:
                    return np.inf
                return val

            # Use tighter tolerance for final polish
            gtol = 1e-8 if tight else 1e-6
            maxiter = min(max_evals, 100 if tight else 60)

            try:
                res = minimize(objective, x0, method='BFGS',
                              options={'maxiter': maxiter, 'gtol': gtol})
                if res.fun < best_f and np.isfinite(res.fun):
                    best_f = float(res.fun)
                    best_x = np.clip(res.x, -5, 5).copy()
            except Exception:
                pass

            if best_x is not None:
                return best_x.copy(), best_f
            return x0, np.inf

        def nelder_mead_polish(x0, f0, budget_left, rng):
            """Nelder-Mead polish from x0, using budgeted function calls."""
            nonlocal best_x, best_f, total_calls

            n = dim
            # Initialize simplex with adaptive step sizes
            simplex = [x0.copy()]
            fs = [f0]
            for i in range(n):
                xi = x0.copy()
                # Adaptive step size based on distance to boundary
                dist_to_bound = min(abs(x0[i] + 5), abs(5 - x0[i]))
                step = min(0.2, 0.5 * dist_to_bound)
                if step < 0.01:
                    step = 0.01
                direction = 1 if x0[i] < 0 else -1
                if x0[i] + direction * step > 5:
                    direction = -1
                if x0[i] + direction * step < -5:
                    direction = 1
                xi[i] += direction * step
                xi = np.clip(xi, -5, 5)
                fi = safe_call(xi)
                if fi is None:
                    break
                simplex.append(xi)
                fs.append(fi)

            if len(simplex) < 2:
                return x0, f0

            alpha = 1.0
            gamma = 2.0
            rho = 0.5
            sigma = 0.5

            max_iters = min(budget_left, 150)
            for _ in range(max_iters):
                if time.time() > deadline or total_calls >= budget:
                    break

                # Sort simplex
                order = np.argsort(fs)
                simplex = [simplex[i] for i in order]
                fs = [fs[i] for i in order]

                # Check convergence
                if len(fs) > 1 and abs(fs[0] - fs[-1]) < 1e-12:
                    break

                # Centroid of all but worst
                centroid = np.mean(simplex[:-1], axis=0)

                # Reflection
                xr = centroid + alpha * (centroid - simplex[-1])
                xr = np.clip(xr, -5, 5)
                fr = safe_call(xr)
                if fr is None:
                    break

                if fr < fs[0]:
                    # Expansion
                    xe = centroid + gamma * (centroid - simplex[-1])
                    xe = np.clip(xe, -5, 5)
                    fe = safe_call(xe)
                    if fe is None:
                        break
                    if fe < fr:
                        simplex[-1] = xe
                        fs[-1] = fe
                    else:
                        simplex[-1] = xr
                        fs[-1] = fr
                elif fr < fs[-2]:
                    # Simple reflection
                    simplex[-1] = xr
                    fs[-1] = fr
                else:
                    # Contraction
                    if fr < fs[-1]:
                        xc = centroid + rho * (simplex[-1] - centroid)
                    else:
                        xc = centroid - rho * (simplex[-1] - centroid)
                    xc = np.clip(xc, -5, 5)
                    fc = safe_call(xc)
                    if fc is None:
                        break
                    if fc <= fr:
                        simplex[-1] = xc
                        fs[-1] = fc
                    else:
                        # Shrink
                        for i in range(1, n + 1):
                            simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], -5, 5)
                            fi = safe_call(simplex[i])
                            if fi is None:
                                break
                            fs[i] = fi
                        if fi is None:
                            break

                # Update best
                min_idx = np.argmin(fs)
                if fs[min_idx] < best_f:
                    best_f = fs[min_idx]
                    best_x = simplex[min_idx].copy()

            return best_x, best_f

        # Generate stratified sample points
        rng = np.random.default_rng(12345)

        # Start with center and a few deterministic points
        initial_points = [np.zeros(dim)]

        # Add stratified Latin Hypercube samples
        n_lhs = 25
        lhs_points = lhd_sample(n_lhs, dim, rng)
        initial_points.extend([p for p in lhs_points])

        # Add some random points for diversity
        n_random = 10
        random_points = rng.uniform(-4, 4, size=(n_random, dim))
        initial_points.extend([p for p in random_points])

        # Phase 1: Evaluate all initial points and track best
        point_values = []
        for x_init in initial_points:
            if total_calls >= budget or time.time() > deadline:
                break
            x_clipped = np.clip(x_init, -5, 5)
            f_val = safe_call(x_clipped)
            if f_val is not None:
                point_values.append((x_clipped, f_val))
                if f_val < best_f:
                    best_f = f_val
                    best_x = x_clipped.copy()

        # Sort by function value to get best candidates
        if point_values:
            point_values.sort(key=lambda x: x[1])

        # Phase 2: BFGS local search from the best few initial points
        if best_x is not None and total_calls < budget and time.time() < deadline:
            # Take top 5 candidates by function value
            candidate_points = [p[0] for p in point_values[:5]]

            # Run BFGS from each candidate
            for i, x_cand in enumerate(candidate_points):
                if total_calls >= budget or time.time() > deadline:
                    break
                remaining_budget = budget - total_calls
                if remaining_budget < 10:
                    break
                # Allocate remaining budget proportionally, with more for earlier candidates
                max_evals_for_this = min(remaining_budget // max(1, len(candidate_points) - i), 45)
                bfgs_local_search(x_cand, max_evals=max_evals_for_this)

        # Phase 3: Nelder-Mead polish from the best point
        if best_x is not None and total_calls < budget and time.time() < deadline:
            remaining = budget - total_calls
            if remaining > 15:
                nelder_mead_polish(best_x, best_f, remaining, rng)

        # Phase 4: Final high-precision BFGS polish from the best point
        if best_x is not None and total_calls < budget and time.time() < deadline:
            remaining = budget - total_calls
            if remaining > 10:
                bfgs_local_search(best_x, max_evals=min(remaining, 50), tight=True)

        # Final fallback
        if best_x is None:
            best_x = np.zeros(dim)
            best_f = safe_call(best_x)
            if best_f is None:
                best_f = np.inf

        return best_x, best_f
# EVOLVE-BLOCK-END
