"""Rewritten optimizer using Nelder-Mead simplex with multi-restart strategy."""
import numpy as np
import time


class Optimizer:
    """Nelder-Mead based optimizer with multi-restart and adaptive strategies."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def _nelder_mead(self, func, x0, fx0, budget, deadline, rng):
        """
        Run Nelder-Mead simplex method from a given starting point.
        Returns (best_x, best_fx, calls_used).
        """
        dim = self.dim
        x = x0.copy()
        fx = fx0
        best_x = x.copy()
        best_fx = fx
        calls = 1  # initial evaluation already done

        # Initialize simplex: x0 and dim perturbed points
        simplex = [x.copy()]
        simplex_f = [fx]
        alpha = 0.5  # perturbation scale
        for i in range(dim):
            xi = x.copy()
            xi[i] += alpha
            xi = np.clip(xi, -5, 5)
            fi = func(xi)
            calls += 1
            simplex.append(xi)
            simplex_f.append(fi)
            if fi < best_fx:
                best_x = xi.copy()
                best_fx = fi

        # Nelder-Mead parameters
        alpha_nm = 1.0   # reflection
        gamma = 2.0      # expansion
        rho = 0.5        # contraction
        sigma_nm = 0.5   # shrink

        stall_count = 0
        stall_limit = 20

        while calls < budget:
            if time.time() > deadline:
                break

            # Sort simplex by fitness
            indices = np.argsort(simplex_f)
            simplex = [simplex[i].copy() for i in indices]
            simplex_f = [simplex_f[i] for i in indices]

            # Update best
            if simplex_f[0] < best_fx:
                best_x = simplex[0].copy()
                best_fx = simplex_f[0]

            # Check for convergence
            if abs(simplex_f[-1] - simplex_f[0]) < 1e-10:
                break

            # Compute centroid of all points except worst
            centroid = np.mean(simplex[:-1], axis=0)

            # Reflection
            x_ref = centroid + alpha_nm * (centroid - simplex[-1])
            x_ref = np.clip(x_ref, -5, 5)
            f_ref = func(x_ref)
            calls += 1

            if f_ref < simplex_f[0]:
                # Try expansion
                x_exp = centroid + gamma * (centroid - simplex[-1])
                x_exp = np.clip(x_exp, -5, 5)
                f_exp = func(x_exp)
                calls += 1

                if f_exp < f_ref:
                    simplex[-1] = x_exp
                    simplex_f[-1] = f_exp
                else:
                    simplex[-1] = x_ref
                    simplex_f[-1] = f_ref
            elif f_ref < simplex_f[-2]:
                # Accept reflection
                simplex[-1] = x_ref
                simplex_f[-1] = f_ref
            else:
                # Try contraction
                x_con = centroid + rho * (simplex[-1] - centroid)
                x_con = np.clip(x_con, -5, 5)
                f_con = func(x_con)
                calls += 1

                if f_con < simplex_f[-1]:
                    simplex[-1] = x_con
                    simplex_f[-1] = f_con
                else:
                    # Shrink
                    for i in range(1, len(simplex)):
                        simplex[i] = simplex[0] + sigma_nm * (simplex[i] - simplex[0])
                        simplex[i] = np.clip(simplex[i], -5, 5)
                        simplex_f[i] = func(simplex[i])
                        calls += 1
                        if simplex_f[i] < best_fx:
                            best_x = simplex[i].copy()
                            best_fx = simplex_f[i]

            # Track stagnation
            if simplex_f[0] >= best_fx + 1e-12:
                stall_count += 1
            else:
                stall_count = 0

            if stall_count >= stall_limit:
                # Perturb the worst point to escape local minimum
                simplex[-1] = np.clip(simplex[0] + 2.0 * rng.standard_normal(dim), -5, 5)
                simplex_f[-1] = func(simplex[-1])
                calls += 1
                if simplex_f[-1] < best_fx:
                    best_x = simplex[-1].copy()
                    best_fx = simplex_f[-1]
                stall_count = 0

            # Early termination if simplex is too small
            spread = max(np.max(np.array(simplex), axis=0) - np.min(np.array(simplex), axis=0))
            if spread < 1e-8:
                break

        return best_x, best_fx, calls

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # 20s safety margin before 300s limit
        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        best_x = None
        best_fx = np.inf
        total_calls = 0

        # Phase 1: Sample multiple random starting points
        n_samples = 15
        samples = []
        sample_f = []

        for i in range(n_samples):
            if total_calls >= budget or time.time() > deadline:
                break
            x = rng.uniform(-5, 5, dim)
            f = func(x)
            total_calls += 1
            samples.append(x.copy())
            sample_f.append(f)
            if f < best_fx:
                best_x = x.copy()
                best_fx = f

        if best_x is None:
            x0 = np.zeros(dim)
            f0 = func(x0)
            best_x = x0
            best_fx = f0
            total_calls += 1

        # Phase 2: Run Nelder-Mead from the best starting point
        remaining = budget - total_calls
        if remaining > 0 and time.time() < deadline:
            # Run Nelder-Mead from best sample
            x0 = samples[np.argmin(sample_f)].copy()
            fx0 = min(sample_f)
            nm_budget = min(remaining, 200)
            best_x, best_fx, calls_used = self._nelder_mead(func, x0, fx0, nm_budget, deadline, rng)
            total_calls += calls_used - 1  # fx0 already counted

        # Phase 3: If budget remains, try Nelder-Mead from second-best sample
        if total_calls < budget and time.time() < deadline:
            sorted_idx = np.argsort(sample_f)
            if len(sorted_idx) > 1:
                x0 = samples[sorted_idx[1]].copy()
                fx0 = sample_f[sorted_idx[1]]
                remaining = budget - total_calls
                if remaining > 10:
                    nm_budget = min(remaining, 80)
                    bx2, bf2, calls_used2 = self._nelder_mead(func, x0, fx0, nm_budget, deadline, rng)
                    total_calls += calls_used2 - 1
                    if bf2 < best_fx:
                        best_x = bx2.copy()
                        best_fx = bf2

        # Phase 4: Final refinement with small perturbations around best
        if best_x is not None and total_calls < budget:
            remaining = budget - total_calls
            if remaining > 5:
                x = best_x.copy()
                fx = best_fx
                sigma = 0.3
                for _ in range(min(remaining, 30)):
                    if time.time() > deadline:
                        break
                    z = rng.standard_normal(dim)
                    x_new = np.clip(x + sigma * z, -5, 5)
                    f_new = func(x_new)
                    total_calls += 1
                    if f_new < fx:
                        x = x_new
                        fx = f_new
                        sigma *= 0.95
                    else:
                        sigma *= 0.9

                if fx < best_fx:
                    best_x = x.copy()
                    best_fx = fx

        if best_x is None:
            x0 = rng.uniform(-5, 5, dim)
            fx0 = func(x0)
            best_x = x0
            best_fx = fx0
            total_calls += 1

        return best_x, best_fx
