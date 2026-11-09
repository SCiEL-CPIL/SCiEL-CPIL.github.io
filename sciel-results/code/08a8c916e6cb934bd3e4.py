"""Rewritten optimizer for BBOB-style Rosenbrock (f8) on [-5,5]^5."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """(1+1)-ES with Nelder-Mead polish and deadline guard."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # safety margin before 300s limit
        rng = np.random.default_rng(0)

        # Start from a reasonable point
        x = np.zeros(self.dim)
        fx = func(x)
        best_x = x.copy()
        best_fx = fx
        calls = 1

        # Phase 1: (1+1)-ES to get close to the basin
        sigma = 1.5
        for i in range(self.budget - 1):
            if time.time() > deadline - 30.0:
                break
            y = np.clip(x + sigma * rng.standard_normal(self.dim), -5, 5)
            fy = func(y)
            calls += 1
            if fy < fx:
                x, fx = y, fy
                if fx < best_fx:
                    best_x, best_fx = x.copy(), fx
                sigma *= 1.2
            else:
                sigma *= 0.85

        # Phase 2: Nelder-Mead polish from the best point found
        if calls < self.budget - 10 and time.time() < deadline - 5.0:
            best_x = self._nelder_mead(func, best_x, best_fx, budget_remaining=self.budget - calls,
                                        deadline=deadline, dim=self.dim)
            # Update best
            fx_final = func(best_x)
            calls += 1
            if fx_final < best_fx:
                best_x, best_fx = best_x, fx_final

        return best_x, best_fx

    def _nelder_mead(self, func, x0, f0, budget_remaining, deadline, dim):
        """Nelder-Mead simplex method for final polish."""
        # Initialize simplex
        simplex = [x0.copy()]
        fvals = [f0]
        # Add perturbed points
        for i in range(dim):
            xi = x0.copy()
            step = 0.1 if abs(x0[i]) < 1e-10 else 0.1 * abs(x0[i])
            xi[i] += step
            xi = np.clip(xi, -5, 5)
            fi = func(xi)
            simplex.append(xi)
            fvals.append(fi)

        alpha = 1.0  # reflection
        gamma = 2.0  # expansion
        rho = 0.5    # contraction
        sigma_nm = 0.5  # shrink

        calls = dim + 1
        max_iter = min(budget_remaining - 1, 500)

        for _ in range(max_iter):
            if time.time() > deadline - 2.0:
                break

            # Sort simplex
            idx = np.argsort(fvals)
            simplex = [simplex[i].copy() for i in idx]
            fvals = [fvals[i] for i in idx]

            # Centroid of all but worst
            centroid = np.mean(simplex[:-1], axis=0)

            # Reflection
            xr = np.clip(centroid + alpha * (centroid - simplex[-1]), -5, 5)
            fr = func(xr)
            calls += 1

            if fr < fvals[0]:
                # Expansion
                xe = np.clip(centroid + gamma * (centroid - simplex[-1]), -5, 5)
                fe = func(xe)
                calls += 1
                if fe < fr:
                    simplex[-1], fvals[-1] = xe, fe
                else:
                    simplex[-1], fvals[-1] = xr, fr
            elif fr < fvals[-2]:
                simplex[-1], fvals[-1] = xr, fr
            else:
                # Contraction
                if fr < fvals[-1]:
                    xc = np.clip(centroid + rho * (simplex[-1] - centroid), -5, 5)
                else:
                    xc = np.clip(centroid - rho * (centroid - simplex[-1]), -5, 5)
                fc = func(xc)
                calls += 1
                if fc <= min(fr, fvals[-1]):
                    simplex[-1], fvals[-1] = xc, fc
                else:
                    # Shrink
                    for i in range(1, dim + 1):
                        simplex[i] = np.clip(simplex[0] + sigma_nm * (simplex[i] - simplex[0]), -5, 5)
                        fvals[i] = func(simplex[i])
                        calls += 1
                        if calls > budget_remaining:
                            break
                    if calls > budget_remaining:
                        break

        # Return best point
        best_idx = np.argmin(fvals)
        return simplex[best_idx]
# EVOLVE-BLOCK-END
