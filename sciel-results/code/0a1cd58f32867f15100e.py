"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """(1+1)-ES with Nelder-Mead polish and time guard."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # safety margin before 330s limit

        rng = np.random.default_rng(0)

        # Initial point
        x = rng.uniform(-5, 5, self.dim)
        fx = func(x)
        best_x, best_fx = x.copy(), fx
        calls = 1

        # (1+1)-ES phase
        sigma = 2.0
        min_sigma = 1e-8

        while calls < self.budget:
            if time.time() > deadline:
                break

            y = np.clip(x + sigma * rng.standard_normal(self.dim), -5, 5)
            fy = func(y)
            calls += 1

            if fy < fx:
                x, fx = y, fy
                sigma *= 1.5
            else:
                sigma *= 0.87

            if fx < best_fx:
                best_x, best_fx = x.copy(), fx

            if sigma < min_sigma:
                break

        # Nelder-Mead polish phase
        if calls < self.budget and time.time() < deadline:
            x, fx = self._nelder_mead(func, best_x, best_fx, calls, self.budget, deadline)
            calls = fx[1]
            if fx[0] < best_fx:
                best_x, best_fx = fx[2].copy(), fx[0]

        return best_x, best_fx

    def _nelder_mead(self, func, x0, fx0, calls, budget, deadline):
        """Nelder-Mead simplex method for local polishing."""
        dim = self.dim
        # Initialize simplex
        simplex = [x0.copy()]
        fvals = [fx0]

        for i in range(dim):
            xi = x0.copy()
            xi[i] += 0.5
            simplex.append(xi)
            fvals.append(func(xi))
            calls += 1

        # Nelder-Mead iterations
        alpha = 1.0  # reflection
        gamma = 2.0  # expansion
        rho = 0.5    # contraction
        sigma_nm = 0.5  # shrink

        max_iter = min(budget - calls, 200)

        for _ in range(max_iter):
            if time.time() > deadline:
                break
            if calls >= budget:
                break

            # Sort simplex by function value
            idx = np.argsort(fvals)
            simplex = [simplex[i] for i in idx]
            fvals = [fvals[i] for i in idx]

            # Compute centroid of all but worst
            centroid = np.mean(simplex[:-1], axis=0)

            # Reflection
            xr = centroid + alpha * (centroid - simplex[-1])
            xr = np.clip(xr, -5, 5)
            fr = func(xr)
            calls += 1

            if fr < fvals[0]:
                # Expansion
                xe = centroid + gamma * (centroid - simplex[-1])
                xe = np.clip(xe, -5, 5)
                fe = func(xe)
                calls += 1
                if fe < fr:
                    simplex[-1] = xe
                    fvals[-1] = fe
                else:
                    simplex[-1] = xr
                    fvals[-1] = fr
            elif fr < fvals[-2]:
                simplex[-1] = xr
                fvals[-1] = fr
            else:
                # Contraction
                if fr < fvals[-1]:
                    xc = centroid + rho * (simplex[-1] - centroid)
                    fc = func(xc)
                    calls += 1
                    if fc <= fr:
                        simplex[-1] = xc
                        fvals[-1] = fc
                    else:
                        # Shrink
                        for i in range(1, len(simplex)):
                            simplex[i] = simplex[0] + sigma_nm * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], -5, 5)
                            fvals[i] = func(simplex[i])
                            calls += 1
                else:
                    # Shrink
                    for i in range(1, len(simplex)):
                        simplex[i] = simplex[0] + sigma_nm * (simplex[i] - simplex[0])
                        simplex[i] = np.clip(simplex[i], -5, 5)
                        fvals[i] = func(simplex[i])
                        calls += 1

            # Check convergence
            fmin, fmax = min(fvals), max(fvals)
            if abs(fmax - fmin) < 1e-12:
                break

        # Return best point
        best_idx = np.argmin(fvals)
        return fvals[best_idx], calls, simplex[best_idx]
# EVOLVE-BLOCK-END
