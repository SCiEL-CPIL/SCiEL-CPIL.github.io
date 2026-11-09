"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-start (1+1)-ES with Nelder-Mead polish for BBOB f21."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285.0  # Safety margin before 300s kill
        rng = np.random.default_rng(12345)

        calls = 0
        best_x = None
        best_f = np.inf

        def count(x):
            nonlocal calls
            fx = func(x)
            calls += 1
            return fx

        # Phase 1: Global exploration with multi-start (1+1)-ES
        # Use a few diverse starting points, each with a short ES run
        n_starts = 5
        calls_per_start = (self.budget - 200) // n_starts  # reserve 200 for polish
        calls_per_start = max(calls_per_start, 20)

        for s in range(n_starts):
            if calls >= self.budget or time.time() > deadline:
                break

            # Diverse starting points: mix of random and structured
            if s == 0:
                x = np.zeros(self.dim)
            elif s == 1:
                x = rng.uniform(-4, 4, self.dim)
            else:
                x = rng.uniform(-5, 5, self.dim)

            x = np.clip(x, -5, 5)
            fx = count(x)

            if fx < best_f:
                best_f = fx
                best_x = x.copy()

            sigma = 1.5
            for _ in range(calls_per_start):
                if calls >= self.budget or time.time() > deadline:
                    break
                y = np.clip(x + sigma * rng.standard_normal(self.dim), -5, 5)
                fy = count(y)
                if fy < fx:
                    x, fx = y, fy
                    sigma *= 1.3
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                else:
                    sigma *= 0.85
                    if sigma < 0.01:
                        break

        # Phase 2: Nelder-Mead polish on the best point
        if best_x is not None and calls < self.budget and time.time() < deadline:
            # Simple Nelder-Mead implementation
            dim = self.dim
            x0 = best_x.copy()
            f0 = best_f

            # Initial simplex
            simplex = [x0.copy()]
            for i in range(dim):
                xi = x0.copy()
                step = 0.1 if abs(x0[i]) > 1e-10 else 0.1
                xi[i] += step
                simplex.append(np.clip(xi, -5, 5))

            fvals = [f0]
            for i in range(1, dim + 1):
                fvals.append(count(simplex[i]))

            alpha = 1.0   # reflection
            gamma = 2.0   # expansion
            rho = 0.5     # contraction
            sigma_nm = 0.5  # shrink

            max_iter = min(self.budget - calls, 100)
            for _ in range(max_iter):
                if calls >= self.budget or time.time() > deadline:
                    break

                # Sort simplex by function value
                order = np.argsort(fvals)
                simplex = [simplex[i] for i in order]
                fvals = [fvals[i] for i in order]

                # Centroid of all but worst
                centroid = np.mean(simplex[:-1], axis=0)

                # Reflection
                xr = centroid + alpha * (centroid - simplex[-1])
                xr = np.clip(xr, -5, 5)
                fr = count(xr)

                if fr < fvals[0]:
                    # Expansion
                    xe = centroid + gamma * (centroid - simplex[-1])
                    xe = np.clip(xe, -5, 5)
                    fe = count(xe)
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
                        # Outside contraction
                        xc = centroid + 0.5 * (xr - centroid)
                        xc = np.clip(xc, -5, 5)
                        fc = count(xc)
                        if fc <= fr:
                            simplex[-1] = xc
                            fvals[-1] = fc
                        else:
                            # Shrink
                            for i in range(1, dim + 1):
                                simplex[i] = simplex[0] + sigma_nm * (simplex[i] - simplex[0])
                                simplex[i] = np.clip(simplex[i], -5, 5)
                                fvals[i] = count(simplex[i])
                    else:
                        # Inside contraction
                        xc = centroid + 0.5 * (simplex[-1] - centroid)
                        xc = np.clip(xc, -5, 5)
                        fc = count(xc)
                        if fc < fvals[-1]:
                            simplex[-1] = xc
                            fvals[-1] = fc
                        else:
                            # Shrink
                            for i in range(1, dim + 1):
                                simplex[i] = simplex[0] + sigma_nm * (simplex[i] - simplex[0])
                                simplex[i] = np.clip(simplex[i], -5, 5)
                                fvals[i] = count(simplex[i])

                # Update best
                min_idx = np.argmin(fvals)
                if fvals[min_idx] < best_f:
                    best_f = fvals[min_idx]
                    best_x = simplex[min_idx].copy()

                # Check convergence
                if max(fvals) - min(fvals) < 1e-10:
                    break

        # Phase 3: Final fine-tuning with small-step (1+1)-ES if budget remains
        if calls < self.budget and time.time() < deadline:
            x = best_x.copy()
            fx = best_f
            sigma = 0.05
            for _ in range(self.budget - calls):
                if time.time() > deadline:
                    break
                y = np.clip(x + sigma * rng.standard_normal(self.dim), -5, 5)
                fy = count(y)
                if fy < fx:
                    x, fx = y, fy
                    sigma *= 1.2
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                else:
                    sigma *= 0.9
                    if sigma < 0.001:
                        break

        return best_x, best_f
# EVOLVE-BLOCK-END
