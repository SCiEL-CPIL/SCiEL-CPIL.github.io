"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """(1+1)-ES with 1/5 success rule, stagnation restarts, and Nelder-Mead polish."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def _nelder_mead(self, func, x0, budget, deadline, rng, sigma_init=0.5):
        """Run a Nelder-Mead local search from x0 with a function-call budget."""
        dim = self.dim
        n = dim + 1
        # Simplex initialization
        simplex = np.zeros((n, dim))
        simplex[0] = x0
        fvals = np.zeros(n)
        fvals[0] = func(x0)
        budget -= 1
        for i in range(1, n):
            step = x0.copy()
            step[i - 1] += sigma_init
            step = np.clip(step, -5, 5)
            simplex[i] = step
            fvals[i] = func(step)
            budget -= 1

        alpha = 1.0  # reflection
        gamma = 2.0  # expansion
        rho = 0.5    # contraction
        sigma = 0.5  # shrink

        best_idx = np.argmin(fvals)
        for _ in range(budget):
            if time.time() > deadline:
                break
            # Sort simplex
            order = np.argsort(fvals)
            simplex = simplex[order]
            fvals = fvals[order]
            best_idx = 0

            # Centroid of all but worst
            centroid = simplex[:-1].mean(axis=0)
            worst = simplex[-1]
            worst_f = fvals[-1]

            # Reflection
            xr = centroid + alpha * (centroid - worst)
            xr = np.clip(xr, -5, 5)
            fr = func(xr)
            budget -= 1
            if time.time() > deadline:
                break

            if fr < fvals[0]:
                # Try expansion
                xe = centroid + gamma * (xr - centroid)
                xe = np.clip(xe, -5, 5)
                fe = func(xe)
                budget -= 1
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
                if fr < worst_f:
                    xc = centroid + rho * (xr - centroid)
                else:
                    xc = centroid - rho * (worst - centroid)
                xc = np.clip(xc, -5, 5)
                fc = func(xc)
                budget -= 1
                if fc <= min(fr, worst_f):
                    simplex[-1] = xc
                    fvals[-1] = fc
                else:
                    # Shrink
                    for i in range(1, n):
                        simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                        simplex[i] = np.clip(simplex[i], -5, 5)
                        fvals[i] = func(simplex[i])
                        budget -= 1
                        if time.time() > deadline:
                            break
                    if time.time() > deadline:
                        break

            best_idx = np.argmin(fvals)

        best_x = simplex[np.argmin(fvals)].copy()
        best_fx = fvals[best_idx]
        return best_x, best_fx, budget

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 270.0  # safety margin before 300s kill
        rng = np.random.default_rng(42)
        dim = self.dim

        # Phase 1: (1+1)-ES for global exploration
        # Use ~70% of budget for ES, 30% for NM polish
        es_budget = int(0.70 * self.budget)
        nm_budget = self.budget - es_budget

        x = rng.uniform(-5, 5, dim)
        fx = func(x)
        best_x = x.copy()
        best_fx = fx
        sigma = 2.0
        calls = 1

        stagnation = 0
        max_stagnation = 30

        for i in range(es_budget - 1):
            if time.time() > deadline:
                break
            calls += 1
            y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
            fy = func(y)
            if fy < fx:
                x, fx = y, fy
                sigma *= 1.3
                stagnation = 0
            else:
                sigma *= 0.9
                stagnation += 1

            if fx < best_fx:
                best_fx = fx
                best_x = x.copy()

            # Restart if stagnated too long
            if stagnation > max_stagnation:
                # Move toward best found
                x = best_x + sigma * 0.5 * rng.standard_normal(dim)
                x = np.clip(x, -5, 5)
                fx = func(x)
                calls += 1
                if fx < best_fx:
                    best_fx = fx
                    best_x = x.copy()
                sigma = max(sigma, 1.0)
                stagnation = 0

        # Phase 2: Nelder-Mead polish from best point
        remaining = nm_budget - max(0, calls - es_budget + 1)  # account for extra restart calls
        remaining = max(remaining, 50)
        if remaining > 10 and time.time() < deadline:
            nm_x, nm_fx, used = self._nelder_mead(func, best_x, remaining, deadline, rng, sigma_init=0.3)
            if nm_fx < best_fx:
                best_fx = nm_fx
                best_x = nm_x

        return best_x, best_fx
# EVOLVE-BLOCK-END
