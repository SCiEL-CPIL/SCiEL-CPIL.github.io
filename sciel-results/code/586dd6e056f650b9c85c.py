"""Optimizer for BBOB f20 (Schwefel) on [-5,5]^5."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-start Nelder-Mead with time guard."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def _nelder_mead(self, func, x0, max_iter=200):
        """Run Nelder-Mead from x0, return best point and value."""
        dim = self.dim
        # Initial simplex
        simplex = np.zeros((dim + 1, dim))
        simplex[0] = x0
        for i in range(dim):
            simplex[i + 1] = x0.copy()
            simplex[i + 1][i] += 0.1 if x0[i] < 4.9 else -0.1

        fvals = np.array([func(simplex[i]) for i in range(dim + 1)])

        alpha, gamma, rho, sigma = 1.0, 2.0, 0.5, 0.5
        best_idx = np.argmin(fvals)

        for _ in range(max_iter):
            # Sort simplex by function value
            order = np.argsort(fvals)
            simplex = simplex[order]
            fvals = fvals[order]

            if fvals[0] - fvals[-1] < 1e-10:
                break

            # Centroid of all but worst
            centroid = np.mean(simplex[:-1], axis=0)

            # Reflection
            xr = centroid + alpha * (centroid - simplex[-1])
            xr = np.clip(xr, -5, 5)
            fr = func(xr)

            if fr < fvals[0]:
                # Expansion
                xe = centroid + gamma * (centroid - simplex[-1])
                xe = np.clip(xe, -5, 5)
                fe = func(xe)
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
                    xc = centroid + rho * (simplex[-1] - centroid)
                else:
                    # Inside contraction
                    xc = centroid - rho * (simplex[-1] - centroid)
                xc = np.clip(xc, -5, 5)
                fc = func(xc)
                if fc < fvals[-1]:
                    simplex[-1] = xc
                    fvals[-1] = fc
                else:
                    # Shrink
                    for i in range(1, dim + 1):
                        simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                        simplex[i] = np.clip(simplex[i], -5, 5)
                        fvals[i] = func(simplex[i])

        best_idx = np.argmin(fvals)
        return simplex[best_idx], fvals[best_idx]

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # safety margin before 300s

        dim = self.dim
        budget = self.budget

        # Multi-start: try several initial points
        rng = np.random.default_rng(42)

        # Generate diverse starting points
        starts = []
        # Center
        starts.append(np.zeros(dim))
        # Random points
        for _ in range(10):
            starts.append(rng.uniform(-4.5, 4.5, dim))
        # Grid-like points
        for v in [-3, 0, 3]:
            for w in [-3, 0, 3]:
                pt = np.array([v, w, 0, 0, 0], dtype=float)
                starts.append(pt)
                pt2 = np.array([0, 0, v, w, 0], dtype=float)
                starts.append(pt2)
                pt3 = np.array([0, 0, 0, v, w], dtype=float)
                starts.append(pt3)

        # Remove duplicates approximately
        unique_starts = [starts[0]]
        for s in starts[1:]:
            if not any(np.allclose(s, u, atol=0.5) for u in unique_starts):
                unique_starts.append(s)

        best_x = None
        best_f = np.inf

        for i, x0 in enumerate(unique_starts):
            if time.time() > deadline:
                break

            # Use remaining budget for this start
            remaining_budget = budget - i * 50  # approximate
            if remaining_budget < 20:
                break

            # Scale max_iter based on remaining time
            elapsed = time.time() - start_time
            time_left = deadline - elapsed
            # Estimate: ~0.5ms per function eval, ~10 evals per NM iteration
            max_iter = min(200, int(time_left / 0.005))
            if max_iter < 10:
                break

            x_opt, f_opt = self._nelder_mead(func, x0, max_iter=max_iter)

            if f_opt < best_f:
                best_f = f_opt
                best_x = x_opt.copy()

            # If we've found something very good, do a polish
            if f_opt < 1e-3:
                # Polish with smaller initial step
                x_polish = x_opt.copy()
                for _ in range(min(50, max(10, int(time_left / 0.01)))):
                    if time.time() > deadline:
                        break
                    for d in range(dim):
                        for delta in [0.01, 0.001, -0.01, -0.001]:
                            x_trial = x_polish.copy()
                            x_trial[d] += delta
                            x_trial = np.clip(x_trial, -5, 5)
                            f_trial = func(x_trial)
                            if f_trial < best_f:
                                best_f = f_trial
                                best_x = x_trial.copy()

        if best_x is None:
            # Fallback: single random point
            best_x = rng.uniform(-5, 5, dim)
            best_f = func(best_x)

        return best_x, best_f
# EVOLVE-BLOCK-END
