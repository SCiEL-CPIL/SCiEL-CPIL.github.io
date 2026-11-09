"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Robust Multi-start Nelder-Mead optimizer for Rosenbrock."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def _nelder_mead(self, func, x0, max_calls, deadline):
        """Run Nelder-Mead simplex method with a strict call budget."""
        n = self.dim
        # Initial simplex
        simplex = [x0.copy()]
        for i in range(n):
            v = x0.copy()
            # Small step in each dimension
            step = 0.2
            v[i] += step
            simplex.append(v)

        # Evaluate initial simplex
        fvals = np.array([func(np.clip(v, -5, 5)) for v in simplex])
        calls_used = n + 1

        best_idx = np.argmin(fvals)
        best_f = fvals[best_idx]
        best_x = simplex[best_idx].copy()

        # Parameters
        alpha = 1.0
        gamma = 2.0
        rho = 0.5
        sigma = 0.5

        while calls_used < max_calls:
            if time.time() >= deadline:
                break

            # Sort
            order = np.argsort(fvals)
            simplex = [simplex[i].copy() for i in order]
            fvals = fvals[order]

            centroid = np.mean(simplex[:-1], axis=0)

            # Reflection
            xr = centroid + alpha * (centroid - simplex[-1])
            xr = np.clip(xr, -5, 5)
            fr = func(xr)
            calls_used += 1

            if fr < fvals[0]:
                # Expansion
                xe = centroid + gamma * (xr - centroid)
                xe = np.clip(xe, -5, 5)
                fe = func(xe)
                calls_used += 1

                if fe < fr:
                    simplex[-1] = xe
                    fvals[-1] = fe
                else:
                    simplex[-1] = xr
                    fvals[-1] = fr
            elif fr < fvals[-2]:
                # Accept Reflection
                simplex[-1] = xr
                fvals[-1] = fr
            else:
                # Contraction
                if fr < fvals[-1]:
                    xc = centroid + rho * (xr - centroid)
                else:
                    xc = centroid - rho * (simplex[-1] - centroid)
                xc = np.clip(xc, -5, 5)
                fc = func(xc)
                calls_used += 1

                if fc < fvals[-1]:
                    simplex[-1] = xc
                    fvals[-1] = fc
                else:
                    # Shrink
                    for i in range(1, n + 1):
                        simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                        simplex[i] = np.clip(simplex[i], -5, 5)
                        fvals[i] = func(simplex[i])
                        calls_used += 1
                        if calls_used >= max_calls:
                            break

            # Update best
            current_best_idx = np.argmin(fvals)
            if fvals[current_best_idx] < best_f:
                best_f = fvals[current_best_idx]
                best_x = simplex[current_best_idx].copy()

            # Convergence check
            if np.max(fvals) - np.min(fvals) < 1e-12:
                break

        return best_x, best_f, calls_used

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0
        rng = np.random.default_rng(42)

        best_x = np.zeros(self.dim)
        best_f = np.inf
        calls_made = 0

        # Generate diverse starting points
        # Include origin, random points, and points near likely minima (Rosenbrock min is often at 1,1,1,1,1 in standard form, but here it's shifted/rotated)
        starts = [
            np.zeros(self.dim),
            np.ones(self.dim) * 1.0,
            np.ones(self.dim) * -1.0,
            np.array([1.0, 1.0, 1.0, 1.0, 1.0]),
            rng.uniform(-4, 4, self.dim),
            rng.uniform(-4, 4, self.dim),
            rng.uniform(-4, 4, self.dim),
            rng.uniform(-4, 4, self.dim),
            rng.uniform(-4, 4, self.dim),
            rng.uniform(-4, 4, self.dim),
        ]

        # Shuffle starts
        indices = np.arange(len(starts))
        rng.shuffle(indices)

        # Allocate budget: 10 starts, 50 calls each = 500 total
        calls_per_start = self.budget // len(starts)

        for idx in indices:
            if calls_made >= self.budget:
                break
            if time.time() >= deadline:
                break

            x0 = np.clip(starts[idx], -5, 5)
            # Ensure we don't exceed total budget
            remaining = self.budget - calls_made
            current_budget = min(calls_per_start, remaining)
            
            if current_budget < 10: # Need at least n+1=6 for NM, 10 is safe
                break

            bx, bf, used = self._nelder_mead(func, x0, current_budget, deadline)
            calls_made += used

            if bf < best_f:
                best_f = bf
                best_x = bx.copy()

        return best_x, best_f
# EVOLVE-BLOCK-END
