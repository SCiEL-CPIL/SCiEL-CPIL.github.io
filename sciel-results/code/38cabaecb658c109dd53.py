"""Two-stage optimizer: Sobol global sampling + Nelder-Mead pattern search."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Sobol global search followed by Nelder-Mead local refinement."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def _sobol_sample(self, n, dim, rng):
        """Generate n Sobol-like low-discrepancy points in [-5, 5]^dim."""
        # Use a simple Halton-like quasi-random sequence for low-discrepancy sampling
        points = np.zeros((n, dim))
        for d in range(dim):
            # Prime number for each dimension
            primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]
            p = primes[d % len(primes)]
            for i in range(n):
                # Van der Corput sequence
                val = 0.0
                f = 1.0
                k = i + 1
                while k > 0:
                    val += f * (k % p)
                    k //= p
                    f /= p
                points[i, d] = val
        # Map to [-5, 5]
        points = points * 10.0 - 5.0
        return points

    def _nelder_mead(self, func, x0, max_iter=200, tol=1e-8, deadline=None):
        """Nelder-Mead simplex method with time guard."""
        dim = self.dim
        n = dim + 1
        # Initialize simplex
        simplex = np.zeros((n, dim))
        simplex[0] = x0
        for i in range(1, n):
            simplex[i] = x0.copy()
            simplex[i, i - 1] = 0.5

        fvals = np.array([func(simplex[i]) for i in range(n)])
        best_x = simplex[np.argmin(fvals)].copy()
        best_f = np.min(fvals)

        for iteration in range(max_iter):
            if deadline is not None and time.time() > deadline:
                break

            # Sort simplex by function value
            order = np.argsort(fvals)
            simplex = simplex[order]
            fvals = fvals[order]

            # Check convergence
            if np.max(fvals) - np.min(fvals) < tol:
                break

            # Centroid of all but worst point
            centroid = np.mean(simplex[:-1], axis=0)

            # Reflection
            alpha = 1.0
            x_ref = centroid + alpha * (centroid - simplex[-1])
            x_ref = np.clip(x_ref, -5, 5)
            f_ref = func(x_ref)

            if f_ref < fvals[0]:
                # Expansion
                gamma = 2.0
                x_exp = centroid + gamma * (centroid - simplex[-1])
                x_exp = np.clip(x_exp, -5, 5)
                f_exp = func(x_exp)
                if f_exp < f_ref:
                    simplex[-1] = x_exp
                    fvals[-1] = f_exp
                else:
                    simplex[-1] = x_ref
                    fvals[-1] = f_ref
            elif f_ref < fvals[-2]:
                # Accept reflection
                simplex[-1] = x_ref
                fvals[-1] = f_ref
            else:
                # Contraction
                beta = 0.5
                x_con = centroid + beta * (simplex[-1] - centroid)
                x_con = np.clip(x_con, -5, 5)
                f_con = func(x_con)
                if f_con < fvals[-1]:
                    simplex[-1] = x_con
                    fvals[-1] = f_con
                else:
                    # Shrink
                    for i in range(1, n):
                        simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                        fvals[i] = func(simplex[i])

            # Track best
            idx = np.argmin(fvals)
            if fvals[idx] < best_f:
                best_f = fvals[idx]
                best_x = simplex[idx].copy()

        return best_x, best_f

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 270  # 30s safety margin before 300s limit
        dim = self.dim
        budget = self.budget

        # Stage 1: Sobol global sampling
        n_sobol = min(100, budget // 5)
        sobol_points = self._sobol_sample(n_sobol, dim, None)

        best_x = None
        best_f = np.inf
        calls_made = 0

        for i in range(n_sobol):
            if calls_made >= budget:
                break
            if time.time() > deadline:
                break
            x = sobol_points[i]
            f = func(x)
            calls_made += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()

        # Stage 2: Nelder-Mead from best Sobol point
        if best_x is not None and calls_made < budget and time.time() < deadline:
            remaining_budget = budget - calls_made
            # Estimate max iterations based on remaining budget
            # Each NM iteration uses up to dim+2 evaluations
            max_iter = max(1, remaining_budget // (dim + 2))
            nm_x, nm_f = self._nelder_mead(func, best_x, max_iter=max_iter, deadline=deadline)
            if nm_f < best_f:
                best_f = nm_f
                best_x = nm_x

        # If we still have budget, do a few random restarts with NM
        if calls_made < budget and time.time() < deadline:
            rng = np.random.default_rng(42)
            remaining = budget - calls_made
            n_restarts = min(5, remaining // 20)
            for _ in range(n_restarts):
                if calls_made >= budget or time.time() > deadline:
                    break
                x0 = rng.uniform(-5, 5, dim)
                f0 = func(x0)
                calls_made += 1
                if f0 < best_f:
                    best_f = f0
                    best_x = x0.copy()
                if calls_made < budget and time.time() < deadline:
                    remaining_iter = (budget - calls_made) // (dim + 2)
                    if remaining_iter > 5:
                        nm_x, nm_f = self._nelder_mead(func, x0, max_iter=min(30, remaining_iter), deadline=deadline)
                        if nm_f < best_f:
                            best_f = nm_f
                            best_x = nm_x

        return best_x, best_f
# EVOLVE-BLOCK-END
