"""Optimizer for BBOB-style Rosenbrock (f8) on [-5,5]^5."""
import numpy as np
import time


class Optimizer:
    """Nelder-Mead optimizer with time guard for Rosenbrock optimization."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        t0 = time.time()
        deadline = t0 + 270  # safety margin before 300s limit

        def safe_call(x):
            nonlocal calls, best_x, best_f
            x = np.clip(np.asarray(x, dtype=float), -5.0, 5.0)
            if calls >= budget:
                return np.inf
            f = func(x)
            if not np.isfinite(f):
                f = np.inf
            calls += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        # Nelder-Mead parameters
        alpha = 1.0  # Reflection
        gamma = 2.0  # Expansion
        rho = 0.5    # Contraction
        sigma = 0.5  # Shrinkage

        calls = 0
        best_x = np.zeros(dim)
        best_f = np.inf

        # Try a few random starting points to find a good initial region
        rng = np.random.default_rng(42)
        # Start with center
        x0 = np.zeros(dim)
        f0 = safe_call(x0)

        # Try a few random points to find a better starting region
        for attempt in range(10):
            if calls >= budget or time.time() > deadline:
                break
            x_try = rng.uniform(-5, 5, size=dim)
            f_try = safe_call(x_try)
            if f_try < f0:
                x0 = x_try
                f0 = f_try

        # Create initial simplex around the best starting point
        step = 0.5
        simplex = [x0.copy()]
        fvals = [f0]

        for i in range(dim):
            if calls >= budget or time.time() > deadline:
                break
            x_new = x0.copy()
            if x0[i] + step <= 5.0:
                x_new[i] += step
            else:
                x_new[i] -= step
            f_new = safe_call(x_new)
            simplex.append(x_new)
            fvals.append(f_new)

        # If we couldn't build a full simplex, return best found so far
        if len(simplex) < dim + 1:
            return best_x, best_f

        # Sort simplex by function value
        sorted_indices = np.argsort(fvals)
        simplex = [simplex[i] for i in sorted_indices]
        fvals = [fvals[i] for i in sorted_indices]

        while calls < budget and time.time() < deadline:
            # Compute centroid of all points except the worst
            centroid = np.sum(simplex[:dim], axis=0) / dim

            # Reflection
            x_r = centroid + alpha * (centroid - simplex[dim])
            x_r = np.clip(x_r, -5, 5)
            f_r = safe_call(x_r)
            if f_r == np.inf:
                break

            if f_r < fvals[0]:
                # Expansion
                x_e = centroid + gamma * (centroid - simplex[dim])
                x_e = np.clip(x_e, -5, 5)
                f_e = safe_call(x_e)
                if f_e == np.inf:
                    # Expansion failed, use reflection
                    simplex[dim] = x_r
                    fvals[dim] = f_r
                elif f_e < f_r:
                    simplex[dim] = x_e
                    fvals[dim] = f_e
                else:
                    simplex[dim] = x_r
                    fvals[dim] = f_r
            elif f_r < fvals[dim - 1]:
                # Accept reflection
                simplex[dim] = x_r
                fvals[dim] = f_r
            else:
                # Contraction
                if f_r < fvals[dim]:
                    # Outside contraction
                    x_c = centroid + rho * (x_r - centroid)
                else:
                    # Inside contraction
                    x_c = centroid - rho * (x_r - centroid)

                x_c = np.clip(x_c, -5, 5)
                f_c = safe_call(x_c)
                if f_c == np.inf:
                    break

                if f_c < min(f_r, fvals[dim]):
                    simplex[dim] = x_c
                    fvals[dim] = f_c
                else:
                    # Shrinkage
                    shrink_ok = True
                    for i in range(1, dim + 1):
                        simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                        simplex[i] = np.clip(simplex[i], -5, 5)
                        f_i = safe_call(simplex[i])
                        if f_i == np.inf:
                            shrink_ok = False
                            break
                        fvals[i] = f_i
                    if not shrink_ok:
                        break

            # Re-sort simplex
            sorted_indices = np.argsort(fvals)
            simplex = [simplex[i] for i in sorted_indices]
            fvals = [fvals[i] for i in sorted_indices]

            # Check if we're stuck (all points very close)
            if np.max(fvals) - np.min(fvals) < 1e-12:
                break

        return best_x, best_f
