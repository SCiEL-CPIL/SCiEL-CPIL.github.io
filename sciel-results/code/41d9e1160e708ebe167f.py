"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """
    Derivative-free optimizer combining random restarts with Nelder-Mead.
    - Uses a small number of random samples to find promising basins.
    - Applies Nelder-Mead simplex method for rapid local convergence.
    - Adapts to the Griewank-Rosenbrock landscape by leveraging local unimodality.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        # Phase 1: Random sampling to find good starting points
        n_random = min(30, budget // 5)
        best_x = None
        best_f = np.inf

        for _ in range(n_random):
            x = rng.uniform(-5, 5, dim)
            fx = func(x)
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        # Phase 2: Nelder-Mead optimization from the best starting point
        remaining_budget = budget - n_random
        if remaining_budget > 0:
            best_x, best_f = self._nelder_mead(
                func, best_x, best_f, dim, remaining_budget, rng
            )

        return best_x, best_f

    def _nelder_mead(self, func, x0, f0, dim, budget, rng):
        """
        Nelder-Mead simplex method with adaptive parameters.
        """
        # Initialize simplex with larger initial step size for better exploration
        alpha = 0.5  # initial step size
        simplex = [x0.copy()]
        fvals = [f0]

        # Add perturbed points with larger steps
        for i in range(dim):
            xi = x0.copy()
            if xi[i] > 0:
                xi[i] += alpha
            elif xi[i] < 0:
                xi[i] -= alpha
            else:
                xi[i] += alpha
            xi = np.clip(xi, -5, 5)
            fi = func(xi)
            simplex.append(xi)
            fvals.append(fi)

        # Track best so far
        best_x = x0.copy()
        best_f = f0
        for i, (xi, fi) in enumerate(zip(simplex, fvals)):
            if fi < best_f:
                best_f = fi
                best_x = xi.copy()

        calls_used = 1 + dim
        calls_left = budget - calls_used

        # Nelder-Mead parameters
        gamma = 1.0  # reflection
        rho = 2.0    # expansion
        sigma = 0.5  # contraction
        delta = 0.5  # shrink

        # Adaptive step size
        step_scale = 1.0

        while calls_left > 0:
            # Sort simplex by function value
            indices = np.argsort(fvals)
            simplex = [simplex[i] for i in indices]
            fvals = [fvals[i] for i in indices]

            # Centroid of all points except the worst
            centroid = np.mean(simplex[:-1], axis=0)

            # Reflection
            x_ref = centroid + gamma * (centroid - simplex[-1])
            x_ref = np.clip(x_ref, -5, 5)
            f_ref = func(x_ref)
            calls_left -= 1

            if f_ref < fvals[0]:
                # Expansion
                x_exp = centroid + rho * (x_ref - centroid)
                x_exp = np.clip(x_exp, -5, 5)
                f_exp = func(x_exp)
                calls_left -= 1

                if f_exp < f_ref:
                    simplex[-1] = x_exp
                    fvals[-1] = f_exp
                else:
                    simplex[-1] = x_ref
                    fvals[-1] = f_ref
            elif f_ref < fvals[-2]:
                # Reflection is good
                simplex[-1] = x_ref
                fvals[-1] = f_ref
            else:
                # Contraction
                if f_ref < fvals[-1]:
                    # Outside contraction
                    x_con = centroid + sigma * (x_ref - centroid)
                else:
                    # Inside contraction
                    x_con = centroid - sigma * (simplex[-1] - centroid)

                x_con = np.clip(x_con, -5, 5)
                f_con = func(x_con)
                calls_left -= 1

                if f_con < min(f_ref, fvals[-1]):
                    simplex[-1] = x_con
                    fvals[-1] = f_con
                else:
                    # Shrink
                    for i in range(1, len(simplex)):
                        simplex[i] = simplex[0] + delta * (simplex[i] - simplex[0])
                        simplex[i] = np.clip(simplex[i], -5, 5)
                        fvals[i] = func(simplex[i])
                        calls_left -= 1
                        if calls_left <= 0:
                            break

            # Update best
            current_best_idx = np.argmin(fvals)
            if fvals[current_best_idx] < best_f:
                best_f = fvals[current_best_idx]
                best_x = simplex[current_best_idx].copy()

            # Check convergence
            if calls_left <= 0:
                break

            f_spread = max(fvals) - min(fvals)
            if f_spread < 1e-12:
                break

        return best_x, best_f
# EVOLVE-BLOCK-END
