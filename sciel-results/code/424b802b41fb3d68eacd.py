"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Nelder-Mead simplex method with adaptive step and time guard."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 300s limit

        rng = np.random.default_rng(42)
        d = self.dim
        budget = self.budget

        # Initial simplex: centered at origin with small perturbation
        x0 = np.zeros(d)
        simplex = [x0.copy()]
        fvals = [func(x0)]
        budget -= 1

        # Create simplex vertices
        for i in range(d):
            v = x0.copy()
            v[i] = 0.5
            simplex.append(v)
            fvals.append(func(v))
            budget -= 1

        # Sort by function value
        order = np.argsort(fvals)
        simplex = [simplex[i] for i in order]
        fvals = [fvals[i] for i in order]

        best_x = simplex[0].copy()
        best_f = fvals[0]

        alpha = 1.0   # reflection
        gamma = 2.0   # expansion
        rho = 0.5     # contraction
        sigma = 0.5   # shrink

        while budget > 0:
            # Check time
            if time.time() > deadline:
                break

            # If simplex has collapsed, restart with smaller sigma
            spread = np.max(fvals) - np.min(fvals)
            if spread < 1e-12:
                # Shrink all points toward the best
                for i in range(1, len(simplex)):
                    simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                    fvals[i] = func(simplex[i])
                    budget -= 1
                    if budget <= 0:
                        break
                sigma *= 0.5
                if sigma < 1e-8:
                    break
                continue

            # Compute centroid of all but worst
            centroid = np.mean(simplex[:-1], axis=0)

            # Reflection
            x_ref = centroid + alpha * (centroid - simplex[-1])
            x_ref = np.clip(x_ref, -5, 5)
            f_ref = func(x_ref)
            budget -= 1

            if f_ref < fvals[0]:
                # Expansion
                x_exp = centroid + gamma * (x_ref - centroid)
                x_exp = np.clip(x_exp, -5, 5)
                f_exp = func(x_exp)
                budget -= 1

                if f_exp < f_ref:
                    simplex[-1] = x_exp
                    fvals[-1] = f_exp
                else:
                    simplex[-1] = x_ref
                    fvals[-1] = f_ref
            elif f_ref < fvals[-2]:
                simplex[-1] = x_ref
                fvals[-1] = f_ref
            else:
                # Contraction
                if f_ref < fvals[-1]:
                    # Outside contraction
                    x_con = centroid + rho * (x_ref - centroid)
                else:
                    # Inside contraction
                    x_con = centroid - rho * (x_ref - centroid)

                x_con = np.clip(x_con, -5, 5)
                f_con = func(x_con)
                budget -= 1

                if f_con <= f_ref:
                    simplex[-1] = x_con
                    fvals[-1] = f_con
                else:
                    # Shrink
                    for i in range(1, len(simplex)):
                        simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                        fvals[i] = func(simplex[i])
                        budget -= 1
                        if budget <= 0:
                            break

            # Re-sort
            order = np.argsort(fvals)
            simplex = [simplex[i] for i in order]
            fvals = [fvals[i] for i in order]

            # Update best
            if fvals[0] < best_f:
                best_f = fvals[0]
                best_x = simplex[0].copy()

            # Check for convergence
            if np.max(fvals) - np.min(fvals) < 1e-10:
                break

        return best_x, best_f
# EVOLVE-BLOCK-END
