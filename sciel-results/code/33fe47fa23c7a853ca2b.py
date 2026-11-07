"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

class Optimizer:
    """Nelder-Mead simplex method for anisotropic functions."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        # Initialize simplex
        # Point 0: origin
        # Points 1..dim: origin + step * basis vectors
        step = 1.0
        simplex = np.zeros((dim + 1, dim))
        simplex[0] = 0.0
        for i in range(dim):
            simplex[i + 1, i] = step

        # Evaluate initial simplex
        fvals = np.array([func(x) for x in simplex])
        calls = dim + 1

        # Nelder-Mead parameters
        alpha = 1.0    # Reflection
        gamma = 2.0    # Expansion
        rho = 0.5      # Contraction
        sigma = 0.5    # Shrinkage

        # Main loop
        while calls < budget:
            # Sort simplex by fvals
            idx = np.argsort(fvals)
            simplex = simplex[idx]
            fvals = fvals[idx]

            # Centroid of all points except the worst
            centroid = np.mean(simplex[:-1], axis=0)

            # Worst point
            x_worst = simplex[-1]
            f_worst = fvals[-1]

            # Reflection
            x_ref = centroid + alpha * (centroid - x_worst)
            x_ref = np.clip(x_ref, -5, 5)
            f_ref = func(x_ref)
            calls += 1

            if f_ref < fvals[0]:
                # Expansion
                x_exp = centroid + gamma * (x_ref - centroid)
                x_exp = np.clip(x_exp, -5, 5)
                f_exp = func(x_exp)
                calls += 1

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
                if f_ref < f_worst:
                    # Outside contraction
                    x_con = centroid + rho * (x_ref - centroid)
                else:
                    # Inside contraction
                    x_con = centroid - rho * (x_worst - centroid)
                
                x_con = np.clip(x_con, -5, 5)
                f_con = func(x_con)
                calls += 1

                if f_con <= min(f_ref, f_worst):
                    simplex[-1] = x_con
                    fvals[-1] = f_con
                else:
                    # Shrinkage
                    for i in range(1, dim + 1):
                        simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                        simplex[i] = np.clip(simplex[i], -5, 5)
                        fvals[i] = func(simplex[i])
                        calls += 1
                        if calls >= budget:
                            break
                    if calls >= budget:
                        break

        return simplex[0], fvals[0]
# EVOLVE-BLOCK-END
