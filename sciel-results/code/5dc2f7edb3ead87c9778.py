import numpy as np


class Optimizer:
    """Nelder-Mead style simplex optimizer for BBOB f17 (Schaffers F7)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget

        # Initial simplex: center at 0, spread to cover a good portion of the box
        center = np.zeros(dim)
        simplex = [center.copy()]
        for i in range(dim):
            pt = center.copy()
            pt[i] = 2.0
            simplex.append(pt)

        # Evaluate initial simplex
        fvals = [func(pt) for pt in simplex]
        calls = dim + 1
        best_x = simplex[np.argmin(fvals)].copy()
        best_f = min(fvals)

        # Standard Nelder-Mead parameters
        alpha = 1.0   # reflection
        gamma = 2.0   # expansion
        rho = 0.5     # contraction
        sigma_shrink = 0.5

        # We'll track the simplex as a list of (point, fval)
        # Sort initially
        order = np.argsort(fvals)
        simplex = [simplex[i] for i in order]
        fvals = [fvals[i] for i in order]

        while calls < budget:
            # Ensure sorted
            order = np.argsort(fvals)
            if len(order) > 1:
                simplex = [simplex[i] for i in order]
                fvals = [fvals[i] for i in order]

            # Centroid of all points except the worst
            centroid = np.mean(simplex[:-1], axis=0)
            worst_pt = simplex[-1]
            worst_f = fvals[-1]

            # Reflection
            x_ref = centroid + alpha * (centroid - worst_pt)
            x_ref = np.clip(x_ref, -5, 5)
            f_ref = func(x_ref)
            calls += 1

            if f_ref < fvals[0]:
                # Try expansion
                x_exp = centroid + gamma * (x_ref - centroid)
                x_exp = np.clip(x_exp, -5, 5)
                if calls < budget:
                    f_exp = func(x_exp)
                    calls += 1
                    if f_exp < f_ref:
                        simplex[-1] = x_exp
                        fvals[-1] = f_exp
                    else:
                        simplex[-1] = x_ref
                        fvals[-1] = f_ref
                else:
                    simplex[-1] = x_ref
                    fvals[-1] = f_ref
            elif f_ref < worst_f:
                # Accept reflection
                simplex[-1] = x_ref
                fvals[-1] = f_ref
            else:
                # Contraction
                # Outside contraction if worst, inside otherwise
                if f_ref > worst_f:
                    x_con = centroid + rho * (worst_pt - centroid)  # outside contraction
                else:
                    x_con = centroid - rho * (centroid - worst_pt)  # inside contraction
                x_con = np.clip(x_con, -5, 5)
                f_con = func(x_con)
                calls += 1

                if f_con <= min(f_ref, worst_f):
                    simplex[-1] = x_con
                    fvals[-1] = f_con
                else:
                    # Shrink
                    new_simplex = [simplex[0].copy()]
                    new_fvals = [fvals[0]]
                    for i in range(1, len(simplex)):
                        pt = simplex[0] + sigma_shrink * (simplex[i] - simplex[0])
                        pt = np.clip(pt, -5, 5)
                        new_simplex.append(pt)
                        if calls < budget:
                            f_pt = func(pt)
                            calls += 1
                            new_fvals.append(f_pt)
                        else:
                            # If we're out of budget, stop
                            break
                    if calls >= budget:
                        break
                    simplex = new_simplex
                    fvals = new_fvals

            # Track best
            current_best = min(fvals)
            if current_best < best_f:
                best_f = current_best
                best_idx = np.argmin(fvals)
                best_x = simplex[best_idx].copy()

        return best_x, best_f
