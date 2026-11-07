# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Nelder-Mead simplex optimizer with restarts."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 280.0  # Safety margin before 300s kill

        rng = np.random.default_rng(42)
        calls_made = 0
        best_x = None
        best_f = np.inf

        def nm_search(x0, f0, remaining_budget):
            nonlocal calls_made, best_x, best_f

            # Nelder-Mead parameters
            alpha = 1.0   # reflection
            gamma = 2.0   # expansion
            rho = 0.5     # contraction
            sigma = 0.5   # shrink
            tol_f = 1e-12
            tol_x = 1e-10
            max_iter = 200

            # Initialize simplex
            simplex = np.zeros((dim + 1, dim))
            simplex[0] = x0
            for i in range(1, dim + 1):
                simplex[i] = x0.copy()
                simplex[i][i - 1] += 0.05

            # Evaluate initial simplex
            fvals = np.zeros(dim + 1)
            fvals[0] = f0
            calls_made += 1
            if fvals[0] < best_f:
                best_f = fvals[0]
                best_x = x0.copy()

            for i in range(1, dim + 1):
                fvals[i] = func(simplex[i])
                calls_made += 1
                if fvals[i] < best_f:
                    best_f = fvals[i]
                    best_x = simplex[i].copy()

            # Clip simplex to box
            simplex = np.clip(simplex, -5, 5)

            for iteration in range(max_iter):
                # Check time and budget
                if time.time() > deadline or calls_made >= remaining_budget - 5:
                    break

                # Sort by function value
                order = np.argsort(fvals)
                simplex = simplex[order]
                fvals = fvals[order]

                # Check convergence
                if np.max(fvals) - np.min(fvals) < tol_f:
                    break
                if np.max(np.abs(simplex - simplex[0])) < tol_x:
                    break

                # Centroid of all but worst
                centroid = np.mean(simplex[:-1], axis=0)

                # Reflection
                x_r = centroid + alpha * (centroid - simplex[-1])
                x_r = np.clip(x_r, -5, 5)
                f_r = func(x_r)
                calls_made += 1
                if f_r < best_f:
                    best_f = f_r
                    best_x = x_r.copy()

                if f_r < fvals[0]:
                    # Expansion
                    x_e = centroid + gamma * (x_r - centroid)
                    x_e = np.clip(x_e, -5, 5)
                    f_e = func(x_e)
                    calls_made += 1
                    if f_e < best_f:
                        best_f = f_e
                        best_x = x_e.copy()

                    if f_e < f_r:
                        simplex[-1] = x_e
                        fvals[-1] = f_e
                    else:
                        simplex[-1] = x_r
                        fvals[-1] = f_r
                elif f_r < fvals[-2]:
                    # Accept reflection
                    simplex[-1] = x_r
                    fvals[-1] = f_r
                else:
                    # Contraction
                    if f_r < fvals[-1]:
                        # Outside contraction
                        x_c = centroid + rho * (x_r - centroid)
                    else:
                        # Inside contraction
                        x_c = centroid - rho * (x_r - centroid)

                    x_c = np.clip(x_c, -5, 5)
                    f_c = func(x_c)
                    calls_made += 1
                    if f_c < best_f:
                        best_f = f_c
                        best_x = x_c.copy()

                    if f_c < min(f_r, fvals[-1]):
                        simplex[-1] = x_c
                        fvals[-1] = f_c
                    else:
                        # Shrink
                        for i in range(1, dim + 1):
                            simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], -5, 5)
                            fvals[i] = func(simplex[i])
                            calls_made += 1
                            if fvals[i] < best_f:
                                best_f = fvals[i]
                                best_x = simplex[i].copy()

                # Check if we've exceeded budget
                if calls_made >= remaining_budget:
                    break

            return best_x, best_f

        # Phase 1: Initial diverse sampling
        n_init = 20
        X_init = rng.uniform(-5, 5, (n_init, dim))
        F_init = np.array([func(x) for x in X_init])
        calls_made += n_init

        idx_best = np.argmin(F_init)
        best_x = X_init[idx_best].copy()
        best_f = F_init[idx_best]

        # Phase 2: Multiple Nelder-Mead restarts from top candidates
        # Sort initial points by fitness
        init_order = np.argsort(F_init)

        # Use top 5 starting points for NM restarts
        n_restarts = min(5, n_init)
        for i in range(n_restarts):
            if time.time() > deadline or calls_made >= budget - 10:
                break

            start_idx = init_order[i]
            x0 = X_init[start_idx].copy()
            f0 = F_init[start_idx]

            # Small perturbation for diversity
            if i > 0:
                perturbation = rng.normal(0, 0.1, dim)
                x0 = np.clip(x0 + perturbation, -5, 5)
                f0 = func(x0)
                calls_made += 1
                if f0 < best_f:
                    best_f = f0
                    best_x = x0.copy()

            nm_search(x0, f0, budget)

        return best_x, best_f
# EVOLVE-BLOCK-END
