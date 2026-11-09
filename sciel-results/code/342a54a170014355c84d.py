"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Hybrid optimizer for the Bent Cigar function (f12).
    Strategy:
    1. Multi-start with a few random initial points.
    2. For each start, run Nelder-Mead with adaptive simplex.
    3. Use a time guard to ensure we finish within budget.
    4. Keep track of the best point found.
    
    The Bent Cigar has one wide axis and many narrow ones. Nelder-Mead
    can adapt to this anisotropy better than isotropic methods.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.calls_made = 0

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin before 300s limit
        dim = self.dim
        budget = self.budget

        # State tracking
        best_x = None
        best_f = np.inf
        self.calls_made = 0

        def call_func(x):
            nonlocal self
            x = np.asarray(x, dtype=np.float64)
            self.calls_made += 1
            return float(func(x))

        # Multi-start: use several random starting points
        # We'll allocate budget across multiple starts
        n_starts = 5
        budget_per_start = max(1, budget // n_starts)

        rng = np.random.default_rng(42)

        for s in range(n_starts):
            if self.calls_made >= budget:
                break
            if time.time() > deadline:
                break

            # Random starting point
            x0 = rng.uniform(-5, 5, dim)
            f0 = call_func(x0)

            if f0 < best_f:
                best_f = f0
                best_x = x0.copy()

            # Nelder-Mead simplex
            # Initialize simplex
            simplex = [x0.copy()]
            f_simplex = [f0]

            # Add perturbed points
            for i in range(dim):
                xi = x0.copy()
                # Perturb in dimension i
                step = 0.5 if abs(xi[i]) < 4.5 else 0.1
                xi[i] += step
                xi[i] = np.clip(xi[i], -5, 5)
                fi = call_func(xi)
                simplex.append(xi)
                f_simplex.append(fi)
                if fi < best_f:
                    best_f = fi
                    best_x = xi.copy()

            # Nelder-Mead parameters
            alpha = 1.0  # reflection
            gamma = 2.0  # expansion
            rho = 0.5    # contraction
            sigma_nm = 0.5  # shrink

            # Local budget for this start
            local_budget = min(budget_per_start, budget - self.calls_made)
            nm_iters = 0

            while nm_iters < local_budget and self.calls_made < budget:
                if time.time() > deadline:
                    break

                # Sort simplex by function value
                order = np.argsort(f_simplex)
                simplex = [simplex[i] for i in order]
                f_simplex = [f_simplex[i] for i in order]

                # Centroid of all but worst
                centroid = np.mean(simplex[:-1], axis=0)

                # Reflection
                x_r = centroid + alpha * (centroid - simplex[-1])
                x_r = np.clip(x_r, -5, 5)
                f_r = call_func(x_r)

                if f_r < f_simplex[0]:
                    # Expansion
                    x_e = centroid + gamma * (centroid - simplex[-1])
                    x_e = np.clip(x_e, -5, 5)
                    f_e = call_func(x_e)
                    if f_e < f_r:
                        simplex[-1] = x_e
                        f_simplex[-1] = f_e
                        if f_e < best_f:
                            best_f = f_e
                            best_x = x_e.copy()
                    else:
                        simplex[-1] = x_r
                        f_simplex[-1] = f_r
                        if f_r < best_f:
                            best_f = f_r
                            best_x = x_r.copy()
                elif f_r < f_simplex[-2]:
                    # Simple reflection
                    simplex[-1] = x_r
                    f_simplex[-1] = f_r
                    if f_r < best_f:
                        best_f = f_r
                        best_x = x_r.copy()
                else:
                    # Contraction
                    if f_r < f_simplex[-1]:
                        # Outside contraction
                        x_c = centroid + rho * (x_r - centroid)
                    else:
                        # Inside contraction
                        x_c = centroid - rho * (centroid - simplex[-1])
                    x_c = np.clip(x_c, -5, 5)
                    f_c = call_func(x_c)

                    if f_c <= min(f_r, f_simplex[-1]):
                        simplex[-1] = x_c
                        f_simplex[-1] = f_c
                        if f_c < best_f:
                            best_f = f_c
                            best_x = x_c.copy()
                    else:
                        # Shrink
                        for i in range(1, len(simplex)):
                            simplex[i] = simplex[0] + sigma_nm * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], -5, 5)
                            f_simplex[i] = call_func(simplex[i])
                            if f_simplex[i] < best_f:
                                best_f = f_simplex[i]
                                best_x = simplex[i].copy()

                nm_iters += 1

        return best_x, best_f
# EVOLVE-BLOCK-END
