"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-start Nelder-Mead with coordinate refinement for Weierstrass f16."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        n = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 285  # 15s safety margin before 300s limit

        best_x = None
        best_f = np.inf
        calls_made = 0

        def evaluate(x):
            nonlocal calls_made, best_x, best_f
            x = np.clip(x, -5.0, 5.0)
            f = func(x)
            calls_made += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        # Phase 1: Coarse random sampling to find a good basin
        # Use ~30% of budget for exploration
        explore_budget = int(budget * 0.3)
        rng = np.random.default_rng(42)

        # Start with a few well-spaced points
        base_points = [
            np.zeros(n),
            np.full(n, 1.0),
            np.full(n, -1.0),
            np.full(n, 2.0),
            np.full(n, -2.0),
            np.full(n, 3.0),
            np.full(n, -3.0),
        ]
        for p in base_points:
            if calls_made >= budget:
                break
            if time.time() > deadline:
                break
            evaluate(p)

        # Random exploration
        while calls_made < explore_budget and time.time() < deadline:
            x = rng.uniform(-4.5, 4.5, n)
            evaluate(x)

        # Phase 2: Nelder-Mead refinement from best point
        def nelder_mead(x0, max_iter=200):
            nonlocal calls_made, best_x, best_f
            if calls_made >= budget:
                return

            # Initialize simplex
            simplex = [x0.copy()]
            fvals = [evaluate(x0)]
            best_f_simplex = fvals[0]
            best_x_simplex = x0.copy()

            # Add perturbation points
            base_step = 0.5
            for i in range(n):
                xi = x0.copy()
                xi[i] += base_step
                if calls_made >= budget:
                    return
                fi = evaluate(xi)
                simplex.append(xi)
                fvals.append(fi)
                if fi < best_f_simplex:
                    best_f_simplex = fi
                    best_x_simplex = xi.copy()

            alpha = 1.0   # reflection
            gamma = 2.0   # expansion
            rho = 0.5     # contraction
            sigma = 0.5   # shrink

            for _ in range(max_iter):
                if calls_made >= budget:
                    break
                if time.time() > deadline:
                    break

                # Sort simplex by function value
                order = np.argsort(fvals)
                simplex = [simplex[i] for i in order]
                fvals = [fvals[i] for i in order]

                # Check convergence
                if abs(fvals[0] - fvals[-1]) < 1e-12:
                    break

                # Centroid of all but worst
                centroid = np.mean(simplex[:-1], axis=0)

                # Reflection
                xr = centroid + alpha * (centroid - simplex[-1])
                fr = evaluate(xr)
                if fvals[0] <= fr < fvals[-2]:
                    simplex[-1] = xr
                    fvals[-1] = fr
                    continue

                # Expansion
                if fr < fvals[0]:
                    xe = centroid + gamma * (centroid - simplex[-1])
                    fe = evaluate(xe)
                    if fe < fr:
                        simplex[-1] = xe
                        fvals[-1] = fe
                    else:
                        simplex[-1] = xr
                        fvals[-1] = fr
                    continue

                # Contraction
                if fr >= fvals[-2]:
                    if fr < fvals[-1]:
                        xc = centroid + rho * (simplex[-1] - centroid)
                    else:
                        xc = centroid - rho * (centroid - simplex[-1])
                    fc = evaluate(xc)
                    if fc < min(fr, fvals[-1]):
                        simplex[-1] = xc
                        fvals[-1] = fc
                        continue

                # Shrink
                for i in range(1, len(simplex)):
                    simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                    fvals[i] = evaluate(simplex[i])

            # Update global best
            if fvals[0] < best_f:
                best_f = fvals[0]
                best_x = simplex[0].copy()

        # Run Nelder-Mead from the best point found in exploration
        if best_x is not None:
            nelder_mead(best_x, max_iter=300)

        # Phase 3: Coordinate-wise local refinement
        def coordinate_refinement(x, max_iter=100):
            nonlocal calls_made, best_x, best_f
            if calls_made >= budget:
                return
            x = x.copy()
            fx = evaluate(x)
            step = 0.1
            for _ in range(max_iter):
                if calls_made >= budget:
                    break
                if time.time() > deadline:
                    break
                improved = False
                for i in range(n):
                    if calls_made >= budget:
                        break
                    # Try positive and negative steps
                    for direction in [1, -1]:
                        y = x.copy()
                        y[i] += direction * step
                        y = np.clip(y, -5, 5)
                        fy = evaluate(y)
                        if fy < fx:
                            x = y
                            fx = fy
                            improved = True
                            break
                if not improved:
                    step *= 0.5
                    if step < 1e-6:
                        break

        if best_x is not None:
            coordinate_refinement(best_x, max_iter=200)

        # Phase 4: If we have budget left, do another round of NM from slightly perturbed best
        if best_x is not None and calls_made < budget and time.time() < deadline:
            for restart in range(3):
                if calls_made >= budget:
                    break
                if time.time() > deadline:
                    break
                x_restart = best_x + rng.normal(0, 0.1, n)
                x_restart = np.clip(x_restart, -5, 5)
                nelder_mead(x_restart, max_iter=100)

        # Ensure we've made at least 1 call
        if calls_made == 0:
            evaluate(np.zeros(n))

        return best_x, best_f
# EVOLVE-BLOCK-END
