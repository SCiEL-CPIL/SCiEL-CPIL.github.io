"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES-inspired (1+1) with success-rate step adaptation and Nelder-Mead polish."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # safety margin before 330s kill
        rng = np.random.default_rng(0)

        # Phase 1: CMA-ES-inspired (1+1) with success-rate adaptation
        # Use 70% of budget for global search, 30% for local polish
        global_budget = int(self.budget * 0.70)
        polish_budget = self.budget - global_budget

        x = rng.uniform(-5, 5, self.dim)
        fx = func(x)
        best_x = x.copy()
        best_fx = fx
        sigma = 2.0
        success_count = 0
        window = 10
        successes_in_window = 0

        for i in range(global_budget):
            if time.time() > deadline:
                break

            y = np.clip(x + sigma * rng.standard_normal(self.dim), -5, 5)
            fy = func(y)

            if fy < fx:
                x = y
                fx = fy
                successes_in_window += 1
                sigma *= 1.2
                if fx < best_fx:
                    best_fx = fx
                    best_x = x.copy()
            else:
                sigma *= 0.87

            # Success-rate based adaptation every 10 steps
            if (i + 1) % window == 0:
                success_rate = successes_in_window / window
                # Target success rate ~0.33 for CMA-ES-like behavior
                if success_rate > 0.4:
                    sigma *= 1.1
                elif success_rate < 0.2:
                    sigma *= 0.9
                successes_in_window = 0

            # Keep sigma in reasonable bounds
            sigma = np.clip(sigma, 0.01, 10.0)

        # Phase 2: Nelder-Mead polish with tight tolerances
        if polish_budget > 0 and time.time() < deadline:
            x_nm = best_x.copy()
            fx_nm = best_fx

            # Initialize simplex
            simplex = [x_nm.copy()]
            for j in range(self.dim):
                xi = x_nm.copy()
                if abs(xi[j]) > 1e-10:
                    xi[j] += 0.05 * abs(xi[j])
                else:
                    xi[j] += 0.05
                simplex.append(xi)

            # Evaluate initial simplex
            simplex_fx = [func(pt) for pt in simplex]

            # Nelder-Mead parameters
            alpha = 1.0
            gamma = 2.0
            rho = 0.5
            sigma_nm = 0.5

            # Tight tolerances for precision
            tol_x = 1e-10
            tol_f = 1e-10
            max_iter = polish_budget

            for _ in range(max_iter):
                if time.time() > deadline:
                    break

                # Sort simplex by function value
                order = np.argsort(simplex_fx)
                simplex = [simplex[i] for i in order]
                simplex_fx = [simplex_fx[i] for i in order]

                # Check convergence
                if np.max(np.abs(np.array(simplex) - simplex[0])) < tol_x:
                    break
                if max(simplex_fx) - min(simplex_fx) < tol_f:
                    break

                # Centroid of all but worst point
                centroid = np.mean(simplex[:-1], axis=0)

                # Reflection
                xr = centroid + alpha * (centroid - simplex[-1])
                xr = np.clip(xr, -5, 5)
                fxr = func(xr)

                if fxr < simplex_fx[0]:
                    # Expansion
                    xe = centroid + gamma * (centroid - simplex[-1])
                    xe = np.clip(xe, -5, 5)
                    fxe = func(xe)
                    if fxe < fxr:
                        simplex[-1] = xe
                        simplex_fx[-1] = fxe
                    else:
                        simplex[-1] = xr
                        simplex_fx[-1] = fxr
                elif fxr < simplex_fx[-2]:
                    simplex[-1] = xr
                    simplex_fx[-1] = fxr
                else:
                    # Contraction
                    if fxr < simplex_fx[-1]:
                        # Outside contraction
                        xc = centroid + 0.5 * (xr - centroid)
                    else:
                        # Inside contraction
                        xc = centroid - 0.5 * (centroid - simplex[-1])
                    xc = np.clip(xc, -5, 5)
                    fxc = func(xc)
                    if fxc < min(fxr, simplex_fx[-1]):
                        simplex[-1] = xc
                        simplex_fx[-1] = fxc
                    else:
                        # Shrink
                        for j in range(1, len(simplex)):
                            simplex[j] = simplex[0] + sigma_nm * (simplex[j] - simplex[0])
                            simplex[j] = np.clip(simplex[j], -5, 5)
                            simplex_fx[j] = func(simplex[j])

                # Track best
                min_idx = np.argmin(simplex_fx)
                if simplex_fx[min_idx] < best_fx:
                    best_fx = simplex_fx[min_idx]
                    best_x = simplex[min_idx].copy()

        return best_x, best_fx
# EVOLVE-BLOCK-END
