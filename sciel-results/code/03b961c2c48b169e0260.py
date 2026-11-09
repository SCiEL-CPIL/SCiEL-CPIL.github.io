"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Hybrid Sobol global exploration + Nelder-Mead local refinement."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 290  # Safety margin before 300s limit

        # Generate Sobol sequence for space-filling exploration
        # Use scipy's Sobol if available, else use a simple deterministic grid + jitter
        try:
            from scipy.stats import qmc
            sobol = qmc.Sobol(d=dim, scramble=False)
            n_sobol = min(256, budget // 2)
            # Use power-of-2 for Sobol
            n_sobol = 2 ** int(np.log2(max(2, n_sobol)))
            points = sobol.random(n_sobol)
            # Map [0,1]^d to [-5,5]^d
            points = points * 10.0 - 5.0
            points = np.clip(points, -5, 5)
        except Exception:
            # Fallback: deterministic grid with jitter
            n_per_dim = 4
            n_sobol = n_per_dim ** dim
            if n_sobol > budget // 2:
                n_per_dim = 3
                n_sobol = n_per_dim ** dim
            grid = np.array(np.meshgrid(*[np.linspace(-5, 5, n_per_dim)] * dim)).T.reshape(-1, dim)
            points = grid[:min(n_sobol, budget // 2)]

        # Evaluate Sobol points and track best
        best_x = points[0].copy()
        best_f = func(best_x)
        calls = 1

        # Store evaluated points for potential reuse
        eval_points = [points[0].copy()]
        eval_vals = [best_f]

        # Evaluate remaining Sobol points
        for i in range(1, len(points)):
            if calls >= budget:
                break
            if time.time() > deadline:
                break
            x = points[i]
            fx = func(x)
            calls += 1
            eval_points.append(x.copy())
            eval_vals.append(fx)
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        # Now do Nelder-Mead refinement from the best point
        # and from a few other promising points
        # We'll do multiple short Nelder-Mead runs from different starting points
        remaining = budget - calls

        # Pick top-k starting points for Nelder-Mead
        n_start = min(5, len(eval_points))
        if n_start > 1:
            # Sort by function value
            order = np.argsort(eval_vals)
            start_indices = order[:n_start]
        else:
            start_indices = [0]

        # Allocate remaining budget among Nelder-Mead runs
        # Each Nelder-Mead step is roughly 1 call (simplex update)
        # A typical Nelder-Mead in 5D needs ~5*dim iterations to converge
        # Let's do short runs: ~remaining/n_start iterations each

        nm_per_run = max(5, remaining // max(1, n_start))

        for idx in start_indices:
            if calls >= budget:
                break
            if time.time() > deadline:
                break

            # Nelder-Mead from this starting point
            x_nm = eval_points[idx].copy()
            f_nm = eval_vals[idx]

            # Build initial simplex
            simplex = [x_nm.copy()]
            simplex_vals = [f_nm]

            for j in range(1, dim + 1):
                if calls >= budget:
                    break
                if time.time() > deadline:
                    break
                step = 0.5 if j == 1 else 0.25
                x_j = x_nm.copy()
                x_j[j - 1] += step
                x_j = np.clip(x_j, -5, 5)
                f_j = func(x_j)
                calls += 1
                simplex.append(x_j)
                simplex_vals.append(f_j)

            # Nelder-Mead iterations
            for _ in range(nm_per_run):
                if calls >= budget:
                    break
                if time.time() > deadline:
                    break

                # Sort simplex by function value
                order = np.argsort(simplex_vals)
                simplex = [simplex[i] for i in order]
                simplex_vals = [simplex_vals[i] for i in order]

                # Centroid of all but worst
                centroid = np.mean(simplex[:-1], axis=0)
                worst = simplex[-1]
                worst_val = simplex_vals[-1]

                # Reflection
                alpha = 1.0
                x_ref = centroid + alpha * (centroid - worst)
                x_ref = np.clip(x_ref, -5, 5)
                f_ref = func(x_ref)
                calls += 1

                if f_ref < simplex_vals[0]:
                    # Expansion
                    if calls < budget and time.time() < deadline:
                        gamma = 2.0
                        x_exp = centroid + gamma * (centroid - worst)
                        x_exp = np.clip(x_exp, -5, 5)
                        f_exp = func(x_exp)
                        calls += 1
                        if f_exp < f_ref:
                            simplex[-1] = x_exp
                            simplex_vals[-1] = f_exp
                        else:
                            simplex[-1] = x_ref
                            simplex_vals[-1] = f_ref
                    else:
                        simplex[-1] = x_ref
                        simplex_vals[-1] = f_ref
                elif f_ref < simplex_vals[-2]:
                    # Accept reflection
                    simplex[-1] = x_ref
                    simplex_vals[-1] = f_ref
                else:
                    # Contraction
                    if calls < budget and time.time() < deadline:
                        rho = 0.5
                        x_con = centroid + rho * (worst - centroid)
                        x_con = np.clip(x_con, -5, 5)
                        f_con = func(x_con)
                        calls += 1
                        if f_con <= worst_val:
                            simplex[-1] = x_con
                            simplex_vals[-1] = f_con
                        else:
                            # Shrink
                            for i in range(1, len(simplex)):
                                simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                                if calls < budget and time.time() < deadline:
                                    simplex_vals[i] = func(simplex[i])
                                    calls += 1
                                else:
                                    break
                    else:
                        # Can't do contraction, just stop
                        break

                # Update best
                for i in range(len(simplex_vals)):
                    if simplex_vals[i] < best_f:
                        best_f = simplex_vals[i]
                        best_x = simplex[i].copy()

        return best_x, best_f
# EVOLVE-BLOCK-END
