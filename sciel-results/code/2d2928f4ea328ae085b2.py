"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Adaptive Nelder-Mead with periodic global restarts for BBOB f14.
    
    Strategy:
    - Phase 1: Quick random sampling to find a decent starting point
    - Phase 2: Nelder-Mead simplex method with adaptive step size
    - When NM stagnates, do a local random search around best, then restart NM
    - Use per-dimension adaptive step sizes based on the function's anisotropy
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 270.0
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(123)

        # Phase 1: Random initialization
        n_init = min(20, budget)
        best_x = None
        best_f = np.inf

        # Try origin and some structured points
        init_points = [np.zeros(dim)]
        for i in range(dim):
            for val in [-3.0, 3.0]:
                p = np.zeros(dim)
                p[i] = val
                init_points.append(p)
        while len(init_points) < n_init:
            init_points.append(rng.uniform(-4.0, 4.0, dim))

        for p in init_points[:n_init]:
            p = np.clip(p, -5, 5)
            fp = func(p)
            if fp < best_f:
                best_f = fp
                best_x = p.copy()
            if time.time() > deadline:
                break

        calls_made = min(n_init, budget)
        if calls_made >= budget or time.time() > deadline:
            return best_x, best_f

        # Phase 2: Adaptive Nelder-Mead with restarts
        x = best_x.copy()
        fx = best_f
        sigma = 0.5  # initial step size for simplex vertices

        # Per-dimension step sizes (adaptive)
        step_sizes = np.ones(dim) * sigma

        while calls_made < budget and time.time() < deadline:
            # Build simplex
            simplex = [x.copy()]
            for i in range(dim):
                v = x.copy()
                v[i] += step_sizes[i]
                v = np.clip(v, -5, 5)
                simplex.append(v)

            # Evaluate simplex vertices
            simplex_f = [fx]
            for i in range(1, dim + 1):
                if calls_made >= budget or time.time() > deadline:
                    break
                fi = func(simplex[i])
                simplex_f.append(fi)
                calls_made += 1

            if len(simplex_f) < dim + 1:
                break

            # Nelder-Mead iterations
            nm_iter = 0
            max_nm_iter = 30
            stagnation = 0

            while nm_iter < max_nm_iter and calls_made < budget and time.time() < deadline:
                nm_iter += 1

                # Sort simplex by function value
                order = np.argsort(simplex_f)
                simplex = [simplex[i] for i in order]
                simplex_f = [simplex_f[i] for i in order]

                # Check convergence
                f_range = simplex_f[-1] - simplex_f[0]
                if f_range < 1e-10 * (1 + abs(simplex_f[0])):
                    break

                # Reflection
                centroid = np.mean(simplex[:-1], axis=0)
                x_reflect = centroid + 1.0 * (centroid - simplex[-1])
                x_reflect = np.clip(x_reflect, -5, 5)
                f_reflect = func(x_reflect)
                calls_made += 1

                if f_reflect < simplex_f[0]:
                    # Expansion
                    x_expand = centroid + 2.0 * (centroid - simplex[-1])
                    x_expand = np.clip(x_expand, -5, 5)
                    f_expand = func(x_expand)
                    calls_made += 1
                    if f_expand < f_reflect:
                        simplex[-1] = x_expand
                        simplex_f[-1] = f_expand
                    else:
                        simplex[-1] = x_reflect
                        simplex_f[-1] = f_reflect
                    stagnation = 0
                elif f_reflect < simplex_f[-2]:
                    # Accept reflection
                    simplex[-1] = x_reflect
                    simplex_f[-1] = f_reflect
                    stagnation = 0
                else:
                    # Contraction
                    x_contract = centroid + 0.5 * (simplex[-1] - centroid)
                    x_contract = np.clip(x_contract, -5, 5)
                    f_contract = func(x_contract)
                    calls_made += 1
                    if f_contract < simplex_f[-1]:
                        simplex[-1] = x_contract
                        simplex_f[-1] = f_contract
                        stagnation = 0
                    else:
                        # Shrink
                        for i in range(1, dim + 1):
                            simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], -5, 5)
                        # Re-evaluate shrunk vertices
                        for i in range(1, dim + 1):
                            if calls_made >= budget or time.time() > deadline:
                                break
                            simplex_f[i] = func(simplex[i])
                            calls_made += 1
                        stagnation += 1

                # Update best
                min_idx = np.argmin(simplex_f)
                if simplex_f[min_idx] < fx:
                    fx = simplex_f[min_idx]
                    x = simplex[min_idx].copy()
                    best_x = x.copy()
                    best_f = fx

                if stagnation >= 5:
                    break

            # After NM phase, adapt step sizes based on progress
            if calls_made >= budget or time.time() > deadline:
                break

            # Check if we made progress in this NM phase
            if nm_iter > 0:
                # Adjust step sizes based on simplex spread
                for i in range(dim):
                    spread = np.std([v[i] for v in simplex])
                    if spread > 0:
                        step_sizes[i] = np.clip(spread * 0.5, 1e-4, 5.0)
                    else:
                        step_sizes[i] *= 0.5
            else:
                # No progress, shrink step sizes
                step_sizes *= 0.5
                step_sizes = np.clip(step_sizes, 1e-5, 5.0)

            # If very small step sizes, do a local random search to escape
            if np.all(step_sizes < 0.01):
                # Local random search
                for _ in range(min(10, budget - calls_made)):
                    if calls_made >= budget or time.time() > deadline:
                        break
                    trial = x + 0.3 * rng.standard_normal(dim)
                    trial = np.clip(trial, -5, 5)
                    ft = func(trial)
                    calls_made += 1
                    if ft < fx:
                        fx = ft
                        x = trial.copy()
                        best_x = x.copy()
                        best_f = fx
                        step_sizes = np.clip(step_sizes * 2, 1e-4, 5.0)

                # Reset step sizes for new NM phase
                step_sizes = np.clip(step_sizes, 0.05, 5.0)

        return best_x, best_f
# EVOLVE-BLOCK-END
