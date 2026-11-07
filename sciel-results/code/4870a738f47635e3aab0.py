"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """
    Derivative-free optimizer combining random restarts with Nelder-Mead.
    - Uses random samples to find promising basins.
    - Applies Nelder-Mead simplex method for rapid local convergence.
    - Multiple restarts from top candidates to handle multi-modality.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        # Phase 1: Random sampling to find good starting points
        n_random = min(30, budget // 4)
        remaining_budget = budget - n_random

        candidates = []
        for _ in range(n_random):
            x = rng.uniform(-5, 5, dim)
            fx = func(x)
            candidates.append((fx, x.copy()))

        # Sort candidates by function value
        candidates.sort(key=lambda t: t[0])

        best_x = candidates[0][1].copy()
        best_f = candidates[0][0]

        # Phase 2: Nelder-Mead from top candidates
        n_restarts = min(5, len(candidates))
        budget_per_restart = remaining_budget // n_restarts

        for i in range(n_restarts):
            if remaining_budget <= 0:
                break

            fx_init, x_init = candidates[i]
            alloc = min(budget_per_restart, remaining_budget)
            if i == n_restarts - 1:
                alloc = remaining_budget  # use all remaining

            x_res, f_res = self._nelder_mead(func, x_init, fx_init, dim, alloc, rng)

            if f_res < best_f:
                best_f = f_res
                best_x = x_res.copy()

            remaining_budget -= alloc

        return best_x, best_f

    def _nelder_mead(self, func, x0, f0, dim, budget, rng):
        """
        Nelder-Mead simplex method with adaptive parameters and convergence detection.
        """
        # Initialize simplex with proper scaling
        simplex = [x0.copy()]
        fvals = [f0]

        # Create simplex vertices with scaled perturbations
        scale = 0.25  # initial perturbation size
        for i in range(dim):
            xi = x0.copy()
            xi[i] += scale
            xi = np.clip(xi, -5, 5)
            fi = func(xi)
            simplex.append(xi)
            fvals.append(fi)

        calls_used = 1 + dim
        calls_left = budget - calls_used

        # Nelder-Mead parameters
        gamma = 1.0  # reflection
        rho = 2.0    # expansion
        sigma = 0.5  # contraction
        delta = 0.5  # shrink

        best_x = min(simplex, key=lambda x: fvals[simplex.index(x)])
        best_f = min(fvals)

        # Track for convergence
        prev_best_f = best_f
        stall_count = 0
        max_stall = 20

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

            # Track stall
            if best_f < prev_best_f - 1e-10:
                prev_best_f = best_f
                stall_count = 0
            else:
                stall_count += 1

            if stall_count >= max_stall:
                break

            # Check simplex size convergence
            f_spread = max(fvals) - min(fvals)
            x_spread = np.max(np.array(simplex) - np.min(np.array(simplex), axis=0))
            if f_spread < 1e-12 and x_spread < 1e-8:
                break

        return best_x, best_f
# EVOLVE-BLOCK-END
