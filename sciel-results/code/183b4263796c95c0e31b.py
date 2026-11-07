"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """
    Derivative-free optimizer combining random restarts with Nelder-Mead.
    - Uses random sampling to identify promising basins.
    - Applies Nelder-Mead simplex method for local convergence.
    - Uses multi-start strategy to avoid local minima.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        # Phase 1: Random sampling to find good starting points
        # Use more samples to better cover the landscape and find the global basin
        n_random = min(50, budget // 10)
        candidates = []

        for _ in range(n_random):
            x = rng.uniform(-5, 5, dim)
            fx = func(x)
            candidates.append((fx, x))

        # Sort by function value
        candidates.sort(key=lambda t: t[0])
        
        # Phase 2: Multi-start Nelder-Mead
        # Select top K candidates
        K = min(3, len(candidates))
        remaining_budget = budget - n_random
        
        # Distribute remaining budget among the K starts
        # Leave a small buffer for safety
        budget_per_start = remaining_budget // K
        
        best_x = candidates[0][1]
        best_f = candidates[0][0]

        for i in range(K):
            fx, x = candidates[i]
            # Only run if we have enough budget for a meaningful NM run
            if budget_per_start < 10:
                break
            
            # Run Nelder-Mead from this starting point
            x_opt, f_opt = self._nelder_mead(
                func, x, fx, dim, budget_per_start, rng
            )
            
            if f_opt < best_f:
                best_f = f_opt
                best_x = x_opt

        return best_x, best_f

    def _nelder_mead(self, func, x0, f0, dim, budget, rng):
        """
        Robust Nelder-Mead simplex method.
        """
        # Initialize simplex
        # Use a step size relative to the domain size or a fixed reasonable value
        # Domain is [-5, 5], so a step of 1.0 is reasonable
        step_size = 1.0
        
        simplex = [x0.copy()]
        fvals = [f0]

        for i in range(dim):
            xi = x0.copy()
            # Perturb in direction i
            # If x0[i] is close to boundary, push inwards
            if xi[i] > 0:
                xi[i] += step_size
            else:
                xi[i] -= step_size
            xi = np.clip(xi, -5, 5)
            fi = func(xi)
            simplex.append(xi)
            fvals.append(fi)

        best_x = x0.copy()
        best_f = f0
        for i in range(len(simplex)):
            if fvals[i] < best_f:
                best_f = fvals[i]
                best_x = simplex[i].copy()

        calls_used = 1 + dim
        calls_left = budget - calls_used

        # Nelder-Mead parameters
        gamma = 1.0  # reflection
        rho = 2.0    # expansion
        sigma = 0.5  # contraction
        delta = 0.5  # shrink

        while calls_left > 0:
            # Sort simplex by function value
            indices = np.argsort(fvals)
            simplex = [simplex[i] for i in indices]
            fvals = [fvals[i] for i in indices]

            # Check for convergence
            f_spread = fvals[-1] - fvals[0]
            if f_spread < 1e-10:
                break

            # Centroid of all points except the worst
            centroid = np.mean(simplex[:-1], axis=0)

            # Reflection
            x_ref = centroid + gamma * (centroid - simplex[-1])
            x_ref = np.clip(x_ref, -5, 5)
            f_ref = func(x_ref)
            calls_left -= 1
            if calls_left <= 0:
                # Update best if reflection was good
                if f_ref < best_f:
                    best_f = f_ref
                    best_x = x_ref.copy()
                break

            if f_ref < fvals[0]:
                # Expansion
                x_exp = centroid + rho * (x_ref - centroid)
                x_exp = np.clip(x_exp, -5, 5)
                f_exp = func(x_exp)
                calls_left -= 1
                if calls_left <= 0:
                    # Update best
                    if f_exp < best_f:
                        best_f = f_exp
                        best_x = x_exp.copy()
                    elif f_ref < best_f:
                        best_f = f_ref
                        best_x = x_ref.copy()
                    break

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
                if calls_left <= 0:
                    # Update best
                    if f_con < best_f:
                        best_f = f_con
                        best_x = x_con.copy()
                    elif f_ref < best_f:
                        best_f = f_ref
                        best_x = x_ref.copy()
                    break

                if f_con < min(f_ref, fvals[-1]):
                    simplex[-1] = x_con
                    fvals[-1] = f_con
                else:
                    # Shrink
                    # Shrink all points except the best towards the best
                    for i in range(1, len(simplex)):
                        if calls_left <= 0:
                            break
                        simplex[i] = simplex[0] + delta * (simplex[i] - simplex[0])
                        simplex[i] = np.clip(simplex[i], -5, 5)
                        fvals[i] = func(simplex[i])
                        calls_left -= 1

            # Update best
            current_best_idx = np.argmin(fvals)
            if fvals[current_best_idx] < best_f:
                best_f = fvals[current_best_idx]
                best_x = simplex[current_best_idx].copy()

        return best_x, best_f
# EVOLVE-BLOCK-END
