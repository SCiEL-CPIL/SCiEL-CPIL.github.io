"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Nelder-Mead optimizer with adaptive restarts for bent cigar."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        rng = np.random.default_rng(0)
        d = self.dim
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin

        # Nelder-Mead parameters
        alpha = 1.0
        gamma = 2.0
        rho = 0.5
        sigma = 0.5
        max_iter = self.budget
        calls = 0

        def nelder_mead(func, x0, budget_left, rng, deadline):
            nonlocal calls
            # Initialize simplex
            simplex = [x0.copy() for _ in range(d + 1)]
            for i in range(1, d + 1):
                simplex[i] = x0.copy()
                simplex[i][i - 1] += 0.1 # Small step in each dimension
            
            # Evaluate initial simplex
            f_simplex = []
            for i in range(d + 1):
                if time.time() > deadline:
                    return x0, f_simplex[0] if f_simplex else float('inf')
                f_val = func(simplex[i])
                calls += 1
                f_simplex.append(f_val)
                
            best_x = simplex[0].copy()
            best_f = f_simplex[0]
            for i in range(d + 1):
                if f_simplex[i] < best_f:
                    best_f = f_simplex[i]
                    best_x = simplex[i].copy()

            it = 0
            no_improve = 0
            while calls < budget_left and it < max_iter:
                if time.time() > deadline:
                    break
                
                # Sort simplex by function value
                order = np.argsort(f_simplex)
                simplex = [simplex[i] for i in order]
                f_simplex = [f_simplex[i] for i in order]

                # Calculate centroid of all but the worst point
                centroid = np.mean(simplex[:-1], axis=0)

                # Reflection
                x_r = centroid + alpha * (centroid - simplex[-1])
                x_r = np.clip(x_r, -5, 5)
                f_r = func(x_r)
                calls += 1

                if f_r < f_simplex[0]:
                    # Expansion
                    x_e = centroid + gamma * (x_r - centroid)
                    x_e = np.clip(x_e, -5, 5)
                    f_e = func(x_e)
                    calls += 1
                    if f_e < f_r:
                        simplex[-1] = x_e
                        f_simplex[-1] = f_e
                    else:
                        simplex[-1] = x_r
                        f_simplex[-1] = f_r
                    no_improve = 0
                elif f_r < f_simplex[-2]:
                    # Reflection accepted
                    simplex[-1] = x_r
                    f_simplex[-1] = f_r
                    no_improve = 0
                else:
                    # Contraction
                    # Outside contraction
                    x_oc = centroid + 0.5 * (x_r - centroid)
                    x_oc = np.clip(x_oc, -5, 5)
                    f_oc = func(x_oc)
                    calls += 1
                    if f_oc < f_r:
                        simplex[-1] = x_oc
                        f_simplex[-1] = f_oc
                    else:
                        # Inside contraction
                        x_ic = centroid - 0.5 * (simplex[-1] - centroid)
                        x_ic = np.clip(x_ic, -5, 5)
                        f_ic = func(x_ic)
                        calls += 1
                        if f_ic < f_simplex[-1]:
                            simplex[-1] = x_ic
                            f_simplex[-1] = f_ic
                        else:
                            # Shrink
                            for i in range(1, d + 1):
                                simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                                simplex[i] = np.clip(simplex[i], -5, 5)
                                f_simplex[i] = func(simplex[i])
                                calls += 1
                            no_improve += 1

                # Update best
                current_best_idx = np.argmin(f_simplex)
                if f_simplex[current_best_idx] < best_f:
                    best_f = f_simplex[current_best_idx]
                    best_x = simplex[current_best_idx].copy()
                    no_improve = 0
                else:
                    no_improve += 1

                it += 1

                # Restart if stuck
                if no_improve > 10:
                    break

            return best_x, best_f

        # Initial point
        x0 = np.zeros(d)
        
        # Run Nelder-Mead
        best_x, best_f = nelder_mead(func, x0, self.budget, rng, deadline)
        
        return None
# EVOLVE-BLOCK-END
