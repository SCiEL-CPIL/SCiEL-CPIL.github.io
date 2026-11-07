"""Nelder-Mead based optimizer for BBOB-style f13 (Sharp ridge)."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """Nelder-Mead simplex method with adaptive step sizing and restarts."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        # Parameters
        alpha, gamma, rho, sigma = 1.0, 2.0, 0.5, 0.5
        initial_step = 1.0
        restart_threshold = 1e-6  # Spread threshold for restart
        restart_multiplier = 2.0  # Multiply step size on restart
        max_restarts = 10
        restarts_used = 0
        
        # State
        best_x = None
        best_f = np.inf
        calls = 0

        def init_simplex(center, step):
            nonlocal calls
            simplex = [center.copy()]
            fx_list = [func(center)]
            calls += 1
            
            for i in range(dim):
                v = center.copy()
                v[i] += step
                simplex.append(v)
                fx_list.append(func(v))
                calls += 1
                
            # Sort
            order = np.argsort(fx_list)
            simplex = [simplex[i] for i in order]
            fx_list = [fx_list[i] for i in order]
            
            return simplex, fx_list

        # Initial setup
        x0 = np.zeros(dim)
        simplex, fx_list = init_simplex(x0, initial_step)
        
        # Update best
        if fx_list[0] < best_f:
            best_f = fx_list[0]
            best_x = simplex[0].copy()

        while calls < budget:
            # Sort simplex by function value
            order = np.argsort(fx_list)
            simplex = [simplex[i] for i in order]
            fx_list = [fx_list[i] for i in order]

            # Update best
            if fx_list[0] < best_f:
                best_f = fx_list[0]
                best_x = simplex[0].copy()

            # Check for convergence/stagnation
            spread = np.max(fx_list) - np.min(fx_list)
            if spread < restart_threshold:
                # Check if we should restart
                if restarts_used < max_restarts:
                    # Restart around best_x with larger step
                    restarts_used += 1
                    new_step = initial_step * (restart_multiplier ** restarts_used)
                    # Cap step size to avoid going out of bounds too much
                    new_step = min(new_step, 5.0)
                    simplex, fx_list = init_simplex(best_x, new_step)
                    continue
                else:
                    break

            # Compute centroid of all but worst
            centroid = np.mean(simplex[:-1], axis=0)

            # Reflection
            xr = centroid + alpha * (centroid - simplex[-1])
            fr = func(np.clip(xr, -5, 5))
            calls += 1

            if fr < fx_list[0]:
                # Try expansion
                xe = centroid + gamma * (centroid - simplex[-1])
                fe = func(np.clip(xe, -5, 5))
                calls += 1
                if fe < fr:
                    simplex[-1] = np.clip(xe, -5, 5)
                    fx_list[-1] = fe
                else:
                    simplex[-1] = np.clip(xr, -5, 5)
                    fx_list[-1] = fr
            elif fr < fx_list[-2]:
                simplex[-1] = np.clip(xr, -5, 5)
                fx_list[-1] = fr
            else:
                # Contraction
                if fr < fx_list[-1]:
                    # Outside contraction
                    xc = centroid + rho * (simplex[-1] - centroid)
                else:
                    # Inside contraction
                    xc = centroid - rho * (simplex[-1] - centroid)
                fc = func(np.clip(xc, -5, 5))
                calls += 1

                if fc <= min(fr, fx_list[-1]):
                    simplex[-1] = np.clip(xc, -5, 5)
                    fx_list[-1] = fc
                else:
                    # Shrink
                    for i in range(1, dim + 1):
                        simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                        fx_list[i] = func(simplex[i])
                        calls += 1
                        if calls >= budget:
                            break
                    if calls >= budget:
                        break

        return best_x, best_f
# EVOLVE-BLOCK-END
