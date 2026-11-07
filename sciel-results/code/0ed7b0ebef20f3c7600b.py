"""Nelder-Mead based optimizer for BBOB-style f13 (Sharp ridge)."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """Nelder-Mead simplex method with adaptive restarts for ridge structures."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        # Track global best
        best_x = np.zeros(dim)
        best_f = np.inf
        calls = 0

        # Initial parameters
        current_scale = 2.0
        x_start = np.zeros(dim)
        
        # We will run NM in a loop, potentially restarting if stuck
        # But to keep it simple and efficient, we implement restarts inside the main loop
        
        def init_simplex(center, scale):
            nonlocal calls
            simp = [center.copy()]
            fxs = [func(center)]
            calls += 1
            for i in range(dim):
                v = center.copy()
                v[i] += scale
                simp.append(v)
                fxs.append(func(v))
                calls += 1
            return simp, fxs

        simplex, fx_list = init_simplex(x_start, current_scale)
        
        if calls >= budget:
            return best_x, best_f

        alpha, gamma, rho, sigma = 1.0, 2.0, 0.5, 0.5
        stagnation_counter = 0
        max_stagnation = 15
        restart_scale_factor = 1.5

        while calls < budget:
            # Sort simplex
            order = np.argsort(fx_list)
            simplex = [simplex[i].copy() for i in order]
            fx_list = [fx_list[i] for i in order]
            
            # Update global best
            if fx_list[0] < best_f:
                best_f = fx_list[0]
                best_x = simplex[0].copy()
                stagnation_counter = 0
            else:
                stagnation_counter += 1

            # Check for stagnation -> Restart
            if stagnation_counter > max_stagnation:
                # Restart around best_x with larger scale
                current_scale *= restart_scale_factor
                if current_scale > 10.0:
                    break # Prevent infinite loops with huge scales
                simplex, fx_list = init_simplex(best_x, current_scale)
                if calls >= budget:
                    break
                stagnation_counter = 0
                continue

            # Check for convergence
            spread = np.max(fx_list) - np.min(fx_list)
            if spread < 1e-12:
                break

            # Compute centroid of all but worst
            centroid = np.mean(simplex[:-1], axis=0)

            # Reflection
            xr = centroid + alpha * (centroid - simplex[-1])
            xr = np.clip(xr, -5, 5)
            fr = func(xr)
            calls += 1

            if fr < fx_list[0]:
                # Try expansion
                xe = centroid + gamma * (centroid - simplex[-1])
                xe = np.clip(xe, -5, 5)
                fe = func(xe)
                calls += 1
                if fe < fr:
                    simplex[-1] = xe
                    fx_list[-1] = fe
                else:
                    simplex[-1] = xr
                    fx_list[-1] = fr
            elif fr < fx_list[-2]:
                simplex[-1] = xr
                fx_list[-1] = fr
            else:
                # Contraction
                if fr < fx_list[-1]:
                    # Outside contraction
                    xc = centroid + rho * (simplex[-1] - centroid)
                else:
                    # Inside contraction
                    xc = centroid - rho * (simplex[-1] - centroid)
                xc = np.clip(xc, -5, 5)
                fc = func(xc)
                calls += 1

                if fc <= min(fr, fx_list[-1]):
                    simplex[-1] = xc
                    fx_list[-1] = fc
                else:
                    # Shrink
                    for i in range(1, dim + 1):
                        simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                        simplex[i] = np.clip(simplex[i], -5, 5)
                        fx_list[i] = func(simplex[i])
                        calls += 1
                        if calls >= budget:
                            return best_x, best_f
                    if calls >= budget:
                        return best_x, best_f

        return best_x, best_f
# EVOLVE-BLOCK-END
