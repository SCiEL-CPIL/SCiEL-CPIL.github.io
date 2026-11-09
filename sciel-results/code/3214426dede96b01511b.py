"""
Strategy: Multi-Start Nelder-Mead with adaptive step sizes and restarts.
Rationale: The Discus function (f11) has a flat plateau and a single narrow basin.
Nelder-Mead is efficient for smooth, unimodal landscapes and can navigate anisotropic
basins better than population-based methods when given a good initial step size.
Multiple restarts from diverse starting points increase the chance of landing in
the basin, and adaptive step sizing helps both exploration and exploitation.
"""

import numpy as np
import time


class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 290  # Safety margin before 300s kill

        best_x = np.zeros(dim)
        best_f = np.inf
        total_calls = 0

        # Helper to safely call func and track budget
        def safe_call(x):
            nonlocal total_calls, best_f, best_x
            if total_calls >= budget:
                return None
            x = np.clip(x, -5.0, 5.0)
            total_calls += 1
            f_val = func(x)
            if f_val < best_f:
                best_f = f_val
                best_x = x.copy()
            return f_val

        def nelder_mead(x0, f0, max_calls, initial_step=1.0):
            """
            Nelder-Mead with adaptive step sizing.
            """
            nonlocal total_calls, best_f, best_x

            if total_calls >= budget:
                return best_x, best_f

            # Initialize simplex
            simplex = [(x0.copy(), f0)]
            start_calls = total_calls

            # Expand simplex
            for i in range(dim):
                if total_calls >= budget or total_calls - start_calls >= max_calls:
                    break
                xi = x0.copy()
                xi[i] += initial_step
                xi = np.clip(xi, -5.0, 5.0)
                fi = safe_call(xi)
                if fi is None:
                    break
                simplex.append((xi, fi))

            if len(simplex) < dim + 1:
                return best_x, best_f

            simplex.sort(key=lambda t: t[1])

            alpha = 0.5
            gamma = 2.0
            rho = 0.5
            sigma = 0.5
            no_improve = 0
            prev_best = best_f

            while total_calls - start_calls < max_calls and total_calls < budget:
                if time.time() > deadline - 5:
                    break

                # Check convergence
                x_center = np.mean([s[0] for s in simplex], axis=0)
                spread = np.max(np.abs(np.array([s[0] for s in simplex]) - x_center))
                if spread < 1e-6:
                    break

                x_worst = simplex[-1][0]
                f_worst = simplex[-1][1]
                x_second_worst = simplex[-2][0]
                f_second_worst = simplex[-2][1]

                x_centroid = np.mean([s[0] for s in simplex[:-1]], axis=0)

                # Reflection
                x_reflect = x_centroid + gamma * (x_centroid - x_worst)
                x_reflect = np.clip(x_reflect, -5.0, 5.0)
                f_reflect = safe_call(x_reflect)
                if f_reflect is None:
                    break

                if f_reflect < simplex[0][1]:
                    # Expansion
                    x_expand = x_centroid + alpha * (x_reflect - x_worst)
                    x_expand = np.clip(x_expand, -5.0, 5.0)
                    f_expand = safe_call(x_expand)
                    if f_expand is None:
                        break
                    if f_expand < f_reflect:
                        simplex[-1] = (x_expand, f_expand)
                    else:
                        simplex[-1] = (x_reflect, f_reflect)
                elif f_reflect < f_second_worst:
                    simplex[-1] = (x_reflect, f_reflect)
                else:
                    # Contraction
                    if f_reflect < f_worst:
                        x_contract = x_centroid + rho * (x_reflect - x_worst)
                    else:
                        x_contract = x_centroid - rho * (x_worst - x_centroid)
                    x_contract = np.clip(x_contract, -5.0, 5.0)
                    f_contract = safe_call(x_contract)
                    if f_contract is None:
                        break
                    if f_contract <= min(f_reflect, f_worst):
                        simplex[-1] = (x_contract, f_contract)
                    else:
                        # Shrink
                        for i in range(1, dim + 1):
                            if total_calls >= budget:
                                break
                            new_x = simplex[0][0] + sigma * (simplex[i][0] - simplex[0][0])
                            new_x = np.clip(new_x, -5.0, 5.0)
                            new_f = safe_call(new_x)
                            if new_f is None:
                                break
                            simplex[i] = (new_x, new_f)
                        if total_calls >= budget:
                            break

                simplex.sort(key=lambda t: t[1])

                # Check for stagnation
                if best_f < prev_best - 1e-8:
                    no_improve = 0
                    prev_best = best_f
                else:
                    no_improve += 1
                    if no_improve > 30:
                        break

            return best_x, best_f

        # --- Multi-Start Strategy ---
        # We'll do multiple restarts with different initial points and step sizes
        rng = np.random.default_rng(42)

        # Allocate budget: reserve some for final refinement
        reserve = 50
        main_budget = budget - reserve

        # Define starting points and initial steps
        starts = [
            (np.zeros(dim), 2.0),
            (rng.uniform(-4.0, 4.0, dim), 2.0),
            (rng.uniform(-3.0, 3.0, dim), 1.5),
            (rng.uniform(-5.0, 5.0, dim), 2.5),
            (rng.uniform(-2.0, 2.0, dim), 1.0),
        ]

        calls_per_start = max(10, main_budget // len(starts))

        for x0, step in starts:
            if total_calls >= budget or time.time() > deadline - 10:
                break
            f0 = safe_call(x0)
            if f0 is None:
                break
            nelder_mead(x0, f0, calls_per_start, initial_step=step)

        # Final refinement on the best point found
        remaining = budget - total_calls
        if remaining > 10 and total_calls < budget:
            nelder_mead(best_x.copy(), best_f, remaining, initial_step=0.3)

        return best_x, best_f
