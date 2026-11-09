"""Deterministic multi-start Nelder-Mead optimizer for BBOB f23 (Katsuura).

Strategy:
1. Generate a small set of random starting points.
2. Run a standard Nelder-Mead simplex method from each starting point.
3. Use a wall-clock deadline guard to ensure termination before the 330s limit.
4. The deterministic nature of Nelder-Mead (no random step sizes) often converges
   faster to local minima than stochastic ES on smooth but multimodal functions,
   provided the starting points are well-chosen.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-start Nelder-Mead optimizer."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        t0 = time.time()
        deadline = t0 + 280.0  # Safety margin before 330s kill

        rng = np.random.default_rng(12345)
        lo, hi = -5.0, 5.0

        best_x = None
        best_f = np.inf
        calls = 0

        def eval_point(x):
            nonlocal calls, best_x, best_f
            if calls >= budget:
                return np.inf
            if time.time() > deadline:
                return np.inf
            calls += 1
            f = float(func(x))
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        # Phase 1: Generate starting points
        # We use a few random points to kick off the local searches.
        # 10 starting points is a good balance between exploration and exploitation budget.
        n_starts = 10
        start_points = []
        for _ in range(n_starts):
            x0 = rng.uniform(lo, hi, dim)
            f0 = eval_point(x0)
            start_points.append((x0, f0))
        
        # Sort starting points by fitness
        start_points.sort(key=lambda x: x[1])

        # Phase 2: Run Nelder-Mead from each starting point
        # We allocate the remaining budget roughly equally among the starts,
        # but prioritize better starting points if time is tight.
        
        # Standard Nelder-Mead parameters
        alpha = 1.0    # reflection
        gamma = 2.0    # expansion
        rho = 0.5      # contraction
        sigma = 0.5    # shrinkage
        tol = 1e-8     # tolerance for convergence
        max_iter_per_start = 50  # Limit iterations per start to avoid getting stuck in one basin for too long

        for x0, f0 in start_points:
            if calls >= budget or time.time() > deadline:
                break
            
            # Initialize simplex
            simplex = [x0.copy()]
            f_values = [f0]
            
            # Add other points in the simplex
            for i in range(dim):
                xi = x0.copy()
                step = 0.05 if x0[i] != 0 else 0.05 # Small step
                xi[i] += step
                xi = np.clip(xi, lo, hi)
                fi = eval_point(xi)
                if np.isinf(fi):
                    break
                simplex.append(xi)
                f_values.append(fi)
            
            if len(simplex) < dim + 1:
                continue

            # Nelder-Mead iterations
            for _ in range(max_iter_per_start):
                if calls >= budget or time.time() > deadline:
                    break
                
                # Sort simplex
                idx = np.argsort(f_values)
                simplex = [simplex[i] for i in idx]
                f_values = [f_values[i] for i in idx]
                
                # Check convergence
                if f_values[0] - f_values[-1] < tol:
                    break
                
                # Centroid of all but worst point
                centroid = np.mean(simplex[:-1], axis=0)
                
                # Reflection
                x_r = centroid + alpha * (centroid - simplex[-1])
                x_r = np.clip(x_r, lo, hi)
                f_r = eval_point(x_r)
                
                if f_r < f_values[0]:
                    # Expansion
                    x_e = centroid + gamma * (x_r - centroid)
                    x_e = np.clip(x_e, lo, hi)
                    f_e = eval_point(x_e)
                    if f_e < f_r:
                        simplex[-1] = x_e
                        f_values[-1] = f_e
                    else:
                        simplex[-1] = x_r
                        f_values[-1] = f_r
                elif f_r < f_values[-2]:
                    # Accept reflection
                    simplex[-1] = x_r
                    f_values[-1] = f_r
                else:
                    # Contraction
                    x_c = centroid + rho * (simplex[-1] - centroid)
                    x_c = np.clip(x_c, lo, hi)
                    f_c = eval_point(x_c)
                    if f_c < f_values[-1]:
                        simplex[-1] = x_c
                        f_values[-1] = f_c
                    else:
                        # Shrinkage
                        for i in range(1, dim + 1):
                            simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], lo, hi)
                            f_values[i] = eval_point(simplex[i])
                            if np.isinf(f_values[i]):
                                break

        return best_x, best_f
# EVOLVE-BLOCK-END
