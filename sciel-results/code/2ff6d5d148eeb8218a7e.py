"""Hierarchical grid search with local polish for Katsuura function."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Hierarchical grid-based search for Katsuura function on [-5,5]^5."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285.0  # safety margin before 300s limit

        dim = self.dim
        budget = self.budget
        calls = 0

        def evaluate(x):
            nonlocal calls
            if calls >= budget:
                return np.inf
            calls += 1
            return func(x)

        best_f = np.inf
        best_x = np.zeros(dim)

        # Phase 1: Coarse grid - 3^5 = 243 points
        # This gives us a broad overview of the landscape
        coarse_sizes = [3, 5, 7, 9]
        current_lower = np.full(dim, -5.0)
        current_upper = np.full(dim, 5.0)

        for grid_size in coarse_sizes:
            if time.time() > deadline - 10:
                break
            if calls >= budget:
                break

            # Create grid points in the current box
            points = []
            for i in range(grid_size):
                for j in range(grid_size):
                    for k in range(grid_size):
                        for l in range(grid_size):
                            for m in range(grid_size):
                                x = np.array([
                                    current_lower[0] + (current_upper[0] - current_lower[0]) * i / (grid_size - 1),
                                    current_lower[1] + (current_upper[1] - current_lower[1]) * j / (grid_size - 1),
                                    current_lower[2] + (current_upper[2] - current_lower[2]) * k / (grid_size - 1),
                                    current_lower[3] + (current_upper[3] - current_lower[3]) * l / (grid_size - 1),
                                    current_lower[4] + (current_upper[4] - current_lower[4]) * m / (grid_size - 1),
                                ])
                                points.append(x)

            # Evaluate all points
            local_best_f = np.inf
            local_best_x = None
            for x in points:
                if calls >= budget or time.time() > deadline - 5:
                    break
                f = evaluate(x)
                if f < local_best_f:
                    local_best_f = f
                    local_best_x = x.copy()

            if local_best_f < best_f:
                best_f = local_best_f
                best_x = local_best_x.copy()

            # Refine around the best point: take a box around it
            if local_best_x is not None and grid_size > 1:
                # The step size in the current grid
                step = (current_upper - current_lower) / (grid_size - 1)
                # Take a box of half-step around the best point
                new_lower = np.maximum(current_lower, local_best_x - step * 0.5)
                new_upper = np.minimum(current_upper, local_best_x + step * 0.5)
                
                # Ensure the box is valid
                if np.all(new_lower < new_upper):
                    current_lower = new_lower
                    current_upper = new_upper
                else:
                    # Box collapsed, use a smaller fraction of the original step
                    frac = 0.3
                    new_lower = np.maximum(current_lower, local_best_x - step * frac)
                    new_upper = np.minimum(current_upper, local_best_x + step * frac)
                    if np.all(new_lower < new_upper):
                        current_lower = new_lower
                        current_upper = new_upper
                    else:
                        break

        # Phase 2: Local polish using Nelder-Mead on the best point
        if best_x is not None and calls < budget and time.time() < deadline - 5:
            remaining_budget = budget - calls
            # Nelder-Mead uses about dim+1 = 6 function calls per iteration
            # Plus initial simplex evaluation
            maxiter = max(1, (remaining_budget - 1) // (dim + 1))
            
            # Simple Nelder-Mead implementation
            x = best_x.copy()
            f = best_f
            
            # Create initial simplex
            simplex = [x.copy()]
            for i in range(dim):
                xi = x.copy()
                xi[i] += 0.1 if xi[i] < 5.0 else -0.1
                xi = np.clip(xi, -5, 5)
                simplex.append(xi)
            
            # Evaluate simplex
            simplex_f = [f]
            for i in range(1, dim + 1):
                if calls >= budget or time.time() > deadline - 2:
                    break
                simplex_f.append(evaluate(simplex[i]))
            
            # Nelder-Mead iterations
            alpha = 1.0
            gamma = 2.0
            rho = 0.5
            sigma = 0.5
            
            for iteration in range(maxiter):
                if calls >= budget or time.time() > deadline - 2:
                    break
                
                # Sort simplex by function value
                indices = np.argsort(simplex_f)
                simplex = [simplex[i] for i in indices]
                simplex_f = [simplex_f[i] for i in indices]
                
                # Centroid of all but worst point
                centroid = np.mean(simplex[:-1], axis=0)
                
                # Reflection
                xr = centroid + alpha * (centroid - simplex[-1])
                xr = np.clip(xr, -5, 5)
                if calls >= budget:
                    break
                fr = evaluate(xr)
                
                if fr < simplex_f[0]:
                    # Expansion
                    xe = centroid + gamma * (centroid - simplex[-1])
                    xe = np.clip(xe, -5, 5)
                    if calls >= budget:
                        break
                    fe = evaluate(xe)
                    if fe < fr:
                        simplex[-1] = xe
                        simplex_f[-1] = fe
                    else:
                        simplex[-1] = xr
                        simplex_f[-1] = fr
                elif fr < simplex_f[-2]:
                    simplex[-1] = xr
                    simplex_f[-1] = fr
                else:
                    # Contraction
                    xc = centroid + rho * (simplex[-1] - centroid)
                    xc = np.clip(xc, -5, 5)
                    if calls >= budget:
                        break
                    fc = evaluate(xc)
                    if fc < simplex_f[-1]:
                        simplex[-1] = xc
                        simplex_f[-1] = fc
                    else:
                        # Shrink
                        for i in range(1, dim + 1):
                            simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                            if calls >= budget:
                                break
                            simplex_f[i] = evaluate(simplex[i])
                
                # Update best
                min_idx = np.argmin(simplex_f)
                if simplex_f[min_idx] < best_f:
                    best_f = simplex_f[min_idx]
                    best_x = simplex[min_idx].copy()

        return best_x, best_f
# EVOLVE-BLOCK-END
