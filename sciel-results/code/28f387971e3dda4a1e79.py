"""Optimizer for the BBOB-style Büche-Rastrigin task (f4).

Multi-start Nelder-Mead optimizer. Uses a coarse random initialization to identify
promising basins and refines them with a fast local search method.
"""
# EVOLVE-BLOCK-START
import time
import numpy as np


class Optimizer:
    """Global black-box minimizer via Multi-start Nelder-Mead on [-5, 5]^dim."""

    def __init__(self, budget, dim):
        self.budget = int(budget)
        self.dim = int(dim)

    def _nelder_mead(self, func, x0, budget, seed, step_size=0.5):
        n = self.dim
        rng = np.random.default_rng(seed)
        
        # Initialize simplex
        # Start with x0 and small perturbations
        simplex = np.empty((n + 1, n))
        simplex[0] = x0
        for i in range(1, n + 1):
            simplex[i] = x0.copy()
            simplex[i, i - 1] += step_size

        # Evaluate initial simplex
        fx = np.empty(n + 1)
        calls = 0
        deadline = time.time() + 290.0
        
        for i in range(n + 1):
            fx[i] = func(simplex[i])
            calls += 1
            if calls >= budget or time.time() > deadline:
                break

        best_idx = np.argmin(fx)
        best_x = simplex[best_idx].copy()
        best_fx = fx[best_idx]

        # Nelder-Mead parameters
        alpha = 1.0  # reflection
        gamma = 2.0  # expansion
        rho = 0.5   # contraction
        sigma = 0.5 # shrink

        while calls < budget and time.time() < deadline:
            # Sort simplex by fitness
            order = np.argsort(fx)
            simplex = simplex[order]
            fx = fx[order]
            best_idx = 0
            worst_idx = n
            
            # If the best is not at index 0 after sorting, it is.
            # Check for convergence
            if fx[worst_idx] - fx[best_idx] < 1e-10:
                break

            # Centroid of best n points
            centroid = simplex[:n].mean(axis=0)
            
            # Reflection
            x_r = centroid + alpha * (centroid - simplex[worst_idx])
            x_r = np.clip(x_r, -5.0, 5.0)
            f_r = func(x_r)
            calls += 1
            if calls >= budget or time.time() > deadline:
                break

            if f_r < fx[best_idx]:
                # Expansion
                x_e = centroid + gamma * (x_r - centroid)
                x_e = np.clip(x_e, -5.0, 5.0)
                f_e = func(x_e)
                calls += 1
                if calls >= budget or time.time() > deadline:
                    break
                
                if f_e < f_r:
                    simplex[worst_idx] = x_e
                    fx[worst_idx] = f_e
                else:
                    simplex[worst_idx] = x_r
                    fx[worst_idx] = f_r
            elif f_r < fx[-2]:
                # Accept reflection
                simplex[worst_idx] = x_r
                fx[worst_idx] = f_r
            else:
                # Contraction
                if f_r < fx[worst_idx]:
                    # Outside contraction
                    x_c = centroid + rho * (x_r - centroid)
                else:
                    # Inside contraction
                    x_c = centroid - rho * (x_r - centroid)
                
                x_c = np.clip(x_c, -5.0, 5.0)
                f_c = func(x_c)
                calls += 1
                if calls >= budget or time.time() > deadline:
                    break

                if f_c < min(f_r, fx[worst_idx]):
                    simplex[worst_idx] = x_c
                    fx[worst_idx] = f_c
                else:
                    # Shrink
                    for i in range(1, n + 1):
                        simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                        simplex[i] = np.clip(simplex[i], -5.0, 5.0)
                        fx[i] = func(simplex[i])
                        calls += 1
                        if calls >= budget or time.time() > deadline:
                            break
                    if calls >= budget or time.time() > deadline:
                        break

            # Update best
            current_best_idx = np.argmin(fx)
            if fx[current_best_idx] < best_fx:
                best_fx = fx[current_best_idx]
                best_x = simplex[current_best_idx].copy()

        return best_x, best_fx

    def __call__(self, func):
        n = self.dim
        budget = self.budget
        rng = np.random.default_rng(0)
        
        # Phase 1: Coarse Global Search
        # Use 20% of budget for initialization
        init_budget = max(10, budget // 5)
        init_points = []
        init_fits = []
        
        # Deterministic grid-like sampling for better coverage in low dims
        # For dim 5, we can sample corners and midpoints
        # But random is safer for hidden shifts
        for i in range(init_budget):
            # Mix of random and structured
            if i < 10:
                # Sample near 0 and corners
                x = rng.uniform(-5, 5, n)
            else:
                x = rng.uniform(-5, 5, n)
            
            fx = func(x)
            init_points.append(x)
            init_fits.append(fx)
        
        # Sort by fitness
        init_indices = np.argsort(init_fits)
        
        # Phase 2: Local Refinement with Restarts
        best_x = init_points[init_indices[0]]
        best_fx = init_fits[init_indices[0]]
        
        remaining_budget = budget - init_budget
        # Allocate budget to top 3 initial points
        # 50% to best, 30% to 2nd, 20% to 3rd
        allocations = [
            int(remaining_budget * 0.5),
            int(remaining_budget * 0.3),
            int(remaining_budget * 0.2)
        ]
        
        # Ensure we don't exceed budget
        for k in range(min(3, len(init_indices))):
            idx = init_indices[k]
            x_start = init_points[idx]
            alloc = allocations[k]
            
            if alloc <= 0:
                break
                
            # Run Nelder-Mead
            x_opt, f_opt = self._nelder_mead(func, x_start, alloc, seed=42+k)
            
            if f_opt < best_fx:
                best_fx = f_opt
                best_x = x_opt
        
        return best_x, best_fx
# EVOLVE-BLOCK-END
