"""Nelder-Mead + (1+1)-ES hybrid optimizer for BBOB f11 (Discus)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Hybrid Nelder-Mead + (1+1)-ES for BBOB f11.
    Nelder-Mead adapts the simplex shape to the discus landscape; (1+1)-ES polishes.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 320  # safety margin before 330s kill

        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        # --- Phase 1: Initial Exploration ---
        # We need to find the basin of attraction.
        # For f11 (Discus), the minimum is shifted.
        
        points = []
        values = []
        
        # Evaluate center
        center = np.zeros(dim)
        fc = func(np.clip(center, -5, 5))
        points.append(center)
        values.append(fc)
        evals = 1
        
        # Explore along each axis to find the shift direction
        # The shift is in [-4, 4]^5.
        # We sample a grid to find a good starting point for the local optimizer.
        # A coarse grid in 5D is expensive, so we use a structured random search.
        
        # 1. Axis-aligned probes
        for i in range(dim):
            for d in [1.0, 2.0, 3.0, 4.0]:
                for sign in [1.0, -1.0]:
                    if evals >= budget:
                        break
                    p = np.zeros(dim)
                    p[i] = sign * d
                    fp = func(np.clip(p, -5, 5))
                    points.append(p)
                    values.append(fp)
                    evals += 1
                if evals >= budget:
                    break
            if evals >= budget:
                break
        
        # 2. Random points for diversity
        # We use a portion of the budget for this phase.
        # Let's reserve 80% for optimization, so use 20% for exploration.
        explore_budget = min(100, budget // 5)
        if evals < explore_budget:
            for _ in range(explore_budget - evals):
                p = rng.uniform(-4.5, 4.5, dim)
                fp = func(np.clip(p, -5, 5))
                points.append(p)
                values.append(fp)
                evals += 1

        if evals >= budget:
            best_idx = int(np.argmin(values))
            return points[best_idx], values[best_idx]

        best_idx = int(np.argmin(values))
        best_x = np.array(points[best_idx], dtype=np.float64)
        best_f = values[best_idx]

        # --- Phase 2: Nelder-Mead Optimization ---
        # Nelder-Mead parameters
        # We run Nelder-Mead until we run out of budget or stagnate.
        
        # Initialize Simplex
        # Start from best_x, expand slightly to create the initial simplex
        step = 0.5
        simplex = [best_x.copy() for _ in range(dim + 1)]
        for i in range(dim):
            simplex[i+1][i] += step
            
        simplex_f = [func(np.clip(s, -5, 5)) for s in simplex]
        evals += dim + 1
        
        # Update best
        min_idx = int(np.argmin(simplex_f))
        if simplex_f[min_idx] < best_f:
            best_f = simplex_f[min_idx]
            best_x = simplex[min_idx].copy()

        # Nelder-Mead Parameters
        alpha = 1.0    # Reflection
        gamma = 2.0    # Expansion
        rho = 0.5      # Contraction
        sigma = 0.5    # Shrinkage
        
        # We will run Nelder-Mead for a while, then switch to (1+1)-ES
        # Nelder-Mead is effective for the "Discus" shape as it stretches the simplex.
        
        # Reserve some budget for the final polish
        polish_budget = 50
        nm_budget = budget - polish_budget
        
        while evals < nm_budget:
            if time.time() > deadline:
                break
                
            # Sort simplex by function value
            order = np.argsort(simplex_f)
            simplex = [simplex[i] for i in order]
            simplex_f = [simplex_f[i] for i in order]
            
            # Update best
            if simplex_f[0] < best_f:
                best_f = simplex_f[0]
                best_x = simplex[0].copy()
                
            # Calculate centroid of all points except the worst
            centroid = np.mean(simplex[:-1], axis=0)
            
            # 1. Reflection
            x_r = centroid + alpha * (centroid - simplex[-1])
            x_r = np.clip(x_r, -5, 5)
            f_r = func(x_r)
            evals += 1
            
            if f_r < simplex_f[0]:
                # 2. Expansion
                x_e = centroid + gamma * (centroid - simplex[-1])
                x_e = np.clip(x_e, -5, 5)
                f_e = func(x_e)
                evals += 1
                
                if f_e < f_r:
                    simplex[-1] = x_e
                    simplex_f[-1] = f_e
                else:
                    simplex[-1] = x_r
                    simplex_f[-1] = f_r
            elif f_r < simplex_f[-2]:
                # Accept reflection
                simplex[-1] = x_r
                simplex_f[-1] = f_r
            else:
                # 3. Contraction
                x_c = centroid + rho * (simplex[-1] - centroid)
                x_c = np.clip(x_c, -5, 5)
                f_c = func(x_c)
                evals += 1
                
                if f_c < simplex_f[-1]:
                    simplex[-1] = x_c
                    simplex_f[-1] = f_c
                else:
                    # 4. Shrinkage
                    for i in range(1, dim + 1):
                        simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                        simplex_f[i] = func(np.clip(simplex[i], -5, 5))
                        evals += 1
                    
                    # Check if we are out of budget
                    if evals >= nm_budget:
                        break

        # --- Phase 3: (1+1)-ES Local Polish ---
        # Start from best found by Nelder-Mead
        x_local = best_x.copy()
        f_local = best_f
        sigma_local = 0.05

        no_improve = 0
        max_no_improve = 20

        while evals < budget:
            if time.time() > deadline:
                break

            z = rng.standard_normal(dim)
            y = x_local + sigma_local * z
            y = np.clip(y, -5, 5)
            fy = func(y)
            evals += 1

            if fy < f_local:
                x_local = y
                f_local = fy
                sigma_local = max(sigma_local * 1.05, 1e-6)
                no_improve = 0
            else:
                sigma_local = max(sigma_local * 0.95, 1e-6)
                no_improve += 1

            if no_improve >= max_no_improve:
                break

            if f_local < best_f:
                best_f = f_local
                best_x = x_local.copy()

        return best_x, best_f
# EVOLVE-BLOCK-END
