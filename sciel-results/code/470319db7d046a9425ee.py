"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Nelder-Mead with multi-start and deadline guard."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # 280s safety margin before 300s limit
        rng = np.random.default_rng(42)
        
        # Multi-start: evaluate 50 random points
        n_starts = 50
        points = rng.uniform(-5, 5, (n_starts, self.dim))
        values = np.empty(n_starts)
        for i in range(n_starts):
            values[i] = func(points[i])
        
        # Select top 5 starting points
        top_indices = np.argsort(values)[:5]
        best_overall = values[top_indices[0]]
        best_point = points[top_indices[0]].copy()
        
        # Allocate remaining budget to 5 parallel Nelder-Mead runs
        remaining_budget = self.budget - n_starts
        budget_per_run = max(1, remaining_budget // 5)
        
        for idx in top_indices:
            if time.time() > deadline:
                break
                
            x = points[idx].copy()
            fx = values[idx]
            
            # Initialize Nelder-Mead simplex
            simplex = [x.copy()]
            simplex_f = [fx]
            
            # Add dim-1 more points with small perturbations
            for j in range(self.dim - 1):
                perturbation = rng.uniform(-0.5, 0.5, self.dim)
                new_x = np.clip(x + perturbation, -5, 5)
                new_f = func(new_x)
                simplex.append(new_x)
                simplex_f.append(new_f)
            
            # Run Nelder-Mead for budget_per_run iterations
            for _ in range(budget_per_run):
                if time.time() > deadline:
                    break
                
                # Sort simplex by function value
                order = np.argsort(simplex_f)
                simplex = [simplex[i] for i in order]
                simplex_f = [simplex_f[i] for i in order]
                
                # Update best overall
                if simplex_f[0] < best_overall:
                    best_overall = simplex_f[0]
                    best_point = simplex[0].copy()
                
                # Compute centroid of all but worst point
                centroid = np.mean(simplex[:-1], axis=0)
                worst_x = simplex[-1]
                worst_f = simplex_f[-1]
                
                # Reflection
                alpha = 1.0
                xr = centroid + alpha * (centroid - worst_x)
                xr = np.clip(xr, -5, 5)
                fr = func(xr)
                
                if fr < simplex_f[0]:
                    # Expansion
                    gamma = 2.0
                    xe = centroid + gamma * (centroid - worst_x)
                    xe = np.clip(xe, -5, 5)
                    fe = func(xe)
                    if fe < fr:
                        simplex[-1] = xe
                        simplex_f[-1] = fe
                    else:
                        simplex[-1] = xr
                        simplex_f[-1] = fr
                elif fr < simplex_f[-2]:
                    # Accept reflection
                    simplex[-1] = xr
                    simplex_f[-1] = fr
                else:
                    # Contraction
                    rho = 0.5
                    if fr < worst_f:
                        # Outside contraction
                        xc = centroid + rho * (xr - centroid)
                    else:
                        # Inside contraction
                        xc = centroid - rho * (centroid - worst_x)
                    xc = np.clip(xc, -5, 5)
                    fc = func(xc)
                    if fc < min(fr, worst_f):
                        simplex[-1] = xc
                        simplex_f[-1] = fc
                    else:
                        # Shrink
                        for i in range(1, len(simplex)):
                            simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], -5, 5)
                            simplex_f[i] = func(simplex[i])
        
        return best_point, best_overall
# EVOLVE-BLOCK-END
