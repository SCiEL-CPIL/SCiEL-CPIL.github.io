"""Rewritten program using Nelder-Mead Simplex with Random Restarts."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """Nelder-Mead optimizer with random restarts for the BBOB-style f16 (Weierstrass) task."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        best_f = np.inf
        best_x = np.zeros(dim)

        # We will use multiple restarts.
        # Each restart gets a portion of the budget.
        # Let's aim for roughly 10-20 evaluations per restart initially, 
        # but Nelder-Mead in 5D typically needs ~20-50 evals to converge locally.
        # Total budget 500. 
        # Let's try 10 restarts of 50 evals each? Or adaptive?
        # Better: Use a fixed number of restarts, e.g., 10.
        # Each restart has budget/10 = 50 evals.
        
        num_restarts = 10
        evals_per_restart = budget // num_restarts
        
        # If budget is small, reduce restarts
        if evals_per_restart < 10:
            num_restarts = max(1, budget // 10)
            evals_per_restart = budget // num_restarts

        for r in range(num_restarts):
            # 1. Initialize simplex
            # Start from a random point in the box
            x0 = rng.uniform(-4, 4, size=dim) # Avoid edges slightly
            f0 = func(x0)
            if f0 < best_f:
                best_f = f0
                best_x = x0.copy()
            
            # Initialize simplex vertices
            # v0 = x0
            # vi = x0 + delta * e_i for i=1..dim
            # Choose delta based on initial scale
            delta = 0.5
            
            simplex = np.zeros((dim + 1, dim))
            simplex[0] = x0
            for i in range(1, dim + 1):
                simplex[i] = x0.copy()
                simplex[i, i-1] += delta
            
            # Evaluate initial simplex
            simplex_f = np.zeros(dim + 1)
            simplex_f[0] = f0
            evals_made = 1
            
            for i in range(1, dim + 1):
                if evals_made >= evals_per_restart:
                    break
                # Clip to box
                x_i = np.clip(simplex[i], -5, 5)
                simplex_f[i] = func(x_i)
                simplex[i] = x_i # Keep the clipped point for consistency
                evals_made += 1
                if simplex_f[i] < best_f:
                    best_f = simplex_f[i]
                    best_x = x_i.copy()

            # Nelder-Mead Loop
            # Parameters
            alpha = 1.0  # reflection
            gamma = 2.0  # expansion
            rho = 0.5    # contraction
            sigma = 0.5  # shrink
            max_iter = evals_per_restart - evals_made

            for _ in range(max_iter):
                if evals_made >= evals_per_restart:
                    break
                
                # Sort simplex by fitness
                order = np.argsort(simplex_f)
                simplex = simplex[order]
                simplex_f = simplex_f[order]
                
                # Check convergence
                # If the spread of the simplex is small, stop
                if np.max(simplex_f) - np.min(simplex_f) < 1e-6 * (1 + np.abs(np.min(simplex_f))):
                    break
                
                # Centroid of the best dim points
                centroid = np.mean(simplex[:dim], axis=0)
                
                # Reflection
                xr = centroid + alpha * (centroid - simplex[-1])
                xr = np.clip(xr, -5, 5)
                fr = func(xr)
                evals_made += 1
                if fr < best_f:
                    best_f = fr
                    best_x = xr.copy()
                
                if fr < simplex_f[0]:
                    # Expansion
                    xe = centroid + gamma * (centroid - simplex[-1])
                    xe = np.clip(xe, -5, 5)
                    fe = func(xe)
                    evals_made += 1
                    if fe < best_f:
                        best_f = fe
                        best_x = xe.copy()
                    
                    if fe < fr:
                        simplex[-1] = xe
                        simplex_f[-1] = fe
                    else:
                        simplex[-1] = xr
                        simplex_f[-1] = fr
                elif fr < simplex_f[-2]:
                    # Reflection is good, but not better than best
                    simplex[-1] = xr
                    simplex_f[-1] = fr
                else:
                    # Contraction
                    # Outside contraction
                    xc = centroid + (1 - rho) * (simplex[-1] - centroid)
                    xc = np.clip(xc, -5, 5)
                    fc = func(xc)
                    evals_made += 1
                    if fc < best_f:
                        best_f = fc
                        best_x = xc.copy()
                    
                    if fc < simplex_f[-1]:
                        simplex[-1] = xc
                        simplex_f[-1] = fc
                    else:
                        # Shrink
                        # Move all points towards x0 (best)
                        for i in range(1, dim + 1):
                            simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], -5, 5)
                            # Re-evaluate
                            fi = func(simplex[i])
                            evals_made += 1
                            if fi < best_f:
                                best_f = fi
                                best_x = simplex[i].copy()
                            simplex_f[i] = fi
                            if evals_made >= evals_per_restart:
                                break
                        if evals_made >= evals_per_restart:
                            break

        return best_x, best_f
# EVOLVE-BLOCK-END
