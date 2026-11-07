"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """(mu+lambda)-ES with per-axis step size adaptation."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin
        
        rng = np.random.default_rng(42)
        n = self.dim
        mu = 2
        lam = 4
        
        # Initialize at origin
        x = np.zeros(n)
        fx = func(x)
        best_x = x.copy()
        best_f = fx
        
        # Per-axis step sizes
        sigma = np.ones(n) * 2.0
        
        calls_made = 1
        for _ in range(self.budget - 1):
            if time.time() > deadline:
                break
            
            # Generate offspring
            offspring = []
            f_values = []
            
            for _ in range(lam):
                z = rng.standard_normal(n)
                y = x + sigma * z
                y = np.clip(y, -5, 5)
                fy = func(y)
                calls_made += 1
                offspring.append(y)
                f_values.append(fy)
                
                if fy < best_f:
                    best_f = fy
                    best_x = y.copy()
            
            # Sort offspring
            idx = np.argsort(f_values)
            offspring = np.array(offspring)[idx]
            f_values = np.array(f_values)[idx]
            
            # Update mean (weighted average of top mu)
            weights = np.zeros(mu)
            for i in range(mu):
                weights[i] = np.log(mu + 0.5) - np.log(i + 1)
            weights /= weights.sum()
            
            x_new = np.sum(weights[:, None] * offspring[:mu], axis=0)
            
            # Adapt per-axis step sizes
            # Compare mean of selected vs all offspring per axis
            # If selected are closer to 0 in a direction, reduce sigma? 
            # Standard rule: if improvement, increase sigma slightly, else decrease.
            # But for anisotropic functions, we want to shrink sigma in directions where progress is fast.
            
            # Simple heuristic: 
            # Calculate step size change based on success rate per axis
            # If the new mean x_new is "better" than x, we might have moved in a good direction.
            
            # A robust approach for Bent Cigar: 
            # Estimate the curvature per axis by looking at the variance of the selected offspring.
            # If variance is small, we are close to the optimum in that axis -> shrink sigma.
            # If variance is large, we are far -> keep or grow sigma.
            
            variances = np.var(offspring[:mu], axis=0)
            # Target variance is proportional to sigma^2.
            # If variances[i] < sigma[i]^2, we are converging, shrink sigma.
            # If variances[i] > sigma[i]^2, we are exploring, grow sigma.
            
            for i in range(n):
                if variances[i] < sigma[i] ** 2:
                    sigma[i] *= 0.9
                else:
                    sigma[i] *= 1.1
                
                # Clamp sigma
                sigma[i] = np.clip(sigma[i], 1e-6, 10.0)
            
            x = x_new
            fx = func(x) # Evaluate the new mean? 
            # Wait, we already used budget for offspring. 
            # The task says "Every call is recorded". 
            # In (mu+lambda)-ES, we usually don't evaluate the parent/mean separately unless it's part of the selection.
            # Here, we generate lam offspring. We have already used lam calls.
            # The next iteration will generate lam more.
            # We do NOT need to evaluate x separately if we don't use it for selection in the next step.
            # But we need to update best_f if x is better.
            # However, x is a convex combination of offspring, so it might not have been evaluated.
            # To be safe and maximize early progress, we can skip evaluating x separately to save budget for more offspring.
            # The best_f is updated during offspring generation.
            
        return best_x, best_f
# EVOLVE-BLOCK-END
