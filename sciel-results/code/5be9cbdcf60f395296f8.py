"""Rewritten optimizer: CMA-ES style with adaptive step-size and time guard."""
import numpy as np
import time


class Optimizer:
    """CMA-ES variant tuned for 5-dim Rosenbrock with hidden affine transforms."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # safety margin before 300s kill

        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        # Initial point: try center first, then a few random starts to pick best
        candidates = [np.zeros(dim)]
        for _ in range(5):
            candidates.append(rng.uniform(-5, 5, dim))
        
        best_x = None
        best_f = np.inf
        for c in candidates:
            fc = func(c)
            if fc < best_f:
                best_f = fc
                best_x = c.copy()
        
        # If we've used too many calls already, return
        calls_made = 6
        if calls_made >= budget:
            return best_x, best_f

        # CMA-ES parameters
        lam = 4  # population size
        # Initial sigma: scale to cover the box reasonably
        sigma = 2.0
        x = best_x.copy()
        fx = best_f
        
        # Covariance matrix (identity initially)
        C = np.eye(dim)
        # Eigen decomposition for sampling
        eigvals, eigvecs = np.linalg.eigh(C)
        # Ensure positive eigenvalues
        eigvals = np.maximum(eigvals, 1e-12)
        
        # Step size control parameters
        sigma = 2.0
        # Success history for sigma adaptation
        successes = []
        
        # Track best
        global_best_x = best_x.copy()
        global_best_f = best_f
        
        for gen in range(budget - calls_made):
            # Check time
            if time.time() > deadline:
                break
            
            # Generate population
            offspring = []
            offspring_f = []
            
            for i in range(lam):
                # Sample from N(0, C)
                z = rng.standard_normal(dim)
                # Transform: eigvecs @ diag(sqrt(eigvals)) @ z
                y = eigvecs @ (np.sqrt(eigvals) * z)
                child = x + sigma * y
                child = np.clip(child, -5, 5)
                fc = func(child)
                offspring.append(child)
                offspring_f.append(fc)
                calls_made += 1
                
                # Update global best
                if fc < global_best_f:
                    global_best_f = fc
                    global_best_x = child.copy()
            
            if calls_made >= budget:
                break
            
            # Sort offspring by fitness
            indices = np.argsort(offspring_f)
            sorted_offspring = [offspring[i] for i in indices]
            sorted_f = [offspring_f[i] for i in indices]
            
            # Update mean (x)
            # Weighted recombination: top mu individuals
            mu = lam // 2
            weights = np.log(mu + 1) - np.log(1 + np.arange(mu))
            weights /= np.sum(weights)
            
            new_x = np.zeros(dim)
            for i in range(mu):
                new_x += weights[i] * sorted_offspring[i]
            
            # Check if best offspring improved current x
            improved = sorted_f[0] < fx
            
            # Update covariance matrix
            # C <- (1 - c1 - cmu) * C + c1 * (y*y^T) + cmu * sum wi*(yi - ybar)(yi - ybar)^T
            y_vec = new_x - x  # step vector (before sigma scaling in the update)
            # Actually, the step is sigma * C^0.5 * z, so the actual step is:
            # step = new_x - x
            # We need to update C such that the distribution matches
            
            c1 = 2 / ((dim + 1)**2 + 10)
            cmu = min(1 - c1, 2 * (mu - 2 + 1/mu) / ((dim + 2)**2 + 10))
            
            # Compute correction for sigma
            # The expected norm of the step
            # For now, use a simpler adaptation
            
            # Update C
            C = (1 - c1 - cmu) * C
            C += c1 * np.outer(y_vec, y_vec) / (sigma ** 2)
            
            # Weighted sum of deviations
            ybar = np.zeros(dim)
            for i in range(mu):
                ybar += weights[i] * (sorted_offspring[i] - x)
            
            for i in range(mu):
                diff = (sorted_offspring[i] - x) - ybar
                C += cmu * weights[i] * np.outer(diff, diff) / (sigma ** 2)
            
            # Symmetrize
            C = (C + C.T) / 2
            
            # Update eigen decomposition
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.maximum(eigvals, 1e-12)
            
            # Adapt sigma
            successes.append(1 if improved else 0)
            if len(successes) > 10:
                successes = successes[-10:]
            success_rate = np.mean(successes)
            if success_rate > 0.5:
                sigma *= 1.1
            else:
                sigma *= 0.95
            sigma = np.clip(sigma, 0.01, 5.0)
            
            # Update mean
            x = new_x
            fx = sorted_f[0]
            
            # If no improvement for a while, do a random restart from best
            # (simple stagnation handling)
            
            # Check time again
            if time.time() > deadline:
                break
        
        return global_best_x, global_best_f
