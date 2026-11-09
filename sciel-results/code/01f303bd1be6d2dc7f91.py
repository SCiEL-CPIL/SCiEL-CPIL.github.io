import numpy as np

class Optimizer:
    """
    CMA-ES (Covariance Matrix Adaptation Evolution Strategy) optimizer.
    
    This algorithm is robust for non-separable functions like the Sharp Ridge (BBOB f13).
    It adapts the search distribution to the local landscape geometry, specifically
    aligning with the ridge direction to escape local plateaus and converge faster
    than isotropic methods like (1+1)-ES.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        # Standard CMA-ES parameters
        D = self.dim
        budget = self.budget
        
        # Population size
        lambda_ = 4 + int(3 * np.log(D))
        
        # Selection size
        mu = lambda_ // 2
        
        # Weights for recombination
        weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        
        # Variance-effective population size
        mueff = 1 / np.sum(weights ** 2)
        
        # Initial step size
        sigma = 2.0  # Start with a relatively large step size to explore
        
        # Initial mean (start at origin or random? Origin is often good if bounds are symmetric)
        # The problem states minimizer is in [-4, 4]^5. Starting at 0 is a safe bet.
        mean = np.zeros(D)
        
        # Initial covariance matrix
        C = np.eye(D)
        Csqrt = np.eye(D)  # Cholesky factor or square root, we use identity for simplicity initially
        # Actually, we need to maintain C and its square root for sampling.
        # For D=5, we can just use Cholesky decomposition every step or maintain it.
        
        # Cumulation path for step size control
        ps = np.zeros(D)
        
        # Cumulation path for covariance adaptation
        pc = np.zeros(D)
        
        # Learning rates
        cs = (mueff + 2) / (D + mueff + 5)
        ds = 1 + 2 * max(0, np.sqrt((mueff - 1) / (D + 1)) - 1) + cs
        c1 = (2 + 2 * (mueff - 2) / (D + 1) ** 2) / ((D + 2) ** 2 + mueff)
        cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((D + 2) ** 2 + mueff))
        
        # Damping for covariance update
        # chiN = expected norm of standard normal
        chiN = np.sqrt(D) * (1 - 1 / (4 * D) + 1 / (21 * D ** 2))
        
        # Evaluation counter
        count = 0
        
        # Initial evaluation
        x0 = np.clip(mean, -5, 5)
        f0 = func(x0)
        count += 1
        
        # Keep track of best solution found so far
        best_x = x0.copy()
        best_f = f0
        
        while count < budget:
            # Determine how many candidates we can generate in this generation
            # We can generate up to lambda_ candidates, but we might have leftover budget
            # For simplicity and robustness, we just generate lambda_ as long as we have budget.
            # If budget < lambda_, we stop or generate fewer.
            if budget - count < lambda_:
                # Generate only as many as we can afford
                num_gen = budget - count
            else:
                num_gen = lambda_
            
            if num_gen == 0:
                break
                
            # Sample candidates
            # We use the current mean, sigma, and C to generate points
            # y = C^{1/2} * z
            # Since D is small (5), we can compute the square root of C via eigen decomposition
            # or just use Cholesky if C is positive definite.
            # To ensure stability, we update C carefully.
            
            # Get square root of C
            # Using eigendecomposition for stability in small D
            eigvals, eigvecs = np.linalg.eigh(C)
            # Clip small negative eigenvalues to 0 due to numerical errors
            eigvals = np.clip(eigvals, 1e-20, None)
            Csqrt = eigvecs @ np.diag(np.sqrt(eigvals))
            
            # Generate random normals
            Z = np.random.randn(num_gen, D)
            
            # Transform to search space
            # X = mean + sigma * Z @ Csqrt.T  (since Z is (num_gen, D), Csqrt is (D, D))
            X = mean + sigma * (Z @ Csqrt.T)
            
            # Clip to bounds
            X = np.clip(X, -5, 5)
            
            # Evaluate
            F = np.array([func(x) for x in X])
            count += num_gen
            
            # Update best
            if np.min(F) < best_f:
                idx_min = np.argmin(F)
                best_f = F[idx_min]
                best_x = X[idx_min].copy()
            
            # If we didn't use full population, we need to handle selection carefully.
            # Standard CMA-ES assumes full population. If num_gen < lambda_, we sort the available ones.
            # For simplicity, if we don't have a full population, we just use the available ones for update.
            # However, the weights are defined for mu out of lambda.
            # If num_gen < lambda_, we should probably just stop or adjust.
            # Given budget 500 and lambda 8, we will have full populations for 62 gens.
            # 62 * 8 = 496. 4 evals left.
            # The last partial generation might mess up CMA-ES updates.
            # Strategy: If remaining budget < lambda_, just do random search or (1+1) for the remainder?
            # Or just evaluate and don't update CMA-ES parameters, just track best.
            
            if num_gen < lambda_:
                # Don't update CMA-ES parameters with incomplete population
                # Just finish
                continue
                
            # Sort indices by fitness
            sorted_indices = np.argsort(F)
            # Take top mu
            selected_indices = sorted_indices[:mu]
            selected_X = X[selected_indices]
            selected_F = F[selected_indices]
            
            # New mean
            new_mean = np.sum(weights[:, None] * selected_X, axis=0)
            
            # Step vector (difference between new mean and old mean)
            step = (new_mean - mean) / sigma
            
            # Update step size control path ps
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) / np.linalg.norm(Csqrt) * Csqrt.T @ step
            
            # Check if ps is within the confidence interval
            if np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** (2 * (count - budget // lambda_))) < (1.4 + 2 / (D + 1)) * chiN:
                # Update sigma
                # Heuristic for sigma update
                # Standard formula: sigma = sigma * exp(cs/ds * (||ps||/chiN - 1))
                # But we need to handle the case where we are near the boundary or stuck.
                # Let's use the standard update.
                sigma = sigma * np.exp(cs / ds * (np.linalg.norm(ps) / chiN - 1))
                
                # Update covariance matrix path pc
                pc = (1 - cs) * pc + np.sqrt(cs * (2 - cs) * mueff) * step
                
                # Update covariance matrix C
                # C = (1 - mu1 - cmu) * C + mu1 * pc @ pc.T + cmu * sum wi * (yi @ yi.T)
                # where yi = (xi - mean_old) / sigma
                Y = selected_X - mean
                Y = Y / sigma  # Standardized steps
                
                # Rank-one update
                C = (1 - c1 - cmu) * C + c1 * (pc @ pc.T + (np.linalg.norm(pc) ** 2 > (1.4 + 2 / (D + 1)) ** 2 * chiN ** 2) * pc @ pc.T)
                
                # Rank-mu update
                C += cmu * np.sum(weights[:, None, None] * (Y * Y.T), axis=0)
                
                # Ensure C is symmetric and positive definite
                C = (C + C.T) / 2
                # Regularize if necessary
                eigvals, eigvecs = np.linalg.eigh(C)
                if np.any(eigvals < 1e-20):
                    C = eigvecs @ np.diag(np.clip(eigvals, 1e-20, None)) @ eigvecs.T
            
            # Update mean
            mean = new_mean
            
            # Optional: If mean goes out of bounds, project it back?
            # The task says func clips. So mean can be anywhere, but generated points are clipped.
            # However, if mean is far out, all points might be clipped to the same boundary point.
            # To avoid this, we can gently pull mean back towards center if it's outside.
            # But standard CMA-ES relies on the clipping in the function.
            # Let's just ensure sigma doesn't grow too large if we are stuck at a boundary.
            # A simple heuristic: if many points are on the boundary, reduce sigma.
            # For now, rely on the standard update.

        return best_x, best_f
