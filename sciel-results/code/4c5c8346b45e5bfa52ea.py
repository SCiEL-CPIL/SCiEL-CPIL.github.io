"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES with a wall-clock deadline guard."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start = time.perf_counter()
        deadline = start + 285.0          # safety margin well under the 330 s kill
        rng = np.random.default_rng(42)
        budget = self.budget
        dim = self.dim

        # CMA-ES parameters
        # Population size lambda
        lam = 4 + int(3 * np.log(dim))
        # Weights for recombination
        mu = lam // 2
        weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mueff = 1.0 / np.sum(weights**2)
        
        # Step size control parameters
        cs = (mueff + 2) / (dim + mueff + 5)
        ds = 1 + 2 * max(0, np.sqrt((mueff - 1) / (dim + 1)) - 1) + cs
        c1 = (2 / ((dim + 1.3) ** 2 + mueff)) * (1 - cs) + 2 / ((dim + 2) ** 2 + mueff)
        cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((dim + 2) ** 2 + mueff))
        chiN = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))

        # Initialize
        # Start at 0,0,0,0,0 as it's often a good heuristic for these types of functions
        # or a random point. Let's try 0 first.
        x_mean = np.zeros(dim)
        sigma = 2.0  # Initial step size
        B = np.eye(dim)
        D = np.ones(dim)
        pc = np.zeros(dim)
        ps = np.zeros(dim)
        C = np.eye(dim)
        
        # Pre-compute for speed
        # We will use a simple loop for CMA-ES steps
        
        global_best_x = x_mean.copy()
        global_best_f = func(global_best_x)
        calls = 1

        while calls < budget:
            if time.perf_counter() > deadline:
                break
            
            # Generate population
            # y = B * D * z
            Z = rng.standard_normal((lam, dim))
            Y = np.einsum('ij,j->ij', np.einsum('ij,jk->ik', B, D), Z)
            P = x_mean + sigma * Y
            
            # Evaluate
            F = np.empty(lam)
            for i in range(lam):
                if time.perf_counter() > deadline:
                    # Fill rest with inf to break loop naturally or handle carefully
                    # Actually, if we hit deadline, we should break out of the main loop
                    # but we are inside the generation loop. 
                    # Let's just break and let the outer check catch it? 
                    # No, we need to finish the current generation to update CMA state correctly 
                    # or just stop. The task says "exception later keeps calls made".
                    # But we want to avoid exceptions. 
                    # If we break here, we might have partial F. 
                    # It's safer to just let it run if it's fast, but 330s is long.
                    # 500 calls is very small. CMA-ES overhead is minimal.
                    pass
                x_clip = np.clip(P[i], -5, 5)
                F[i] = func(x_clip)
                calls += 1
                
            if time.perf_counter() > deadline:
                break

            # Sort by fitness
            order = np.argsort(F)
            # Update best
            if F[order[0]] < global_best_f:
                global_best_f = F[order[0]]
                global_best_x = np.clip(P[order[0]], -5, 5).copy()

            # Reuse population for next generation (standard CMA-ES)
            Y = Y[order]
            P = P[order]
            F = F[order]
            
            # Update mean
            x_mean_new = np.sum(weights[:, None] * P[:mu], axis=0)
            
            # Update step size (CSA)
            # pc = (1-cs)*pc + cs/sigma * (sqrt(mueff)*B*D * y_w - x_mean)
            # y_w is the weighted mean of the selected Ys
            y_w = np.sum(weights[:, None] * Y[:mu], axis=0)
            # Note: B and D are the current ones. 
            # We need B*D applied to y_w? 
            # Standard formula: pc = (1-cs)*pc + cs * (sqrt(mueff)/sigma) * (B*D*y_w - x_mean)
            # Wait, the mean update uses P. The pc update uses the step in the z-space transformed by B*D.
            # Let's stick to the standard implementation details.
            
            # Actually, simpler:
            # pc = (1-cs)*pc + cs/sigma * (sqrt(mueff) * (B @ (D * y_w)) - x_mean)
            # Note: B @ (D * y_w) is the step in x-space corresponding to the weighted mean of z's.
            
            pc = (1 - cs) * pc + (cs / sigma) * (np.sqrt(mueff) * (B @ (D * y_w)) - x_mean)
            
            # ps update
            norm_pc = np.linalg.norm(pc)
            hsig = (norm_pc / np.sqrt(1 - (1 - cs) ** (2 * (calls + 1) / (dim * cs))) < (1.4 + 2 / (dim + 1)) * chiN)
            ps = (1 - ds) * ps + hsig * np.sqrt(ds * (2 - ds) * cs) / sigma * (B @ (D * y_w) - x_mean)
            
            # Update Covariance Matrix
            # C = (1 + c1 * hsig * (2 - c1) - c1 - cmu) * C + c1 * pc * pc.T + cmu * sum(weights * (y_i - x_mean) * (y_i - x_mean).T)
            # Wait, the formula is usually on the z-space or x-space?
            # C is in z-space? No, C is the covariance of the search distribution in the x-space?
            # No, CMA-ES usually maintains C in the z-space (standard normal space) and transforms it.
            # Or it maintains C in x-space?
            # Standard: C is the covariance matrix of the distribution N(x_mean, sigma^2 * C).
            # So C is in the same space as x.
            
            # Correction: 
            # The update of C uses the differences in the x-space (or the transformed z-space).
            # Let y_i be the z-vector. The x-vector is x_mean + sigma * B @ (D * y_i).
            # The difference is sigma * B @ (D * (y_i - y_w)).
            # So the outer product is sigma^2 * (B @ D * (y_i - y_w)) @ (B @ D * (y_i - y_w)).T
            # This is expensive. 
            # Standard efficient implementation:
            # C = (1 + c1*hsig*(2-c1) - c1 - cmu)*C + c1*pc*pc.T + cmu/sigma^2 * sum(weights * (x_i - x_mean)*(x_i - x_mean).T)
            # Wait, the term is usually:
            # C_new = (1 + c1 * hsig * (2 - c1) - c1 - cmu) * C + c1 * pc * pc.T + cmu * sum( w_i * (y_i - y_w) * (y_i - y_w).T )
            # Where y_i are the *selected* z-vectors? 
            # Let's check the formula.
            # C <- (1 + c1 hsig (2 - c1) - c1 - cmu) C + c1 pc pc^T + cmu sum_{i=1}^mu w_i (z_i - z_w)(z_i - z_w)^T
            # Here z_i are the z-vectors of the selected parents.
            
            # Let's use this formula.
            Z_selected = Y[:mu]
            Z_w = y_w
            # Calculate the sum of outer products
            # diff = Z_selected - Z_w
            # sum_outer = sum(weights * diff[:, i] * diff[:, j])
            # This can be done as:
            # W_sqrt = np.sqrt(weights)
            # Diff_scaled = (Z_selected - Z_w) * W_sqrt[:, None]
            # Sum_outer = Diff_scaled.T @ Diff_scaled
            
            W_sqrt = np.sqrt(weights)
            Diff_scaled = (Z_selected - Z_w) * W_sqrt[:, None]
            Sum_outer = Diff_scaled.T @ Diff_scaled
            
            # hsig is boolean, treat as 1.0 or 0.0
            hsig_val = 1.0 if hsig else 0.0
            c1_term = c1 * hsig_val * (2 - c1)
            
            C = (1 + c1_term - c1 - cmu) * C + c1 * np.outer(pc, pc) + cmu * Sum_outer
            
            # Update B and D (eigendecomposition)
            # C is symmetric
            eigvals, eigvecs = np.linalg.eigh(C)
            # Ensure positive eigenvalues
            eigvals = np.maximum(eigvals, 1e-12)
            D = np.sqrt(eigvals)
            B = eigvecs
            
            x_mean = x_mean_new
            
            # Optional: restart if sigma is too large or small
            if sigma > 1e6 or sigma < 1e-12:
                break

        return global_best_x, global_best_f
# EVOLVE-BLOCK-END
