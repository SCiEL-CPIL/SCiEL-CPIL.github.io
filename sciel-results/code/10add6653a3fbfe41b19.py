"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """CMA-ES with time guard."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280
        rng = np.random.default_rng(0)

        # Initial point and sigma
        x = np.zeros(self.dim)
        sigma = 4.0
        pop_size = max(4, int(4 + 3 * np.log(self.dim)))
        
        # CMA-ES parameters
        c_sigma = (pop_size + 2) / (self.dim + pop_size + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((pop_size - 1) / (self.dim + 1)) - 1) + c_sigma
        c_c = (4 + pop_size / self.dim) / (self.dim + 4 + 2 * pop_size / self.dim)
        c_1 = 2 / ((self.dim + 1.3) ** 2 + pop_size)
        c_mu = min(1 - c_1, 2 * ((pop_size / 2 + 1) - self.dim - 2) / ((self.dim + 2) ** 2 + pop_size))
        mu = pop_size // 2
        w = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
        w = w / np.sum(w)
        mu_eff = 1 / np.sum(w ** 2)
        c_1 += 2 * (c_mu - c_1)
        c_mu = min(c_mu, (1 - c_1) / 2)

        C = np.eye(self.dim)
        pc = np.zeros(self.dim)
        ps = np.zeros(self.dim)
        chi_n = np.sqrt(self.dim) * (1 - 1 / (4 * self.dim) + 1 / (21 * self.dim ** 2))

        # Precompute sqrt of C via Cholesky (approximate via SVD or just identity initially, 
        # but CMA-ES usually updates C and uses Cholesky or eigendecomposition for sampling.
        # For dim=5, Cholesky is cheap.
        try:
            D, B = np.linalg.eigh(C)
            sqrt_C = B @ np.diag(np.sqrt(np.maximum(D, 1e-12))) @ B.T
        except:
            sqrt_C = np.eye(self.dim)

        best_x = x
        best_f = func(best_x)
        
        # Initial population evaluation to warm up if possible? 
        # Standard CMA-ES starts with 1 eval for x0, then loops.
        # Budget is 500.
        # Loop until budget or time
        
        for _ in range(self.budget - 1):
            if time.time() > deadline:
                break
            
            # Sample
            # Generate m normal vectors
            z = rng.standard_normal((pop_size, self.dim))
            # Transform to step
            # y = x + sigma * z @ sqrt_C.T (or similar depending on convention)
            # Standard: y_i = x + sigma * B @ (D * z_i)
            # Let's use z @ sqrt_C.T for efficiency if sqrt_C is symmetric
            steps = z @ sqrt_C.T
            pop = x + sigma * steps
            
            # Evaluate
            fvals = np.array([func(np.clip(p, -5, 5)) for p in pop])
            
            # Sort
            idx = np.argsort(fvals)
            fvals_sorted = fvals[idx]
            pop_sorted = pop[idx]
            
            # Update best
            if fvals_sorted[0] < best_f:
                best_f = fvals_sorted[0]
                best_x = pop_sorted[0].copy()
                
            # Weighted mean
            x_new = np.sum(pop_sorted[:mu] * w[:, None], axis=0)
            
            # Update C
            y = (x_new - x) / sigma
            pc = (1 - c_c) * pc + c_c * y
            hsig = (np.sqrt((1 - (1 - c_c) ** (2 * (self.budget - 1))) / (1 - (1 - c_c) ** 2)) * chi_n < np.linalg.norm(ps) / np.sqrt(1 - (1 - c_c) ** 2)) if self.budget > 1 else False
            # Simplified hsig check, often just checking norm
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * y / np.linalg.norm(pc) if np.linalg.norm(pc) > 0 else (1 - c_sigma) * ps
            
            # Check termination criteria roughly
            if np.linalg.norm(y) < 1e-12:
                break
                
            # Update C
            # C = (1 - c_1 - c_mu) * C + c_1 * (pc @ pc.T + (1 - (1 - c_c)**2) * C) + c_mu * sum w_i (y_i @ y_i.T)
            # Actually standard update:
            # C = (1 - c_1 - c_mu) * C + c_1 * (pc @ pc.T + (1 - (1 - c_c)**2) * C) + c_mu * sum(w_i * (y_i - pc) @ (y_i - pc).T)
            
            y_diffs = pop_sorted[:mu] - x_new
            C = (1 - c_1 - c_mu) * C + c_1 * (pc @ pc.T + (1 - (1 - c_c) ** 2) * C) + c_mu * np.sum((y_diffs * w[:, None]) @ y_diffs.T, axis=0)
            
            # Symmetrize
            C = (C + C.T) / 2
            
            # Update sigma
            sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / chi_n - 1))
            
            # Update x
            x = x_new
            
            # Update sqrt_C
            try:
                D, B = np.linalg.eigh(C)
                sqrt_C = B @ np.diag(np.sqrt(np.maximum(D, 1e-12))) @ B.T
            except:
                sqrt_C = np.eye(self.dim)
                
            # If sigma becomes too small or large, might need to restart, but budget is small.
            if sigma < 1e-12:
                break

        return best_x, best_f
# EVOLVE-BLOCK-END
