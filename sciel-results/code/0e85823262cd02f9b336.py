import numpy as np
import time

class Optimizer:
    """CMA-ES Optimizer for continuous black-box optimization."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.boundaries = np.array([-5.0, 5.0])

    def _reflect(self, x):
        """Reflect points outside the boundaries back into the box."""
        # x is a 1D array
        # If x < lower, reflect: lower + (lower - x) = 2*lower - x
        # If x > upper, reflect: upper - (x - upper) = 2*upper - x
        lower = self.boundaries[0]
        upper = self.boundaries[1]
        
        # Handle multiple reflections if step size is huge, but usually one reflection is enough
        # For robustness, we can loop, but typically one pass is fine for reasonable sigma
        while np.any(x < lower) or np.any(x > upper):
            x = np.where(x < lower, 2 * lower - x, x)
            x = np.where(x > upper, 2 * upper - x, x)
        return x

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        
        # CMA-ES Parameters
        # Population size
        lam = 4 + int(3 * np.log(dim))
        if lam < 4: lam = 4
        if lam > 2 * dim: lam = 2 * dim
        
        # Weights for the top mu individuals
        mu = lam // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mueff = 1 / np.sum(weights**2)
        
        # Learning rates
        c_sigma = (mueff + 2) / (dim + mueff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mueff - 1) / (dim + 1)) - 1) + c_sigma
        c_c = (4 + mueff / dim) / (dim + 4 + 2 * mueff / dim)
        c_1 = 2 / ((dim + 1.3)**2 + mueff)
        c_mu = min(1 - c_1, 2 * (mueff - 2 + 1 / mueff) / ((dim + 2)**2 + mueff))
        
        # Initial values
        x_mean = np.zeros(dim)
        sigma = 1.0
        C = np.eye(dim)
        pc = np.zeros(dim)
        ps = np.zeros(dim)
        
        # Pre-compute constants
        chi_n = np.sqrt(dim) * (1 - 1/(4*dim) + 1/(21*dim**2))
        
        best_x = x_mean.copy()
        best_f = func(best_x)
        
        start_time = time.time()
        time_limit = 290.0 # Safety margin before 300s limit
        
        calls_made = 1
        
        for gen in range(budget // lam + 1):
            if calls_made >= budget:
                break
            
            # Check time
            if time.time() - start_time > time_limit:
                break
            
            # Sample points
            # Standard normal samples
            # We need to transform standard normal to N(0, sigma^2 C)
            # z ~ N(0, I) -> y = sigma * C^0.5 @ z
            # We can use Cholesky decomposition of C
            try:
                L = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                # If C is not positive definite, re-initialize or fix
                C = np.eye(dim)
                L = np.eye(dim)
            
            # Generate lam samples
            z = np.random.randn(lam, dim)
            # Transform to C space
            # y = z @ L.T (since L is lower triangular, C = L L^T)
            ys = z @ L.T
            # Scale by sigma
            ys *= sigma
            # Add mean
            xs = x_mean[None, :] + ys
            
            # Reflect points outside boundaries
            for i in range(lam):
                xs[i] = self._reflect(xs[i])
            
            # Evaluate
            fs = np.zeros(lam)
            for i in range(lam):
                fs[i] = func(xs[i])
                calls_made += 1
                if calls_made >= budget:
                    break
            
            if calls_made >= budget:
                # If we ran out of budget mid-generation, update best and break
                # Note: We might have evaluated fewer than lam points in this gen
                # We still need to update the state with the available points if we want to be consistent,
                # but since we break, we just return best.
                # However, to be precise, if we broke early, we should probably not update the CMA state 
                # with incomplete data, or just stop.
                # The task says "Every call is recorded".
                # Let's just break out.
                pass

            # Sort by fitness
            order = np.argsort(fs)
            xs_sorted = xs[order]
            fs_sorted = fs[order]
            
            # Update best
            if fs_sorted[0] < best_f:
                best_f = fs_sorted[0]
                best_x = xs_sorted[0].copy()
            
            # Update mean
            x_mean_new = np.zeros(dim)
            for i in range(mu):
                x_mean_new += weights[i] * xs_sorted[i]
            x_mean = x_mean_new
            
            # Update step size
            # ps = (1 - c_sigma) ps + c_sigma * sqrt(mueff) / sigma * (x_mean - x_mean_old)
            # Wait, standard formula uses the step direction in the C-space?
            # No, ps is in the original space? 
            # Actually, ps is updated with the normalized step.
            # Let's stick to the standard implementation.
            # The step taken is (x_mean_new - x_mean_old)
            # But we need to be careful with the definition.
            
            # Standard CMA-ES update:
            # 1. x_mean_new = sum w_i x_(i)
            # 2. ps = (1-c_sigma)ps + h_sigma * c_sigma * sqrt(mueff) / sigma * (x_mean_new - x_mean_old)
            #    where h_sigma is a hysteresis operator based on ||ps||/chi_n
            # 3. sigma = sigma * exp(c_sigma/d_sigma * (||ps||/chi_n - 1))
            # 4. pc = (1-c_c)pc + h_c * c_c * sqrt(mueff) * (x_mean_new - x_mean_old)/sigma
            #    where h_c is based on ||pc||/sqrt(1-(1-c_c)^(2*(gen+1)))
            # 5. C = (1 - mu_1 - mu_mu)C + mu_1 * pc pc^T + mu_mu * sum w_i y_(i) y_(i)^T
            #    where y_(i) = (x_(i) - x_mean_old)/sigma
            
            # Let's implement this carefully.
            
            # We need x_mean_old
            # We didn't store it explicitly in the loop above, but x_mean was the old one.
            # Let's rename variables for clarity in the update step.
            # We have x_mean (old), x_mean_new (calculated)
            
            # Calculate step
            step = x_mean_new - x_mean
            
            # Update ps
            ps = (1 - c_sigma) * ps + c_sigma * np.sqrt(mueff) / sigma * step
            # Hysteresis operator h_sigma
            h_sigma = (np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma)**(2 * (gen + 1))) < (1.4 + 2 / (dim + 1)) * chi_n)
            # Update sigma
            sigma = sigma * np.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / chi_n - 1))
            
            # Update pc
            pc = (1 - c_c) * pc + h_sigma * c_c * np.sqrt(mueff) * step / sigma
            # Hysteresis operator h_c
            # Note: The condition for h_c is slightly different in some implementations, often tied to ps or just a norm check.
            # Standard: h_c = (||pc|| / sqrt(1 - (1 - c_c)^(2*(gen+1)))) < (1.4 + 2/(dim+1)) * sqrt((mueff+2)/dim - 1) * chi_n
            # Let's use the standard one.
            h_c = (np.linalg.norm(pc) / np.sqrt(1 - (1 - c_c)**(2 * (gen + 1))) < (1.4 + 2 / (dim + 1)) * np.sqrt((mueff + 2) / dim - 1) * chi_n)
            
            # Update C
            # y_i = (x_(i) - x_mean_old) / sigma
            # We have xs_sorted and x_mean (old)
            ys_sorted = (xs_sorted[:mu] - x_mean[None, :]) / sigma
            
            # Rank-1 update
            C_rank1 = np.outer(pc, pc)
            
            # Rank-mu update
            C_rank_mu = np.zeros((dim, dim))
            for i in range(mu):
                C_rank_mu += weights[i] * np.outer(ys_sorted[i], ys_sorted[i])
            
            # Update C
            C = (1 - c_1 - c_mu) * C + c_1 * C_rank1 + c_mu * C_rank_mu
            
            # Ensure C is symmetric
            C = (C + C.T) / 2
            
            # Check for termination
            if sigma < 1e-12:
                break
            if np.linalg.norm(C, ord='fro') < 1e-12:
                break
            
            # Check if we have converged (step size very small)
            if sigma < 1e-8:
                break

        return best_x, best_f
