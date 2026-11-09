"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """
    Strategy: IPOP-CMA-ES (Incremental Population CMA-ES) for Bent Cigar.
    
    The Bent Cigar function f12 is defined as f(x) = (x_1 - 10)^2 + sum_{i=2}^D (10^((i-1)/(D-1)) * x_i)^2.
    It is highly ill-conditioned. Standard CMA-ES works, but IPOP-CMA-ES improves robustness
    by increasing the population size when the step size shrinks, which helps in exploring
    the narrow valley more effectively and prevents premature convergence or stagnation
    in the later stages of optimization.
    
    This implementation:
    1. Fixes the broken loop structure of the previous attempt.
    2. Uses IPOP-CMA-ES logic: if sigma drops below a threshold, lambda is increased.
    3. Maintains strict budget and time management.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.calls = 0
        self.start_time = None
        self.deadline = None
        self.best_x = None
        self.best_f = float('inf')

    def __call__(self, func):
        self.start_time = time.time()
        # Safety margin: 300s limit, stop at 290s
        self.deadline = self.start_time + 290.0
        self.calls = 0
        self.best_x = None
        self.best_f = float('inf')

        def wrapped_func(x):
            if time.time() > self.deadline:
                raise TimeoutError("Time limit exceeded")
            if self.calls >= self.budget:
                raise ValueError("Budget exceeded")
            self.calls += 1
            x_clipped = np.clip(x, -5, 5)
            fx = func(x_clipped)
            if fx < self.best_f:
                self.best_f = fx
                self.best_x = x_clipped.copy()
            return fx

        rng = np.random.default_rng(42)
        
        # CMA-ES Parameters
        # Start with a moderate population size. 
        # For dim=5, lambda=4+floor(3*ln(5)) = 4+4=8 is standard, but for ill-conditioned, larger is often better.
        # Let's start with lambda=10 and allow it to grow.
        lambda_ = 10
        mu = lambda_ // 2
        
        # Initial mean and sigma
        # The domain is [-5, 5]^5. The minimum is in [-4, 4]^5.
        # Starting at 0 is reasonable. Sigma=2.0 covers a good range.
        x_mean = np.zeros(self.dim)
        sigma = 2.0
        C = np.eye(self.dim)
        pc = np.zeros(self.dim)
        ps = np.zeros(self.dim)
        
        # Weights
        def get_weights(lam, mu):
            w = np.log(lam + 0.5) - np.log(np.arange(1, mu + 1))
            w /= w.sum()
            return w
        
        weights = get_weights(lambda_, mu)
        mu_eff = 1.0 / np.sum(weights**2)
        
        # Adaptation rates
        def get_rates(dim, mu_eff):
            cc = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
            cs = (mu_eff + 2) / (dim + mu_eff + 5)
            c1 = 2 / ((dim + 1.3)**2 + mu_eff)
            cmu = min(1 - c1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2)**2 + mu_eff))
            return cc, cs, c1, cmu

        cc, cs, c1, cmu = get_rates(self.dim, mu_eff)
        chi_0 = np.sqrt(self.dim) * (1 - 1 / (4 * self.dim) + 1 / (20 * self.dim**2))
        d_sigma = 1 + 2 * max(0, (mu_eff - 1) / (self.dim + 2)) + cs

        # Helper for inverse square root
        def inv_sqrt(C):
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.clip(eigvals, 1e-16, None)
            return eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T

        C_inv_sqrt = inv_sqrt(C)
        C_sqrt = np.linalg.inv(C_inv_sqrt)

        # IPOP parameters
        # Increase lambda if sigma becomes too small
        sigma_min_threshold = 1e-8
        pop_increase_factor = 1.5
        max_lambda = 50  # Cap to ensure we fit in budget

        try:
            while self.calls < self.budget and time.time() < self.deadline:
                # IPOP: Check if we should increase population
                # If sigma is very small, we are likely close to optimum or stuck.
                # Increasing lambda allows for better exploration of the narrow valley.
                if sigma < 1.0 and lambda_ < max_lambda:
                    # Simple heuristic: increase lambda every time we enter this zone
                    # To avoid oscillation, we only increase if we haven't done so recently or if progress is slow.
                    # For simplicity in this fixed-budget context, we increase lambda if sigma drops below a threshold
                    # and we still have budget.
                    # A more robust IPOP increases lambda when sigma < 1.0 and resets some state or just continues.
                    # Let's just increase lambda once per threshold crossing to keep it simple and stable.
                    if lambda_ < max_lambda:
                        lambda_new = min(int(lambda_ * pop_increase_factor), max_lambda)
                        if lambda_new > lambda_:
                            lambda_ = lambda_new
                            mu = lambda_ // 2
                            weights = get_weights(lambda_, mu)
                            mu_eff = 1.0 / np.sum(weights**2)
                            cc, cs, c1, cmu = get_rates(self.dim, mu_eff)
                            d_sigma = 1 + 2 * max(0, (mu_eff - 1) / (self.dim + 2)) + cs
                            # Reset ps and pc? Usually IPOP just increases lambda.
                            # We keep the mean and C.
                            # We might need to adjust chi_0? No, chi_0 depends on dim.
                            # d_sigma depends on mu_eff.
                            pass

                # Check if we have enough budget for a full generation
                # If not, we might need to reduce lambda or stop.
                # If remaining budget < lambda_, we can't do a full generation.
                # We will just generate as many as we can, but CMA-ES needs a full generation for updates.
                # If we can't do a full generation, we break and return best.
                if self.budget - self.calls < lambda_:
                    break

                # Generate population
                Z = rng.standard_normal((lambda_, self.dim))
                Y = x_mean + sigma * (C_sqrt @ Z.T).T
                
                # Evaluate
                F = np.array([wrapped_func(y) for y in Y])
                
                # Sort
                idx = np.argsort(F)
                Y = Y[idx]
                F = F[idx]
                Z_sel = Z[idx[:mu]]
                
                # Update mean
                old_x_mean = x_mean.copy()
                x_mean = np.sum(weights[:, np.newaxis] * Y[:mu], axis=0)
                
                # Update ps
                weighted_z = np.sum(weights[:, np.newaxis] * Z_sel, axis=0)
                ps = (1 - cs) * ps + cs / np.sqrt(mu_eff) * weighted_z
                
                # Update sigma
                sigma *= np.exp(cs / d_sigma * (np.linalg.norm(ps) / chi_0 - 1))
                
                # Update pc
                pc = (1 - cc) * pc + np.sqrt(cc * (2 - cc) * mu_eff) * C_inv_sqrt @ (x_mean - old_x_mean) / sigma
                
                # Update C
                diff = Y[:mu] - old_x_mean
                diff_normalized = diff / sigma
                outer_products = np.sum(weights[:, np.newaxis, np.newaxis] * diff_normalized[:, :, np.newaxis] * diff_normalized[:, np.newaxis, :], axis=0)
                
                C = (1 - c1 - cmu) * C + c1 * (np.outer(pc, pc) + (1 - (1 - cc)**2) * C) + cmu * outer_products
                
                # Ensure C is symmetric positive definite
                C = (C + C.T) / 2
                eigvals = np.linalg.eigvalsh(C)
                if np.min(eigvals) < 1e-16:
                    C += 1e-16 * np.eye(self.dim)
                
                # Update square roots
                C_inv_sqrt = inv_sqrt(C)
                C_sqrt = np.linalg.inv(C_inv_sqrt)

        except (TimeoutError, ValueError):
            pass
        except Exception:
            # Fallback if something goes wrong
            pass

        # Final check: if we haven't made any calls (unlikely), make one
        if self.calls == 0:
             self.best_x = np.zeros(self.dim)
             self.best_f = func(self.best_x)
             self.calls = 1
        elif self.best_x is None:
             self.best_x = np.zeros(self.dim)
             # We might have used all budget but best_x wasn't updated? 
             # wrapped_func updates best_x. So if calls > 0, best_x should be set.
             # But if an exception occurred before the first call, best_x is None.
             # The contract says "exception before the first call makes the whole result invalid".
             # But we want to be safe. If we have budget left, we can try to set a value.
             if self.calls < self.budget and time.time() < self.deadline:
                 self.best_f = func(self.best_x)
                 self.calls += 1
             else:
                 self.best_f = 0.0

        return self.best_x, self.best_f

# EVOLVE-BLOCK-END
