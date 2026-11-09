"""CMA-ES Optimizer for BBOB f8 (Rosenbrock)."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """CMA-ES Optimizer for continuous black-box functions."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.best_x = None
        self.best_f = np.inf
        self.calls_made = 0
        self.start_time = 0.0
        self.deadline = 0.0

    def __call__(self, func):
        self.start_time = time.time()
        # Use 290s limit to be safe within 300s
        self.deadline = self.start_time + 290.0

        def wrapped_func(x):
            if self.calls_made >= self.budget:
                raise StopIteration("Budget exhausted")
            if time.time() > self.deadline:
                raise StopIteration("Time limit exceeded")

            x = np.clip(x, -5, 5)
            self.calls_made += 1
            val = func(x)
            
            if val < self.best_f:
                self.best_f = val
                self.best_x = x.copy()
            
            return val

        # CMA-ES Parameters
        dim = self.dim
        # Initial population size
        lambda_ = max(4, int(4 + 3 * np.log(dim)))
        
        # Initial step size
        # For Rosenbrock, starting with a moderate sigma is good.
        # The search space is [-5, 5].
        sigma = 1.0
        
        # Initial mean
        # Start at 0, which is a common good start for BBOB shifted functions if not too far.
        # However, the minimum is in [-4, 4]. 0 is a safe bet.
        mean = np.zeros(dim)
        
        # Covariance matrix (identity)
        C = np.eye(dim)
        
        # Eigenvalues and eigenvectors of C
        eigvals, eigvecs = np.linalg.eigh(C)
        
        # Adaptation parameters
        chi_n = np.sqrt(dim) * (1 - 1/(4*dim) + 1/(21*dim**2))
        cs = (np.sqrt((2*dim + 1/3)) + 1) / (dim + np.sqrt(2) + 3)
        ds = 1 + 2*max(0, np.sqrt((cs - 1)/(cs + 1)) - 1) + cs
        c1 = 2 / ((dim + 1.3)**2 + lambda_)
        cmu = min(1 - c1, 2 * ((lambda_//2 + 1) - 2 - c1) / ((dim + 2)**2 + lambda_))
        cc = (4 + dim/4) / (dim + 4 + 2*lambda_/dim)
        
        # State variables
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        
        # Budget allocation: Use almost all budget for CMA-ES
        # CMA-ES is very efficient.
        
        try:
            while self.calls_made < self.budget and time.time() < self.deadline:
                
                # Generate population
                Y = []
                F = []
                
                for i in range(lambda_):
                    # Sample from N(mean, sigma^2 * C)
                    # y ~ N(0, I)
                    z = np.random.normal(0, 1, dim)
                    # Transform to normal with covariance C
                    # y = eigvecs @ (eigvals**0.5 * z)
                    y = eigvecs @ (np.sqrt(eigvals) * z)
                    
                    x = mean + sigma * y
                    x = np.clip(x, -5, 5)
                    
                    try:
                        f_val = wrapped_func(x)
                    except StopIteration:
                        break
                    
                    Y.append(x)
                    F.append(f_val)
                
                if not Y:
                    break
                    
                Y = np.array(Y)
                F = np.array(F)
                
                # Sort by fitness
                indices = np.argsort(F)
                Y = Y[indices]
                F = F[indices]
                
                # Update best
                if F[0] < self.best_f:
                    self.best_f = F[0]
                    self.best_x = Y[0].copy()
                
                # Update mean (weighted recombination)
                # Weights: log-weighted
                mu = lambda_ // 2
                weights = np.log(np.arange(mu, 0, -1))
                weights /= np.sum(weights)
                mu_eff = 1 / np.sum(weights**2)
                
                new_mean = np.sum(weights[:, None] * Y[:mu], axis=0)
                
                # Update path evolution strategies
                # Standard deviation of the mean
                inv_sqrt_C = eigvecs @ (np.diag(1 / np.sqrt(eigvals))) @ eigvecs.T
                d_sigma = np.linalg.inv(inv_sqrt_C) @ (new_mean - mean) / sigma
                p_sigma = (1 - cs) * p_sigma + np.sqrt(cs * (2 - cs) * mu_eff) * d_sigma
                norm_p_sigma = np.linalg.norm(p_sigma)
                
                # Hessian matrix update
                h_sigma = (norm_p_sigma / np.sqrt(1 - (1 - cs)**2) / chi_n < 1.4 + 2 / (dim + 1))
                
                p_c = (1 - cc) * p_c + h_sigma * np.sqrt(cc * (2 - cc) * mu_eff) * d_sigma
                
                # Update covariance matrix
                delta_h = (1 - h_sigma) * cc * (2 - cc)
                rank_one = np.outer(p_c, p_c)
                
                # Rank-mu update
                rank_mu = np.zeros((dim, dim))
                for i in range(mu):
                    z_i = inv_sqrt_C @ (Y[i] - mean) / sigma
                    rank_mu += weights[i] * np.outer(z_i, z_i)
                
                C = (1 + c1 * delta_h - c1 - cmu) * C + c1 * rank_one + cmu * rank_mu
                
                # Symmetrize and ensure positive definiteness
                C = (C + C.T) / 2
                eigvals, eigvecs = np.linalg.eigh(C)
                
                # Clip eigenvalues to prevent numerical issues
                eigvals = np.clip(eigvals, 1e-12, 1e12)
                
                # Update sigma
                sigma *= np.exp(cs / ds * (norm_p_sigma / chi_n - 1))
                
                # Clip sigma
                sigma = np.clip(sigma, 1e-8, 1e8)
                
                # Update mean
                mean = new_mean
                
                # Check convergence
                if sigma < 1e-8:
                    break
                    
        except StopIteration:
            pass
        except Exception:
            # If CMA-ES fails, fall back to simple (1+1)-ES if budget remains
            if self.calls_made < self.budget and time.time() < self.deadline:
                self._fallback_es(func, wrapped_func)

        if self.best_x is None:
            self.best_x = np.zeros(self.dim)
            self.best_f = wrapped_func(self.best_x)

        return self.best_x, self.best_f

    def _fallback_es(self, func, wrapped_func):
        """Simple (1+1)-ES fallback."""
        if self.best_x is None:
            x = np.zeros(self.dim)
            fx = wrapped_func(x)
        else:
            x = self.best_x.copy()
            fx = self.best_f
            
        sigma = 0.1
        rng = np.random.default_rng(42)
        
        while self.calls_made < self.budget and time.time() < self.deadline:
            y = np.clip(x + sigma * rng.standard_normal(self.dim), -5, 5)
            fy = wrapped_func(y)
            
            if fy < fx:
                x, fx = y, fy
                sigma *= 1.2
            else:
                sigma *= 0.85
                
            if sigma < 1e-6:
                break

# EVOLVE-BLOCK-END
