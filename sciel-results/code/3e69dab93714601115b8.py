"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import warnings
warnings.filterwarnings('ignore')


class Optimizer:
    """Hybrid GP-EI explorer followed by CMA-ES local refinement."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        # Phase 1: Gaussian Process with Expected Improvement for global exploration
        # We use a small subset of the budget for GP-based exploration
        gp_budget = min(80, budget // 5)
        
        # Initialize with a few random points plus center
        X = np.zeros((0, dim))
        Y = np.zeros(0)

        def add_point(x, f):
            nonlocal X, Y
            X = np.vstack([X, x.reshape(1, -1)])
            Y = np.append(Y, f)

        # Initial sampling: center and a few random points
        init_points = [np.zeros(dim)]
        for _ in range(4):
            init_points.append(rng.uniform(-5, 5, dim))
        
        best_x = None
        best_f = np.inf
        
        for x0 in init_points:
            x0 = np.clip(x0, -5, 5)
            f0 = func(x0)
            add_point(x0, f0)
            if f0 < best_f:
                best_f = f0
                best_x = x0.copy()

        # GP-based exploration loop
        # We'll use a simple GP implementation with RBF kernel
        # For efficiency, we limit the GP training set size
        
        max_gp_points = 30  # Cap the GP training set for speed
        
        for i in range(gp_budget - len(init_points)):
            if len(X) >= max_gp_points:
                # Thin the GP model: keep only the most informative points
                # Simple strategy: keep points with lowest f and some random ones
                # Or just keep the last max_gp_points points
                # Better: keep points that are spread out. For simplicity, subsample.
                idx = np.random.choice(len(X), max_gp_points, replace=False)
                X = X[idx]
                Y = Y[idx]
            
            # Compute GP mean and variance at candidate points
            # To find the next point, we sample a set of candidates and pick the best EI
            
            # Use a grid or random candidates
            n_candidates = 100
            candidates = rng.uniform(-5, 5, (n_candidates, dim))
            
            # Compute GP predictions
            # Kernel: RBF
            length_scale = 1.0
            sigma_f = 1.0
            
            # For numerical stability, standardize Y
            mu_y = np.mean(Y)
            std_y = np.std(Y)
            if std_y < 1e-10:
                std_y = 1.0
            Y_std = (Y - mu_y) / std_y
            
            # Kernel matrix K
            # K_ij = sigma_f^2 * exp(-||x_i - x_j||^2 / (2 * l^2))
            
            def rbf_kernel(A, B, l):
                # A: (n, d), B: (m, d)
                # Returns (n, m)
                sq_dists = np.sum(A**2, axis=1)[:, None] + np.sum(B**2, axis=1)[None, :] - 2 * A @ B.T
                sq_dists = np.maximum(sq_dists, 0)
                return np.exp(-sq_dists / (2 * l**2))
            
            K = rbf_kernel(X, X, length_scale) + 1e-8 * np.eye(len(X))
            Kf = rbf_kernel(candidates, X, length_scale)
            
            # GP mean and variance
            try:
                # Solve K @ alpha = Y_std
                alpha = np.linalg.solve(K, Y_std)
                mu = Kf @ alpha
                v = np.linalg.solve(K, Kf.T)
                var = 1.0 - np.sum(v**2, axis=0)
                var = np.maximum(var, 0.0)
                std = np.sqrt(var)
                
                # Unstandardize
                mu = mu * std_y + mu_y
                std = std * std_y
                
                # Expected Improvement
                # EI(x) = (f_best - mu(x)) * Phi((f_best - mu(x))/std(x)) + std(x) * phi((f_best - mu(x))/std(x))
                # where f_best is the current best (we want to minimize, so we use -f)
                # For minimization, EI = (f_best - mu) * Phi((f_best - mu)/std) + std * phi((f_best - mu)/std)
                # But if std is very small, EI is small.
                
                f_best_gp = np.min(Y)
                z = (f_best_gp - mu) / (std + 1e-10)
                ei = (f_best_gp - mu) * norm.cdf(z) + std * norm.pdf(z)
                
                # Handle case where all std are very small
                ei = np.where(std < 1e-8, 0.0, ei)
                
                best_idx = np.argmax(ei)
                x_next = candidates[best_idx]
                
            except np.linalg.LinAlgError:
                # Fallback to random
                x_next = rng.uniform(-5, 5, dim)
            
            x_next = np.clip(x_next, -5, 5)
            f_next = func(x_next)
            add_point(x_next, f_next)
            
            if f_next < best_f:
                best_f = f_next
                best_x = x_next.copy()
        
        # Phase 2: Local refinement using CMA-ES
        # Use the remaining budget
        remaining_budget = budget - (len(init_points) + gp_budget - len(init_points))
        # Actually, we used len(init_points) + (gp_budget - len(init_points)) = gp_budget calls
        # So remaining = budget - gp_budget
        
        if remaining_budget > 10 and best_x is not None:
            # Implement a simple CMA-ES or use scipy's minimize
            # Let's use a simple (1+1)-ES with adaptive sigma, starting from best_x
            # Or use Nelder-Mead if dim is small (dim=5)
            
            # Let's use a simple CMA-ES implementation
            # Since dim=5, we can afford a small population size
            
            pop_size = 4 + int(3 * np.log(dim))
            pop_size = min(pop_size, remaining_budget // 5)
            if pop_size < 2:
                pop_size = 2
            
            x = best_x.copy()
            sigma = 0.5  # Initial step size
            remaining_calls = remaining_budget
            
            # Simple (1+lambda)-ES
            while remaining_calls > 0 and sigma > 1e-6:
                # Generate population
                offspring = []
                for j in range(pop_size):
                    if remaining_calls <= 0:
                        break
                    y = x + sigma * rng.standard_normal(dim)
                    y = np.clip(y, -5, 5)
                    fy = func(y)
                    remaining_calls -= 1
                    offspring.append((fy, y))
                
                if not offspring:
                    break
                
                offspring.sort(key=lambda t: t[0])
                best_off_f, best_off_x = offspring[0]
                
                if best_off_f < best_f:
                    # Move
                    x = best_off_x
                    best_f = best_off_f
                    sigma *= 1.1
                else:
                    sigma *= 0.9
                
                if remaining_calls <= 0:
                    break
        
        return best_x, best_f

# EVOLVE-BLOCK-END
