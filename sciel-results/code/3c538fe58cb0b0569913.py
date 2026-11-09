"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

class Optimizer:
    """GP Surrogate with Deterministic Grid Survey Initialization."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 295.0
        
        rng = np.random.default_rng(42)
        dim = self.dim
        total_budget = self.budget
        
        best_x = None
        best_f = np.inf
        calls_made = 0
        
        X = []
        Y = []
        
        def update_best(x, f):
            nonlocal best_x, best_f
            if f < best_f:
                best_f = f
                best_x = x.copy()

        def add_point(x, f):
            nonlocal calls_made
            X.append(x)
            Y.append(f)
            calls_made += 1
            update_best(x, f)

        # Phase 1: Deterministic Multi-Scale Grid Survey
        # The goal is to sample the global structure.
        # With dim=5, a full grid is infeasible. We use a stratified approach.
        # 1. Coarse grid: 3 points per dimension -> 3^5 = 243 points.
        # This fits in the first ~200-250 budget.
        
        # We will use a specific sequence to ensure early coverage.
        # Let's use a Latin Hypercube or a specific grid.
        # A simple grid on [-4, 4] (since min is in [-4,4]) is good.
        # 3^5 = 243.
        
        grid_vals = np.array([-4.0, 0.0, 4.0])
        
        # Generate all combinations
        # This might be too many if we want to save budget for GP.
        # Let's try 243 points.
        
        import itertools
        grid_points = []
        for combo in itertools.product(grid_vals, repeat=dim):
            grid_points.append(np.array(combo))
            
        # Shuffle to break symmetry? No, deterministic order might help or hurt.
        # Let's shuffle with a fixed seed for reproducibility but avoid immediate clustering.
        indices = np.arange(len(grid_points))
        rng.shuffle(indices)
        grid_points = [grid_points[i] for i in indices]
        
        # Evaluate grid
        for x in grid_points:
            if calls_made >= total_budget:
                break
            if time.time() >= deadline:
                break
            f = func(x)
            add_point(x, f)
            
        if calls_made >= total_budget or time.time() >= deadline:
            return best_x, best_f

        # Phase 2: GP Surrogate Optimization
        # Fit GP on the observed data
        # Standardize Y for numerical stability
        Y_arr = np.array(Y)
        X_arr = np.array(X)
        
        y_mean = np.mean(Y_arr)
        y_std = np.std(Y_arr) + 1e-8
        Y_std = (Y_arr - y_mean) / y_std
        
        # GP Parameters
        # Length scales: initialize to 1.0, maybe adapt later? 
        # For simplicity and speed, fixed length scales or a simple optimization.
        # Given the "rotated" nature, anisotropic length scales might be needed, 
        # but standard CMA-ES handles rotation via covariance. GP handles it via kernel.
        # Let's use a Matern 5/2 kernel.
        
        def matern52_kernel(X1, X2, l):
            """Compute Matern 5/2 kernel matrix."""
            # X1: (n, d), X2: (m, d)
            # l: (d,) or scalar
            if np.isscalar(l):
                l = np.full(X1.shape[1], l)
            
            # Euclidean distance
            # dist^2 = sum((x1-x2)^2)
            # Using broadcasting
            X1_norm = np.sum((X1 / l) ** 2, axis=1)[:, np.newaxis]
            X2_norm = np.sum((X2 / l) ** 2, axis=1)[np.newaxis, :]
            dist_sq = X1_norm + X2_norm - 2 * (X1 / l) @ (X2 / l).T
            dist_sq = np.maximum(dist_sq, 0) # Numerical stability
            d = np.sqrt(dist_sq)
            
            # Matern 5/2
            # K = (1 + sqrt(5)*d + 5/3*d^2) * exp(-sqrt(5)*d)
            sqrt5 = np.sqrt(5.0)
            K = (1 + sqrt5 * d + 5/3 * d**2) * np.exp(-sqrt5 * d)
            return K
        
        # Hyperparameters
        # Signal variance sigma_f^2
        sigma_f = 1.0
        # Noise variance sigma_n^2
        sigma_n = 1e-4
        
        # Initial length scales: 1.0 in all directions
        l = np.ones(dim)
        
        # We might want to optimize hyperparameters, but that's expensive.
        # Let's stick to fixed reasonable hyperparameters for the first pass,
        # or a simple grid search over a few length scales.
        
        # Given the budget (remaining ~250 calls), we can afford a few GP iterations.
        # However, fitting GP in 5D is fast.
        
        # Let's try to optimize length scales slightly.
        # Or just use fixed l=1.0, sigma_f=1.0.
        # The grid survey should have revealed the scale.
        
        # Let's refine length scales based on the grid spacing?
        # Grid spacing is 4.0. So l=1.0 is reasonable.
        
        # GP Prediction Function
        def gp_predict(X_new):
            """
            Predict mean and variance for X_new.
            X_new: (n, d)
            Returns: mean (n,), var (n,)
            """
            K = matern52_kernel(X_arr, X_new, l) # (N, n)
            K_ss = matern52_kernel(X_new, X_new, l) # (n, n)
            
            # Add noise to training data covariance
            K_nn = matern52_kernel(X_arr, X_arr, l) + sigma_n * np.eye(len(X_arr))
            
            # Solve K_nn * alpha = Y_std
            try:
                L = np.linalg.cholesky(K_nn)
                alpha = np.linalg.solve(L.T, np.linalg.solve(L, Y_std))
            except np.linalg.LinAlgError:
                # Fallback to pinv if singular
                alpha = np.linalg.pinv(K_nn) @ Y_std
                
            # Mean prediction
            mu = K.T @ alpha
            
            # Variance prediction
            v = np.linalg.solve(L, K.T) if 'L' in locals() else np.linalg.pinv(K_nn) @ K.T
            var = np.diag(K_ss) - np.sum(v**2, axis=0)
            
            # Denormalize
            mu = mu * y_std + y_mean
            var = var * y_std**2
            
            return mu, np.maximum(var, 1e-10)
        
        # Acquisition Function: Expected Improvement (EI)
        def expected_improvement(x):
            """
            Calculate EI for a single point x.
            """
            mu, var = gp_predict(x[np.newaxis, :])
            mu = mu[0]
            var = var[0]
            std = np.sqrt(var)
            
            if std < 1e-8:
                return 0.0
                
            z = (best_f - mu) / std
            EI = (best_f - mu) * norm.cdf(z) + std * norm.pdf(z)
            
            return EI
        
        # Optimization of Acquisition Function
        # We need to find the max of EI.
        # Use a local search method. Since EI is smooth but multimodal,
        # we can use multiple starts or a coarse grid followed by Nelder-Mead.
        
        # Remaining budget
        while calls_made < total_budget:
            if time.time() >= deadline:
                break
                
            # 1. Coarse search for EI max
            # Sample random points and maybe perturb best points
            candidates = []
            
            # Random points
            for _ in range(50):
                candidates.append(rng.uniform(-5, 5, dim))
                
            # Perturbations of best point
            for _ in range(20):
                candidates.append(best_x + rng.normal(0, 0.5, dim))
                
            # Perturbations of other good points (top 5 from Y)
            Y_sorted_idx = np.argsort(Y_arr)
            for idx in Y_sorted_idx[:5]:
                for _ in range(5):
                    candidates.append(X_arr[idx] + rng.normal(0, 0.5, dim))
            
            candidates = np.array(candidates)
            
            # Evaluate EI for candidates
            eis = np.array([expected_improvement(x) for x in candidates])
            
            # Pick top 5 candidates for local refinement
            top_idx = np.argsort(eis)[::-1][:5]
            
            best_ei_val = -np.inf
            best_x_ei = None
            
            for idx in top_idx:
                x_init = candidates[idx]
                
                # Local optimization using Nelder-Mead
                # Minimize -EI
                from scipy.optimize import minimize
                
                def neg_ei(x):
                    return -expected_improvement(x)
                
                try:
                    res = minimize(neg_ei, x_init, method='Nelder-Mead', 
                                   options={'maxiter': 100, 'xatol': 1e-4, 'fatol': 1e-6})
                    if res.fun < best_ei_val:
                        best_ei_val = -res.fun
                        best_x_ei = res.x
                except:
                    pass
            
            if best_x_ei is None:
                # Fallback to best candidate
                best_x_ei = candidates[top_idx[0]]
                
            # Clip to bounds
            best_x_ei = np.clip(best_x_ei, -5, 5)
            
            # Evaluate the true function
            f_new = func(best_x_ei)
            add_point(best_x_ei, f_new)
            
            # Update GP data
            # Note: X_arr and Y_arr are updated via add_point? No, add_point updates lists.
            # We need to update the numpy arrays for the GP.
            X_arr = np.array(X)
            Y_arr = np.array(Y)
            
            # Re-normalize Y for GP
            y_mean = np.mean(Y_arr)
            y_std = np.std(Y_arr) + 1e-8
            Y_std = (Y_arr - y_mean) / y_std
            
            # Optional: Update hyperparameters?
            # For simplicity, we keep l fixed. If we want to adapt, we could do a simple search.
            # But given time constraints, fixed l=1.0 with a good grid survey should be robust.
            
            # Check for convergence
            if best_f < -99.9: # Close to theoretical min?
                break

        return best_x, best_f

# EVOLVE-BLOCK-END
