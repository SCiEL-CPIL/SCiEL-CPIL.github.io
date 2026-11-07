"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import t as student_t

class Optimizer:
    """Bayesian Optimizer with GP surrogate and Student-t EI acquisition."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def _rbf_kernel(self, X1, X2, length_scales):
        """
        Compute RBF kernel between X1 and X2.
        X1: (n1, d), X2: (n2, d)
        length_scales: (d,)
        Returns: (n1, n2)
        """
        # Standardize by length scales
        X1_scaled = X1 / length_scales
        X2_scaled = X2 / length_scales
        
        # Squared distances
        sq_norms1 = np.sum(X1_scaled**2, axis=1)[:, None]
        sq_norms2 = np.sum(X2_scaled**2, axis=1)[None, :]
        sq_dists = sq_norms1 + sq_norms2 - 2 * X1_scaled @ X2_scaled.T
        sq_dists = np.maximum(sq_dists, 0) # Numerical stability
        
        return np.exp(-0.5 * sq_dists)

    def _student_t_ei(self, mu, sigma, best_f, nu=3.0):
        """
        Compute Student-t Expected Improvement.
        mu: (n,) mean predictions
        sigma: (n,) std predictions
        best_f: float
        nu: float degrees of freedom
        """
        # Avoid division by zero
        sigma = np.maximum(sigma, 1e-8)
        y = (best_f - mu) / sigma
        cdf = student_t.cdf(y, nu)
        pdf = student_t.pdf(y, nu)
        
        # EI = (best_f - mu) * cdf + sigma * pdf * (1 + y^2 / nu)
        # Note: Standard EI uses Normal. Student-t EI is slightly different but often approximated or used with t-distribution properties.
        # A common approximation for t-EI when using t-distribution for uncertainty:
        # If Y ~ t_nu, E[(Y - mu)_+] ... 
        # Let's use the standard formula adjusted for t-distribution.
        # Actually, a simpler robust approach often used in BO is just Normal EI, but the prompt asks for Student-t.
        # Let's implement the exact expectation for a t-distributed variable.
        # Let Z ~ t_nu. We want E[(Z - mu)_+] given Z has mean 0, var nu/(nu-2).
        # This is complex. A common heuristic is to use the Normal EI formula but with a larger effective sigma or just use the t-cdf/pdf in the standard EI structure.
        # Let's stick to the standard EI formula but using the t-distribution's CDF and PDF for the probability and density terms, 
        # which effectively makes the acquisition function heavier-tailed in its exploration.
        
        ei = (best_f - mu) * cdf + sigma * pdf * (1 + y**2 / nu)
        
        return ei

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0 # Safety margin
        budget = self.budget
        n = self.dim
        
        # Initial points: LHS or random
        # Let's use a mix of center and random
        X = np.zeros((0, n))
        Y = np.zeros(0)
        
        # Initial design: 5 points
        n_init = 5
        # One at center
        init_points = [np.zeros(n)]
        # Rest random in [-5, 5]
        for _ in range(n_init - 1):
            init_points.append(np.random.uniform(-5, 5, n))
        
        X_init = np.array(init_points)
        Y_init = np.array([func(x) for x in X_init])
        
        X = X_init
        Y = Y_init
        calls = n_init
        
        best_f = np.min(Y)
        best_x = X[np.argmin(Y)]
        
        # GP Hyperparameters
        # Length scales: initialize to range of each dimension
        length_scales = np.ones(n) * 2.0
        # Signal variance
        sigma_f = 1.0
        # Noise variance (small)
        sigma_n = 1e-6
        
        # Adaptive hyperparameters? Keep simple for now, maybe tune sigma_f based on Y scale
        if len(Y) > 0:
            y_range = np.max(Y) - np.min(Y)
            if y_range > 0:
                sigma_f = y_range / 10.0
            else:
                sigma_f = 1.0

        # Main loop
        while calls < budget:
            if time.time() > deadline:
                break
            
            # Update hyperparameters occasionally? 
            # For simplicity and speed, we will fix hyperparameters but scale sigma_f if Y changes drastically.
            # Actually, let's do a simple hyperparameter update every 10 steps.
            if calls % 10 == 0 and len(Y) > 10:
                # Simple heuristic update
                y_range = np.max(Y) - np.min(Y)
                if y_range > 0:
                    sigma_f = y_range / 10.0
                
                # Adjust length scales based on spread of X? 
                # Keep length scales fixed for stability in 5D with small budget.
                pass
            
            # Predict on a grid or candidate set
            # Since dim=5, a full grid is expensive. Use a candidate set.
            # Generate candidates:
            # 1. Random points
            # 2. Points around best_x
            n_candidates = 200
            
            # 50% random
            n_rand = n_candidates // 2
            cand_rand = np.random.uniform(-5, 5, (n_rand, n))
            
            # 50% local around best
            n_local = n_candidates - n_rand
            # Adaptive radius
            radius = 1.0 * (1.0 - calls / budget) + 0.1
            cand_local = best_x + radius * np.random.randn(n_local, n)
            cand_local = np.clip(cand_local, -5, 5)
            
            X_cand = np.vstack([cand_rand, cand_local])
            
            # Compute GP mean and variance at candidates
            # K(X, X)
            K = self._rbf_kernel(X, X, length_scales)
            K += sigma_n * np.eye(len(X))
            
            # K(X, X_cand)
            K_s = self._rbf_kernel(X, X_cand, length_scales)
            
            # Solve K alpha = Y
            try:
                alpha = np.linalg.solve(K, Y)
            except np.linalg.LinAlgError:
                # Fallback to pinv if singular
                alpha = np.linalg.pinv(K) @ Y
            
            # Mean
            mu = K_s.T @ alpha
            
            # Variance
            # Var = K(X_cand, X_cand) - K(X_cand, X) K(X, X)^-1 K(X, X_cand)^T
            K_ss = self._rbf_kernel(X_cand, X_cand, length_scales)
            var = np.diag(K_ss) - np.sum(K_s * (K_s.T @ np.linalg.solve(K, np.eye(len(X)))) * 0, axis=1) # This is wrong, need K_s @ K_inv
            
            # Efficient variance calculation:
            # v = K_s.T @ alpha is mean.
            # For variance: v_s = K_s.T @ K_inv (where K_inv is inverse of K)
            # But we don't have K_inv explicitly.
            # var_i = K_ss[i,i] - K_s[i,:] @ K_inv @ K_s[i,:]^T
            # Let's compute K_inv or use Cholesky.
            
            try:
                L = np.linalg.cholesky(K)
                # Solve L L^T v = K_s^T  =>  L^T v = K_s^T  =>  v = L^-1 K_s^T
                # Actually: K = L L^T. K^{-1} = L^{-T} L^{-1}.
                # We need K_s @ K^{-1} @ K_s^T.
                # Let M = K_s @ K^{-1} @ K_s^T.
                # K^{-1} A = (L^{-T} L^{-1}) A.
                # Let's solve L z = K_s^T => z = L^{-1} K_s^T.
                # Then L^T w = z => w = L^{-T} z = L^{-T} L^{-1} K_s^T = K^{-1} K_s^T.
                # Then M = K_s @ w.
                
                # z = solve(L, K_s.T)
                z = np.linalg.solve(L, K_s.T)
                # w = solve(L.T, z)
                w = np.linalg.solve(L.T, z)
                # M = K_s @ w
                M = K_s @ w
                var = np.diag(K_ss) - np.sum(M, axis=1)
                var = np.maximum(var, 1e-8)
                sigma = np.sqrt(var)
            except np.linalg.LinAlgError:
                # Fallback: use diagonal only (ignoring correlations)
                var = sigma_f**2
                sigma = np.sqrt(var)
                mu = np.zeros(len(X_cand)) # If GP fails, fallback to random search effectively

            # Compute EI
            ei = self._student_t_ei(mu, sigma, best_f, nu=3.0)
            
            # Select best candidate
            idx = np.argmax(ei)
            x_new = X_cand[idx]
            
            # Evaluate
            f_new = func(x_new)
            calls += 1
            
            # Update best
            if f_new < best_f:
                best_f = f_new
                best_x = x_new.copy()
            
            # Add to dataset
            X = np.vstack([X, x_new])
            Y = np.append(Y, f_new)
            
            # Optional: Local refinement if close to convergence?
            # With BO, we rely on the GP to zoom in.
            
        return best_x, best_f

# EVOLVE-BLOCK-END
