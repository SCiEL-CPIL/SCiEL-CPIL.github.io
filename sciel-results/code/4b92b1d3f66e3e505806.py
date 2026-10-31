import numpy as np
from scipy.optimize import minimize
import time
import math

N = 16
D = 2

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin for 150s limit

    def get_pairwise_dists_sq(pts):
        """Compute squared pairwise distances."""
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists_sq = np.sum(diff**2, axis=2)
        # Zero out diagonal
        np.fill_diagonal(dists_sq, 0)
        # Get upper triangle
        iu = np.triu_indices(N, 1)
        return dists_sq[iu]

    def objective(pts, beta=10.0, alpha=0.01):
        """
        Composite objective:
        1. Smoothed ratio of max to min distance (using log-sum-exp).
        2. Isotropy penalty based on covariance matrix eigenvalues.
        """
        d_sq = get_pairwise_dists_sq(pts)
        
        # Prevent zero distances
        min_d_sq = np.min(d_sq)
        if min_d_sq < 1e-10:
            return 1e10

        # Smooth max and min using log-sum-exp
        # log(sum(exp(beta * x))) / beta approximates max(x)
        # We want ratio (max/min)^2 = exp(2 * (log(max) - log(min)))
        
        # To avoid overflow, shift by max value
        max_d_sq_val = np.max(d_sq)
        
        # Soft max
        # log(sum(exp(beta * (d_sq - max_d_sq_val)))) + beta * max_d_sq_val
        # Divide by beta to get approximate max
        # But we want the ratio.
        
        # Let's use the approximation:
        # log_max = (1/beta) * log(sum(exp(beta * d_sq)))
        # log_min = -(1/beta) * log(sum(exp(-beta * d_sq)))
        
        # For numerical stability:
        # log_max = max_d_sq_val + (1/beta) * log(sum(exp(beta * (d_sq - max_d_sq_val))))
        
        # log_min:
        min_d_sq_val = min_d_sq
        # log_min = min_d_sq_val + (1/beta) * log(sum(exp(-beta * (d_sq - min_d_sq_val))))
        # Note: exp(-beta * (d_sq - min)) is large for d_sq close to min.
        
        # Let's calculate log_max and log_min
        shifted_max = d_sq - max_d_sq_val
        log_sum_max = np.log(np.sum(np.exp(beta * shifted_max)))
        log_max_approx = max_d_sq_val + (1.0 / beta) * log_sum_max
        
        shifted_min = d_sq - min_d_sq_val
        # We want log(sum(exp(-beta * d_sq)))
        # = -beta * min_d_sq_val + log(sum(exp(-beta * (d_sq - min_d_sq_val))))
        # So log_min_approx = -(1/beta) * [ -beta * min_d_sq_val + log(sum(exp(-beta * (d_sq - min_d_sq_val)))) ]
        # = min_d_sq_val - (1/beta) * log(sum(exp(-beta * (d_sq - min_d_sq_val))))
        
        log_sum_min = np.log(np.sum(np.exp(-beta * shifted_min)))
        log_min_approx = min_d_sq_val - (1.0 / beta) * log_sum_min
        
        # Objective term 1: (max/min)^2 = exp(2 * (log_max - log_min))
        ratio_term = np.exp(2.0 * (log_max_approx - log_min_approx))
        
        # Objective term 2: Isotropy Penalty
        # Center the points
        center = np.mean(pts, axis=0)
        centered = pts - center
        # Covariance matrix
        cov = np.cov(centered.T)
        if cov.shape == ():
            # Degenerate case
            cov = np.array([[cov, 0], [0, cov]])
        else:
            # Ensure 2x2
            if cov.shape != (2, 2):
                cov = np.eye(2)
        
        # Eigenvalues of covariance
        eigvals = np.linalg.eigvalsh(cov)
        # eigvals are sorted ascending
        mu_min = eigvals[0]
        mu_max = eigvals[1]
        
        # Penalty for anisotropy: (mu_max - mu_min) / (mu_max + mu_min)
        # We want this to be 0 (isotropic)
        denom = mu_max + mu_min + 1e-8
        anisotropy = (mu_max - mu_min) / denom
        isotropy_penalty = anisotropy ** 2
        
        return ratio_term + alpha * isotropy_penalty

    def make_random_init():
        """Generate a random initial configuration."""
        # Random points in a unit square, then scale
        pts = np.random.rand(N, D) * 2.0 - 1.0
        # Add some structure: mix of random and grid
        # Let's try a perturbed hexagonal lattice
        # v1 = [1, 0], v2 = [0.5, sqrt(3)/2]
        v1 = np.array([1.0, 0.0])
        v2 = np.array([0.5, np.sqrt(3) / 2])
        
        pts = []
        for i in range(-3, 4):
            for j in range(-3, 4):
                p = i * v1 + j * v2
                pts.append(p)
        pts = np.array(pts)
        pts -= pts.mean(axis=0)
        
        # Pick 16 closest to origin with random perturbation
        dists = np.linalg.norm(pts, axis=1)
        order = np.argsort(dists)
        selected = pts[order[:N]]
        
        # Add noise
        noise = np.random.normal(0, 0.1, (N, D))
        selected += noise
        
        return selected

    best_val = np.inf
    best_pts = make_random_init()
    
    # Multi-start optimization
    num_restarts = 50
    # We have ~145s. Each BFGS run should be fast (<1s).
    # 50 restarts * 1s = 50s. Leaves plenty of time.
    
    for i in range(num_restarts):
        if time.time() > deadline - 10:
            break
            
        # Increase beta (sharpness) over time to refine
        # Start with low beta (smooth) to find basin, increase to high beta (sharp) to refine
        beta = 5.0 + 10.0 * (i / num_restarts)
        alpha = 0.01
        
        init = make_random_init()
        
        try:
            res = minimize(
                lambda x: objective(x.reshape(N, D), beta, alpha),
                init.flatten(),
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-8}
            )
            
            # Evaluate the true (non-smooth) objective for the result
            true_val = objective(res.x.reshape(N, D), beta=1000.0, alpha=0.0) # High beta approximates true max/min, alpha=0 for final score
            
            if true_val < best_val:
                best_val = true_val
                best_pts = res.x.reshape(N, D)
                
        except Exception:
            pass

    # Final polish with high beta and no isotropy penalty to ensure exact ratio minimization
    if time.time() < deadline - 5:
        try:
            # Use the best point found
            res = minimize(
                lambda x: objective(x.reshape(N, D), beta=1000.0, alpha=0.0),
                best_pts.flatten(),
                method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-10}
            )
            
            true_val = objective(res.x.reshape(N, D), beta=1000.0, alpha=0.0)
            if true_val < best_val:
                best_val = true_val
                best_pts = res.x.reshape(N, D)
        except Exception:
            pass

    return best_pts
