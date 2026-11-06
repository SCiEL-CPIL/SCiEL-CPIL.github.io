"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin before 150s limit
    
    # Regular octagon as initial guess
    n = 8
    theta = 2 * np.pi * np.arange(n) / n
    points = np.column_stack([np.cos(theta), np.sin(theta)])
    
    def log_score(pts):
        """Compute log of the score Delta/n^n where Delta is product of distances."""
        # Compute all pairwise distances
        diffs = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diffs ** 2, axis=2))
        # Avoid log(0) for self-distances
        np.fill_diagonal(dists, 1.0)
        # Product over ordered pairs = (product over unordered pairs)^2
        # So log(Delta) = 2 * sum over i<j log(d_ij)
        iu = np.triu_indices(n, k=1)
        log_delta = 2.0 * np.sum(np.log(dists[iu]))
        # Score = Delta / n^n
        log_score_val = log_delta - n * np.log(n)
        return log_score_val, dists
    
    def diameter(pts):
        diffs = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diffs ** 2, axis=2))
        return np.max(dists)
    
    def neg_log_score_constrained(pts, t):
        """
        Smooth surrogate: maximize log_score - penalty * max(0, diameter - 2)^2
        Using soft-max via log-sum-exp for smooth diameter.
        """
        log_s, dists = log_score(pts)
        # Soft diameter: use log-sum-exp of squared distances (smooth max)
        # Actually, let's use a different approach: penalize if any distance > 2
        # Smooth penalty: sum of softplus(d_ij - 2) for all pairs
        iu = np.triu_indices(n, k=1)
        d_ij = dists[iu]
        # Soft penalty for distances exceeding 2
        penalty = np.sum(np.logaddexp(0, (d_ij - 2.0) * 100.0) / 100.0)
        return -log_s + t * penalty
    
    # Start with a moderate penalty weight
    t = 10.0
    
    # Use scipy's L-BFGS-B for optimization
    from scipy.optimize import minimize
    
    def objective(x):
        if time.time() > deadline - 5:
            return np.inf
        pts = x.reshape(n, 2)
        return neg_log_score_constrained(pts, t)
    
    def gradient(x):
        if time.time() > deadline - 5:
            return np.ones_like(x)
        pts = x.reshape(n, 2)
        # Numerical gradient
        eps = 1e-7
        grad = np.zeros_like(x)
        f0 = objective(x)
        for i in range(len(x)):
            x_plus = x.copy()
            x_plus[i] += eps
            grad[i] = (objective(x_plus) - f0) / eps
        return grad
    
    # Try multiple starting points and penalty weights
    best_score = -np.inf
    best_points = points.copy()
    
    # First evaluate the regular octagon
    ls, _ = log_score(points)
    best_score = ls
    best_points = points.copy()
    
    # Optimization attempts
    attempts = [
        (points.copy(), 10.0),
        (points.copy(), 50.0),
        (points.copy(), 100.0),
        (points.copy(), 1000.0),
    ]
    
    # Add some perturbed starting points
    rng = np.random.RandomState(42)
    for _ in range(4):
        perturbed = points + rng.randn(n, 2) * 0.1
        attempts.append((perturbed, 50.0))
    
    for init_pts, t_val in attempts:
        if time.time() > deadline - 10:
            break
        
        x0 = init_pts.flatten()
        
        try:
            result = minimize(
                objective,
                x0,
                method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            
            if time.time() > deadline - 5:
                continue
            
            opt_pts = result.x.reshape(n, 2)
            opt_ls, _ = log_score(opt_pts)
            
            if opt_ls > best_score:
                best_score = opt_ls
                best_points = opt_pts.copy()
                
        except Exception:
            continue
    
    # Final polish: try a few more iterations with tighter tolerances
    if time.time() < deadline - 15:
        x0 = best_points.flatten()
        try:
            result = minimize(
                objective,
                x0,
                method='L-BFGS-B',
                options={'maxiter': 1000, 'ftol': 1e-15, 'gtol': 1e-12}
            )
            opt_pts = result.x.reshape(n, 2)
            opt_ls, _ = log_score(opt_pts)
            if opt_ls > best_score:
                best_score = opt_ls
                best_points = opt_pts.copy()
        except Exception:
            pass
    
    # Scale to have diameter exactly 2
    d = diameter(best_points)
    if d > 0:
        best_points = best_points * (2.0 / d)
    
    # Return as list of lists
    return [[float(p[0]), float(p[1])] for p in best_points]
# EVOLVE-BLOCK-END
