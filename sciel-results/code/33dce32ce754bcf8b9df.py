import numpy as np
import time
from scipy.spatial.distance import pdist
from scipy.optimize import minimize

def min_max_dist_dim3_14() -> np.ndarray:
    n = 14
    d = 3
    start_time = time.time()
    deadline = start_time + 580  # safety margin for 630s kill

    def score(pts):
        """Compute (d_min / d_max)^2 for the point configuration."""
        pts = np.asarray(pts, dtype=float)
        if pts.shape != (n, d) or not np.all(np.isfinite(pts)):
            return 0.0
        dists = pdist(pts)
        if len(dists) == 0:
            return 0.0
        d_min = np.min(dists)
        d_max = np.max(dists)
        if d_max < 1e-12:
            return 0.0
        return (d_min / d_max) ** 2

    def make_penalty_objective(beta):
        """Create a smooth penalty objective that penalizes distance variance."""
        def objective(x):
            pts = x.reshape((n, d))
            pts = pts - np.mean(pts, axis=0)
            
            # Compute all pairwise distances
            dists = pdist(pts)
            if len(dists) == 0:
                return 0.0
            
            d_min = np.min(dists)
            d_max = np.max(dists)
            if d_max < 1e-12:
                return 0.0
            
            # Penalty for non-uniform distances: variance of squared distances
            mean_d = np.mean(dists)
            var_d = np.var(dists)
            
            # Primary objective: maximize d_min/d_max ratio
            ratio = d_min / d_max
            
            # Penalty: encourage all distances to be similar
            penalty = beta * var_d / (mean_d ** 2 + 1e-12)
            
            return -(ratio ** 2) + penalty
        
        def gradient(x):
            pts = x.reshape((n, d))
            pts = pts - np.mean(pts, axis=0)
            
            dists = pdist(pts)
            if len(dists) == 0:
                return np.zeros_like(x)
            
            d_min = np.min(dists)
            d_max = np.max(dists)
            if d_max < 1e-12:
                return np.zeros_like(x)
            
            mean_d = np.mean(dists)
            
            # Numerical gradient via finite differences
            grad = np.zeros_like(x)
            eps = 1e-7
            for i in range(len(x)):
                x_plus = x.copy()
                x_plus[i] += eps
                x_minus = x.copy()
                x_minus[i] -= eps
                grad[i] = (objective(x_plus) - objective(x_minus)) / (2 * eps)
            
            return grad
        
        return objective, gradient

    def fibonacci_sphere_init(n_pts, seed=42):
        """Generate initial points using Fibonacci sphere."""
        rng = np.random.RandomState(seed)
        indices = np.arange(0, n_pts, dtype=float) + 0.5
        phi = np.arccos(1 - 2 * indices / n_pts)
        theta = np.pi * (1 + 5**0.5) * indices
        
        x = np.cos(theta) * np.sin(phi)
        y = np.sin(theta) * np.sin(phi)
        z = np.cos(phi)
        
        pts = np.column_stack([x, y, z])
        
        # Add small perturbation
        pert = rng.randn(n_pts, d) * 0.05
        pts = pts + pert
        
        return pts

    def coulomb_repulsion_optimize(pts, max_iter=2000, lr_schedule=None):
        """
        Optimize using Coulomb repulsion: each pair of points repels with force
        proportional to 1/r^2. This naturally pushes points apart and tends to
        equalize distances.
        """
        pts = pts.copy().astype(float)
        
        if lr_schedule is None:
            lr_schedule = lambda it: 0.01 * (0.995 ** (it // 50))
        
        for it in range(max_iter):
            if time.time() > deadline - 30:
                break
            
            # Compute all pairwise distances and forces
            dists = pdist(pts)
            if np.any(dists < 1e-12):
                # Add small random jitter to avoid division by zero
                pts += np.random.randn(n, d) * 1e-6
                dists = pdist(pts)
            
            # Compute forces: F_ij = k * (r_i - r_j) / |r_i - r_j|^3
            # This is the gradient of sum 1/|r_i - r_j|
            forces = np.zeros((n, d))
            
            # Build distance matrix for efficient computation
            for i in range(n):
                for j in range(i + 1, n):
                    diff = pts[i] - pts[j]
                    dist = np.linalg.norm(diff)
                    if dist > 1e-12:
                        # Coulomb force: F = k * diff / dist^3
                        force_mag = 1.0 / (dist ** 3)
                        forces[i] += force_mag * diff
                        forces[j] -= force_mag * diff
            
            # Also add a gentle attraction to center to prevent drift
            forces -= 0.001 * pts
            
            # Update positions with adaptive learning rate
            lr = lr_schedule(it)
            pts += lr * forces
            
            # Remove center of mass
            pts = pts - np.mean(pts, axis=0)
        
        return pts

    def smooth_penalty_optimize(pts, beta_start=10.0, beta_end=0.1, max_iter=500):
        """
        Optimize using smooth penalty method: minimize variance of distances
        while maximizing the min/max ratio.
        """
        pts = pts.copy().astype(float)
        
        for stage in range(10):
            beta = beta_start * (beta_end / beta_start) ** (stage / 9.0)
            objective, gradient = make_penalty_objective(beta)
            
            try:
                result = minimize(
                    objective,
                    pts.flatten(),
                    jac=gradient,
                    method='L-BFGS-B',
                    options={'maxiter': max_iter // 10, 'ftol': 1e-12, 'gtol': 1e-8}
                )
                if result.success or result.fun < objective(pts.flatten()):
                    pts = result.x.reshape((n, d))
                    pts = pts - np.mean(pts, axis=0)
            except Exception:
                pass
            
            if time.time() > deadline - 30:
                break
        
        return pts

    # Generate multiple initial configurations
    rng = np.random.RandomState(12345)
    candidates = []
    
    # Fibonacci sphere with different seeds
    for seed in range(10):
        candidates.append(fibonacci_sphere_init(n, seed=seed))
    
    # Random configurations on sphere
    for i in range(10):
        pts = rng.randn(n, d)
        pts = pts / np.linalg.norm(pts, axis=1, keepdims=True)
        candidates.append(pts)
    
    # Mix of sphere and interior points
    for i in range(5):
        pts = np.zeros((n, d))
        # 12 points on sphere
        for j in range(12):
            p = rng.randn(d)
            pts[j] = p / np.linalg.norm(p)
        # 2 points inside
        pts[12] = rng.randn(d) * 0.3
        pts[13] = rng.randn(d) * 0.3
        candidates.append(pts)
    
    # Evaluate all candidates
    best_score = -np.inf
    best_points = None
    
    for cand in candidates:
        if time.time() > deadline - 120:
            break
        s = score(cand)
        if s > best_score:
            best_score = s
            best_points = cand.copy()
    
    if best_points is None:
        best_points = candidates[0]
    
    # Strategy 1: Coulomb repulsion optimization from best point
    if time.time() < deadline - 60:
        coulomb_pts = coulomb_repulsion_optimize(best_points, max_iter=3000)
        s_coulomb = score(coulomb_pts)
        if s_coulomb > best_score:
            best_score = s_coulomb
            best_points = coulomb_pts.copy()
    
    # Strategy 2: Smooth penalty optimization
    if time.time() < deadline - 30:
        penalty_pts = smooth_penalty_optimize(best_points, beta_start=50.0, beta_end=0.5, max_iter=1000)
        s_penalty = score(penalty_pts)
        if s_penalty > best_score:
            best_score = s_penalty
            best_points = penalty_pts.copy()
    
    # Strategy 3: Combined - Coulomb then penalty refinement
    if time.time() < deadline - 20:
        combined_pts = coulomb_repulsion_optimize(best_points, max_iter=1000)
        combined_pts = smooth_penalty_optimize(combined_pts, beta_start=20.0, beta_end=0.2, max_iter=500)
        s_combined = score(combined_pts)
        if s_combined > best_score:
            best_score = s_combined
            best_points = combined_pts.copy()
    
    # Final: try multiple Coulomb runs from different perturbations of best point
    for i in range(5):
        if time.time() > deadline - 15:
            break
        perturbed = best_points + rng.randn(n, d) * 0.05
        coulomb_pts = coulomb_repulsion_optimize(perturbed, max_iter=1500)
        s = score(coulomb_pts)
        if s > best_score:
            best_score = s
            best_points = coulomb_pts.copy()
    
    # Final normalization
    best_points = best_points - np.mean(best_points, axis=0)
    norm_max = np.max(np.linalg.norm(best_points, axis=1))
    if norm_max > 0:
        best_points = best_points / norm_max * 100
    
    if not np.all(np.isfinite(best_points)):
        best_points = np.zeros((n, d))
    
    return best_points
