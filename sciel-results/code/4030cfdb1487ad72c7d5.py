# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist
from scipy.optimize import minimize
import time


def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions to maximize the ratio of minimum to maximum distance.
    
    Returns
        points: np.ndarray of shape (14,3) containing the coordinates of the 14 points.
    """
    n = 14
    deadline = time.time() + 580.0  # Safety margin before 630s limit

    def score(pts):
        """Compute (d_min / d_max)^2"""
        if pts.shape != (14, 3):
            return 0.0
        if not np.all(np.isfinite(pts)):
            return 0.0
        d = pdist(pts)
        dmin = np.min(d)
        dmax = np.max(d)
        if dmax < 1e-12:
            return 0.0
        return (dmin / dmax) ** 2

    def neg_score(pts):
        """Negative score for minimization"""
        return -score(pts)

    def neg_score_grad(pts):
        """
        Numerical gradient of negative score using central differences.
        """
        pts = pts.reshape(-1)
        grad = np.zeros_like(pts)
        eps = 1e-6
        for i in range(len(pts)):
            pts_plus = pts.copy()
            pts_minus = pts.copy()
            pts_plus[i] += eps
            pts_minus[i] -= eps
            f_plus = -score(pts_plus.reshape(14, 3))
            f_minus = -score(pts_minus.reshape(14, 3))
            grad[i] = (f_plus - f_minus) / (2 * eps)
        return grad.reshape(14, 3)

    # Generate initial configurations
    def fibonacci_sphere(n_points):
        """Generate points on a Fibonacci sphere"""
        phi = np.arccos(1 - 2 * (np.arange(n_points) + 0.5) / n_points)
        theta = np.pi * (1 + 5**0.5) * np.arange(n_points)
        return np.column_stack((
            np.sin(phi) * np.cos(theta),
            np.sin(phi) * np.sin(theta),
            np.cos(phi)
        ))

    def random_sphere(n_points, seed=None):
        """Generate random points on unit sphere"""
        rng = np.random.default_rng(seed)
        pts = rng.normal(size=(n_points, 3))
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        norms[norms < 1e-10] = 1e-10
        return pts / norms

    def d3h_symmetric(ring1_r, ring1_z, ring2_r, ring2_z, pole_z):
        """
        D3h symmetric configuration:
        - 2 poles at (0, 0, ±pole_z)
        - 6 points in a hexagonal ring at radius ring1_r, height ring1_z
        - 6 points in a hexagonal ring at radius ring2_r, height -ring2_z
        Total: 14 points
        """
        pts = []
        # Poles
        pts.append([0, 0, pole_z])
        pts.append([0, 0, -pole_z])
        
        # Ring 1 (6 points at angle offset 0)
        for k in range(6):
            angle = 2 * np.pi * k / 6
            pts.append([ring1_r * np.cos(angle), ring1_r * np.sin(angle), ring1_z])
        
        # Ring 2 (6 points at angle offset pi/6)
        for k in range(6):
            angle = 2 * np.pi * k / 6 + np.pi / 6
            pts.append([ring2_r * np.cos(angle), ring2_r * np.sin(angle), -ring2_z])
        
        return np.array(pts)

    # Initial configurations to try
    initial_configs = []
    
    # Fibonacci sphere
    initial_configs.append(fibonacci_sphere(n))
    
    # Random sphere with different seeds
    for seed in range(5):
        initial_configs.append(random_sphere(n, seed=seed))
    
    # D3h symmetric configurations
    for r1 in [0.5, 0.7, 0.8, 0.9]:
        for r2 in [0.5, 0.7, 0.8, 0.9]:
            for z1 in [0.2, 0.4, 0.5, 0.6]:
                for z2 in [0.2, 0.4, 0.5, 0.6]:
                    for pz in [0.8, 0.9, 0.95, 1.0]:
                        initial_configs.append(d3h_symmetric(r1, z1, r2, z2, pz))
    
    # Optimize each initial configuration
    best_points = initial_configs[0].copy()
    best_score = score(best_points)
    
    for init_idx, init_pts in enumerate(initial_configs):
        if time.time() > deadline - 30:
            break
        
        # Try SLSQP with numerical gradient
        try:
            res = minimize(
                neg_score,
                init_pts,
                method='SLSQP',
                jac=neg_score_grad,
                options={'maxiter': 200, 'ftol': 1e-12}
            )
            
            if res.success:
                final_pts = res.x.reshape(14, 3)
                final_score = score(final_pts)
                
                if final_score > best_score:
                    best_score = final_score
                    best_points = final_pts.copy()
                    
        except Exception:
            continue
    
    # Final local refinement using small perturbations
    if time.time() < deadline - 10:
        current = best_points.copy()
        current_score = best_score
        lr = 0.001
        for iteration in range(2000):
            if time.time() > deadline - 5:
                break
            
            # Compute forces (repulsion)
            forces = np.zeros_like(current)
            for i in range(n):
                for j in range(n):
                    if i == j:
                        continue
                    diff = current[i] - current[j]
                    dist_sq = np.sum(diff**2)
                    if dist_sq < 1e-10:
                        continue
                    dist = np.sqrt(dist_sq)
                    forces[i] += diff / (dist_sq * dist)
            
            # Update
            new_pts = current + lr * forces
            # Project to sphere
            norms = np.linalg.norm(new_pts, axis=1, keepdims=True)
            norms[norms < 1e-10] = 1e-10
            new_pts = new_pts / norms
            
            new_score = score(new_pts)
            if new_score > current_score:
                current = new_pts
                current_score = new_score
                if current_score > best_score:
                    best_score = current_score
                    best_points = current.copy()
            else:
                # Reduce learning rate
                lr *= 0.95
    
    return best_points


# EVOLVE-BLOCK-END
