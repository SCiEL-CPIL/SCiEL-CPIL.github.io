import numpy as np
import time
from scipy.spatial.distance import pdist
from scipy.optimize import minimize

def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions in order to maximize the ratio of minimum to maximum distance.
    
    Returns
        points: np.ndarray of shape (14,3) containing the coordinates of the 14 points.
    """
    n = 14
    d = 3
    start_time = time.time()
    deadline = start_time + 610  # 10s margin
    
    best_score = -1.0
    best_points = np.zeros((n, d))
    
    # Helper to calculate score based on squared distances
    def get_score(pts):
        # Calculate squared pairwise distances
        dists = pdist(pts, metric='sqeuclidean')
        if len(dists) == 0:
            return 0.0
        d_min_sq = np.min(dists)
        d_max_sq = np.max(dists)
        
        # If points coincide or are identical
        if d_max_sq < 1e-12:
            return 0.0
        
        # Metric is (d_min / d_max)^2 = d_min_sq / d_max_sq
        return d_min_sq / d_max_sq

    # Objective function for L-BFGS-B
    # We want to maximize d_min subject to d_max = 1.
    # Equivalent to minimizing -softmin(distances) on normalized points.
    # softmin(x, k) = -1/k * log(sum(exp(-k*x)))
    # We minimize -softmin => maximize softmin.
    # To make it smooth, we use a power mean or log-sum-exp.
    
    def objective(x):
        pts = x.reshape(n, d)
        
        # Normalize points so that max distance is 1
        # This ensures the constraint d_max = 1 is met, and we just need to maximize d_min.
        # However, normalization is non-differentiable if we use argmax.
        # Instead, we can penalize deviations from d_max=1? 
        # Or simply, since the ratio is scale invariant, we can fix the scale by 
        # normalizing the vector x itself? No, d_max is not a norm of x.
        
        # Alternative: Minimize d_max - lambda * d_min?
        # Let's stick to the "Repulsive Potential" + "Normalization" idea but carefully.
        
        # If we just minimize the repulsive potential V = sum(1/d^2), the points will fly apart.
        # We need a bound.
        
        # Let's try minimizing: d_max - d_min.
        # This is non-smooth.
        
        # Let's use the "Soft" approach:
        # Minimize: - (sum exp(-k * d))^(1/k)  approximates -d_min for large k.
        # But we also need to control d_max.
        
        # A common trick: Minimize (d_max^2 + d_min^2) / 2 - d_min * d_max? No.
        
        # Let's use the method from Inspiration 1: "Smooth gradient descent on a repulsive potential".
        # Usually, this implies minimizing V = sum 1/d_ij^p.
        # To prevent explosion, we can normalize the points to have unit RMS radius or similar, 
        # but that doesn't fix d_max.
        
        # Let's try: Minimize V_repulsive - lambda * V_attractive?
        
        # Actually, the simplest robust method for this specific metric is:
        # 1. Normalize so d_max = 1.
        # 2. Maximize d_min.
        
        # Since normalization is tricky for gradients, let's define the objective as:
        # f(x) = d_max(x) - softmin(x, k)
        # This is still non-smooth due to d_max.
        
        # Let's try a different objective:
        # Minimize: log-sum-exp(-d) which approximates -d_min.
        # And constrain d_max <= 1.
        
        # Let's use L-BFGS-B with bounds? No, bounds are per coordinate.
        
        # Let's go back to: Minimize Repulsive Potential V = sum 1/d^2.
        # This configuration will have a certain d_min and d_max.
        # The ratio might be good.
        # We can run multiple restarts with different initializations and powers.
        
        # Let's implement a function that minimizes V = sum 1/d^p.
        # And then we check the score.
        
        # We will do this outside the objective for the main search, 
        # but inside the objective we need a smooth function.
        
        # Let's use: Objective = sum(1/d^2)
        # This pushes points apart.
        # We need to stop them from going to infinity.
        # We can add a penalty if any distance > R_max?
        # Or simply normalize the points to have norm 1?
        # If we constrain points to be on the unit sphere, d_max <= 2.
        # The optimal configuration for max d_min on a sphere is well-studied.
        # For N=14, the optimal arrangement on the sphere is likely close to an icosahedron + 2 points.
        
        # Let's assume the optimal points lie on the unit sphere.
        # If we force them on the sphere, d_max is fixed (<=2).
        # Then we just maximize d_min.
        
        # Constraint: ||p_i|| = 1.
        # L-BFGS-B doesn't handle non-linear constraints well.
        # We can parameterize points on the sphere?
        # Or use a penalty: sum (||p_i|| - 1)^2.
        
        def sphere_obj(x):
            pts = x.reshape(n, d)
            # Penalty for not being on sphere
            pen = np.sum((np.linalg.norm(pts, axis=1) - 1.0)**2)
            
            # Repulsive potential to maximize d_min
            # We want to maximize min distance.
            # Minimize -softmin(dist)
            dists = pdist(pts)
            # Avoid zero distance
            dists = np.maximum(dists, 1e-6)
            
            # Soft minimum approximation
            # Using log-sum-exp trick for stability
            k = 20.0
            # -1/k * log(sum(exp(-k*d)))
            # To avoid overflow/underflow:
            min_d = np.min(dists)
            exp_terms = np.exp(-k * (dists - min_d))
            soft_min_val = -min_d - (1.0/k) * np.log(np.sum(exp_terms))
            
            # We want to maximize soft_min_val, so minimize -soft_min_val
            return pen - soft_min_val

        return sphere_obj(x)

    # This objective is complex. Let's simplify.
    # The metric is scale invariant.
    # Let's just use the Repulsive Potential V = sum(1/d^2) without sphere constraint, 
    # but normalize the result before scoring.
    # Wait, if we minimize V, the points expand. The ratio d_min/d_max might stay constant or change.
    # For a regular simplex, V is minimized when it's as "spread" as possible?
    
    # Let's try a direct approach:
    # Objective: Minimize (d_max - d_min).
    # Smoothed: Minimize (logsumexp(d) - logsumexp(-d))? No.
    
    # Let's use the "Soft" objective:
    # f(x) = logsumexp(dists) - logsumexp(-dists)  ?
    
    # Actually, let's look at the inspiration again.
    # "Smooth gradient descent (L-BFGS-B) on a repulsive potential".
    # This usually means:
    # Minimize V(x) = sum_{i<j} 1/||x_i - x_j||^p
    # This is a standard technique for Tammes problem (max min distance).
    # It finds a local maximum of the minimum distance.
    # Since the metric is (d_min/d_max)^2, and d_max is also affected, 
    # for points on a sphere, d_max is bounded, so maximizing d_min is good.
    # If points are not on a sphere, d_max grows.
    
    # Let's assume the optimal solution is on the sphere.
    # We will use L-BFGS-B to minimize V(x) = sum 1/d^2 + lambda * sum(||x_i||^2 - 1)^2.
    
    def repulsive_obj(x, p=2.0, lambda_sphere=100.0):
        pts = x.reshape(n, d)
        
        # Sphere constraint penalty
        norms = np.linalg.norm(pts, axis=1)
        sphere_pen = lambda_sphere * np.sum((norms - 1.0)**2)
        
        # Repulsive potential
        dists = pdist(pts)
        # Avoid division by zero
        dists = np.maximum(dists, 1e-6)
        
        # Sum of inverse powers
        # We want to maximize d_min, which corresponds to minimizing 1/d_min.
        # Sum 1/d^p is a good proxy.
        V = np.sum(1.0 / np.power(dists, p))
        
        return V + sphere_pen

    def run_l_bfgs(start_pts):
        x0 = start_pts.flatten()
        
        # Define objective with fixed params
        def obj(x):
            return repulsive_obj(x, p=2.0, lambda_sphere=100.0)
            
        # Gradient is not provided, so L-BFGS-B will use numerical differentiation.
        # For N=14, D=3, 42 variables. Numerical gradient is expensive but manageable?
        # 42 function calls per gradient step.
        # Let's provide an analytical gradient if possible?
        # dV/dx_i = sum_{j != i} -p * (x_i - x_j) / ||x_i - x_j||^(p+2)
        # dPen/dx_i = 2 * lambda * (||x_i|| - 1) * x_i / ||x_i||
        
        def grad(x):
            pts = x.reshape(n, d)
            g = np.zeros_like(pts)
            
            # Sphere gradient
            norms = np.linalg.norm(pts, axis=1, keepdims=True)
            # Avoid div by 0
            norms_safe = np.maximum(norms, 1e-6)
            sphere_grad = 2.0 * 100.0 * (norms - 1.0) * pts / norms_safe
            g += sphere_grad
            
            # Repulsive gradient
            # V = sum_{i<j} (1/d_ij^p)
            # dV/dx_i = sum_{j!=i} -p * (x_i - x_j) * d_ij^(-p-2)
            
            # Vectorized calculation
            # Diff matrix: (n, n, d)
            diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
            dists_mat = np.linalg.norm(diff, axis=2)
            # Mask diagonal
            mask = ~np.eye(n, dtype=bool)
            
            # We need sum over j for each i
            # Term: -p * (x_i - x_j) / d_ij^(p+2)
            # p = 2
            p = 2.0
            # Avoid div by 0 on diagonal
            dists_mat_safe = np.where(mask, dists_mat, 1.0)
            
            # Factor: 1 / d^(p+2)
            factor = np.where(mask, 1.0 / np.power(dists_mat_safe, p+2), 0.0)
            
            # Gradient contribution from repulsive force
            # g_i = sum_j -p * diff_ij * factor_ij
            repulsive_grad = -p * np.sum(diff * factor[:, :, np.newaxis], axis=1)
            
            g += repulsive_grad
            
            return g.flatten()

        res = minimize(obj, x0, jac=grad, method='L-BFGS-B', 
                       options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-8})
        
        best_pts = res.x.reshape(n, d)
        return best_pts

    # Initial population
    def get_initial_points():
        # Icosahedron + 2 random on sphere
        phi = (1 + np.sqrt(5)) / 2
        ico = np.array([
            [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
            [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
            [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
        ])
        ico = ico / np.linalg.norm(ico, axis=1, keepdims=True)
        
        # 2 random points on sphere
        v1 = np.random.randn(3)
        v1 /= np.linalg.norm(v1)
        v2 = np.random.randn(3)
        v2 /= np.linalg.norm(v2)
        
        pts = np.vstack([ico, [v1, v2]])
        return pts

    # Run multiple restarts
    num_restarts = 0
    while time.time() < deadline:
        start_pts = get_initial_points()
        
        # Run L-BFGS-B
        try:
            local_best = run_l_bfgs(start_pts)
            local_score = get_score(local_best)
            
            if local_score > best_score:
                best_score = local_score
                best_points = local_best
        except Exception:
            pass
        
        num_restarts += 1
        
        # If we are very good, break?
        if best_score > 0.9:
            break

    return best_points
