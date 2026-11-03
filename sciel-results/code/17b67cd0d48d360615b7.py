import numpy as np
import time
from scipy.spatial import ConvexHull
from scipy.optimize import minimize

N = 130

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    def fibonacci_sphere(n):
        i = np.arange(n) + 0.5
        z = 1.0 - 2.0 * i / n
        phi = np.pi * (1.0 + np.sqrt(5.0)) * i
        r = np.sqrt(1.0 - z * z)
        return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    points = fibonacci_sphere(N)
    best_points = points.copy()
    
    def compute_radius_and_grad(x):
        # x is (N, 3)
        # Returns radius, gradient (N, 3)
        pts = x / np.linalg.norm(x, axis=1, keepdims=True)
        
        try:
            hull = ConvexHull(pts)
            faces = hull.simplices
        except Exception:
            return float('inf'), np.zeros_like(pts)
            
        F = len(faces)
        if F == 0:
            return float('inf'), np.zeros_like(pts)
            
        p1 = pts[faces[:, 0]]
        p2 = pts[faces[:, 1]]
        p3 = pts[faces[:, 2]]
        
        v1 = p1 - p2
        v2 = p3 - p2
        cross = np.cross(v1, v2)
        norms = np.linalg.norm(cross, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        centers = cross / norms
        
        # Compute min distance for each center
        dots = centers @ pts.T
        dots = np.clip(dots, -1.0, 1.0)
        dists = np.arccos(dots)
        min_dists = np.min(dists, axis=1)
        min_indices = np.argmin(dists, axis=1)
        
        max_dist = np.max(min_dists)
        
        # Smooth max approximation for gradient
        # J = logsumexp(beta * r^2) / beta
        # We want grad of J
        # Let's use a high beta
        beta = 100.0
        r_sq = min_dists ** 2
        exp_terms = np.exp(beta * (r_sq - np.max(r_sq)))
        weights = exp_terms / np.sum(exp_terms)
        
        # Grad of J w.r.t pts
        # J = sum_f w_f * r_f^2
        # dJ/dp_k = sum_f w_f * 2 r_f * d(r_f)/dp_k
        # r_f = arccos(c_f . p_{m_f})
        # d(r_f)/dp_k = -1/sqrt(1-s^2) * d(c_f . p_{m_f})/dp_k
        
        grad = np.zeros_like(pts)
        
        # We only need to compute gradient for points involved in faces with significant weight
        # For simplicity and speed, we might approximate
        # But let's try to compute for all
        
        # This is expensive. 
        # Alternative: Use FD for gradient in L-BFGS-B.
        
        return max_dist, grad # Placeholder

    # Given the complexity of the analytical gradient, 
    # I will use L-BFGS-B with FD gradient.
    
    def objective_and_grad(x):
        if time.time() > deadline:
            raise ValueError("Time limit exceeded")
        pts = x.reshape(N, 3)
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        pts = pts / norms
        
        # Compute radius
        try:
            hull = ConvexHull(pts)
            faces = hull.simplices
            p1 = pts[faces[:, 0]]
            p2 = pts[faces[:, 1]]
            p3 = pts[faces[:, 2]]
            v1 = p1 - p2
            v2 = p3 - p2
            cross = np.cross(v1, v2)
            norms_c = np.linalg.norm(cross, axis=1, keepdims=True)
            norms_c[norms_c == 0] = 1.0
            centers = cross / norms_c
            dots = centers @ pts.T
            dots = np.clip(dots, -1.0, 1.0)
            dists = np.arccos(dots)
            min_dists = np.min(dists, axis=1)
            val = np.max(min_dists)
        except Exception:
            val = float('inf')
            
        # FD Gradient
        eps = 1e-5
        grad = np.zeros_like(x)
        # Only compute gradient for a subset of points to save time?
        # No, L-BFGS needs full grad.
        # But we can stop if time is low.
        
        for i in range(N):
            if time.time() > deadline - 5.0:
                break
            for j in range(3):
                orig = x[i, j]
                x[i, j] = orig + eps
                val_plus = objective_and_grad_helper(x)
                x[i, j] = orig - eps
                val_minus = objective_and_grad_helper(x)
                x[i, j] = orig
                grad[i, j] = (val_plus - val_minus) / (2 * eps)
                
        return val, grad

    def objective_and_grad_helper(x):
        pts = x.reshape(N, 3)
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        pts = pts / norms
        try:
            hull = ConvexHull(pts)
            faces = hull.simplices
            p1 = pts[faces[:, 0]]
            p2 = pts[faces[:, 1]]
            p3 = pts[faces[:, 2]]
            v1 = p1 - p2
            v2 = p3 - p2
            cross = np.cross(v1, v2)
            norms_c = np.linalg.norm(cross, axis=1, keepdims=True)
            norms_c[norms_c == 0] = 1.0
            centers = cross / norms_c
            dots = centers @ pts.T
            dots = np.clip(dots, -1.0, 1.0)
            dists = np.arccos(dots)
            min_dists = np.min(dists, axis=1)
            return np.max(min_dists)
        except Exception:
            return float('inf')

    x0 = points.flatten()
    
    def callback(xk):
        if time.time() > deadline:
            raise ValueError("Time limit exceeded")
        return False

    try:
        res = minimize(objective_and_grad, x0, method='L-BFGS-B', jac=True,
                       options={'maxiter': 100, 'ftol': 1e-6, 'disp': False},
                       callback=callback)
        if res.fun < float('inf'):
            pts = res.x.reshape(N, 3)
            norms = np.linalg.norm(pts, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            pts = pts / norms
            # Verify
            best_points = pts
    except Exception:
        pass

    return best_points
# EVOLVE-BLOCK-END
