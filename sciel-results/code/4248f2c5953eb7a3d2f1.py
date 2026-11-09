import numpy as np
from scipy.spatial.distance import pdist
import time
from scipy.optimize import minimize

def min_max_dist_dim3_14() -> np.ndarray:
    start_time = time.time()
    timeout = 600
    
    def calculate_score(points):
        dists = pdist(points)
        if len(dists) == 0:
            return 0.0
        d_min = np.min(dists)
        d_max = np.max(dists)
        if d_max == 0:
            return 0.0
        return (d_min / d_max) ** 2

    def project_to_sphere(points):
        norms = np.linalg.norm(points, axis=1, keepdims=True)
        norms[norms == 0] = 1
        return points / norms

    def optimize_energy(points, iterations=5000, lr=0.01, beta=2.0):
        """
        Minimize repulsive energy E = - sum 1/d_ij^beta on the sphere.
        This helps distribute points evenly.
        """
        points = project_to_sphere(points.copy())
        n = len(points)
        
        for i in range(iterations):
            if time.time() - start_time > timeout - 5:
                break
                
            # Calculate distances
            diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
            dist_matrix = np.linalg.norm(diff, axis=2)
            mask = ~np.eye(n, dtype=bool)
            dist_matrix[~mask] = 1e9
            
            grad = np.zeros_like(points)
            for k in range(n):
                p_k = points[k]
                d_kj = dist_matrix[k]
                d_kj[k] = 1e9
                diff_k = diff[k]
                weights = np.power(d_kj, -beta - 2)
                weights[k] = 0
                grad_k = np.sum(diff_k * weights[:, np.newaxis], axis=0)
                grad[k] = beta * grad_k
            
            # Project to tangent space
            for k in range(n):
                p_k = points[k]
                g_k = grad[k]
                proj = np.dot(g_k, p_k)
                grad[k] = g_k - proj * p_k
            
            points = points - lr * grad
            points = project_to_sphere(points)
            
            if i % 100 == 0:
                lr *= 0.95
                
        return points

    def polish_min_dist(points, iterations=2000, lr=0.005):
        """
        Final polish to strictly maximize min distance.
        """
        points = project_to_sphere(points.copy())
        n = len(points)
        
        for i in range(iterations):
            if time.time() - start_time > timeout - 2:
                break
                
            diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
            dist_matrix = np.linalg.norm(diff, axis=2)
            mask = ~np.eye(n, dtype=bool)
            dist_matrix[~mask] = np.inf
            d_min = np.min(dist_matrix)
            
            min_pairs = np.argwhere(dist_matrix == d_min)
            
            grad = np.zeros_like(points)
            for idx in min_pairs:
                i, j = idx[0], idx[1]
                vec = points[i] - points[j]
                d = np.linalg.norm(vec)
                if d > 1e-8:
                    unit = vec / d
                    grad[i] += unit
                    grad[j] -= unit
            
            for k in range(n):
                p_k = points[k]
                g_k = grad[k]
                proj = np.dot(g_k, p_k)
                grad[k] = g_k - proj * p_k
            
            points = points + lr * grad
            points = project_to_sphere(points)
            
            if i > 500:
                lr *= 0.999
                
        return points

    def high_precision_polish(points, iterations=5000, lr=0.001):
        """
        High-precision polish with smaller step size and more iterations.
        """
        points = project_to_sphere(points.copy())
        n = len(points)
        
        for i in range(iterations):
            if time.time() - start_time > timeout - 1:
                break
                
            diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
            dist_matrix = np.linalg.norm(diff, axis=2)
            mask = ~np.eye(n, dtype=bool)
            dist_matrix[~mask] = np.inf
            d_min = np.min(dist_matrix)
            
            min_pairs = np.argwhere(dist_matrix == d_min)
            
            grad = np.zeros_like(points)
            for idx in min_pairs:
                i, j = idx[0], idx[1]
                vec = points[i] - points[j]
                d = np.linalg.norm(vec)
                if d > 1e-8:
                    unit = vec / d
                    grad[i] += unit
                    grad[j] -= unit
            
            for k in range(n):
                p_k = points[k]
                g_k = grad[k]
                proj = np.dot(g_k, p_k)
                grad[k] = g_k - proj * p_k
            
            points = points + lr * grad
            points = project_to_sphere(points)
            
            if i > 2000:
                lr *= 0.9995
                
        return points

    def soft_min_objective(params, temperature=0.01):
        """
        Smooth surrogate for minimizing -log(soft_min(distances)).
        Uses log-sum-exp to approximate min distance smoothly.
        """
        n = 14
        points = params.reshape(n, 3)
        points = project_to_sphere(points)
        
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dist_matrix = np.linalg.norm(diff, axis=2)
        mask = ~np.eye(n, dtype=bool)
        dists = dist_matrix[mask]
        
        # Soft min using log-sum-exp: -log(sum(exp(-d/T))) * T
        # This approximates min(d) when T is small
        exp_neg_d = np.exp(-dists / temperature)
        soft_min = -temperature * np.log(np.sum(exp_neg_d))
        
        # We want to maximize soft_min, so minimize -soft_min
        return -soft_min

    def soft_min_gradient(params, temperature=0.01):
        """
        Gradient of soft_min_objective.
        """
        n = 14
        points = params.reshape(n, 3)
        points = project_to_sphere(points)
        
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dist_matrix = np.linalg.norm(diff, axis=2)
        mask = ~np.eye(n, dtype=bool)
        
        # Compute soft min weights
        exp_neg_d = np.exp(-dist_matrix / temperature)
        exp_neg_d[~mask] = 0
        sum_exp = np.sum(exp_neg_d, axis=1, keepdims=True)
        weights = exp_neg_d / sum_exp
        
        # Gradient of -soft_min w.r.t. points
        # d(-soft_min)/dp_i = -d(soft_min)/dp_i
        # soft_min = -T * log(sum_j exp(-d_ij/T))
        # d(soft_min)/dp_i = -T * (1/sum_exp) * sum_j exp(-d_ij/T) * d(-d_ij/T)/dp_i
        #                  = -T * sum_j weights_ij * (-1/T) * d(d_ij)/dp_i
        #                  = sum_j weights_ij * d(d_ij)/dp_i
        # d(d_ij)/dp_i = (p_i - p_j) / d_ij
        
        grad = np.zeros_like(points)
        for k in range(n):
            d_kj = dist_matrix[k].copy()
            d_kj[k] = 1e9
            weights_k = weights[k]
            weights_k[k] = 0
            diff_k = diff[k]
            
            # d(d_kj)/dp_k = (p_k - p_j) / d_kj
            unit_vecs = diff_k / d_kj[:, np.newaxis]
            grad[k] = np.sum(weights_k[:, np.newaxis] * unit_vecs, axis=0)
        
        # Project to tangent space
        for k in range(n):
            p_k = points[k]
            g_k = grad[k]
            proj = np.dot(g_k, p_k)
            grad[k] = g_k - proj * p_k
            
        return grad.reshape(-1)

    def polish_with_soft_min(points, iterations=2000, lr=0.01, temperature=0.01):
        """
        Polish using soft-min surrogate with gradient descent.
        """
        points = project_to_sphere(points.copy())
        n = 14
        
        for i in range(iterations):
            if time.time() - start_time > timeout - 2:
                break
                
            params = points.reshape(-1)
            grad = soft_min_gradient(params, temperature)
            
            # Move in direction of increasing soft_min (decreasing objective)
            points = points - lr * grad.reshape(n, 3)
            points = project_to_sphere(points)
            
            # Decay learning rate
            if i % 100 == 0:
                lr *= 0.98
                
            # Decrease temperature for sharper min approximation
            if i % 500 == 0 and i > 0:
                temperature *= 0.5
                
        return points

    n = 14
    best_score = 0.0
    best_points = None
    
    # Initializations
    inits = []
    
    # 1. Fibonacci
    phi = np.pi * (3.0 - np.sqrt(5.0))
    y = np.linspace(-1, 1, n, endpoint=False) + 0.5/n
    r = np.sqrt(1 - y**2)
    theta = phi * np.arange(n)
    x = np.cos(theta) * r
    z = np.sin(theta) * r
    inits.append(np.column_stack((x, y, z)))
    
    # 2. Random
    np.random.seed(42)
    inits.append(np.random.randn(n, 3))
    
    # 3. Random with different seed
    np.random.seed(123)
    inits.append(np.random.randn(n, 3))
    
    # 4. More random seeds for diversity
    for seed in [7, 99, 2024]:
        np.random.seed(seed)
        inits.append(np.random.randn(n, 3))

    for init in inits:
        if time.time() - start_time > timeout - 30:
            break
            
        # Phase 1: Spread out using energy minimization
        p1 = optimize_energy(init, iterations=2000, lr=0.02, beta=4.0)
        
        # Phase 2: Polish min distance
        p2 = polish_min_dist(p1, iterations=2000, lr=0.005)
        
        # Phase 3: Soft-min polish for smoother optimization
        p3 = polish_with_soft_min(p2, iterations=1500, lr=0.01, temperature=0.02)
        
        # Phase 4: High-precision final polish
        p4 = high_precision_polish(p3, iterations=3000, lr=0.001)
        
        score = calculate_score(p4)
        if score > best_score:
            best_score = score
            best_points = p4

    if best_points is None:
        best_points = np.random.randn(n, 3)
        
    return best_points
