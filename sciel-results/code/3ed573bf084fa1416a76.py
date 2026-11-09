import numpy as np
from scipy.spatial.distance import pdist
import time

def min_max_dist_dim3_14() -> np.ndarray:
    start_time = time.time()
    timeout = 600
    n = 14

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

    def soft_min_gradient(dist_matrix, mask, temp=0.01):
        """
        Compute gradient of -log(sum(exp(-d_ij/temp))) with respect to points.
        This is a smooth approximation to max(d_ij) minimization (i.e. minimizing the largest distance).
        Actually, we want to maximize the minimum distance.
        Let's use a smooth approximation for the minimum distance maximization.
        Objective: maximize min_{i<j} d_ij.
        Surrogate: minimize -log(sum_{i<j} exp(-d_ij / temp))
        Gradient of this surrogate w.r.t p_i:
        f = -log(sum exp(-d_ij/temp))
        df/dp_i = -1/sum * sum [ exp(-d_ij/temp) * (-1/temp) * df_ij/dp_i ]
        f_ij = -d_ij / temp
        exp(f_ij) * df_ij/dp_i = exp(-d_ij/temp) * (-1/temp) * (p_i - p_j)/d_ij
        So df/dp_i = -1/S * sum_j [ exp(-d_ij/temp) * (-1/temp) * (p_i - p_j)/d_ij ]
        df/dp_i = (1/(S*temp)) * sum_j [ exp(-d_ij/temp) * (p_i - p_j)/d_ij ]
        where S = sum_j exp(-d_ij/temp)
        
        We want to MAXIMIZE min dist, so we ASCEND this gradient.
        """
        n = len(points)
        # dist_matrix is n x n
        # mask is n x n boolean, True for i != j
        # We need to compute weights w_ij = exp(-d_ij / temp) for i != j
        
        # To avoid underflow/overflow, subtract max exponent
        # exponents = -dist_matrix / temp
        # But dist_matrix has inf on diagonal. We mask it.
        
        # Let's work with the masked matrix
        # d_masked = np.where(mask, dist_matrix, np.inf)
        # exponents = -d_masked / temp
        # max_exp = np.max(exponents) # This is 0 for the smallest distance (largest exponent)
        # Actually, since d > 0, -d/temp < 0. The max exponent is closest to 0, corresponding to min d.
        
        # For numerical stability:
        # weights = exp(-d_ij / temp)
        # If d_ij is large, weight is ~0. If d_ij is small, weight is ~1.
        
        # Let's compute the gradient efficiently.
        # grad_i = (1/(S_i * temp)) * sum_{j != i} [ w_ij * (p_i - p_j) / d_ij ]
        # where S_i = sum_{j != i} w_ij
        
        grad = np.zeros_like(points)
        
        # Compute weights matrix
        # dist_matrix has inf on diagonal (due to mask in caller? No, mask is passed)
        # Let's create a safe distance matrix for exponentiation
        d_safe = np.where(mask, dist_matrix, 1e9)
        
        # Compute exponents
        exponents = -d_safe / temp
        
        # For stability, subtract the max exponent for each row? 
        # The sum S_i is per row i.
        # max_exp_i = max(exponents[i, :])
        # w_ij = exp(exponents[i,j] - max_exp_i)
        # Then S_i = sum(w_ij)
        # And the term inside sum is w_ij * (p_i - p_j) / d_ij
        
        max_exp = np.max(exponents, axis=1, keepdims=True)
        # Handle case where all are -inf? Not possible since d is finite for i!=j.
        # But if mask is all False? No, i!=j is always true for some j.
        
        # If max_exp is -inf (all d are inf), handle separately.
        # But d_safe is 1e9 for diagonal, so exponents are -1e9/temp, which is very small.
        # The max will be from the actual neighbors.
        
        # Shift exponents
        shifted_exponents = exponents - max_exp
        weights = np.exp(shifted_exponents)
        
        # Zero out diagonal (and any masked entries, though d_safe handles diagonal)
        weights[~mask] = 0
        
        # Compute S_i
        S = np.sum(weights, axis=1, keepdims=True)
        # Avoid division by zero
        S[S == 0] = 1e-9
        
        # Compute weighted sum of vectors
        # We need sum_j [ w_ij * (p_i - p_j) / d_ij ]
        # (p_i - p_j) is diff[i, j]
        # d_ij is d_safe[i, j] (for i!=j)
        
        # Create a matrix of 1/d_ij
        inv_d = np.where(mask, 1.0 / d_safe, 0.0)
        
        # Weighted vectors: weights[i,j] * inv_d[i,j] * diff[i,j]
        # diff is n x n x 3
        # weights is n x n
        # inv_d is n x n
        
        # Let's compute the sum for each i
        # term_ij = weights[i,j] * inv_d[i,j] * (p_i - p_j)
        # grad_i = (1/(S_i * temp)) * sum_j term_ij
        
        # Vectorized approach:
        # coeff[i,j] = weights[i,j] * inv_d[i,j] / (S_i * temp)
        coeff = weights * inv_d / (S * temp)
        
        # Now, grad_i = sum_j coeff[i,j] * (p_i - p_j)
        # = (sum_j coeff[i,j]) * p_i - sum_j coeff[i,j] * p_j
        
        # sum_coeff_i = sum_j coeff[i,j]
        sum_coeff = np.sum(coeff, axis=1, keepdims=True)
        
        # sum_j coeff[i,j] * p_j
        # This is coeff @ points
        weighted_points = coeff @ points
        
        grad = sum_coeff * points - weighted_points
        
        return grad

    def optimize_smooth_min_dist(points, iterations=10000, lr=0.01, temp_init=0.05, temp_decay=0.999):
        """
        Maximize the minimum distance using a smooth surrogate (log-sum-exp).
        """
        points = project_to_sphere(points.copy())
        n = len(points)
        temp = temp_init
        
        for i in range(iterations):
            if time.time() - start_time > timeout - 5:
                break
                
            diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
            dist_matrix = np.linalg.norm(diff, axis=2)
            mask = ~np.eye(n, dtype=bool)
            dist_matrix[~mask] = np.inf # Ensure diagonal is ignored in min/max logic, but we use d_safe in gradient
            
            # Calculate gradient of the smooth surrogate
            grad = soft_min_gradient(dist_matrix, mask, temp)
            
            # Project to tangent space
            for k in range(n):
                p_k = points[k]
                g_k = grad[k]
                proj = np.dot(g_k, p_k)
                grad[k] = g_k - proj * p_k
            
            # Update: Ascend the gradient (maximize the surrogate)
            points = points + lr * grad
            points = project_to_sphere(points)
            
            # Decay temperature
            temp *= temp_decay
            # Decay learning rate slowly
            if i % 100 == 0:
                lr *= 0.98

        return points

    def polish_min_dist(points, iterations=2000, lr=0.005):
        """
        Final polish to strictly maximize min distance using exact min pairs.
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
            
            # Find all pairs with distance close to d_min
            # Use a small tolerance to include near-min pairs
            tol = 1e-8
            min_pairs = np.argwhere(np.abs(dist_matrix - d_min) < tol)
            
            # Filter out self-pairs just in case (though dist_matrix diag is inf)
            min_pairs = min_pairs[(min_pairs[:, 0] != min_pairs[:, 1])]
            
            grad = np.zeros_like(points)
            for idx in min_pairs:
                i_idx, j_idx = idx[0], idx[1]
                vec = points[i_idx] - points[j_idx]
                d = np.linalg.norm(vec)
                if d > 1e-8:
                    unit = vec / d
                    grad[i_idx] += unit
                    grad[j_idx] -= unit
            
            for k in range(n):
                p_k = points[k]
                g_k = grad[k]
                proj = np.dot(g_k, p_k)
                grad[k] = g_k - proj * p_k
            
            points = points + lr * grad # Ascend on min dist
            points = project_to_sphere(points)
            
            if i > 500:
                lr *= 0.999
                
        return points

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
    
    # 2. Random seeds
    for seed in [42, 123, 7, 99, 2024]:
        np.random.seed(seed)
        inits.append(np.random.randn(n, 3))

    for init in inits:
        if time.time() - start_time > timeout - 20:
            break
            
        # Phase 1: Spread out using energy minimization
        p1 = optimize_energy(init, iterations=2000, lr=0.02, beta=4.0)
        
        # Phase 2: Smooth optimization of min distance
        p2 = optimize_smooth_min_dist(p1, iterations=5000, lr=0.01, temp_init=0.05, temp_decay=0.999)
        
        # Phase 3: Polish min distance
        p3 = polish_min_dist(p2, iterations=2000, lr=0.005)
        
        score = calculate_score(p3)
        if score > best_score:
            best_score = score
            best_points = p3

    if best_points is None:
        best_points = np.random.randn(n, 3)
        
    return best_points
