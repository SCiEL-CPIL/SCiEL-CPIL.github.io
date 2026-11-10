# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial.distance import pdist
from scipy.optimize import minimize

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.
    
    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.
    """
    n = 16
    d = 2
    start_time = time.time()
    deadline = start_time + 600  # 600 seconds
    
    def compute_score(points):
        """Compute R = (d_min / d_max)^2"""
        if len(points) < 2:
            return 0.0
        dists = pdist(points)
        d_min = np.min(dists)
        d_max = np.max(dists)
        if d_max == 0:
            return 0.0
        return (d_min / d_max) ** 2
    
    def objective(x, soft_param=10.0):
        """
        Objective function: minimize -log(soft_min) + log(soft_max)
        This is equivalent to minimizing log(d_max/d_min) with smooth approximations.
        """
        points = x.reshape(n, 2)
        # Compute all pairwise distances
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        # Zero out diagonal
        np.fill_diagonal(dists, np.inf)
        
        # Soft min and max using log-sum-exp trick
        # soft_min = -logsumexp(-dists/soft_param) * soft_param
        # soft_max = logsumexp(dists/soft_param) * soft_param
        
        # For numerical stability, use the log-sum-exp formula
        # soft_min: we want to approximate min(dists)
        # min(x) ≈ -1/t * log(sum(exp(-t*x)))
        # soft_max: we want to approximate max(dists)
        # max(x) ≈ 1/t * log(sum(exp(t*x)))
        
        t = soft_param
        
        # For soft_min:
        neg_dists = -dists / t
        # Subtract max for numerical stability
        neg_dists_max = np.max(neg_dists)
        soft_min_log = -t * (neg_dists_max + np.log(np.sum(np.exp(neg_dists - neg_dists_max))))
        soft_min = -soft_min_log  # This gives us the soft_min value
        
        # For soft_max:
        dists_scaled = dists / t
        dists_max = np.max(dists_scaled)
        soft_max_log = t * (dists_max + np.log(np.sum(np.exp(dists_scaled - dists_max))))
        soft_max = soft_max_log
        
        if soft_min <= 0 or soft_max <= 0:
            return 1e10
            
        return np.log(soft_max) - np.log(soft_min)
    
    def gradient(x, soft_param=10.0):
        """Gradient of the objective function"""
        points = x.reshape(n, 2)
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2) + 1e-12)
        np.fill_diagonal(dists, np.inf)
        
        t = soft_param
        
        # Gradients for soft_min
        neg_dists = -dists / t
        neg_dists_max = np.max(neg_dists)
        weights_min = np.exp(neg_dists - neg_dists_max) / np.sum(np.exp(neg_dists - neg_dists_max))
        
        # Gradients for soft_max
        dists_scaled = dists / t
        dists_max = np.max(dists_scaled)
        weights_max = np.exp(dists_scaled - dists_max) / np.sum(np.exp(dists_scaled - dists_max))
        
        # Gradient of log(soft_max) - log(soft_min)
        # d/dx log(soft_max) = (1/soft_max) * d(soft_max)/dx
        # d/dx log(soft_min) = (1/soft_min) * d(soft_min)/dx
        
        # soft_max = t * logsumexp(dists/t)
        # d(soft_max)/dx_i = sum_j weights_max[i,j] * (diff[i,j] / dists[i,j])
        
        # soft_min = -t * logsumexp(-dists/t)
        # d(soft_min)/dx_i = -sum_j weights_min[i,j] * (diff[i,j] / dists[i,j])
        
        grad = np.zeros_like(x)
        
        # Compute contribution to gradient from each pair
        for i in range(n):
            for j in range(i+1, n):
                if dists[i,j] < 1e-10:
                    continue
                # Direction from i to j
                dir_ij = diff[i, j] / dists[i, j]
                
                # Contribution to point i from soft_max
                grad[2*i:2*i+2] += weights_max[i,j] * dir_ij / t
                # Contribution to point j from soft_max
                grad[2*j:2*j+2] -= weights_max[i,j] * dir_ij / t
                
                # Contribution to point i from soft_min (with negative sign)
                grad[2*i:2*i+2] += weights_min[i,j] * dir_ij / t
                # Contribution to point j from soft_min (with negative sign)
                grad[2*j:2*j+2] -= weights_min[i,j] * dir_ij / t
        
        return grad
    
    # Initial guess: hexagonal lattice pattern
    def generate_hexagonal_grid():
        """Generate points in a hexagonal lattice pattern"""
        points = []
        # Use a hexagonal packing arrangement
        # For 16 points, use a 4x4 arrangement with hexagonal offsets
        for i in range(4):
            for j in range(4):
                x = i * 1.0
                y = j * np.sqrt(3) / 2
                if i % 2 == 1:
                    y += np.sqrt(3) / 4
                points.append([x, y])
        points = np.array(points)
        # Center the points
        points -= np.mean(points, axis=0)
        # Scale to unit variance
        std = np.std(points)
        if std > 0:
            points /= std
        return points
    
    def generate_circular_grid():
        """Generate points in concentric circles"""
        points = []
        # Center point
        points.append([0, 0])
        # Ring 1: 5 points
        for k in range(5):
            angle = 2 * np.pi * k / 5
            points.append([np.cos(angle), np.sin(angle)])
        # Ring 2: 10 points
        for k in range(10):
            angle = 2 * np.pi * k / 10 + np.pi / 10
            points.append([1.8 * np.cos(angle), 1.8 * np.sin(angle)])
        points = np.array(points)
        # Center the points
        points -= np.mean(points, axis=0)
        # Scale to unit variance
        std = np.std(points)
        if std > 0:
            points /= std
        return points
    
    # Try multiple initializations
    initializations = []
    
    # Hexagonal grid
    initializations.append(generate_hexagonal_grid())
    
    # Circular grid
    initializations.append(generate_circular_grid())
    
    # Random initializations
    np.random.seed(42)
    for _ in range(5):
        points = np.random.randn(n, 2)
        points -= np.mean(points, axis=0)
        std = np.std(points)
        if std > 0:
            points /= std
        initializations.append(points)
    
    # Grid initialization
    x_grid = np.linspace(-1.5, 1.5, 4)
    y_grid = np.linspace(-1.5, 1.5, 4)
    xx, yy = np.meshgrid(x_grid, y_grid)
    grid_points = np.column_stack((xx.ravel(), yy.ravel()))
    grid_points -= np.mean(grid_points, axis=0)
    std = np.std(grid_points)
    if std > 0:
        grid_points /= std
    initializations.append(grid_points)
    
    best_points = None
    best_score = -1
    
    for init in initializations:
        if time.time() > deadline - 30:  # Leave 30 seconds margin
            break
        
        x0 = init.flatten()
        
        # Multi-stage optimization with decreasing soft_param
        for soft_param in [20.0, 10.0, 5.0, 2.0, 1.0]:
            if time.time() > deadline - 30:
                break
                
            result = minimize(
                objective,
                x0,
                args=(soft_param,),
                jac=gradient,
                method='L-BFGS-B',
                options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            
            if result.success or result.fun < 1e10:
                x0 = result.x
        
        # Evaluate final score
        points = x0.reshape(n, 2)
        score = compute_score(points)
        
        if score > best_score:
            best_score = score
            best_points = points.copy()
    
    # If we have time, do a final polish with exact min/max
    if time.time() < deadline - 10 and best_points is not None:
        x0 = best_points.flatten()
        
        def exact_objective(x):
            """Minimize -log(d_min) + log(d_max) using exact min/max"""
            points = x.reshape(n, 2)
            dists = pdist(points)
            d_min = np.min(dists)
            d_max = np.max(dists)
            if d_min <= 0 or d_max <= 0:
                return 1e10
            return np.log(d_max) - np.log(d_min)
        
        # Use SLSQP for final polish
        result = minimize(
            exact_objective,
            x0,
            method='SLSQP',
            options={'maxiter': 500, 'ftol': 1e-14}
        )
        
        if result.fun < 1e10:
            points = result.x.reshape(n, 2)
            score = compute_score(points)
            if score > best_score:
                best_score = score
                best_points = points.copy()
    
    # If no valid points found, return a default
    if best_points is None:
        best_points = generate_hexagonal_grid()
    
    return best_points

# EVOLVE-BLOCK-END
