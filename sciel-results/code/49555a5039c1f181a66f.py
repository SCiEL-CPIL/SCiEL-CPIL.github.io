import numpy as np
from scipy.optimize import minimize

def construct():
    N = 16
    
    # 1. Generate initial configuration: 16 closest points to origin in triangular lattice
    v1 = np.array([1.0, 0.0])
    v2 = np.array([0.5, np.sqrt(3.0)/2.0])
    
    points = []
    for i in range(-10, 11):
        for j in range(-10, 11):
            p = i * v1 + j * v2
            points.append(p)
            
    points = np.array(points)
    dists = np.linalg.norm(points, axis=1)
    idx = np.argsort(dists)[:N]
    selected_points = points[idx]
    
    # Center the initial points to keep them near the origin for numerical stability
    centroid = np.mean(selected_points, axis=0)
    x0 = (selected_points - centroid).flatten()
    
    # 2. Define the objective function
    def objective(x):
        pts = x.reshape(N, 2)
        # Compute pairwise distances using broadcasting
        dist_matrix = np.linalg.norm(pts[:, np.newaxis, :] - pts[np.newaxis, :, :], axis=2)
        # Get upper triangle (excluding diagonal)
        iu = np.triu_indices(N, 1)
        pairwise_dists = dist_matrix[iu]
        
        min_d = np.min(pairwise_dists)
        max_d = np.max(pairwise_dists)
        
        # Avoid division by zero
        if min_d < 1e-10:
            return 1e10
            
        return (max_d / min_d) ** 2
    
    # 3. Optimize using Nelder-Mead
    # This method does not require gradients and handles non-smooth objectives reasonably well
    result = minimize(
        objective, 
        x0, 
        method='Nelder-Mead',
        options={
            'maxiter': 5000,
            'xatol': 1e-10,
            'fatol': 1e-10,
            'adaptive': True
        }
    )
    
    # 4. Return the optimized points
    x_opt = result.x.reshape(N, 2)
    
    # Re-center to ensure no drift
    centroid_opt = np.mean(x_opt, axis=0)
    x_opt = x_opt - centroid_opt
    
    return x_opt
