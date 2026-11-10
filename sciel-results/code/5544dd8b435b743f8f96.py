import numpy as np
from scipy.optimize import minimize

def min_max_dist_dim3_14() -> np.ndarray:
    n = 14
    
    # Initial points: Icosahedron vertices + 2 more
    # Icosahedron vertices
    phi = (1 + np.sqrt(5)) / 2
    ico = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ])
    ico = ico / np.linalg.norm(ico, axis=1, keepdims=True)
    
    # Add 2 points, e.g., at (1,0,0) and (-1,0,0) initially, then optimize
    p13 = np.array([1.0, 0.0, 0.0])
    p14 = np.array([-1.0, 0.0, 0.0])
    
    points = np.vstack([ico, p13, p14])
    
    # Local optimization to maximize min distance
    # We will use a simple iterative method: 
    # For each point, try to move it to increase the min distance.
    
    def min_dist(pts):
        from scipy.spatial.distance import pdist
        return np.min(pdist(pts))
    
    # We'll do a few iterations of local improvement
    for _ in range(1000):
        improved = False
        for i in range(n):
            original_point = points[i].copy()
            best_point = original_point
            best_min = min_dist(points)
            
            # Try moving in random directions
            for _ in range(20):
                # Random small perturbation
                delta = np.random.randn(3) * 0.05
                new_point = original_point + delta
                new_point = new_point / np.linalg.norm(new_point)
                
                points[i] = new_point
                current_min = min_dist(points)
                
                if current_min > best_min:
                    best_min = current_min
                    best_point = new_point
                    improved = True
            points[i] = best_point
            
        if not improved:
            break
    
    return points
