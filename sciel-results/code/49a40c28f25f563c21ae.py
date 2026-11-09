import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import pdist

def min_max_dist_dim3_14() -> np.ndarray:
    n = 14
    d = 3
    
    # Objective function to maximize: (d_min / d_max)^2
    def objective(points):
        # points is (14, 3)
        dists = pdist(points)
        if len(dists) == 0:
            return 0.0
        d_min = np.min(dists)
        d_max = np.max(dists)
        if d_max == 0:
            return 0.0
        ratio = d_min / d_max
        return ratio * ratio

    # Parametrization:
    # Point 0: Fixed at North Pole (0, 0, 1). phi_0 = 0, theta_0 = 0.
    # Point 1: Fixed in XZ plane. theta_1 = 0. phi_1 is variable.
    # Points 2..13: Variable phi_i, theta_i.
    # Total variables: 1 (phi_1) + 2 * 12 (phi_2..13, theta_2..13) = 25 variables.
    
    def sph_to_cart(sph_params):
        # sph_params: [phi_1, theta_2, phi_2, theta_3, phi_3, ..., theta_13, phi_13]
        # Wait, let's order them clearly.
        # idx 0: phi_1
        # idx 1: theta_2
        # idx 2: phi_2
        # idx 3: theta_3
        # ...
        # idx 2*k-1: theta_{k+1}
        # idx 2*k: phi_{k+1}
        
        pts = np.zeros((n, 3))
        
        # Point 0
        pts[0] = [0.0, 0.0, 1.0]
        
        # Point 1
        phi_1 = sph_params[0]
        pts[1] = [np.sin(phi_1), 0.0, np.cos(phi_1)]
        
        # Points 2 to 13
        for i in range(2, n):
            idx_theta = 1 + 2 * (i - 2)
            idx_phi = idx_theta + 1
            
            theta_i = sph_params[idx_theta]
            phi_i = sph_params[idx_phi]
            
            pts[i] = [
                np.sin(phi_i) * np.cos(theta_i),
                np.sin(phi_i) * np.sin(theta_i),
                np.cos(phi_i)
            ]
            
        return pts

    def sph_objective(sph_params):
        pts = sph_to_cart(sph_params)
        # Normalize just in case (should be unit sphere)
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        pts = pts / norms
        
        return -objective(pts) # Minimize negative

    best_score = -1
    best_points = None
    
    # Multi-start
    np.random.seed(42)
    
    num_starts = 50
    maxiter = 1000
    
    for start in range(num_starts):
        # Initialize variables
        # phi in [0, pi], theta in [0, 2*pi]
        
        # Random init for phi_1
        phi_1_init = np.random.uniform(0, np.pi)
        
        # Random init for others
        params_init = np.zeros(25)
        params_init[0] = phi_1_init
        
        for i in range(2, n):
            idx_theta = 1 + 2 * (i - 2)
            idx_phi = idx_theta + 1
            params_init[idx_theta] = np.random.uniform(0, 2 * np.pi)
            params_init[idx_phi] = np.random.uniform(0, np.pi)
            
        try:
            res = minimize(sph_objective, params_init, method='Nelder-Mead', 
                           options={'maxiter': maxiter, 'xatol': 1e-10, 'fatol': 1e-10, 'adaptive': True})
            
            final_params = res.x
            final_pts = sph_to_cart(final_params)
            final_pts /= np.linalg.norm(final_pts, axis=1, keepdims=True)
            
            final_score = objective(final_pts)
            
            if final_score > best_score:
                best_score = final_score
                best_points = final_pts.copy()
                
        except Exception as e:
            pass

    if best_points is None:
        # Fallback: Random points
        pts = np.random.randn(n, d)
        pts /= np.linalg.norm(pts, axis=1, keepdims=True)
        return pts

    return best_points
