"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import math

N = 32

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # safety margin before 150s kill
    
    # Initial grid placement
    side = int(math.ceil(math.sqrt(N)))
    # 6x6 grid, take first 32
    centers = []
    for i in range(side):
        for j in range(side):
            if len(centers) < N:
                centers.append([(i + 0.5) / side, (j + 0.5) / side])
    
    centers = np.array(centers, dtype=np.float64)
    n = len(centers)
    
    # Compute initial radii
    def compute_radii(centers):
        radii = np.zeros(n)
        for i in range(n):
            # Distance to walls
            r_wall = min(centers[i][0], 1 - centers[i][0], centers[i][1], 1 - centers[i][1])
            # Distance to other circles
            min_dist = r_wall
            for j in range(n):
                if i != j:
                    d = np.linalg.norm(centers[i] - centers[j])
                    min_dist = min(min_dist, d / 2)
            radii[i] = min_dist
        return radii
    
    radii = compute_radii(centers)
    best_centers = centers.copy()
    best_radii = radii.copy()
    best_sum = np.sum(radii)
    
    # Gradient-based optimization
    # We want to maximize sum(radii) where radii depend on centers
    # Use a penalty-based approach with smooth min
    
    def smooth_min(values, beta=100):
        """Smooth approximation of min using log-sum-exp"""
        values = np.asarray(values)
        m = np.min(values)
        return m - np.log(np.sum(np.exp(beta * (values - m)))) / beta
    
    def objective(centers_flat):
        c = centers_flat.reshape(n, 2)
        # Clip to [0,1] with small margin
        c = np.clip(c, 1e-6, 1 - 1e-6)
        
        radii = np.zeros(n)
        for i in range(n):
            # Distance to walls
            r_wall = min(c[i][0], 1 - c[i][0], c[i][1], 1 - c[i][1])
            # Distances to other circles
            dists = []
            for j in range(n):
                if i != j:
                    d = np.linalg.norm(c[i] - c[j])
                    dists.append(d / 2)
            dists.append(r_wall)
            radii[i] = min(dists)
        
        return -np.sum(radii), c
    
    # Use scipy if available, otherwise simple gradient descent
    try:
        from scipy.optimize import minimize
        use_scipy = True
    except ImportError:
        use_scipy = False
    
    if use_scipy:
        # Multi-start with different initial perturbations
        np.random.seed(42)
        for restart in range(5):
            if time.time() > deadline - 5:
                break
            
            if restart == 0:
                c0 = centers.copy()
            else:
                c0 = centers + np.random.randn(n, 2) * 0.05
                c0 = np.clip(c0, 0.01, 0.99)
            
            def obj_and_grad(x):
                c = x.reshape(n, 2)
                c = np.clip(c, 1e-6, 1 - 1e-6)
                
                # Compute radii and gradients numerically
                radii = np.zeros(n)
                for i in range(n):
                    r_wall = min(c[i][0], 1 - c[i][0], c[i][1], 1 - c[i][1])
                    min_d = r_wall
                    for j in range(n):
                        if i != j:
                            d = np.linalg.norm(c[i] - c[j])
                            min_d = min(min_d, d / 2)
                    radii[i] = min_d
                
                f = -np.sum(radii)
                
                # Numerical gradient
                eps = 1e-7
                grad = np.zeros_like(x)
                for k in range(len(x)):
                    x_plus = x.copy()
                    x_plus[k] += eps
                    f_plus = obj_and_grad(x_plus)[0]
                    grad[k] = (f_plus - f) / eps
                
                return f, grad
            
            try:
                result = minimize(
                    lambda x: obj_and_grad(x)[0],
                    c0.flatten(),
                    jac=lambda x: obj_and_grad(x)[1],
                    method='L-BFGS-B',
                    options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-8}
                )
                
                if result.success or result.fun < objective(c0.flatten())[0]:
                    c_new = np.clip(result.x.reshape(n, 2), 1e-6, 1 - 1e-6)
                    r_new = compute_radii(c_new)
                    if np.sum(r_new) > best_sum:
                        best_centers = c_new
                        best_radii = r_new
                        best_sum = np.sum(r_new)
            except Exception:
                pass
    
    # Final polish: local optimization with tighter tolerance
    if time.time() < deadline - 5:
        # Simple iterative refinement
        for iteration in range(200):
            if time.time() > deadline - 2:
                break
            
            improved = False
            for i in range(n):
                # Try small moves for circle i
                best_pos = best_centers[i].copy()
                best_r = best_radii[i]
                
                for dx in [-0.01, -0.005, 0, 0.005, 0.01]:
                    for dy in [-0.01, -0.005, 0, 0.005, 0.01]:
                        new_pos = best_centers[i] + np.array([dx, dy])
                        if new_pos[0] < 0.001 or new_pos[0] > 0.999 or new_pos[1] < 0.001 or new_pos[1] > 0.999:
                            continue
                        
                        # Check if this improves the minimum radius for circle i
                        new_c = best_centers.copy()
                        new_c[i] = new_pos
                        r_wall = min(new_pos[0], 1 - new_pos[0], new_pos[1], 1 - new_pos[1])
                        min_d = r_wall
                        for j in range(n):
                            if i != j:
                                d = np.linalg.norm(new_pos - new_c[j])
                                min_d = min(min_d, d / 2)
                        
                        if min_d > best_r + 1e-10:
                            best_r = min_d
                            best_pos = new_pos.copy()
                            improved = True
            
            if improved:
                best_centers[i] = best_pos
                best_radii = compute_radii(best_centers)
    
    # Build result
    result = np.zeros((n, 3))
    for i in range(n):
        result[i] = [best_centers[i][0], best_centers[i][1], best_radii[i]]
    
    return result
# EVOLVE-BLOCK-END
