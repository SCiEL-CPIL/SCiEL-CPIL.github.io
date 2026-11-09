import numpy as np
import time
from scipy.optimize import minimize

N = 50

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    # 1. Initialization: Hexagonal Lattice
    def get_hex_grid(spacing):
        points = []
        k = int(1.0 / spacing) + 2
        for i in range(-k, k + 1):
            for j in range(-k, k + 1):
                x = (i + 0.5 * (j % 2)) * spacing
                y = j * spacing * np.sqrt(3.0) / 2.0
                if x**2 + y**2 <= 1.0:
                    points.append([x, y])
        points = np.array(points)
        if len(points) < N:
            return None
        # Select N points closest to origin
        dists = np.hypot(points[:, 0], points[:, 1])
        indices = np.argsort(dists)[:N]
        return points[indices]
    
    # Try to find a good initial spacing
    best_init = None
    best_init_r = 0.0
    for s in np.linspace(0.1, 0.3, 50):
        pts = get_hex_grid(s)
        if pts is None:
            continue
        # Calculate r for this grid
        d_min = np.inf
        for i in range(N):
            for j in range(i+1, N):
                d = np.linalg.norm(pts[i] - pts[j])
                if d < d_min:
                    d_min = d
        b_min = np.min(1.0 - np.linalg.norm(pts, axis=1))
        r = min(d_min / 2.0, b_min)
        if r > best_init_r:
            best_init_r = r
            best_init = pts.copy()
            
    if best_init is None:
        # Fallback: Random
        best_init = np.random.rand(N, 2) * 2.0 - 1.0
        best_init = best_init / np.linalg.norm(best_init, axis=1, keepdims=True) * 0.9
        best_init_r = 0.01

    current_points = best_init
    current_r = best_init_r
    
    # 2. Optimization Setup
    def energy_and_grad(c, R):
        c = c.reshape(N, 2)
        # Pairwise distances
        # d[i,j] = ||c_i - c_j||
        diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        d = np.linalg.norm(diff, axis=2)
        # Mask diagonal
        np.fill_diagonal(d, np.inf)
        
        # Boundary distances
        r_norm = np.linalg.norm(c, axis=1)
        b_dist = 1.0 - r_norm
        
        # Violations
        pair_viol = np.maximum(0.0, 2.0 * R - d)
        bound_viol = np.maximum(0.0, R - b_dist)
        
        # Energy
        E = np.sum(pair_viol**2) + np.sum(bound_viol**2)
        
        # Gradient
        grad = np.zeros_like(c)
        
        # Pairwise gradient
        # dE/dd_ij = -2 * pair_viol_ij
        # dd_ij/dc_i = (c_i - c_j) / d_ij
        # Sum over j for each i
        # We can vectorize this partially, but for N=50, loop is fine and safer for clarity/debug
        # However, vectorized is faster.
        
        # Let's do vectorized gradient for pairwise
        # grad_i = sum_{j != i} 2 * pair_viol_ij * (c_i - c_j) / d_ij
        # Note: pair_viol is symmetric.
        
        # Create a matrix of coefficients for the gradient
        # coeff[i,j] = 2 * pair_viol[i,j] / d[i,j]
        coeff = np.zeros((N, N))
        mask = d < np.inf
        coeff[mask] = 2.0 * pair_viol[mask] / d[mask]
        
        # grad_i = sum_j coeff[i,j] * (c_i - c_j)
        # sum_j coeff[i,j] * c_i - sum_j coeff[i,j] * c_j
        # = c_i * sum_j coeff[i,j] - (coeff @ c)_i
        
        sum_coeff = np.sum(coeff, axis=1)
        grad_pair = c * sum_coeff[:, np.newaxis] - (coeff @ c)
        
        # Boundary gradient
        # dE/db_dist_i = -2 * bound_viol_i
        # db_dist_i/dc_i = -c_i / r_norm_i
        # grad_bound_i = -2 * bound_viol_i * (-c_i / r_norm_i) = 2 * bound_viol_i * c_i / r_norm_i
        
        grad_bound = np.zeros_like(c)
        valid_bound = r_norm > 1e-8
        grad_bound[valid_bound] = (2.0 * bound_viol[valid_bound] * c[valid_bound]) / r_norm[valid_bound, np.newaxis]
        
        grad = grad_pair + grad_bound
        
        return E, grad

    def objective(c, R):
        E, _ = energy_and_grad(c, R)
        return E

    def jac(c, R):
        _, g = energy_and_grad(c, R)
        return g.flatten()

    # 3. Bisection Search on R
    R_low = current_r
    R_high = 0.126  # Slightly above best known
    
    best_valid_points = current_points.copy()
    best_valid_r = current_r
    
    # We will try to increase R
    # If we fail to find a packing for R, we lower R_high
    # If we succeed, we raise R_low
    
    # To avoid getting stuck in local minima, we might restart optimization from best_valid_points
    # or current_points
    
    max_iter_bisection = 30
    
    for _ in range(max_iter_bisection):
        if time.time() > deadline - 10:
            break
            
        R_mid = (R_low + R_high) / 2.0
        
        # Optimize for R_mid
        # Start from best_valid_points if we have a valid one, otherwise current
        x0 = best_valid_points.copy()
        
        try:
            res = minimize(
                objective, x0, args=(R_mid,),
                jac=lambda c: jac(c, R_mid),
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            
            if res.success or res.fun < 1e-8:
                # Check if it's actually valid (energy close to 0)
                E_final, _ = energy_and_grad(res.x, R_mid)
                if E_final < 1e-6:
                    R_low = R_mid
                    best_valid_points = res.x.copy()
                    best_valid_r = R_mid
                    # Update current points to this good one
                    current_points = res.x.copy()
                else:
                    # Energy not low enough, treat as failure
                    R_high = R_mid
            else:
                R_high = R_mid
                
        except Exception:
            R_high = R_mid
            
    # Final check: calculate the actual radius of the best points
    pts = best_valid_points
    d_min = np.inf
    for i in range(N):
        for j in range(i+1, N):
            d = np.linalg.norm(pts[i] - pts[j])
            if d < d_min:
                d_min = d
    b_min = np.min(1.0 - np.linalg.norm(pts, axis=1))
    final_r = min(d_min / 2.0, b_min)
    
    return pts
