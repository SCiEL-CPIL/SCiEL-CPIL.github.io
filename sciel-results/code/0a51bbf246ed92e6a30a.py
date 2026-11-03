"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 200

# EVOLVE-BLOCK-START
import numpy as np
import time
import scipy.optimize

def construct():
    start_time = time.time()
    timeout = 140  # Safety margin for 150s limit
    
    # 1. Initialization: Jittered Hexagonal Grid
    # Estimate spacing for 200 circles in unit square
    # Area per circle ~ 1/200 = 0.005
    # Hex packing area per circle ~ s^2 * sqrt(3)/2
    # s^2 ~ 0.005 * 2 / sqrt(3) ~ 0.0057 -> s ~ 0.075
    s = 0.075
    h = s * np.sqrt(3) / 2
    
    # Generate hex grid points
    points = []
    # Range to cover [-0.5, 0.5]
    # x from -0.5 to 0.5
    # y from -0.5 to 0.5
    # Number of rows ~ 1 / h ~ 13
    # Number of cols ~ 1 / s ~ 13
    # Total ~ 169. Need more.
    # Let's make s smaller to get more candidates, then pick 200.
    s = 0.06
    h = s * np.sqrt(3) / 2
    
    rows = int(1.0 / h) + 2
    cols = int(1.0 / s) + 2
    
    # Center the grid
    y_start = -rows * h / 2 + h/2
    x_start = -cols * s / 2 + s/2
    
    candidates = []
    for i in range(rows):
        y = y_start + i * h
        offset = 0 if i % 2 == 0 else s / 2
        for j in range(cols):
            x = x_start + j * s + offset
            if -0.5 <= x <= 0.5 and -0.5 <= y <= 0.5:
                candidates.append([x, y])
                
    candidates = np.array(candidates)
    
    if len(candidates) < N:
        # Fallback to random if grid is too small (should not happen with s=0.06)
        candidates = np.random.uniform(-0.49, 0.49, (N, 2))
    else:
        # Randomly select N points from candidates to break symmetry
        indices = np.random.choice(len(candidates), N, replace=False)
        x = candidates[indices].copy()
        
    # Add small random jitter to avoid exact symmetries
    x += np.random.normal(0, 0.001, x.shape)
    
    # Clip to square (with small epsilon to avoid boundary issues in gradient calc if needed, 
    # but L-BFGS-B handles bounds)
    x = np.clip(x, -0.5, 0.5)
    
    def energy_and_grad(x_flat):
        x = x_flat.reshape(-1, 2)
        n = len(x)
        E = 0.0
        grad = np.zeros_like(x)
        
        # Pairwise interactions
        # Vectorized distance calculation for energy
        # d_ij^2
        diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
        d2 = np.sum(diff**2, axis=2)
        # Avoid division by zero on diagonal
        np.fill_diagonal(d2, 1.0)
        d = np.sqrt(d2)
        
        # Energy term: sum_{i<j} 1/d^4
        # We compute full sum and divide by 2, or just sum i<j
        # For gradient, we need sum over j != i
        inv_d4 = 1.0 / (d2**2) # 1/d^4
        np.fill_diagonal(inv_d4, 0.0)
        
        # E_pair = 0.5 * sum_{i,j} 1/d_{ij}^4
        E += 0.5 * np.sum(inv_d4)
        
        # Gradient for pair terms:
        # dE/dx_i = sum_{j != i} -4 * (x_i - x_j) / d_{ij}^6
        # d^6 = (d2)^3
        inv_d6 = 1.0 / (d2**3)
        np.fill_diagonal(inv_d6, 0.0)
        
        # grad_i = -4 * sum_j (x_i - x_j) * inv_d6_{ij}
        # Note: diff = x_i - x_j
        # grad = -4 * sum_j diff_{ij} * inv_d6_{ij}
        grad += -4.0 * np.sum(diff * inv_d6[:, :, np.newaxis], axis=1)
        
        # Wall interactions
        # V_wall = sum_i [ 1/(x_i+0.5)^4 + 1/(0.5-x_i)^4 + 1/(y_i+0.5)^4 + 1/(0.5-y_i)^4 ]
        # Let's use power 4 for walls too to match stiffness
        px = x[:, 0]
        py = x[:, 1]
        
        d_left = px + 0.5
        d_right = 0.5 - px
        d_down = py + 0.5
        d_up = 0.5 - py
        
        # Avoid zero division if on boundary (should not happen with bounds)
        eps = 1e-6
        d_left = np.maximum(d_left, eps)
        d_right = np.maximum(d_right, eps)
        d_down = np.maximum(d_down, eps)
        d_up = np.maximum(d_up, eps)
        
        E += np.sum(1.0 / d_left**4 + 1.0 / d_right**4 + 1.0 / d_down**4 + 1.0 / d_up**4)
        
        # Gradients for walls
        # d/dx 1/(x+0.5)^4 = -4/(x+0.5)^5
        # d/dx 1/(0.5-x)^4 = 4/(0.5-x)^5
        grad[:, 0] += -4.0 / d_left**5 + 4.0 / d_right**5
        grad[:, 1] += -4.0 / d_down**5 + 4.0 / d_up**5
        
        return E, grad.flatten()
    
    # Optimization
    best_x = x
    best_score = -1
    
    # Try L-BFGS-B
    try:
        res = scipy.optimize.minimize(
            energy_and_grad, 
            x.flatten(), 
            method='L-BFGS-B', 
            jac=True,
            bounds=[(-0.5, 0.5)] * (2 * N),
            options={'maxiter': 200, 'maxfun': 200, 'ftol': 1e-10, 'gtol': 1e-8}
        )
        if res.success or res.fun < energy_and_grad(x.flatten())[0]:
            best_x = res.x.reshape(-1, 2)
            
        # Check time
        if time.time() - start_time > timeout - 10:
            return best_x
            
        # Try Nelder-Mead as a local polish if time permits
        # Nelder-Mead is slower but can escape shallow local minima
        # Let's try a few iterations of Nelder-Mead
        res2 = scipy.optimize.minimize(
            lambda xf: energy_and_grad(xf)[0], 
            best_x.flatten(), 
            method='Nelder-Mead',
            options={'maxiter': 100, 'xatol': 1e-6, 'fatol': 1e-8}
        )
        if res2.fun < energy_and_grad(best_x.flatten())[0]:
            best_x = res2.x.reshape(-1, 2)
            
    except Exception as e:
        # Fallback to simple gradient descent if optimization fails
        # Or just return the initial jittered grid
        pass
    
    # Final validation and clipping
    best_x = np.clip(best_x, -0.5, 0.5)
    
    return best_x

# EVOLVE-BLOCK-END
