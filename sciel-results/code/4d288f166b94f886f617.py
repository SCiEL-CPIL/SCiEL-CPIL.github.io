import numpy as np
from scipy.optimize import minimize
import time

N = 100
DEADLINE = 110.0

def compute_energy(x):
    # x is (N, 3)
    d2 = np.sum((x[:, np.newaxis, :] - x[np.newaxis, :, :])**2, axis=2)
    np.fill_diagonal(d2, 1.0)
    d2 = np.maximum(d2, 1e-12)
    E_pair = np.sum(1.0 / d2)
    
    dx = 0.5 - np.abs(x)
    dx = np.maximum(dx, 1e-12)
    E_wall = np.sum(1.0 / (dx**2))
    
    return E_pair + E_wall

def compute_grad(x):
    d2 = np.sum((x[:, np.newaxis, :] - x[np.newaxis, :, :])**2, axis=2)
    np.fill_diagonal(d2, 1.0)
    d2 = np.maximum(d2, 1e-12)
    
    diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
    weights = 2.0 / (d2**2)
    np.fill_diagonal(weights, 0.0)
    grad_pair = np.sum(weights[:, :, np.newaxis] * diff, axis=1)
    
    dx = 0.5 - np.abs(x)
    dx = np.maximum(dx, 1e-12)
    signs = np.sign(x)
    signs[x == 0] = 1.0
    grad_wall = 2.0 * signs / (dx**3)
    
    return grad_pair + grad_wall

def construct():
    start_time = time.time()
    best_x = None
    best_score = -np.inf
    
    bounds = [(-0.5, 0.5)] * (N * 3)
    
    # Initialization 1: Grid based
    # 5x5x5 grid, take first 100
    ticks = (np.arange(5) + 0.5) / 5.0 - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    x0 = grid[:N].copy()
    # Add small noise to break symmetry
    x0 += np.random.normal(0, 0.01, x0.shape)
    
    # Try a few random restarts as well
    restarts = 5
    
    for i in range(restarts):
        if time.time() > DEADLINE:
            break
            
        if i == 0:
            x_start = x0
        else:
            x_start = np.random.uniform(-0.45, 0.45, (N, 3))
            
        # Optimize
        try:
            res = minimize(
                lambda xf: compute_energy(xf.reshape(N, 3)),
                x_start.flatten(),
                jac=lambda xf: compute_grad(xf.reshape(N, 3)).flatten(),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 2000, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            x_res = res.x.reshape(N, 3)
        except Exception:
            continue
            
        # Calculate score
        d2 = np.sum((x_res[:, np.newaxis, :] - x_res[np.newaxis, :, :])**2, axis=2)
        np.fill_diagonal(d2, np.inf)
        min_pair = np.sqrt(np.min(d2)) / 2.0
        
        min_wall = np.min(0.5 - np.abs(x_res))
        
        score = min(min_pair, min_wall)
        
        if score > best_score:
            best_score = score
            best_x = x_res
            
    # If no valid result found (shouldn't happen), return grid
    if best_x is None:
        return grid[:N]
        
    return best_x
