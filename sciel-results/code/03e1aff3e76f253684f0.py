import numpy as np
import itertools
from scipy.optimize import minimize
import time

N = 14
D = 3

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10s margin
    
    # Generate initial points on a sphere (Fibonacci lattice)
    # This ensures a good spread
    golden_angle = np.pi * (3.0 - np.sqrt(5.0))
    y = np.linspace(1, -1, N)
    radius = np.sqrt(1 - np.square(y))
    theta = golden_angle * np.arange(N)
    
    x = np.cos(theta) * radius
    z = np.sin(theta) * radius
    
    # Initial points
    P_init = np.column_stack((x, y, z))
    
    # Center the points
    P_init = P_init - np.mean(P_init, axis=0)
    
    def smooth_ratio_squared(x_flat, tau=10.0):
        P = x_flat.reshape(N, D)
        # Center again to handle drift
        P = P - np.mean(P, axis=0)
        
        # Pairwise distances
        # Use broadcasting for efficiency
        # dists[i, j] = ||P[i] - P[j]||
        diff = P[:, np.newaxis, :] - P[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        
        # Mask diagonal
        mask = ~np.eye(N, dtype=bool)
        dists_flat = dists[mask]
        
        # Smooth max and min
        # Soft max: (1/tau) * log(sum(exp(tau * dist)))
        # Soft min: -(1/tau) * log(sum(exp(-tau * dist)))
        
        # To avoid overflow, subtract max for soft max
        m_max = np.max(dists_flat)
        soft_max = (1.0 / tau) * np.log(np.sum(np.exp(tau * (dists_flat - m_max)))) + m_max
        
        m_min = np.min(dists_flat)
        soft_min = -(1.0 / tau) * np.log(np.sum(np.exp(-tau * (dists_flat - m_min)))) + m_min
        
        if soft_min < 1e-10:
            return 1e10
            
        ratio = soft_max / soft_min
        return ratio ** 2

    # Optimization
    best_score = np.inf
    best_P = P_init
    
    # Try a few different taus or just one
    # L-BFGS-B is efficient for this size
    try:
        res = minimize(
            smooth_ratio_squared,
            P_init.flatten(),
            method='L-BFGS-B',
            options={'maxiter': 1000, 'ftol': 1e-10}
        )
        
        if res.success and res.fun < best_score:
            best_score = res.fun
            best_P = res.x.reshape(N, D)
            
    except Exception:
        pass
    
    # If time remains, try Nelder-Mead for refinement or different start
    if time.time() < deadline:
        try:
            res2 = minimize(
                lambda x: smooth_ratio_squared(x, tau=50.0), # Higher tau for sharper min/max
                best_P.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-10}
            )
            if res2.success and res2.fun < best_score:
                best_score = res2.fun
                best_P = res2.x.reshape(N, D)
        except Exception:
            pass

    # Final check and centering
    best_P = best_P - np.mean(best_P, axis=0)
    
    return best_P
