import numpy as np
import time
import scipy.optimize as opt

N = 50

def _generate_grid(n_points):
    """Generate a dense grid of points on the unit sphere."""
    # Use a Fibonacci spiral for good distribution
    i = np.arange(n_points) + 0.5
    z = 1.0 - 2.0 * i / n_points
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    grid = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    return grid

def _covering_radius_estimate(P, grid):
    """
    Estimate the covering radius by finding the max distance from grid points
    to the nearest point in P.
    P: (N, 3) array of points (not necessarily normalized).
    grid: (M, 3) array of unit vectors.
    """
    # Normalize P
    norms = np.linalg.norm(P, axis=1, keepdims=True)
    norms[norms == 0] = 1.0 # Avoid division by zero
    P_norm = P / norms
    
    # Compute dot products: grid (M, 1, 3) dot P_norm (1, N, 3) -> (M, N)
    # This is memory intensive if M and N are large. 
    # M=20000, N=50 -> 1M floats = 8MB. Acceptable.
    dots = grid @ P_norm.T
    # Clip to [-1, 1] for numerical stability in arccos
    dots = np.clip(dots, -1.0, 1.0)
    
    # Distance in radians: arccos(dot)
    # We want min distance for each grid point
    min_dots = np.max(dots, axis=1) # Max dot corresponds to min angle
    angles = np.arccos(min_dots)
    
    return np.max(angles)

def construct():
    start_time = time.time()
    deadline = start_time + 110.0 # Safety margin
    
    # Grid for evaluating covering radius
    # 20,000 points is a good balance. More points = better estimate but slower eval.
    grid = _generate_grid(20000)
    
    best_score = np.inf
    best_P = None
    
    # Initial configuration: Fibonacci spiral
    def initial_fibonacci():
        i = np.arange(N) + 0.5
        z = 1.0 - 2.0 * i / N
        phi = np.pi * (1.0 + np.sqrt(5.0)) * i
        r = np.sqrt(1.0 - z * z)
        return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # Optimization function
    def objective(P_flat):
        P = P_flat.reshape(N, 3)
        return _covering_radius_estimate(P, grid)
    
    # We will try a few restarts
    num_restarts = 5
    
    for restart in range(num_restarts):
        if time.time() > deadline:
            break
            
        # Initialize
        if restart == 0:
            P0 = initial_fibonacci()
        else:
            # Random initialization
            P0 = np.random.randn(N, 3)
            P0 /= np.linalg.norm(P0, axis=1, keepdims=True)
            
        P0_flat = P0.flatten()
        
        # Use L-BFGS-B
        # L-BFGS-B is fast but might get stuck in local minima due to non-smoothness.
        # We can also try Powell if L-BFGS-B fails or for diversity.
        # Let's stick to L-BFGS-B for speed, maybe with a higher maxiter.
        
        try:
            res = opt.minimize(objective, P0_flat, method='L-BFGS-B', 
                               options={'maxiter': 500, 'ftol': 1e-8, 'gtol': 1e-8})
            
            if res.success or res.fun < best_score:
                P_opt = res.x.reshape(N, 3)
                # Normalize
                P_opt /= np.linalg.norm(P_opt, axis=1, keepdims=True)
                score = _covering_radius_estimate(P_opt, grid)
                
                if score < best_score:
                    best_score = score
                    best_P = P_opt
                    
        except Exception as e:
            # If optimization fails, try a simple local search or just keep current
            pass

        # Check time
        if time.time() > deadline:
            break

    # If we have time, do a final refinement with Powell on the best candidate
    if best_P is not None and time.time() < deadline - 10:
        try:
            P_best_flat = best_P.flatten()
            res = opt.minimize(objective, P_best_flat, method='Powell',
                               options={'maxiter': 200, 'ftol': 1e-8})
            P_powell = res.x.reshape(N, 3)
            P_powell /= np.linalg.norm(P_powell, axis=1, keepdims=True)
            score_powell = _covering_radius_estimate(P_powell, grid)
            
            if score_powell < best_score:
                best_P = P_powell
                best_score = score_powell
        except Exception:
            pass

    if best_P is None:
        # Fallback to Fibonacci
        best_P = initial_fibonacci()
        
    return best_P
