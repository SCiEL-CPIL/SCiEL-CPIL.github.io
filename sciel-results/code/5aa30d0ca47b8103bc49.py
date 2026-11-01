import numpy as np
import time
from scipy.optimize import minimize

N = 50

def compute_radius(points):
    """
    Compute the maximum radius for the given points.
    points: (N, 3) array
    """
    # Pairwise distances
    # Using broadcasting
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    
    # Set diagonal to infinity to ignore self-distance
    np.fill_diagonal(dists, np.inf)
    
    min_pair_dist = np.min(dists)
    
    # Distance to walls
    # Cube is [-0.5, 0.5]^3
    # dist to wall for point i, coord k is 0.5 - |p_ik|
    wall_dists = 0.5 - np.abs(points)
    min_wall_dist = np.min(wall_dists)
    
    return min(min_pair_dist / 2.0, min_wall_dist)

def optimize_with_nm(points, max_iter=2000):
    """
    Optimize points using Nelder-Mead to maximize radius.
    """
    def objective(x):
        p = x.reshape(-1, 3)
        # Add penalty if points go outside? 
        # The radius calculation handles boundaries by reducing r, 
        # so the optimizer will naturally keep them inside to maximize r.
        return -compute_radius(p)
    
    x0 = points.flatten()
    res = minimize(objective, x0, method='Nelder-Mead', 
                   options={'maxiter': max_iter, 'xatol': 1e-8, 'fatol': 1e-8})
    return res.x.reshape(-1, 3)

def run_sa(points, time_limit=10.0):
    """
    Simple Simulated Annealing.
    """
    current_points = points.copy()
    current_r = compute_radius(current_points)
    best_points = current_points.copy()
    best_r = current_r
    
    T = 0.05
    cooling = 0.99
    step_size = 0.05
    start_time = time.time()
    
    while time.time() - start_time < time_limit:
        # Pick random point
        i = np.random.randint(N)
        # Random step
        delta = np.random.normal(0, step_size, 3)
        new_points = current_points.copy()
        new_points[i] += delta
        
        # Keep within bounds? 
        # If we go out, radius drops significantly, which is a valid signal for SA.
        # But clipping might help convergence.
        # Let's not clip, let the objective function handle it.
        
        new_r = compute_radius(new_points)
        
        if new_r > current_r:
            current_points = new_points
            current_r = new_r
            if current_r > best_r:
                best_points = new_points.copy()
                best_r = current_r
        else:
            # Accept with probability
            if new_r < 1e-9:
                prob = 0
            else:
                prob = np.exp((new_r - current_r) / T)
            if np.random.rand() < prob:
                current_points = new_points
                current_r = new_r
        
        T *= cooling
        # Adaptive step size
        if T < 0.001:
            break
            
    return best_points, best_r

def construct():
    deadline = time.time() + 140.0 # Leave 10s margin
    
    # Initial guesses
    # 1. Jittered Grid
    # 5x5x2 grid = 50 points
    # x: -0.4, -0.2, 0, 0.2, 0.4
    # y: -0.4, -0.2, 0, 0.2, 0.4
    # z: -0.25, 0.25
    xs = np.array([-0.4, -0.2, 0.0, 0.2, 0.4])
    ys = np.array([-0.4, -0.2, 0.0, 0.2, 0.4])
    zs = np.array([-0.25, 0.25])
    grid_points = np.array([(x, y, z) for z in zs for y in ys for x in xs])
    
    # Add jitter
    np.random.seed(42)
    jittered_grid = grid_points + np.random.normal(0, 0.02, grid_points.shape)
    
    # 2. Random points
    random_points = np.random.uniform(-0.45, 0.45, (N, 3))
    
    # 3. Another random points
    np.random.seed(123)
    random_points2 = np.random.uniform(-0.45, 0.45, (N, 3))
    
    candidates = [jittered_grid, random_points, random_points2]
    
    best_points = None
    best_r = -1
    
    for i, init_points in enumerate(candidates):
        if time.time() > deadline - 20:
            break
            
        # Run SA
        sa_points, sa_r = run_sa(init_points, time_limit=10.0)
        
        # Run NM polishing
        nm_points = optimize_with_nm(sa_points, max_iter=1000)
        nm_r = compute_radius(nm_points)
        
        if nm_r > best_r:
            best_r = nm_r
            best_points = nm_points
            
        # If we have a good result, maybe try another round?
        # For now, just keep the best of these 3.
        
    if best_points is None:
        # Fallback
        best_points = jittered_grid
        best_r = compute_radius(best_points)
        
    return best_points
