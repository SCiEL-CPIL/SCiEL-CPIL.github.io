import numpy as np
import time
import random
from scipy.optimize import minimize

N = 11
A = np.array([0.0, 0.0])
B = np.array([1.0, 0.0])
C = np.array([0.5, np.sqrt(3)/2.0])

# Precompute indices for area calculation
idx = np.arange(N)
i_idx, j_idx, k_idx = np.meshgrid(idx, idx, idx, indexing='ij')
mask = (i_idx < j_idx) & (j_idx < k_idx)
I_F = i_idx[mask]
J_F = j_idx[mask]
K_F = k_idx[mask]

def compute_areas_fast(points):
    p1 = points[I_F]
    p2 = points[J_F]
    p3 = points[K_F]
    v1 = p2 - p1
    v2 = p3 - p1
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    return 0.5 * np.abs(cross)

def penalty(points):
    pen = 0.0
    for p in points:
        x, y = p
        if y < 0: pen += y**2
        d2 = y - np.sqrt(3)*x
        if d2 > 0: pen += d2**2
        d3 = y + np.sqrt(3)*x - np.sqrt(3)
        if d3 > 0: pen += d3**2
    return pen

def objective_soft(x, beta):
    points = x.reshape(N, 2)
    areas = compute_areas_fast(points)
    max_a = np.max(areas)
    # If all areas are 0, soft min is 0
    if max_a == 0:
        return 0.0
    log_sum_exp = max_a + np.log(np.sum(np.exp(-beta * (areas - max_a))))
    soft_min = -log_sum_exp / beta
    pen = penalty(points)
    return -soft_min + 10.0 * pen

def objective_exact(x):
    points = x.reshape(N, 2)
    areas = compute_areas_fast(points)
    min_a = np.min(areas)
    pen = penalty(points)
    return -min_a + 10.0 * pen

def random_initialization(rng):
    points = []
    while len(points) < N:
        u = rng.random()
        v = rng.random()
        if u + v < 1.0:
            x = v * 1.0 + (1-u-v) * 0.5
            y = v * 0.0 + (1-u-v) * (np.sqrt(3)/2.0)
            points.append([x, y])
    return np.array(points).flatten()

def optimize_single_start(x0, time_limit):
    current_x = x0.copy()
    betas = [10, 20, 50, 100, 200, 500]
    
    for beta in betas:
        if time.time() > time_limit:
            break
        res = minimize(
            objective_soft, 
            current_x, 
            args=(beta,),
            method='Powell', 
            options={'maxiter': 500, 'xatol': 1e-5, 'fatol': 1e-7}
        )
        # Check if improvement
        if res.fun < objective_soft(current_x, beta):
            current_x = res.x
            
    # Final polish
    if time.time() < time_limit - 2:
        res = minimize(
            objective_exact, 
            current_x, 
            method='Nelder-Mead', 
            options={'maxiter': 1000, 'xatol': 1e-7, 'fatol': 1e-9}
        )
        if res.fun < objective_exact(current_x):
            current_x = res.x
            
    return current_x

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    rng = random.Random(123)
    
    # Initial best
    best_x = random_initialization(rng)
    best_val = -objective_exact(best_x) # min_area - penalty
    
    # Iterated Local Search
    noise_scale = 0.1
    
    while time.time() < deadline - 5:
        # Generate new start
        if random.random() < 0.2:
            # Random restart
            x0 = random_initialization(rng)
        else:
            # Perturb best
            x0 = best_x + rng.gauss(0, noise_scale, N*2)
            # Clip to reasonable bounds to avoid huge penalties
            x0 = np.clip(x0, -0.1, 1.1)
            
        x_opt = optimize_single_start(x0, deadline)
        
        # Evaluate
        val = -objective_exact(x_opt)
        
        if val > best_val:
            best_val = val
            best_x = x_opt.copy()
            # Reduce noise scale if improving
            noise_scale *= 0.9
        else:
            # Increase noise scale if stuck?
            noise_scale *= 1.1
            if noise_scale > 0.5: noise_scale = 0.5
            if noise_scale < 0.01: noise_scale = 0.01

    return best_x.reshape(N, 2)
