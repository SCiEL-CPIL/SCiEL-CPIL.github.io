import numpy as np
import time
from scipy.optimize import minimize

N = 50

def construct():
    # Start time
    start_time = time.time()
    time_limit = 110.0  # Leave 10s buffer for overhead
    
    # Initial position: Grid with noise
    # 7x7 grid = 49 points, add 1 more
    m = 7
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    pos = grid[:49].copy()
    # Add 50th point at center with slight offset or random
    # Actually, a better initialization for 50 points is often a distorted grid
    # Let's use a 5x10 grid? 5*10=50.
    # Square is 1x1. 5 rows, 10 cols.
    # Spacing: 1/10 = 0.1 in x, 1/5 = 0.2 in y.
    # This is anisotropic. Hexagonal is better.
    
    # Let's stick to a robust random restart + SA strategy.
    
    best_global_r = 0.0
    best_global_pos = None
    
    n_restarts = 5
    # Each restart will run SA for a certain amount of time or iterations
    
    for restart in range(n_restarts):
        # Check time
        if time.time() - start_time > time_limit:
            break
            
        # Initialization
        if restart == 0:
            # Grid init
            m = 7
            ticks = (np.arange(m) + 0.5) / m - 0.5
            grid = np.array([(x, y) for y in ticks for x in ticks])
            pos = grid[:49].copy()
            # Add 50th point. In a 7x7 grid, the center is (0,0).
            # The grid points are at (x,y) where x,y in [-0.5, 0.5].
            # Let's place the 50th point at (0, 0) if not occupied, or near center.
            # Actually, 7x7=49. The center of the square is (0,0).
            # Is (0,0) in the grid?
            # ticks = [ -0.5+0.5/7, ... 0, ... 0.5-0.5/7 ]
            # Yes, 0 is the middle element.
            # So (0,0) is already in the grid.
            # We need to displace one point or add a new one.
            # Let's just add a random point and let SA fix it.
            pos = np.vstack([pos, np.array([0.0, 0.0]) + np.random.normal(0, 0.01, 2)])
        else:
            # Random init with some structure (hexagonal)
            s = 0.14
            xs = []
            ys = []
            max_j = int(1.0 / (s * np.sqrt(3) / 2)) + 2
            max_i = int(1.0 / s) + 2
            for j in range(-max_j, max_j + 1):
                for i in range(-max_i, max_i + 1):
                    x = i * s + (j % 2) * s / 2.0
                    y = j * s * np.sqrt(3) / 2.0
                    xs.append(x)
                    ys.append(y)
            
            xs = np.array(xs)
            ys = np.array(ys)
            mask = (np.abs(xs) <= 0.45) & (np.abs(ys) <= 0.45)
            xs = xs[mask]
            ys = ys[mask]
            
            if len(xs) >= N:
                indices = np.random.choice(len(xs), N, replace=False)
                pos = np.column_stack((xs[indices], ys[indices]))
            else:
                pos = np.random.uniform(-0.4, 0.4, (N, 2))
                
        pos += np.random.normal(0, 0.005, pos.shape)
        pos = np.clip(pos, -0.5, 0.5)
        
        # Simulated Annealing
        current_r = calculate_radius(pos)
        best_local_r = current_r
        best_local_pos = pos.copy()
        
        T = 0.01  # Initial temperature
        T_min = 1e-6
        alpha = 0.999  # Cooling rate
        step_size = 0.05
        
        # Run SA until time limit for this restart or T is very low
        # We will run for a fixed number of iterations or time chunk
        max_iters = 20000
        
        for it in range(max_iters):
            if time.time() - start_time > time_limit:
                break
                
            # Pick random point
            idx = np.random.randint(N)
            old_pos = pos[idx].copy()
            
            # Random step
            step = np.random.normal(0, step_size, 2)
            new_pos = old_pos + step
            new_pos = np.clip(new_pos, -0.5, 0.5)
            
            pos[idx] = new_pos
            new_r = calculate_radius(pos)
            
            delta_r = new_r - current_r
            
            # Metropolis acceptance
            if delta_r > 0 or np.random.rand() < np.exp(delta_r / T):
                current_r = new_r
                if new_r > best_local_r:
                    best_local_r = new_r
                    best_local_pos = pos.copy()
            else:
                pos[idx] = old_pos
                
            # Cool down
            T = max(T * alpha, T_min)
            # Reduce step size slowly
            step_size = max(0.001, step_size * 0.9999)
            
        # Update global best
        if best_local_r > best_global_r:
            best_global_r = best_local_r
            best_global_pos = best_local_pos.copy()
            
    # Final Polish: Use Nelder-Mead on the best solution
    # The objective function for scipy.optimize.minimize must be minimized.
    # We want to maximize r, so we minimize -r.
    
    if best_global_pos is not None:
        def objective(x):
            p = x.reshape(-1, 2)
            p = np.clip(p, -0.5, 0.5)
            r = calculate_radius(p)
            return -r
            
        x0 = best_global_pos.flatten()
        # Bounds are handled by clipping in objective, but NM might go out.
        # We can use bounds in minimize, but NM doesn't support bounds directly in all versions?
        # Actually, scipy.optimize.minimize with method='Nelder-Mead' does not support bounds.
        # We rely on the clipping in the objective function.
        
        # Run minimize
        # maxiter to keep it fast
        res = minimize(objective, x0, method='Nelder-Mead', options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-10})
        
        if res.success or res.fun < -best_global_r:
            best_pos = res.x.reshape(-1, 2)
            best_pos = np.clip(best_pos, -0.5, 0.5)
            final_r = calculate_radius(best_pos)
            if final_r > best_global_r:
                best_global_pos = best_pos
                best_global_r = final_r
                
    return best_global_pos

def calculate_radius(pos):
    # Calculate min pairwise distance
    # N=50, so 50x50 distance matrix is fine
    dists = np.linalg.norm(pos[:, np.newaxis, :] - pos[np.newaxis, :, :], axis=-1)
    # Exclude self-distances
    np.fill_diagonal(dists, np.inf)
    min_pairwise = np.min(dists)
    
    # Calculate min boundary distance
    max_abs = np.max(np.abs(pos))
    boundary_dist = 0.5 - max_abs
    
    return min(min_pairwise / 2.0, boundary_dist)
