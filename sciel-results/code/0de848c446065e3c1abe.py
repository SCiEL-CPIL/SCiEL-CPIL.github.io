import numpy as np
import time
import random
from scipy.optimize import minimize

N = 50

def compute_radius(centers):
    # Returns the max radius for the given centers
    # r = min(min_pairwise/2, min_wall)
    if len(centers) < 2:
        return 0.5
    # Pairwise
    # Efficiently compute min pairwise distance
    # For N=50, O(N^2) is fine
    min_dist = float('inf')
    for i in range(N):
        for j in range(i+1, N):
            d = np.linalg.norm(centers[i] - centers[j])
            if d < min_dist:
                min_dist = d
    min_pairwise = min_dist / 2.0
    
    # Walls
    # min_i min_k (0.5 - |c_ik|)
    # min_k (0.5 - |c_ik|) is 0.5 - max_k |c_ik|
    wall_dists = 0.5 - np.max(np.abs(centers), axis=1)
    min_wall = np.min(wall_dists)
    
    return min(min_pairwise, min_wall)

def hex_init():
    # Generate hexagonal points
    # Spacing s
    s = 0.15 
    points = []
    y = -0.5 + s/2
    row = 0
    while y <= 0.5 and len(points) < N:
        x_offset = 0 if row % 2 == 0 else s * np.sqrt(3) / 2
        x = -0.5 + s/2 + x_offset
        while x <= 0.5 and len(points) < N:
            if -0.5 <= x <= 0.5 and -0.5 <= y <= 0.5:
                points.append([x, y])
            x += s
        y += s * np.sqrt(3) / 2
        row += 1
    # If not enough, add random? Or just repeat?
    # Better: If we run out, just stop. But we need N.
    # Let's ensure we generate enough.
    while len(points) < N:
        points.append([random.uniform(-0.5, 0.5), random.uniform(-0.5, 0.5)])
    return np.array(points[:N])

def construct():
    start_time = time.time()
    deadline = 110.0 # 10s margin
    
    centers = hex_init()
    # Initial local mins
    local_mins = np.zeros(N)
    for i in range(N):
        # Compute min dist to others and walls for i
        dists = np.linalg.norm(centers - centers[i], axis=1)
        dists[i] = float('inf')
        min_others = np.min(dists)
        min_walls = 0.5 - np.max(np.abs(centers[i]))
        local_mins[i] = min(min_others, min_walls)
        
    global_r = np.min(local_mins)
    
    T = 0.01
    cooling = 0.999
    steps = 500000
    
    best_r = global_r
    best_centers = centers.copy()
    
    for step in range(steps):
        if time.time() > deadline:
            break
            
        if step % 1000 == 0:
            T *= cooling
            if T < 1e-6:
                break
                
        i = random.randint(0, N-1)
        
        # Propose move
        # Random step size
        step_size = 0.05 * T + 0.001
        delta = np.random.normal(0, step_size, 2)
        new_c_i = centers[i] + delta
        
        # Clamp to box
        new_c_i = np.clip(new_c_i, -0.5, 0.5)
        
        # Compute new local min for i
        dists = np.linalg.norm(centers - new_c_i, axis=1)
        dists[i] = float('inf')
        new_min_others = np.min(dists)
        new_min_walls = 0.5 - np.max(np.abs(new_c_i))
        new_local_min_i = min(new_min_others, new_min_walls)
        
        # Compute min of other locals
        # local_mins is up to date for all j
        mask = np.arange(N) != i
        min_other_locals = np.min(local_mins[mask])
        
        new_global_r = min(new_local_min_i, min_other_locals)
        
        delta_r = new_global_r - global_r
        
        if delta_r > 0 or random.random() < np.exp(delta_r / T):
            # Accept
            centers[i] = new_c_i
            local_mins[i] = new_local_min_i
            global_r = new_global_r
            
            if global_r > best_r:
                best_r = global_r
                best_centers = centers.copy()
                
            # Update local_mins for all j != i
            # This is O(N^2)
            for j in range(N):
                if j == i:
                    continue
                # Recompute local min for j
                dists_j = np.linalg.norm(centers - centers[j], axis=1)
                dists_j[j] = float('inf')
                min_others_j = np.min(dists_j)
                min_walls_j = 0.5 - np.max(np.abs(centers[j]))
                local_mins[j] = min(min_others_j, min_walls_j)
                
            # Update global_r from local_mins
            # It should be min(local_mins)
            # But we already computed new_global_r.
            # However, updating local_mins[j] might have changed the min?
            # No, new_global_r was computed using OLD local_mins[j] for j!=i.
            # But we just updated them.
            # If local_mins[j] increased, the min might increase?
            # Wait.
            # new_global_r = min(new_local_min_i, min_{j!=i} OLD_local_min_j)
            # If we update local_mins[j], they might change.
            # If they increase, the true global min might be higher than new_global_r?
            # No, global min is min of ALL local mins.
            # If some local mins increased, the global min (which is the minimum of them) cannot increase unless the *minimum* one increased.
            # But we are tracking the minimum.
            # Actually, it's safer to just recompute global_r = np.min(local_mins) after updates.
            global_r = np.min(local_mins)
            
            if global_r > best_r:
                best_r = global_r
                best_centers = centers.copy()

    # Polish
    if time.time() < deadline - 5:
        def objective(x):
            c = x.reshape(N, 2)
            c = np.clip(c, -0.5, 0.5)
            return -compute_radius(c)
            
        res = minimize(objective, best_centers.flatten(), method='Nelder-Mead', 
                       options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-10})
        
        if res.fun < -best_r:
            best_centers = res.x.reshape(N, 2)
            best_r = -res.fun
            
    return best_centers
