import numpy as np
import time

N = 50

def compute_radius(P):
    """
    Compute the maximum radius r for the given centers P.
    P is (N, 2) array.
    """
    # Pairwise distances
    diff = P[:, np.newaxis, :] - P[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=-1))
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists)
    
    # Wall distances
    # Distance to nearest wall for each point
    # min(0.5 - |x|, 0.5 - |y|)
    min_wall = np.min(0.5 - np.abs(P))
    
    # Radius is half the minimum distance between centers, 
    # limited by the distance to the wall
    return min(min_pair / 2.0, min_wall)

def init_random():
    return np.random.uniform(-0.5, 0.5, (N, 2))

def simulated_annealing(P, time_limit):
    """
    Run SA on P for the given time limit.
    Returns best P found.
    """
    start_time = time.time()
    T = 0.01
    T_min = 1e-6
    cooling = 0.999
    
    P_curr = P.copy()
    r_curr = compute_radius(P_curr)
    P_best = P_curr.copy()
    r_best = r_curr
    
    step_size = 0.05
    
    while time.time() - start_time < time_limit:
        # Propose move
        i = np.random.randint(N)
        P_new = P_curr.copy()
        P_new[i] += np.random.normal(0, step_size, 2)
        
        r_new = compute_radius(P_new)
        
        delta = r_new - r_curr
        if delta > 0 or np.random.rand() < np.exp(delta / T):
            P_curr = P_new
            r_curr = r_new
            if r_curr > r_best:
                r_best = r_curr
                P_best = P_curr.copy()
        
        T *= cooling
        if T < T_min:
            T = T_min
            # Maybe reheat?
            # For simplicity, just stop or restart?
            # Let's just keep T at min and continue, it becomes local search.
            
    return P_best, r_best

def local_refinement(P, max_iters=500):
    """
    Simple local refinement to polish the solution.
    """
    P = P.copy()
    r_curr = compute_radius(P)
    
    for _ in range(max_iters):
        # Find the bottleneck
        diff = P[:, np.newaxis, :] - P[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=-1))
        np.fill_diagonal(dists, np.inf)
        
        min_pair_val = np.min(dists)
        idx = np.unravel_index(np.argmin(dists), dists.shape)
        i, j = idx
        if i > j: i, j = j, i
        
        wall_dists = 0.5 - np.abs(P)
        min_wall_val = np.min(wall_dists)
        w_idx = np.unravel_index(np.argmin(wall_dists), wall_dists.shape)
        k, coord = w_idx
        
        # Compare contributions to radius
        # r = min(min_pair/2, min_wall)
        val_pair = min_pair_val / 2.0
        val_wall = min_wall_val
        
        moved = False
        if val_pair < val_wall:
            # Move i and j apart
            v = P[j] - P[i]
            norm = np.linalg.norm(v)
            if norm > 1e-9:
                v = v / norm
                step = 0.005
                P[i] -= v * step
                P[j] += v * step
                moved = True
        else:
            # Move k away from wall
            # If P[k, coord] > 0, move left (decrease)
            # If P[k, coord] < 0, move right (increase)
            step = 0.005
            if P[k, coord] > 0:
                P[k, coord] -= step
            else:
                P[k, coord] += step
            moved = True
            
        if not moved:
            break
            
        r_new = compute_radius(P)
        if r_new > r_curr:
            r_curr = r_new
        else:
            # If we didn't improve, maybe try smaller step? 
            # For simplicity, just stop if no improvement in a row?
            # Or just continue, might oscillate.
            # Let's break if no improvement for 50 steps.
            # We don't track history here, so just break on first non-improvement? 
            # That might be too aggressive.
            # Let's just do a fixed number of iterations.
            pass
            
    return P, r_curr

def construct():
    total_time_limit = 110 # 10s margin
    start_time = time.time()
    
    # Run multiple SA restarts
    best_P = None
    best_r = 0
    
    num_restarts = 5
    time_per_restart = (total_time_limit - 10) / num_restarts # Reserve 10s for refinement
    
    for i in range(num_restarts):
        # Check time
        elapsed = time.time() - start_time
        if elapsed > total_time_limit - 10:
            break
            
        P_init = init_random()
        # Run SA
        P_sa, r_sa = simulated_annealing(P_init, time_per_restart)
        
        # Refine
        P_ref, r_ref = local_refinement(P_sa, max_iters=200)
        
        if r_ref > best_r:
            best_r = r_ref
            best_P = P_ref
            
    # Final check and return
    if best_P is None:
        return init_random()
        
    return best_P
