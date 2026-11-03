import numpy as np
import time

N = 26

def calc_radii(P):
    """
    Calculate maximum feasible radii for given centers P.
    """
    dist_walls = np.min(P, 1 - P, axis=1)
    
    # Faster distance calculation
    diff = P[:, np.newaxis, :] - P[np.newaxis, :, :]
    D = np.sqrt(np.sum(diff * diff, axis=2))
    
    R = dist_walls.copy()
    
    for _ in range(15):
        M = D - R[np.newaxis, :]
        np.fill_diagonal(M, np.inf)
        R_new = np.minimum(R, np.min(M, axis=1))
        if np.allclose(R, R_new, atol=1e-7):
            R = R_new
            break
        R = R_new
        
    return R

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    best_score = 0.0
    best_P = None
    best_R = None
    
    def random_init():
        # Use a jittered grid for better initial spread
        # 5x5 grid = 25 points. Add 1 random.
        # Or just 26 random points.
        # Let's use random points in [0.05, 0.95]
        P = np.random.uniform(0.05, 0.95, (N, 2))
        return P

    def local_search(P_init, steps):
        P = P_init.copy()
        R = calc_radii(P)
        score = np.sum(R)
        
        step_size = 0.05
        
        for s in range(steps):
            if time.time() > deadline:
                break
                
            k = np.random.randint(N)
            P_old = P[k].copy()
            
            P[k] += np.random.normal(0, step_size, 2)
            P[k] = np.clip(P[k], 0, 1)
            
            R_new = calc_radii(P)
            score_new = np.sum(R_new)
            
            if score_new > score:
                score = score_new
                R = R_new
            else:
                P[k] = P_old
        
        return P, R, score

    # Main loop
    current_best_P = random_init()
    
    # Run a few full searches
    while time.time() < deadline - 5.0:
        P_init = current_best_P.copy() if best_P is not None else random_init()
        
        if best_P is not None:
            # Perturb best slightly
            P_init += np.random.normal(0, 0.02, (N, 2))
            P_init = np.clip(P_init, 0, 1)
            
        P, R, score = local_search(P_init, steps=15000)
        
        if score > best_score:
            best_score = score
            best_P = P
            best_R = R
            current_best_P = P.copy()
            
    # Final polish on the best solution found
    if best_P is not None:
        P, R, score = local_search(best_P, steps=100000)
        if score > best_score:
            best_score = score
            best_P = P
            best_R = R

    result = np.column_stack((best_P, best_R))
    return result
