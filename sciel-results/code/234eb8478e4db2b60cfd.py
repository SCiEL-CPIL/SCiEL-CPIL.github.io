import numpy as np
import time

N = 300
START_TIME = time.time()
TIME_LIMIT = 145.0 # 5s margin

def construct():
    # Initial hexagonal lattice
    s_init = np.sqrt(2 * np.pi / (N * 3 * np.sqrt(3)))
    points = []
    k = int(1.0 / s_init) + 2
    for j in range(-k, k + 1):
        for i in range(-k, k + 1):
            x = (i + 0.5 * (j % 2)) * s_init
            y = j * s_init * np.sqrt(3.0) / 2.0
            if x*x + y*y <= 1.0:
                points.append([x, y])
                
    points = np.array(points)
    if len(points) > N:
        radii = np.hypot(points[:, 0], points[:, 1])
        indices = np.argsort(radii)
        points = points[indices[:N]]
    elif len(points) < N:
        # This shouldn't happen with k calculation, but just in case
        # Pad with random points
        while len(points) < N:
            points.append([np.random.uniform(-0.9, 0.9), np.random.uniform(-0.9, 0.9)])
        points = np.array(points)
        
    # Add small jitter
    points += np.random.normal(0, 0.01, points.shape)

    # Function to check feasibility for a given r
    def is_feasible(r, pts, max_iters=200, tol=1e-5):
        pts = pts.copy()
        for it in range(max_iters):
            if time.time() - START_TIME > TIME_LIMIT:
                return False, pts
            
            # Vectorized force calculation
            D = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
            Dist = np.linalg.norm(D, axis=2)
            np.fill_diagonal(Dist, 1e9)
            
            Mask = Dist < 2.0 * r
            np.fill_diagonal(Mask, False)
            
            # Weights
            Dist_safe = np.where(Mask, Dist, 1.0)
            W = np.where(Mask, (2.0 * r - Dist_safe) / Dist_safe, 0.0)
            
            # Forces from pairs
            # F_i = sum_j W_ij * D_ij
            F_pairs = np.sum(W[:, :, np.newaxis] * D, axis=1)
            
            # Boundary forces
            Rad = np.linalg.norm(pts, axis=1)
            BoundViol = Rad - (1.0 - r)
            # Force is -pts / Rad * BoundViol if BoundViol > 0
            # Vectorized
            F_bound = np.where(BoundViol[:, np.newaxis] > 0, 
                               -pts * (BoundViol / np.maximum(Rad, 1e-9))[:, np.newaxis], 
                               0)
            
            F = F_pairs + F_bound
            
            # Update
            step = 0.5 / (1.0 + it * 0.05)
            pts += step * F
            
            # Check convergence
            if np.max(np.linalg.norm(F, axis=1)) < tol:
                return True, pts
        
        return False, pts

    # Binary search for max r
    r_guess = 0.0531542991469
    
    best_pts = points
    best_r = 0.0
    
    # Try to increase from r_guess
    r = r_guess
    feasible, pts = is_feasible(r, points, max_iters=150)
    
    if feasible:
        best_pts = pts
        best_r = r
        # Increase r
        r_step = 0.00005
        for _ in range(100):
            if time.time() - START_TIME > TIME_LIMIT:
                break
            r += r_step
            feasible, pts = is_feasible(r, best_pts, max_iters=150)
            if feasible:
                best_pts = pts
                best_r = r
            else:
                break
    else:
        # Decrease r
        r_step = 0.00005
        for _ in range(100):
            if time.time() - START_TIME > TIME_LIMIT:
                break
            r -= r_step
            feasible, pts = is_feasible(r, points, max_iters=150)
            if feasible:
                best_pts = pts
                best_r = r
                break

    return best_pts
