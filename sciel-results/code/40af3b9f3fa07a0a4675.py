import numpy as np
import time
from scipy.optimize import minimize
import math

def score_from_points(P):
    """
    Calculate the score for a set of points P (8x2).
    Score = Delta / n^n, where Delta is product of ordered pair distances,
    after scaling P to diameter 2.
    """
    n = P.shape[0]
    diff = P[:, np.newaxis, :] - P[np.newaxis, :, :]
    dists_sq = np.sum(diff**2, axis=2)
    
    # Mask diagonal
    mask = ~np.eye(n, dtype=bool)
    dists_sq = dists_sq[mask]
    
    if np.any(dists_sq < 1e-12):
        return 0.0
        
    dists = np.sqrt(dists_sq)
    D = np.max(dists)
    
    if D < 1e-12:
        return 0.0
        
    # Scale to diameter 2
    scale = 2.0 / D
    # Delta is product of all ordered pairs (i != j)
    # log(Delta) = sum(log(dists))
    # We want Delta_scaled / n^n
    # Delta_scaled = Delta * scale^(n*(n-1))
    # log(Score) = log(Delta) + n*(n-1)*log(scale) - n*log(n)
    
    sum_log_dists = np.sum(np.log(dists))
    log_score = sum_log_dists + n*(n-1)*np.log(scale) - n*np.log(n)
    
    return np.exp(log_score)

def objective_angles(theta):
    """
    theta: 8 angles
    Points are on the unit circle: (cos(theta_i), sin(theta_i))
    Since scaling is invariant to the initial radius, we can fix radius=1 
    and let the diameter scaling handle the rest. 
    However, points might not be on the unit circle in the optimal solution?
    Actually, for maximizing product of distances with fixed diameter, 
    the optimal points lie on the boundary of the convex hull. 
    For n=8, it is highly probable they lie on a circle (or close to it).
    Let's assume they lie on a circle of radius R. 
    If we fix R=1, the diameter D is determined by the max distance.
    The score is scale-invariant. 
    So we can just optimize the angles of 8 points on the unit circle.
    """
    P = np.column_stack([np.cos(theta), np.sin(theta)])
    
    # Check for duplicates
    dists_sq = np.sum((P[:, np.newaxis, :] - P[np.newaxis, :, :])**2, axis=2)
    mask = ~np.eye(8, dtype=bool)
    if np.any(dists_sq[mask] < 1e-10):
        return 1e100
        
    dists = np.sqrt(dists_sq[mask])
    D = np.max(dists)
    
    if D < 1e-12:
        return 1e100
        
    sum_log_dists = np.sum(np.log(dists))
    n = 8
    log_score = sum_log_dists + n*(n-1)*np.log(2.0/D) - n*np.log(n)
    
    return -log_score

def objective_general(x):
    """
    General 2D objective for polishing.
    x: 16 parameters
    """
    P = x.reshape(8, 2)
    dists_sq = np.sum((P[:, np.newaxis, :] - P[np.newaxis, :, :])**2, axis=2)
    mask = ~np.eye(8, dtype=bool)
    dists_sq = dists_sq[mask]
    
    if np.any(dists_sq < 1e-12):
        return 1e100
        
    dists = np.sqrt(dists_sq)
    D = np.max(dists)
    
    if D < 1e-12:
        return 1e100
        
    sum_log_dists = np.sum(np.log(dists))
    n = 8
    log_score = sum_log_dists + n*(n-1)*np.log(2.0/D) - n*np.log(n)
    
    return -log_score

def construct():
    start_time = time.time()
    deadline = start_time + 140.0 # Safety margin for 150s limit
    
    best_val = -1e100
    best_points = None
    
    # Initial guess: Regular Octagon
    base_angles = np.arange(8) * (2 * np.pi / 8)
    P_reg = np.column_stack([np.cos(base_angles), np.sin(base_angles)])
    best_val = score_from_points(P_reg)
    best_points = P_reg.copy()
    
    # Strategy: Multi-start local optimization in angle space
    # The space of 8 angles on a circle has symmetries (rotation, reflection).
    # We can fix the first angle to 0 to break rotational symmetry, 
    # but Powell/Nelder-Mead handles symmetries okay if we provide good starts.
    
    # Generate diverse starting points for angle optimization
    starts = []
    
    # 1. Regular Octagon
    starts.append(base_angles.copy())
    
    # 2. Perturbed Regular Octagon (small random shifts)
    for _ in range(10):
        perturbed = base_angles + np.random.uniform(-0.1, 0.1, 8)
        starts.append(perturbed)
        
    # 3. Random angles (sorted to maintain some structure, though order doesn't matter for the set)
    for _ in range(20):
        random_angles = np.sort(np.random.uniform(0, 2*np.pi, 8))
        starts.append(random_angles)
        
    # 4. "Square-like" configurations (4 points near corners of a square, 4 near another)
    # e.g., two squares rotated by 45 degrees (which is the regular octagon)
    # Let's try slightly off-square
    for _ in range(10):
        base_sq = np.array([0, np.pi/2, np.pi, 3*np.pi/2])
        base_sq2 = base_sq + np.pi/4
        all_angles = np.sort(np.concatenate([base_sq, base_sq2]))
        perturbed = all_angles + np.random.uniform(-0.05, 0.05, 8)
        starts.append(perturbed)

    # Optimize each start
    for i, start_theta in enumerate(starts):
        if time.time() > deadline - 5:
            break
            
        try:
            # Use Powell's method which is derivative-free and robust
            res = minimize(
                objective_angles, 
                start_theta, 
                method='Powell', 
                options={'maxiter': 100, 'ftol': 1e-10, 'xtol': 1e-10}
            )
            
            if res.fun < 1e99:
                theta_opt = res.x
                P_opt = np.column_stack([np.cos(theta_opt), np.sin(theta_opt)])
                
                # Check validity
                dists_sq = np.sum((P_opt[:, np.newaxis, :] - P_opt[np.newaxis, :, :])**2, axis=2)
                if np.any(dists_sq[~np.eye(8, dtype=bool)] < 1e-9):
                    continue
                    
                val = score_from_points(P_opt)
                
                if val > best_val:
                    best_val = val
                    best_points = P_opt.copy()
                    
        except Exception:
            continue
            
    # Strategy 2: Refine the best point found in full 2D space
    # This allows points to move off the unit circle if that improves the score
    # (though theory suggests they should be on the boundary of the convex hull)
    if best_points is not None and time.time() < deadline - 10:
        try:
            # Flatten points
            x0 = best_points.reshape(-1)
            
            # Use Nelder-Mead or Powell in 2D
            res_2d = minimize(
                objective_general, 
                x0, 
                method='Powell', 
                options={'maxiter': 200, 'ftol': 1e-12, 'xtol': 1e-12}
            )
            
            if res_2d.fun < 1e99:
                P_2d = res_2d.x.reshape(8, 2)
                val_2d = score_from_points(P_2d)
                
                if val_2d > best_val:
                    best_val = val_2d
                    best_points = P_2d.copy()
                    
        except Exception:
            pass
            
    # Strategy 3: If we have time, do a few random 2D restarts from the best point
    if best_points is not None and time.time() < deadline - 20:
        for _ in range(5):
            if time.time() > deadline - 5:
                break
            try:
                # Add small noise to best points
                P_noisy = best_points + np.random.normal(0, 0.01, best_points.shape)
                x0 = P_noisy.reshape(-1)
                
                res_2d = minimize(
                    objective_general, 
                    x0, 
                    method='Powell', 
                    options={'maxiter': 100, 'ftol': 1e-10, 'xtol': 1e-10}
                )
                
                if res_2d.fun < 1e99:
                    P_2d = res_2d.x.reshape(8, 2)
                    val_2d = score_from_points(P_2d)
                    
                    if val_2d > best_val:
                        best_val = val_2d
                        best_points = P_2d.copy()
                        
            except Exception:
                continue
                
    # Final safety check
    if best_points is None:
        return P_reg.tolist()
        
    # Ensure distinct points
    dists_sq = np.sum((best_points[:, np.newaxis, :] - best_points[np.newaxis, :, :])**2, axis=2)
    if np.any(dists_sq[~np.eye(8, dtype=bool)] < 1e-9):
        return P_reg.tolist()
        
    return best_points.tolist()
