import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial.distance import cdist

N = 100

def compute_score(centers):
    # centers shape (N, 3)
    if not np.all(np.isfinite(centers)):
        return 0.0
    # Check bounds
    max_coord = np.max(np.abs(centers))
    if max_coord > 0.5:
        return 0.0 # Or handle in penalty, but for final score it's invalid if outside? 
    # The task says: "A centre on or outside the cube ... make the candidate invalid."
    # Wait, "on or outside" usually means $|c| > 0.5$. 
    # But the score formula is $r = \min(\dots, 0.5 - |c_{ik}|)$.
    # If $|c_{ik}| > 0.5$, the term is negative. 
    # The task says "A centre on or outside the cube ... make the candidate invalid."
    # So we must ensure $|c_{ik}| \le 0.5$.
    
    # Calculate min distance
    dists = cdist(centers, centers)
    # Zero out diagonal
    np.fill_diagonal(dists, np.inf)
    min_dist = np.min(dists)
    
    # Calculate min distance to wall
    min_wall_dist = 0.5 - np.max(np.abs(centers))
    
    r = min(min_dist / 2.0, min_wall_dist)
    
    if r < 0:
        return 0.0
        
    return r

def construct():
    start_time = time.time()
    deadline = start_time + 140.0 # 10s margin
    
    # Initial Grid 5x5x4
    # x, y: 5 points, z: 4 points
    # To maximize initial r, we place them as far apart as possible.
    # If we use the full range [-0.5, 0.5], the boundary constraint is tight.
    # Let's start with r=0.1 grid.
    x = np.linspace(-0.4, 0.4, 5)
    y = np.linspace(-0.4, 0.4, 5)
    z = np.linspace(-0.375, 0.375, 4) # Spacing 0.25? 
    # Wait, 4 points in [-0.4, 0.4] with spacing 0.2?
    # -0.3, -0.1, 0.1, 0.3. Spacing 0.2.
    # Let's use uniform spacing that fits in [-0.4, 0.4].
    # For 5 points: -0.4, -0.2, 0, 0.2, 0.4.
    # For 4 points: -0.3, -0.1, 0.1, 0.3.
    z = np.linspace(-0.3, 0.3, 4)
    
    centers = np.array([(x[i], y[j], z[k]) for k in range(4) for j in range(5) for i in range(5)])
    # This is 4*5*5 = 100 points.
    
    best_centers = centers
    best_score = compute_score(best_centers)
    
    def objective(params):
        c = params[:300].reshape(100, 3)
        r = params[300]
        
        # Penalty for boundary violation
        # We want |c_ik| <= 0.5 - r
        # Violation: |c_ik| - (0.5 - r)
        wall_viol = np.maximum(0, np.abs(c) - (0.5 - r))
        wall_pen = np.sum(wall_viol**2)
        
        # Penalty for overlap
        # We want |c_i - c_j| >= 2r
        # Violation: 2r - |c_i - c_j|
        dists = cdist(c, c)
        np.fill_diagonal(dists, np.inf)
        overlap_viol = np.maximum(0, 2*r - dists)
        overlap_pen = np.sum(overlap_viol**2)
        
        # We want to maximize r, so minimize -r
        # Add penalties
        return -r + 100.0 * (wall_pen + overlap_pen)
        
    # Flatten initial centers and r
    r_init = 0.1
    x0 = np.concatenate([centers.flatten(), [r_init]])
    
    bounds = [(-0.5, 0.5)] * 300 + [(0.0, 0.5)]
    
    # Run optimization
    while time.time() < deadline:
        res = minimize(objective, x0, method='L-BFGS-B', bounds=bounds, 
                       options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8})
        
        c_new = res.x[:300].reshape(100, 3)
        r_new = res.x[300]
        
        # Clip centers to valid range to ensure validity
        c_new = np.clip(c_new, -0.5, 0.5)
        
        score_new = compute_score(c_new)
        
        if score_new > best_score:
            best_score = score_new
            best_centers = c_new
            x0 = np.concatenate([c_new.flatten(), [score_new]]) # Use current best r as init for next?
            # Actually, if we improved, we can try to push further.
            # But if we are stuck, we should perturb.
            
        # Check if we are close to the best known or if optimization converged
        if res.status == 1: # Optimal
            # Try a random perturbation to escape local minimum
            if time.time() < deadline - 10:
                noise = np.random.normal(0, 0.01, (100, 3))
                c_perturbed = np.clip(c_new + noise, -0.5, 0.5)
                x0 = np.concatenate([c_perturbed.flatten(), [best_score]])
            else:
                break
        else:
            # If it didn't converge, maybe increase penalty or restart
            break

    return best_centers
