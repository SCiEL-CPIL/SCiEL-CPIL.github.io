"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import time
import numpy as np
from scipy.optimize import minimize

def construct():
    start_time = time.time()
    timeout = 145.0 # Safety margin for 150s limit
    
    n = 8
    
    # Helper to compute the score (log product of distances) after scaling to diameter 2
    def compute_log_product(points):
        # points: (n, 2) array
        # Compute all pairwise distances
        # Using broadcasting
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        
        # Find diameter
        max_dist = np.max(dists)
        if max_dist < 1e-10:
            return -np.inf
        
        # Scale to diameter 2
        scale = 2.0 / max_dist
        scaled_dists = dists * scale
        
        # We want product of all ordered pairs? 
        # "Delta is the product over ordered pairs of distances"
        # Usually product over pairs (i,j) with i!=j.
        # There are n*(n-1) such pairs.
        # Score = Delta / n^n.
        # Let's maximize log(Delta) = sum(log(d_ij)) for i!=j.
        # Note: log(a*b) = log(a) + log(b).
        # The set of ordered pairs includes (i,j) and (j,i).
        # So sum is 2 * sum_{i<j} log(d_ij).
        # We can just sum all off-diagonal elements.
        
        # Mask diagonal
        mask = ~np.eye(n, dtype=bool)
        off_diag_dists = scaled_dists[mask]
        
        # Avoid log(0)
        off_diag_dists = np.maximum(off_diag_dists, 1e-15)
        
        log_prod = np.sum(np.log(off_diag_dists))
        return log_prod

    def objective(x):
        # x: flattened (n, 2) points
        points = x.reshape(n, 2)
        
        # Check time limit
        if time.time() - start_time > timeout:
            return np.inf # Signal to stop, or return current best? 
            # Minimize will handle this if we raise an exception or just return large value.
            # Better to check in the loop if possible, but minimize is black box.
            # We'll rely on the number of iterations and time checks in the wrapper if we do manual loops, 
            # but scipy minimize is faster. Let's add a check.
            
        log_prod = compute_log_product(points)
        
        # We want to maximize log_prod, so minimize -log_prod
        return -log_prod

    best_score = -np.inf
    best_points = None

    # Generate initial guesses
    inits = []
    
    # 1. Regular Octagon
    angles = 2 * np.pi * np.arange(n) / n
    reg_oct = np.column_stack([np.cos(angles), np.sin(angles)])
    inits.append(reg_oct.flatten())
    
    # 2. Perturbed Regular Octagon
    np.random.seed(42)
    for _ in range(10):
        perturb = np.random.normal(0, 0.1, (n, 2))
        inits.append((reg_oct + perturb).flatten())
        
    # 3. Two Squares
    # Square 1: angles 0, pi/2, pi, 3pi/2
    # Square 2: angles pi/4, 3pi/4, 5pi/4, 7pi/4
    # This is just the regular octagon.
    # Let's try rotating the second square by a different amount
    a1 = 2 * np.pi * np.arange(4) / 4
    a2 = 2 * np.pi * (np.arange(4) / 4 + 0.1) # Slightly off
    sq1 = np.column_stack([np.cos(a1), np.sin(a1)])
    sq2 = np.column_stack([np.cos(a2), np.sin(a2)])
    inits.append(np.vstack([sq1, sq2]).flatten())
    
    # 4. Random points on unit circle
    for _ in range(5):
        angles = np.sort(np.random.uniform(0, 2*np.pi, n))
        pts = np.column_stack([np.cos(angles), np.sin(angles)])
        inits.append(pts.flatten())
        
    # 5. Points in a square grid-like pattern?
    # Just random 2D points
    for _ in range(5):
        pts = np.random.uniform(-1, 1, (n, 2))
        inits.append(pts.flatten())

    # Optimize each initial guess
    # We use Powell method which is derivative-free and robust for small n
    # Or BFGS which is faster but needs gradients. 
    # Let's use BFGS with numerical gradient or just Powell.
    # Powell is slower per iteration but robust.
    # Given n=8 (16 variables), BFGS should be very fast.
    
    # Let's define a bounds to keep coordinates reasonable, e.g., [-10, 10]
    # Since we scale internally, the absolute scale doesn't matter, but it helps optimizer.
    # Actually, scaling inside objective makes the landscape flat in radial directions?
    # No, scaling normalizes. The optimizer will find the shape.
    
    # To help the optimizer, we can fix one point or center of mass?
    # Not necessary if we scale.
    
    for x0 in inits:
        if time.time() - start_time > timeout - 5:
            break
            
        try:
            res = minimize(objective, x0, method='BFGS', options={'maxiter': 500, 'gtol': 1e-10})
            if res.success or res.fun < best_score:
                # res.fun is -log_prod
                current_log_prod = -res.fun
                if current_log_prod > best_score:
                    best_score = current_log_prod
                    best_points = res.x.reshape(n, 2)
        except Exception:
            pass

    # If no optimization succeeded or best is still regular octagon score, 
    # ensure we return a valid set.
    if best_points is None:
        best_points = reg_oct

    # Final check: ensure distinct points and valid coordinates
    # The problem requires distinct points.
    # If optimization collapsed points, we should fallback.
    # Let's verify the score of best_points
    final_log_prod = compute_log_product(best_points)
    
    # Compare with regular octagon
    reg_log_prod = compute_log_product(reg_oct)
    
    if final_log_prod < reg_log_prod - 1e-6:
        # Fallback to regular octagon if optimization failed or worsened
        best_points = reg_oct

    # Convert to list of lists of floats
    result = [[float(p[0]), float(p[1])] for p in best_points]
    
    return result

# EVOLVE-BLOCK-END
