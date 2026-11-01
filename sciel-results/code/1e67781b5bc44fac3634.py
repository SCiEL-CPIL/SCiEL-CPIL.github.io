"""
Packing 50 equal circles in a unit square using Nelder-Mead optimization
on a smooth Log-Sum-Exp approximation of the minimum distance objective.
"""
import numpy as np
import time
from scipy.optimize import minimize

N = 50
HALF = 0.5
DEADLINE_MARGIN = 10.0  # seconds to leave before kill

def _soft_min(vals, beta):
    """
    Compute the soft minimum of an array of values.
    soft_min(v) ≈ -1/beta * log(sum(exp(-beta * v)))
    As beta -> infinity, this approaches min(v).
    """
    # For numerical stability, subtract the min first? 
    # Actually, for soft MIN, we want to penalize small values.
    # min(v) = - max(-v).
    # max(-v) ≈ 1/beta * log(sum(exp(beta * -v))).
    # So min(v) ≈ -1/beta * log(sum(exp(-beta * v))).
    
    # To avoid overflow/underflow, we can use the standard log-sum-exp trick.
    # Let z = -beta * vals. We want log(sum(exp(z))).
    # max_z = max(z).
    # log_sum_exp = max_z + log(sum(exp(z - max_z))).
    
    z = -beta * vals
    max_z = np.max(z)
    # If max_z is very large, exp(z - max_z) is safe.
    # If max_z is very negative, exp(z - max_z) might underflow to 0, which is fine.
    
    # Handle case where all vals are -inf or similar? Not expected here.
    exp_sum = np.sum(np.exp(z - max_z))
    log_sum_exp = max_z + np.log(exp_sum)
    
    return -log_sum_exp / beta

def _score_hard(c):
    """
    Calculate the actual hard score (minimum radius) for a configuration.
    """
    c = np.asarray(c, dtype=np.float64)
    if c.shape != (N, 2) or not np.all(np.isfinite(c)):
        return 0.0
    
    # Pairwise distances
    d = c[:, None, :] - c[None, :, :]
    dist_sq = np.sum(d * d, axis=2)
    np.fill_diagonal(dist_sq, np.inf)
    min_pair_sq = np.min(dist_sq)
    min_pair = np.sqrt(min_pair_sq) / 2.0
    
    # Wall distances
    # Distance to boundary is min(0.5 - x, 0.5 + x) for each coord
    # Which is 0.5 - |x|
    min_wall = np.min(HALF - np.abs(c))
    
    return min(min_pair, min_wall)

def _objective(x, beta):
    """
    Objective function for Nelder-Mead (minimization).
    We want to maximize the soft-min of distances, so we return negative.
    """
    c = x.reshape(N, 2)
    
    # Check bounds strictly to avoid invalid states if NM wanders too far
    # Although NM usually stays close, we clip to a slightly larger box to be safe
    # and add a penalty if it goes outside the valid region for radius r_min?
    # Actually, the soft-min will naturally penalize going out of bounds if wall dist becomes small.
    
    # Compute pairwise squared distances
    d = c[:, None, :] - c[None, :, :]
    dist_sq = np.sum(d * d, axis=2)
    np.fill_diagonal(dist_sq, np.inf)
    
    # We want soft min of distances, not squared distances, to match the objective.
    # dist = sqrt(dist_sq)
    # Computing sqrt for all pairs is expensive but N=50 is small (2500 pairs).
    dist = np.sqrt(dist_sq)
    
    # Wall distances
    wall_dists = HALF - np.abs(c) # shape (N, 2)
    
    # Combine all distances into a single vector
    # Pairwise: 1225 values
    # Wall: 100 values
    # Total: 1325 values
    
    # Flatten pairwise distances (upper triangle to avoid duplicates, though symmetric doesn't matter for min)
    # Using all is fine, just doubles the count.
    pair_dists = dist.flatten()
    
    # Filter out inf
    pair_dists = pair_dists[np.isfinite(pair_dists)]
    
    all_dists = np.concatenate([pair_dists, wall_dists.flatten()])
    
    # Calculate soft min
    # We want to maximize the minimum distance.
    # soft_min returns the approximate minimum distance.
    s_min = _soft_min(all_dists, beta)
    
    # We want to maximize s_min, so minimize -s_min
    return -s_min

def _generate_hex_seed():
    """
    Generate a hexagonal lattice seed.
    """
    # Approximate radius for 50 circles in unit square
    # Area = 1, N=50, r ~ sqrt(1/(50*pi)) ~ 0.08
    # But packing efficiency ~ 0.9, so r ~ 0.07
    
    r_est = 0.07
    dx = 2 * r_est
    dy = np.sqrt(3) * r_est
    
    pts = []
    row = 0
    y = -HALF + r_est
    while y <= HALF - r_est:
        x_offset = 0 if row % 2 == 0 else dx / 2
        x = -HALF + r_est + x_offset
        while x <= HALF - r_est:
            pts.append([x, y])
            x += dx
        y += dy
        row += 1
    
    pts = np.array(pts)
    
    if len(pts) < N:
        # Pad with random points if needed (unlikely for this density)
        while len(pts) < N:
            pts = np.vstack([pts, np.random.uniform(-HALF+r_est, HALF-r_est, 2)])
    elif len(pts) > N:
        # Select N points closest to center
        dists = np.sum(pts**2, axis=1)
        idx = np.argsort(dists)[:N]
        pts = pts[idx]
        
    return pts

def construct():
    start_time = time.time()
    deadline = start_time + 150.0 - DEADLINE_MARGIN
    
    # Initial guess
    pts = _generate_hex_seed()
    x0 = pts.flatten()
    
    best_x = x0.copy()
    best_score = _score_hard(best_x)
    
    # Annealing schedule for beta
    # Start with low beta (smooth), increase to high beta (sharp)
    betas = [10, 20, 50, 100, 200, 500, 1000]
    
    # Nelder-Mead options
    # maxiter needs to be high enough to converge
    # We check time in the callback
    
    for beta in betas:
        if time.time() > deadline:
            break
            
        # Define objective with current beta
        def obj(x):
            return _objective(x, beta)
        
        # Callback to check time and update best
        def callback(xk):
            nonlocal best_x, best_score
            if time.time() > deadline:
                return True # Stop optimization
            
            # Check hard score of current point
            c = xk.reshape(N, 2)
            # Ensure points are roughly within bounds for a valid score calculation
            # If they are way out, score is 0
            if np.any(np.abs(c) > HALF + 0.5):
                return False
            
            s = _score_hard(c)
            if s > best_score:
                best_score = s
                best_x = xk.copy()
            return False
        
        # Run Nelder-Mead
        try:
            res = minimize(
                obj, 
                x0, 
                method='Nelder-Mead', 
                options={
                    'maxiter': 2000,
                    'xatol': 1e-8,
                    'fatol': 1e-8,
                    'adaptive': True
                },
                callback=callback
            )
            
            # Check if this result is better
            x_res = res.x
            c_res = x_res.reshape(N, 2)
            s_res = _score_hard(c_res)
            
            if s_res > best_score:
                best_score = s_res
                best_x = x_res.copy()
            
            # Use the result as the starting point for the next beta
            x0 = x_res.copy()
            
        except Exception as e:
            print(f"Optimization error: {e}")
            # If it fails, keep the previous x0 and try next beta with same x0?
            # Or break? Let's keep x0 and try next beta.
            
    # Final check
    final_pts = best_x.reshape(N, 2)
    
    # Optional: A final quick polish with a very high beta if time remains?
    # Or just return the best found.
    
    return final_pts

# To allow running as a script for quick testing
if __name__ == "__main__":
    pts = construct()
    score = _score_hard(pts)
    print(f"Score: {score}")
    print(f"Best known: 0.071377103865")
    print(f"Improvement: {score / 0.071377103865}")
