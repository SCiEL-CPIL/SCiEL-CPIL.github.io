import numpy as np
from scipy.optimize import differential_evolution, minimize
import time
import math

def construct():
    # Regular octagon as fallback
    def regular_octagon():
        return [[math.cos(2 * math.pi * k / 8), math.sin(2 * math.pi * k / 8)] for k in range(8)]
        
    start_time = time.time()
    
    def neg_score(params):
        pts = params.reshape(8, 2)
        # Center to help with numerical stability, though not strictly necessary for DE
        pts -= np.mean(pts, axis=0)
        
        # Calculate pairwise distances
        # We need sum of logs and max distance
        max_d = 0.0
        log_sum = 0.0
        
        for i in range(8):
            for j in range(i + 1, 8):
                d = np.linalg.norm(pts[i] - pts[j])
                if d < 1e-12:
                    return 1e100
                max_d = max(max_d, d)
                log_sum += np.log(d)
                
        if max_d < 1e-12:
            return 1e100
            
        # Objective: maximize log(prod d_ij) - 28 * log(D)
        # Minimize negative of that
        return -(log_sum - 28.0 * np.log(max_d))

    bounds = [(-2.0, 2.0)] * 16
    
    best_result = None
    best_score = float('inf')
    
    # Run Differential Evolution
    try:
        # Use latinhypercube initialization for better spread
        de_res = differential_evolution(
            neg_score, 
            bounds, 
            maxiter=500, 
            popsize=25, 
            tol=1e-9, 
            seed=42,
            init='latinhypercube',
            polish=False # We will do our own polish or skip
        )
        if de_res.success or de_res.fun < best_score:
            best_result = de_res.x
            best_score = de_res.fun
            
        # Check time
        elapsed = time.time() - start_time
        if elapsed < 120:
            # Try a local polish with L-BFGS-B
            # To handle scale invariance, we add a small penalty to fix scale
            # Let's target diameter = 2
            def neg_score_polished(params):
                pts = params.reshape(8, 2)
                pts -= np.mean(pts, axis=0)
                max_d = 0.0
                log_sum = 0.0
                for i in range(8):
                    for j in range(i + 1, 8):
                        d = np.linalg.norm(pts[i] - pts[j])
                        if d < 1e-12:
                            return 1e100
                        max_d = max(max_d, d)
                        log_sum += np.log(d)
                if max_d < 1e-12:
                    return 1e100
                
                base_val = -(log_sum - 28.0 * np.log(max_d))
                
                # Scale penalty: force diameter to be close to 2
                # This helps L-BFGS-B converge
                penalty = 100.0 * (max_d - 2.0) ** 2
                return base_val + penalty

            local_res = minimize(
                neg_score_polished, 
                best_result, 
                method='L-BFGS-B', 
                bounds=bounds,
                options={'maxiter': 1000, 'ftol': 1e-12}
            )
            
            if local_res.fun < best_score:
                # Verify the actual score without penalty
                # The penalty might have distorted the optimum slightly
                # But it should be very close
                # Let's evaluate the true objective
                true_score = neg_score(local_res.x)
                if true_score < best_score:
                    best_result = local_res.x
                    best_score = true_score
                else:
                    # If penalty distorted it, stick with DE result
                    pass
                    
    except Exception:
        pass
        
    elapsed = time.time() - start_time
    if elapsed > 145:
        # Return best found so far
        if best_result is not None:
            return best_result.reshape(8, 2).tolist()
        return regular_octagon()
        
    if best_result is not None:
        return best_result.reshape(8, 2).tolist()
        
    return regular_octagon()
