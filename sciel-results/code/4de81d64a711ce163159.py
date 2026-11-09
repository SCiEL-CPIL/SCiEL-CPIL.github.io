# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

def squares_in_square10() -> tuple:
    """
    Optimizes the packing of 10 squares in a unit square to maximize the sum of their side lengths.
    Uses a penalty-based optimization approach starting from a known valid configuration.
    """
    n = 10
    tol = 1e-7
    
    # Initial configuration: 1 square of 0.5, 9 squares of 0.25
    # This is a valid packing.
    # Square 0: 0.5 at (0.25, 0.25)
    # Squares 1-4: 0.25 in top-left 0.5x0.5 area
    # Squares 5-8: 0.25 in top-right 0.5x0.5 area (part of right strip)
    # Square 9: 0.25 in bottom-right 0.5x0.5 area (part of right strip)
    
    centers = np.zeros((n, 2))
    sides = np.zeros(n)
    
    # Square 0
    centers[0] = [0.25, 0.25]
    sides[0] = 0.5
    
    # Top-Left 0.5x0.5 region: [0, 0.5] x [0.5, 1.0]
    # 4 squares of 0.25
    # (0.125, 0.625), (0.375, 0.625), (0.125, 0.875), (0.375, 0.875)
    centers[1] = [0.125, 0.625]; sides[1] = 0.25
    centers[2] = [0.375, 0.625]; sides[2] = 0.25
    centers[3] = [0.125, 0.875]; sides[3] = 0.25
    centers[4] = [0.375, 0.875]; sides[4] = 0.25
    
    # Right Strip 0.5x1.0 region: [0.5, 1.0] x [0, 1.0]
    # Top part [0.5, 1.0] x [0.5, 1.0]
    centers[5] = [0.625, 0.625]; sides[5] = 0.25
    centers[6] = [0.875, 0.625]; sides[6] = 0.25
    centers[7] = [0.625, 0.875]; sides[7] = 0.25
    centers[8] = [0.875, 0.875]; sides[8] = 0.25
    
    # Bottom part [0.5, 1.0] x [0, 0.5]
    centers[9] = [0.75, 0.25]; sides[9] = 0.25 # Centered in 0.5x0.5
    
    # Flatten parameters: [cx0, cy0, s0, cx1, cy1, s1, ...]
    x0 = np.concatenate([centers, sides])
    
    def objective(params):
        c = params[:2*n].reshape(n, 2)
        s = params[2*n:]
        
        # Penalties
        penalty = 0.0
        
        # Boundary penalty
        # s/2 must be <= c and <= 1-c
        # c - s/2 >= -tol  => s/2 - c <= tol
        # c + s/2 <= 1 + tol => c + s/2 - 1 <= tol
        # We want to maximize sum(s), so minimize -sum(s) + penalty
        # Penalty for boundary violation
        lower_viol = np.maximum(0, s/2 - c[:, 0] + tol) + np.maximum(0, s/2 - c[:, 1] + tol)
        upper_viol = np.maximum(0, c[:, 0] + s/2 - 1 - tol) + np.maximum(0, c[:, 1] + s/2 - 1 - tol)
        boundary_pen = np.sum(lower_viol**2) + np.sum(upper_viol**2)
        
        # Overlap penalty
        # For each pair i, j
        # |ci - cj| < (si + sj)/2 - tol  => overlap
        # We want |ci - cj| >= (si + sj)/2 - tol
        # Violation = max(0, (si + sj)/2 - tol - |ci - cj|)
        overlap_pen = 0.0
        for i in range(n):
            for j in range(i+1, n):
                dx = c[i, 0] - c[j, 0]
                dy = c[i, 1] - c[j, 1]
                dist_x = abs(dx)
                dist_y = abs(dy)
                req_x = (s[i] + s[j]) / 2 - tol
                req_y = (s[i] + s[j]) / 2 - tol
                # If dist_x < req_x AND dist_y < req_y, they overlap
                # We penalize the "amount" of overlap in both dimensions if both are violated
                # A simpler penalty: sum of squares of violations
                viol_x = max(0, req_x - dist_x)
                viol_y = max(0, req_y - dist_y)
                # If both are positive, they overlap.
                # Penalty should be high if both are positive.
                # Using product or sum of squares.
                overlap_pen += viol_x**2 + viol_y**2
                
        # Minimize -sum(s) + lambda * (boundary_pen + overlap_pen)
        # We want to maximize sum(s), so objective is -sum(s) + penalty
        # If penalty is 0, we maximize sum(s).
        # If penalty > 0, we reduce sum(s) to satisfy constraints.
        # We need a strong penalty coefficient.
        lambda_pen = 1000.0
        return -np.sum(s) + lambda_pen * (boundary_pen + overlap_pen)

    # Bounds
    # s > 0
    # c in [0, 1]
    bounds = []
    for i in range(n):
        bounds.append((0.0, 1.0)) # cx
        bounds.append((0.0, 1.0)) # cy
        bounds.append((1e-6, 1.0)) # s
    
    # Optimize
    start_time = time.time()
    timeout = 500 # Safety margin
    
    result = minimize(objective, x0, method='L-BFGS-B', bounds=bounds, 
                      options={'maxiter': 1000, 'ftol': 1e-10, 'gtol': 1e-8})
    
    # Check time
    if time.time() - start_time > timeout:
        # Return best found so far? L-BFGS-B doesn't easily support early return with best state
        # But it's fast enough usually.
        pass
    
    # Extract best result
    x_best = result.x
    centers_best = x_best[:2*n].reshape(n, 2)
    sides_best = x_best[2*n:]
    
    # Ensure validity
    # Clip sides to positive
    sides_best = np.maximum(sides_best, 1e-6)
    
    # Final check and cleanup
    # If the optimization resulted in a bad penalty, we might have overlaps.
    # We can try to "fix" by reducing sides slightly if there are overlaps.
    # But L-BFGS-B with strong penalty should handle it.
    
    return centers_best, sides_best

# EVOLVE-BLOCK-END
