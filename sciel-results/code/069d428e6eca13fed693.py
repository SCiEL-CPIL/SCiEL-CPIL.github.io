"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from scipy.optimize import minimize
import random

N = 21

# EVOLVE-BLOCK-START
import numpy as np
import time
import random
from scipy.optimize import minimize

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # 5s safety margin
    
    best_score = -np.inf
    best_result = None
    
    # Initialize with a random grid-like structure to provide a decent starting point
    # We will optimize w, h, and circle parameters (cx, cy, r) for each circle
    
    # Objective function for L-BFGS-B
    def objective(params, w, h):
        # params: [cx1, cy1, r1, cx2, cy2, r2, ..., cxN, cyN, rN]
        # We fix w and h for the local optimization step, or optimize them jointly?
        # Optimizing w, h jointly with 21 circles (63 vars) might be slow for L-BFGS-B.
        # Let's optimize positions and radii for fixed w, h first, then adjust w, h in SA.
        
        cx = params[0::3]
        cy = params[1::3]
        r = params[2::3]
        
        # Penalty for out of bounds
        # Rectangle is centered at (0,0) for symmetry, width w, height h
        # Bounds: -w/2 <= cx <= w/2, -h/2 <= cy <= h/2
        # Circle inside: cx + r <= w/2 => cx - w/2 + r <= 0 is violation? No.
        # Violation if cx + r > w/2 or cx - r < -w/2
        # dist_right = cx + r - w/2
        # dist_left = -w/2 - (cx - r) = r - cx - w/2
        
        pen_x = np.maximum(0, cx + r - w/2) + np.maximum(0, -w/2 - cx + r)
        pen_y = np.maximum(0, cy + r - h/2) + np.maximum(0, -h/2 - cy + r)
        
        # Penalty for overlaps
        # d_ij < r_i + r_j
        # Vectorized distance calculation
        cx_diff = cx[:, None] - cx[None, :]
        cy_diff = cy[:, None] - cy[None, :]
        dist = np.sqrt(cx_diff**2 + cy_diff**2)
        r_sum = r[:, None] + r[None, :]
        
        # Only upper triangle to avoid double counting and self-interaction
        iu = np.triu_indices(N, k=1)
        overlap_pen = np.maximum(0, r_sum[iu] - dist[iu])
        
        penalty = 100.0 * (np.sum(pen_x) + np.sum(pen_y) + np.sum(overlap_pen))
        
        # Objective: Minimize -sum(r) + penalty
        return -np.sum(r) + penalty

    def run_sa_iteration(w, h, cx, cy, r, temp):
        # Perturb parameters
        new_w = w + random.gauss(0, 0.1)
        new_h = 2.0 - new_w # Perimeter 4 => 2(w+h)=4 => w+h=2
        if new_w <= 0.1 or new_h <= 0.1:
            return w, h, cx, cy, r, False
        
        new_cx = cx.copy()
        new_cy = cy.copy()
        new_r = r.copy()
        
        # Perturb positions
        new_cx += random.gauss(0, 0.1, N)
        new_cy += random.gauss(0, 0.1, N)
        
        # Perturb radii
        new_r += random.gauss(0, 0.05, N)
        new_r = np.maximum(new_r, 0.01) # Keep radii positive
        
        return new_w, new_h, new_cx, new_cy, new_r, True

    def evaluate_config(w, h, cx, cy, r):
        # Calculate score and validity
        # Check bounds
        if np.any(cx + r > w/2) or np.any(cx - r < -w/2) or np.any(cy + r > h/2) or np.any(cy - r < -h/2):
            return -np.inf, False
            
        # Check overlaps
        cx_diff = cx[:, None] - cx[None, :]
        cy_diff = cy[:, None] - cy[None, :]
        dist = np.sqrt(cx_diff**2 + cy_diff**2)
        r_sum = r[:, None] + r[None, :]
        
        iu = np.triu_indices(N, k=1)
        if np.any(dist[iu] < r_sum[iu] - 1e-9):
            return -np.inf, False
            
        return np.sum(r), True

    # Initial Setup
    # Start with a square-ish rectangle
    w = 1.0
    h = 1.0
    
    # Initial grid
    side = int(np.ceil(np.sqrt(N)))
    cx = np.zeros(N)
    cy = np.zeros(N)
    r = np.zeros(N)
    
    idx = 0
    for i in range(side):
        for j in range(side):
            if idx >= N:
                break
            # Map grid indices to [-w/2, w/2]
            cx[idx] = -w/2 + (i + 0.5) * (w / side)
            cy[idx] = -h/2 + (j + 0.5) * (h / side)
            r[idx] = 0.5 * min(w, h) / side * 0.95 # Slightly smaller to avoid initial overlaps
            idx += 1
            
    # Fill remaining if N > side*side? No, N=21, side=5, 5x5=25. So we use first 21.
    # Actually, let's just use a random init if grid is too simple.
    # Let's stick to grid for stability, then optimize.
    
    current_score, _ = evaluate_config(w, h, cx, cy, r)
    if current_score > best_score:
        best_score = current_score
        best_result = (w, h, cx.copy(), cy.copy(), r.copy())
        
    temp = 0.5
    cool_rate = 0.999
    min_temp = 0.01
    
    # Main SA Loop
    while time.time() < deadline:
        # Check time inside loop for quick exit
        if time.time() > deadline - 2.0:
            break
            
        new_w, new_h, new_cx, new_cy, new_r, valid_perturb = run_sa_iteration(w, h, cx, cy, r, temp)
        
        if not valid_perturb:
            continue
            
        # Evaluate new config
        new_score, is_valid = evaluate_config(new_w, new_h, new_cx, new_cy, new_r)
        
        # Metropolis criterion
        if new_score > current_score:
            w, h, cx, cy, r = new_w, new_h, new_cx, new_cy, new_r
            current_score = new_score
            if new_score > best_score:
                best_score = new_score
                best_result = (w, h, cx.copy(), cy.copy(), r.copy())
                
                # Optional: Trigger L-BFGS-B refinement on new best
                # This is expensive, so do it sparingly or if time allows
                if random.random() < 0.05 and time.time() < deadline - 10.0:
                    # Prepare params for L-BFGS-B
                    # We fix w, h to current best for local refinement
                    params = np.zeros(N * 3)
                    params[0::3] = cx
                    params[1::3] = cy
                    params[2::3] = r
                    
                    # Use a smaller penalty weight for L-BFGS-B to allow it to push boundaries
                    # But we need strict validity for the final result.
                    # L-BFGS-B minimizes the objective.
                    res = minimize(objective, params, args=(w, h), method='L-BFGS-B', options={'maxiter': 50})
                    
                    if res.success:
                        new_cx_lb = res.x[0::3]
                        new_cy_lb = res.x[1::3]
                        new_r_lb = res.x[2::3]
                        
                        # Validate the L-BFGS-B result
                        lb_score, lb_valid = evaluate_config(w, h, new_cx_lb, new_cy_lb, new_r_lb)
                        if lb_valid and lb_score > best_score:
                            best_score = lb_score
                            best_result = (w, h, new_cx_lb.copy(), new_cy_lb.copy(), new_r_lb.copy())
                            # Update current state to this better local optimum
                            cx, cy, r = new_cx_lb, new_cy_lb, new_r_lb
                            current_score = lb_score
        else:
            # Accept with probability exp((new_score - current_score) / temp)
            # Note: Scores are sums of radii (higher is better).
            # So delta = new_score - current_score. If negative, we might accept.
            delta = new_score - current_score
            if delta < 0:
                prob = np.exp(delta / temp)
                if random.random() < prob:
                    w, h, cx, cy, r = new_w, new_h, new_cx, new_cy, new_r
                    current_score = new_score
            # If new_score is -inf, delta is very negative, prob is 0.
        
        temp *= cool_rate
        if temp < min_temp:
            temp = min_temp

    # Final refinement of the best result found
    if best_result is not None and time.time() < deadline - 5.0:
        w, h, cx, cy, r = best_result
        params = np.zeros(N * 3)
        params[0::3] = cx
        params[1::3] = cy
        params[2::3] = r
        
        # Try a few L-BFGS-B restarts with different initial perturbations or just refine
        res = minimize(objective, params, args=(w, h), method='L-BFGS-B', options={'maxiter': 100})
        
        if res.success:
            new_cx_lb = res.x[0::3]
            new_cy_lb = res.x[1::3]
            new_r_lb = res.x[2::3]
            
            lb_score, lb_valid = evaluate_config(w, h, new_cx_lb, new_cy_lb, new_r_lb)
            if lb_valid and lb_score > best_score:
                best_score = lb_score
                best_result = (w, h, new_cx_lb.copy(), new_cy_lb.copy(), new_r_lb.copy())

    if best_result is None:
        # Fallback to seed if something went wrong
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        # Convert to centered coordinates? The seed returns [x, y, r] in [0,1]x[0,1] presumably?
        # The task says "bounding rectangle has perimeter at most 4".
        # The seed returns coordinates in [0,1] range?
        # Let's check the seed: `((i + 0.5) / side, (j + 0.5) / side, ...)`
        # If side=5, coords are 0.1, 0.3, 0.5, 0.7, 0.9.
        # This implies a rectangle of width 1 and height 1? Perimeter 4.
        # My optimization uses centered coords [-w/2, w/2].
        # I must convert back to [0, w] x [0, h] for the output?
        # Or does the evaluator expect specific coordinate ranges?
        # "array of shape (n, 3): n disjoint circles whose bounding rectangle has perimeter at most 4"
        # Usually, this implies the coordinates are absolute positions.
        # The seed uses [0,1] range. If I return centered coords, the bounding box might be negative.
        # The evaluator likely calculates the bounding box from the returned coordinates.
        # So I should return coordinates such that min(x)-r >= 0? No, just disjoint and perimeter <= 4.
        # It's safer to return coordinates in the positive quadrant, e.g., shifted by w/2, h/2.
        
        # Let's assume the evaluator computes bbox from the returned x,y.
        # So I should shift my centered coords to start at 0.
        
        # Re-evaluating the output format.
        # If I return centered coords, x can be negative.
        # The bounding box would be [min(x)-max(r), max(x)+max(r)]?
        # No, "bounding rectangle" usually means the smallest axis-aligned rectangle containing all circles.
        # So min_x = min(cx - r), max_x = max(cx + r).
        # Width = max_x - min_x.
        # Perimeter = 2 * (Width + Height).
        # My optimization ensures circles are inside [ -w/2, w/2 ] x [ -h/2, h/2 ].
        # So the bounding box is exactly w x h.
        # The coordinates returned should probably be the actual coordinates.
        # If I return centered coords, the "bounding rectangle" is still w x h, just shifted.
        # Is there a constraint that x,y > 0? The seed has x,y > 0.
        # To be safe, I will shift the coordinates so that min_x - max_r >= 0 is not required, but usually packing problems assume positive coordinates or just relative.
        # However, if the evaluator checks `x >= 0` or similar, it might fail.
        # The seed returns positive values. I will shift my results to be positive.
        # Shift: x_new = x + w/2, y_new = y + h/2.
        # Then min x will be close to 0 (if circle touches left wall).
        
        return np.array(cells[:N])

    w, h, cx, cy, r = best_result
    
    # Shift to positive quadrant
    cx_out = cx + w/2
    cy_out = cy + h/2
    
    result = np.column_stack((cx_out, cy_out, r))
    return result

# EVOLVE-BLOCK-END
