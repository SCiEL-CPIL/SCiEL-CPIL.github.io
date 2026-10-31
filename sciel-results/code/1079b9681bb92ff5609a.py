"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import random
from scipy.optimize import minimize

N = 21

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    timeout = 115  # Safety margin for 120s limit
    
    best_score = 0.0
    best_solution = None
    best_w = 1.0
    best_h = 1.0
    
    # Helper to validate and calculate score
    def evaluate_solution(x, w, h):
        # x is shape (N, 3): [cx, cy, r]
        # Check boundaries
        for i in range(N):
            cx, cy, r = x[i]
            if cx - r < -1e-6 or cx + r > w + 1e-6:
                return -np.inf, False
            if cy - r < -1e-6 or cy + r > h + 1e-6:
                return -np.inf, False
        # Check overlaps
        for i in range(N):
            for j in range(i+1, N):
                dx = x[i, 0] - x[j, 0]
                dy = x[i, 1] - x[j, 1]
                dist = np.sqrt(dx*dx + dy*dy)
                min_dist = x[i, 2] + x[j, 2]
                if dist < min_dist - 1e-6:
                    return -np.inf, False
        return np.sum(x[:, 2]), True

    # Objective function for scipy
    # Variables: [cx_0, cy_0, r_0, ..., cx_N, cy_N, r_N]
    # We optimize for a fixed w, h in each inner loop, or jointly?
    # Joint optimization of w, h and circle params is complex.
    # Strategy: Iterate over a set of aspect ratios.
    
    # Define a range of aspect ratios to try
    # w + h = 2. Let w = 2 * a / (a + 1), h = 2 / (a + 1) where a = w/h
    # Or simply sample w in [0.5, 1.5] (since h = 2-w, h>0)
    
    num_aspect_ratios = 10
    num_starts = 20
    
    # Pre-generate random seeds for reproducibility within the run
    random.seed(42)
    np.random.seed(42)
    
    for _ in range(num_aspect_ratios):
        # Sample width
        w = random.uniform(0.6, 1.4)
        h = 2.0 - w
        
        for start_idx in range(num_starts):
            if time.time() - start_time > timeout:
                break
                
            # Initialize circles
            # Strategy: Place circles randomly, then push them inside and resolve overlaps roughly
            # Or start with small radii and grow them?
            # Let's try: Random centers, small radii.
            
            x = np.zeros((N, 3))
            # Random positions inside box
            x[:, 0] = np.random.uniform(0.1, w - 0.1, N)
            x[:, 1] = np.random.uniform(0.1, h - 0.1, N)
            # Small initial radii
            x[:, 2] = np.random.uniform(0.05, 0.15, N)
            
            # Define penalty function
            def objective(params):
                # params shape (3*N,)
                p = params.reshape(N, 3)
                cx, cy, r = p[:, 0], p[:, 1], p[:, 2]
                
                # Penalties for boundaries
                pen = 0.0
                # Left
                viol = np.clip(r - cx, 0, None)
                pen += np.sum(viol**2)
                # Right
                viol = np.clip(cx + r - w, 0, None)
                pen += np.sum(viol**2)
                # Bottom
                viol = np.clip(r - cy, 0, None)
                pen += np.sum(viol**2)
                # Top
                viol = np.clip(cy + r - h, 0, None)
                pen += np.sum(viol**2)
                
                # Overlaps
                for i in range(N):
                    for j in range(i+1, N):
                        dx = cx[i] - cx[j]
                        dy = cy[i] - cy[j]
                        dist = np.sqrt(dx*dx + dy*dy)
                        min_d = r[i] + r[j]
                        viol = np.clip(min_d - dist, 0, None)
                        pen += viol**2
                
                # We want to maximize sum(r), so minimize -sum(r) + penalty
                return -np.sum(r) + 100.0 * pen

            # Initial guess
            x0 = x.flatten()
            
            # Bounds
            # cx in [0, w], cy in [0, h], r in [0, min(w,h)/2]
            max_r = min(w, h) / 2.0
            bounds = []
            for i in range(N):
                bounds.append((0, w))
                bounds.append((0, h))
                bounds.append((0.01, max_r))
            
            # Optimize
            try:
                res = minimize(objective, x0, method='L-BFGS-B', bounds=bounds, options={'maxiter': 500, 'maxfun': 5000})
                
                # Evaluate result
                x_opt = res.x.reshape(N, 3)
                score, valid = evaluate_solution(x_opt, w, h)
                
                if valid and score > best_score:
                    best_score = score
                    best_solution = x_opt
                    best_w = w
                    best_h = h
                    
            except Exception:
                pass

        if time.time() - start_time > timeout:
            break

    # If no valid solution found, return a trivial valid one (small circles in a corner)
    if best_solution is None:
        # Fallback: Grid in a square
        w = 1.0
        h = 1.0
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        best_solution = np.array(cells[:N])
        best_w = w
        best_h = h

    # Final check: ensure perimeter constraint is met (it is by construction w+h=2)
    # Return array of shape (N, 3)
    return best_solution
# EVOLVE-BLOCK-END
