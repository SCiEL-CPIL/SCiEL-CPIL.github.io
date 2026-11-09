import numpy as np
from scipy.optimize import minimize
import time

def squares_in_square10() -> tuple:
    n = 10
    start_time = time.time()
    deadline = 590.0  # Safety margin

    # Baseline: 2x5 grid
    cx_base = np.array([0.1, 0.3, 0.5, 0.7, 0.9] * 2)
    cy_base = np.array([0.1] * 5 + [0.9] * 5)
    s_base = np.full(10, 0.2)
    best_centers = np.stack([cx_base, cy_base], axis=1)
    best_sides = s_base.copy()
    best_score = np.sum(s_base)

    def unpack(params):
        cx = params[0:n]
        cy = params[n:2*n]
        s = params[2*n:3*n]
        return cx, cy, s

    def is_valid(params):
        cx, cy, s = unpack(params)
        if not np.all(np.isfinite(params)):
            return False
        if not np.all(s > 0):
            return False
        for i in range(n):
            if cx[i] - s[i]/2 < -1e-7 or cx[i] + s[i]/2 > 1 + 1e-7:
                return False
            if cy[i] - s[i]/2 < -1e-7 or cy[i] + s[i]/2 > 1 + 1e-7:
                return False
        for i in range(n):
            for j in range(i+1, n):
                dx = abs(cx[i] - cx[j])
                dy = abs(cy[i] - cy[j])
                t = (s[i] + s[j]) / 2.0
                if dx < t - 1e-7 and dy < t - 1e-7:
                    return False
        return True

    def objective(params):
        return -np.sum(params[2*n:3*n])

    def make_constraints():
        cons = []
        # Boundary constraints
        for i in range(n):
            # cx - s/2 >= 0
            cons.append({'type': 'ineq', 'fun': lambda x, i=i: x[i] - x[2*n+i]/2})
            # cx + s/2 <= 1
            cons.append({'type': 'ineq', 'fun': lambda x, i=i: 1 - (x[i] + x[2*n+i]/2)})
            # cy - s/2 >= 0
            cons.append({'type': 'ineq', 'fun': lambda x, i=i: x[n+i] - x[2*n+i]/2})
            # cy + s/2 <= 1
            cons.append({'type': 'ineq', 'fun': lambda x, i=i: 1 - (x[n+i] + x[2*n+i]/2)})
            
        # Non-overlap constraints
        # For each pair (i, j), we need to ensure they are separated in X or Y.
        # We will try both orientations for each pair and let the solver pick?
        # No, SLSQP doesn't handle OR.
        # We must fix the orientation.
        # Strategy: For a given initial configuration, determine the separation axis for each pair.
        # If |cx_i - cx_j| > |cy_i - cy_j|, assume X-separation. Else Y-separation.
        # Then add the corresponding constraint.
        # This creates a non-convex problem but we are doing local optimization.
        
        return cons

    def make_constraints_with_orientation(params_init):
        cx, cy, s = unpack(params_init)
        cons = []
        # Boundary constraints
        for i in range(n):
            cons.append({'type': 'ineq', 'fun': lambda x, i=i: x[i] - x[2*n+i]/2})
            cons.append({'type': 'ineq', 'fun': lambda x, i=i: 1 - (x[i] + x[2*n+i]/2)})
            cons.append({'type': 'ineq', 'fun': lambda x, i=i: x[n+i] - x[2*n+i]/2})
            cons.append({'type': 'ineq', 'fun': lambda x, i=i: 1 - (x[n+i] + x[2*n+i]/2)})
            
        # Non-overlap
        for i in range(n):
            for j in range(i+1, n):
                # Determine orientation from initial config
                dx = abs(cx[i] - cx[j])
                dy = abs(cy[i] - cy[j])
                # If dx >= dy, enforce X separation: |cx_i - cx_j| >= (s_i + s_j)/2
                # This is equivalent to:
                # cx_i - cx_j >= (s_i + s_j)/2  OR  cx_j - cx_i >= (s_i + s_j)/2
                # We need to know the sign.
                if cx[i] >= cx[j]:
                    # Enforce cx_i - cx_j >= (s_i + s_j)/2
                    cons.append({'type': 'ineq', 'fun': lambda x, i=i, j=j: (x[i] - x[j]) - (x[2*n+i] + x[2*n+j])/2})
                else:
                    # Enforce cx_j - cx_i >= (s_i + s_j)/2
                    cons.append({'type': 'ineq', 'fun': lambda x, i=i, j=j: (x[j] - x[i]) - (x[2*n+i] + x[2*n+j])/2})
                
                # Wait, this forces them to stay separated in that specific direction.
                # If they were originally separated in X, they must remain so.
                # If they were separated in Y, we should enforce Y separation.
                # Let's refine:
                # If dx > dy, assume X separation.
                # If dy > dx, assume Y separation.
                # If dx == dy, assume X.
                
        # The above logic is flawed because I'm adding constraints based on cx[i] >= cx[j] 
        # without checking if they were actually separated in X.
        # Let's restart the constraint generation logic.
        
        cons = []
        for i in range(n):
            cons.append({'type': 'ineq', 'fun': lambda x, i=i: x[i] - x[2*n+i]/2})
            cons.append({'type': 'ineq', 'fun': lambda x, i=i: 1 - (x[i] + x[2*n+i]/2)})
            cons.append({'type': 'ineq', 'fun': lambda x, i=i: x[n+i] - x[2*n+i]/2})
            cons.append({'type': 'ineq', 'fun': lambda x, i=i: 1 - (x[n+i] + x[2*n+i]/2)})
            
        for i in range(n):
            for j in range(i+1, n):
                dx = abs(cx[i] - cx[j])
                dy = abs(cy[i] - cy[j])
                
                if dx >= dy:
                    # Enforce X separation
                    if cx[i] >= cx[j]:
                        cons.append({'type': 'ineq', 'fun': lambda x, i=i, j=j: (x[i] - x[j]) - (x[2*n+i] + x[2*n+j])/2})
                    else:
                        cons.append({'type': 'ineq', 'fun': lambda x, i=i, j=j: (x[j] - x[i]) - (x[2*n+i] + x[2*n+j])/2})
                else:
                    # Enforce Y separation
                    if cy[i] >= cy[j]:
                        cons.append({'type': 'ineq', 'fun': lambda x, i=i, j=j: (x[n+i] - x[n+j]) - (x[2*n+i] + x[2*n+j])/2})
                    else:
                        cons.append({'type': 'ineq', 'fun': lambda x, i=i, j=j: (x[n+j] - x[n+i]) - (x[2*n+i] + x[2*n+j])/2})
                        
        return cons

    def bounds():
        b = []
        for i in range(n):
            b.append((0, 1)) # cx
            b.append((0, 1)) # cy
            b.append((0.01, 1.0)) # s
        return b

    # Generate diverse initial configurations
    configs = []
    
    # 1. 2x5 Grid
    configs.append(np.hstack([cx_base, cy_base, s_base]))
    
    # 2. 5x2 Grid
    c2_x = np.array([0.1] * 5 + [0.9] * 5)
    c2_y = np.array([0.1, 0.3, 0.5, 0.7, 0.9] * 2)
    c2_s = np.full(10, 0.2)
    configs.append(np.hstack([c2_x, c2_y, c2_s]))
    
    # 3. 3x3 Grid + 1 small (Shrink 3x3 to fit 10th)
    # 3x3 grid of side 0.3.
    # Centers: 0.15, 0.5, 0.85
    c3_x = np.array([0.15, 0.5, 0.85] * 3)
    c3_y = np.array([0.15] * 3 + [0.5] * 3 + [0.85] * 3)
    c3_s = np.full(10, 0.3)
    # Move the last one (index 9, top-right) to make room?
    # Or just shrink all to 0.29 and put 10th in center?
    # Center is occupied by index 4 (0.5, 0.5).
    # Let's just use the 2x5 grid as the primary robust baseline.
    # And a "mixed" layout.
    
    # 4. 4 Large Corners + 6 Small
    c4_x = [0.15, 0.85, 0.15, 0.85, 0.4, 0.6, 0.4, 0.6, 0.15, 0.85]
    c4_y = [0.15, 0.15, 0.85, 0.85, 0.15, 0.15, 0.85, 0.85, 0.5, 0.5]
    c4_s = [0.3, 0.3, 0.3, 0.3, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2]
    configs.append(np.hstack([c4_x, c4_y, c4_s]))

    # Refine each config
    for i, cfg in enumerate(configs):
        if time.time() > deadline:
            break
            
        # Try multiple random perturbations for each config
        for trial in range(5):
            if time.time() > deadline:
                break
                
            np.random.seed(42 + i * 10 + trial)
            perturbed = cfg + np.random.normal(0, 0.01, 3*n)
            # Clip to bounds
            perturbed[0:n] = np.clip(perturbed[0:n], 0, 1)
            perturbed[n:2*n] = np.clip(perturbed[n:2*n], 0, 1)
            perturbed[2*n:3*n] = np.clip(perturbed[2*n:3*n], 0.01, 1.0)
            
            if not is_valid(perturbed):
                continue
                
            try:
                cons = make_constraints_with_orientation(perturbed)
                res = minimize(
                    objective, 
                    perturbed, 
                    method='SLSQP',
                    bounds=bounds(),
                    constraints=cons,
                    options={'maxiter': 100, 'ftol': 1e-8}
                )
                if is_valid(res.x):
                    score = np.sum(res.x[2*n:3*n])
                    if score > best_score:
                        best_score = score
                        best_params = res.x.copy()
                        cx, cy, s = unpack(res.x)
                        best_centers = np.stack([cx, cy], axis=1)
                        best_sides = s.copy()
            except Exception:
                pass

    # Final check
    if not np.all(np.isfinite(best_centers)) or not np.all(np.isfinite(best_sides)):
        # Return a safe default
        cx = np.array([0.1, 0.3, 0.5, 0.7, 0.9] * 2)
        cy = np.array([0.1] * 5 + [0.9] * 5)
        s = np.full(10, 0.2)
        return np.stack([cx, cy], axis=1), s

    return best_centers, best_sides
