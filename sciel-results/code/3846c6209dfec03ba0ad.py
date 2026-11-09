import numpy as np
import time
from scipy.optimize import differential_evolution, minimize

def squares_in_square10() -> tuple:
    """
    Maximizes the sum of side lengths of 10 non-overlapping axis-aligned squares
    inside the unit square.
    """
    n = 10
    start_time = time.time()
    deadline = start_time + 550  # Safety margin before 600s limit

    # Variables: [cx0, cy0, s0, cx1, cy1, s1, ..., cx9, cy9, s9]
    # Total 30 variables.

    def unpack(params):
        cx = params[0::3]
        cy = params[1::3]
        s = params[2::3]
        return cx, cy, s

    def is_valid(cx, cy, s, tol=1e-7):
        if np.any(s <= 0):
            return False
        if np.any(cx - s/2 < -tol) or np.any(cx + s/2 > 1 + tol):
            return False
        if np.any(cy - s/2 < -tol) or np.any(cy + s/2 > 1 + tol):
            return False
        for i in range(n):
            for j in range(i+1, n):
                dx = abs(cx[i] - cx[j])
                dy = abs(cy[i] - cy[j])
                req_x = (s[i] + s[j]) / 2 - tol
                req_y = (s[i] + s[j]) / 2 - tol
                if dx < req_x and dy < req_y:
                    return False
        return True

    def objective(params):
        cx, cy, s = unpack(params)
        
        # Hard bounds penalty (should be handled by DE bounds, but just in case)
        if np.any(s <= 0):
            return 1e10
        if np.any(cx - s/2 < -0.01) or np.any(cx + s/2 > 1.01):
            return 1e10
        if np.any(cy - s/2 < -0.01) or np.any(cy + s/2 > 1.01):
            return 1e10

        # Non-overlap penalty
        penalty = 0.0
        for i in range(n):
            for j in range(i+1, n):
                dx = abs(cx[i] - cx[j])
                dy = abs(cy[i] - cy[j])
                req_x = (s[i] + s[j]) / 2
                req_y = (s[i] + s[j]) / 2
                
                # Penalize overlap
                # If dx < req_x, push in x
                # If dy < req_y, push in y
                # We want to minimize overlap.
                # Using squared error of the "violation"
                px = max(0, req_x - dx)
                py = max(0, req_y - dy)
                # If both are positive, they overlap.
                # The penalty should be proportional to the overlap area or volume.
                # Simple sum of squares works well for gradient-free methods.
                penalty += px**2 + py**2

        # Objective: Minimize -sum(s) + penalty
        # High penalty weight ensures validity is prioritized over size initially
        return -np.sum(s) + penalty * 100.0

    bounds = []
    for i in range(n):
        bounds.append((0.0, 1.0))  # cx
        bounds.append((0.0, 1.0))  # cy
        bounds.append((0.05, 0.45))  # s

    best_valid_val = -np.inf
    best_valid_params = None
    best_obj_val = np.inf
    best_obj_params = None

    def callback(xk, convergence):
        nonlocal best_valid_val, best_valid_params, best_obj_val, best_obj_params
        # xk is the current best individual in the population
        val = objective(xk)
        if val < best_obj_val:
            best_obj_val = val
            best_obj_params = xk.copy()
        
        cx, cy, s = unpack(xk)
        if is_valid(cx, cy, s):
            current_sum = np.sum(s)
            if current_sum > best_valid_val:
                best_valid_val = current_sum
                best_valid_params = xk.copy()
                
        if time.time() > deadline:
            return True  # Stop DE

    # Run Differential Evolution
    # Use a seed for reproducibility
    try:
        res = differential_evolution(
            objective, 
            bounds, 
            maxiter=3000, 
            popsize=15, 
            tol=1e-6,
            seed=42,
            callback=callback,
            polish=False  # We do our own polish
        )
        # Check final result
        val = objective(res.x)
        if val < best_obj_val:
            best_obj_val = val
            best_obj_params = res.x.copy()
        cx, cy, s = unpack(res.x)
        if is_valid(cx, cy, s):
            current_sum = np.sum(s)
            if current_sum > best_valid_val:
                best_valid_val = current_sum
                best_valid_params = res.x.copy()
    except Exception:
        pass

    # Local Refinement (Nelder-Mead) if we have a good starting point and time
    if best_obj_params is not None and time.time() < deadline - 30:
        try:
            res_nm = minimize(
                objective, 
                best_obj_params, 
                method='Nelder-Mead', 
                options={'maxiter': 5000, 'xatol': 1e-8, 'fatol': 1e-8, 'adaptive': True}
            )
            val = objective(res_nm.x)
            if val < best_obj_val:
                best_obj_val = val
                best_obj_params = res_nm.x.copy()
            
            cx, cy, s = unpack(res_nm.x)
            if is_valid(cx, cy, s):
                current_sum = np.sum(s)
                if current_sum > best_valid_val:
                    best_valid_val = current_sum
                    best_valid_params = res_nm.x.copy()
        except Exception:
            pass

    # If no valid solution found, use a fallback
    if best_valid_params is None:
        # Fallback: 10 squares of side 0.25 in a 4x3 grid
        side = 0.25
        cx = np.array([0.125, 0.375, 0.625, 0.875, 0.125, 0.375, 0.625, 0.875, 0.125, 0.375])
        cy = np.array([0.125, 0.125, 0.125, 0.125, 0.375, 0.375, 0.375, 0.375, 0.625, 0.625])
        s = np.full(10, side)
        return np.stack([cx, cy], axis=1), s

    cx = best_valid_params[0::3]
    cy = best_valid_params[1::3]
    s = best_valid_params[2::3]
    
    # Final clamping to ensure strict validity against numerical noise
    s = np.maximum(s, 1e-6)
    for i in range(n):
        half_s = s[i] / 2
        if cx[i] - half_s < 0: cx[i] = half_s
        if cx[i] + half_s > 1: cx[i] = 1 - half_s
        if cy[i] - half_s < 0: cy[i] = half_s
        if cy[i] + half_s > 1: cy[i] = 1 - half_s
        
    centers = np.stack([cx, cy], axis=1)
    return centers, s

# EVOLVE-BLOCK-END
