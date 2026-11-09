"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 21

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin for 150s limit
    
    # Initial configuration: 5x4 grid + 1 extra circle
    # Aspect ratio for 5x4 grid is 5/4 = 1.25
    # W + H = 2
    # W = 1.25 H -> 2.25 H = 2 -> H = 8/9, W = 10/9
    W_init = 10.0 / 9.0
    H_init = 8.0 / 9.0
    
    # Create 5x4 grid
    xs = []
    ys = []
    for i in range(5):
        for j in range(4):
            xs.append((i + 0.5) * W_init / 5.0)
            ys.append((j + 0.5) * H_init / 4.0)
    
    # Add 21st circle in the center of the top gap or just center of the whole box?
    # Better: Place it in the middle of the top row, between columns 2 and 3?
    # Or just place it at (W/2, H/2) and let optimization sort it out.
    # Actually, a 5x4 grid is dense. The 21st circle will cause significant overlap.
    # Let's start with a 5x5 grid, remove one corner, and optimize.
    # Or simply start with 21 circles in a 5x5 grid area, but only 21 exist.
    # Let's use a 5x5 grid positions for the first 21, effectively a 5x4 full + 1.
    # Positions:
    # (0,0) to (4,3) are filled. (0,4) is the 21st.
    # This creates a "hole" at the top right or similar.
    
    x0_list = []
    y0_list = []
    r0_list = []
    
    # 5x5 grid spacing
    # Let's assume W=1, H=1 initially for relative positions, then scale.
    # No, let's stick to the W_init, H_init estimate.
    
    # Generate 5x5 grid points
    for i in range(5):
        for j in range(5):
            if len(x0_list) < 21:
                x0_list.append((i + 0.5) * W_init / 5.0)
                y0_list.append((j + 0.5) * H_init / 5.0) # Note: 5 rows here, but H is for 4 rows in previous calc.
                # If we use 5 rows, H should be adjusted.
                # Let's restart initialization logic.
                
    # Better Initialization:
    # Use a 5x4 grid (20 circles) and put the 21st in the center.
    # This will cause overlap, but optimization will resolve it by moving things.
    
    W_init = 1.1
    H_init = 0.9 # Sum = 2.0
    
    x0 = []
    y0 = []
    # 5 cols, 4 rows
    for i in range(5):
        for j in range(4):
            x0.append((i + 0.5) * W_init / 5.0)
            y0.append((j + 0.5) * H_init / 4.0)
    
    # 21st circle: Center
    x0.append(W_init / 2.0)
    y0.append(H_init / 2.0)
    
    # Initial radii: Small enough to not overlap too badly
    # Grid spacing dx = 0.22, dy = 0.225.
    # Radius approx 0.1
    r0 = np.full(21, 0.1)
    
    # Variables: x0, y0, r0, W (H is derived)
    # Total variables: 21*2 + 21 + 1 = 64
    
    x_flat = np.array(x0)
    y_flat = np.array(y0)
    r_flat = np.array(r0)
    W_var = np.array([W_init])
    
    x_initial = np.concatenate([x_flat, y_flat, r_flat, W_var])
    
    def objective(vars):
        if time.time() > deadline:
            return 1e10 # Force stop or return bad value
        
        x = vars[:21]
        y = vars[21:42]
        r = vars[42:63]
        W = vars[63]
        H = 2.0 - W
        
        # Check validity
        if W <= 0 or H <= 0:
            return 1e10
        if np.any(r <= 0):
            return 1e10
        
        # Objective: Maximize sum of radii -> Minimize -sum(r)
        score = -np.sum(r)
        
        # Penalties
        penalty = 0.0
        pen_weight = 100.0
        
        # Boundary Penalties
        # x in [r, W-r], y in [r, H-r]
        for i in range(21):
            if x[i] < r[i]:
                penalty += (r[i] - x[i])**2
            if x[i] > W - r[i]:
                penalty += (x[i] - (W - r[i]))**2
            if y[i] < r[i]:
                penalty += (r[i] - y[i])**2
            if y[i] > H - r[i]:
                penalty += (y[i] - (H - r[i]))**2
        
        # Overlap Penalties
        for i in range(21):
            for j in range(i+1, 21):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dist = np.sqrt(dx*dx + dy*dy)
                min_dist = r[i] + r[j]
                if dist < min_dist:
                    penalty += (min_dist - dist)**2
        
        return score + pen_weight * penalty

    # Constraints for W
    # W > 0, H > 0 => W < 2
    bounds = []
    for i in range(21):
        bounds.append((1e-6, None)) # x
    for i in range(21):
        bounds.append((1e-6, None)) # y
    for i in range(21):
        bounds.append((1e-6, 1.0)) # r (max radius can't be 1 if others exist, but 1 is safe upper bound)
    bounds.append((0.1, 1.9)) # W
    
    # Run optimization
    try:
        res = minimize(objective, x_initial, method='L-BFGS-B', bounds=bounds, 
                       options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-8})
    except Exception:
        # Fallback to initial
        pass

    # Extract best result
    if hasattr(res, 'x') and res.success is not False and time.time() < deadline:
        x_best = res.x[:21]
        y_best = res.x[21:42]
        r_best = res.x[42:63]
        W_best = res.x[63]
        H_best = 2.0 - W_best
    else:
        x_best = x_flat
        y_best = y_flat
        r_best = r_flat
        W_best = W_init
        H_best = H_init

    # Final Validation and Cleanup
    # Ensure circles are strictly inside and non-overlapping
    # If validation fails, shrink radii slightly to ensure validity.
    
    valid = True
    for i in range(21):
        if x_best[i] < r_best[i] or x_best[i] > W_best - r_best[i]:
            valid = False
        if y_best[i] < r_best[i] or y_best[i] > H_best - r_best[i]:
            valid = False
    
    for i in range(21):
        for j in range(i+1, 21):
            dx = x_best[i] - x_best[j]
            dy = y_best[i] - y_best[j]
            dist = np.sqrt(dx*dx + dy*dy)
            if dist < r_best[i] + r_best[j] - 1e-9:
                valid = False
                
    if not valid:
        # Shrink all radii by 10% to ensure validity, or just return a safe grid
        # Simple fix: Scale down radii until valid
        scale = 1.0
        for _ in range(10):
            r_test = r_best * scale
            ok = True
            for i in range(21):
                if x_best[i] < r_test[i] or x_best[i] > W_best - r_test[i]:
                    ok = False; break
                if y_best[i] < r_test[i] or y_best[i] > H_best - r_test[i]:
                    ok = False; break
            if ok:
                for i in range(21):
                    for j in range(i+1, 21):
                        dx = x_best[i] - x_best[j]
                        dy = y_best[i] - y_best[j]
                        dist = np.sqrt(dx*dx + dy*dy)
                        if dist < r_test[i] + r_test[j] - 1e-9:
                            ok = False; break
                    if not ok: break
            if ok:
                r_best = r_test
                break
            scale *= 0.9
        
        # If still invalid, fallback to a very safe grid
        if not ok:
             W_safe = 1.0
             H_safe = 1.0
             side = 5
             r_safe = 0.1
             x_safe = np.array([(i + 0.5) * W_safe / side for i in range(5) for _ in range(4)])
             y_safe = np.array([(j + 0.5) * H_safe / 4.0 for _ in range(5) for j in range(4)])
             x_safe = np.append(x_safe, W_safe/2)
             y_safe = np.append(y_safe, H_safe/2)
             # This fallback might have overlaps, so let's just use the grid without the 21st if it fails?
             # No, N must be 21.
             # Let's use a smaller radius for the grid to ensure no overlap.
             # 5x4 grid in 1x1 box: spacing 0.2, 0.25. Min spacing 0.2. Radius 0.09.
             r_safe = 0.09
             x_best = x_safe
             y_best = y_safe
             r_best = np.full(21, r_safe)
             W_best = 1.0
             H_best = 1.0
             # Check perimeter: 2(1+1)=4. OK.
             # Check overlaps: dist 0.2, sum r 0.18. OK.
             # Check bounds: x in [0.09, 0.91]. Grid x: 0.1, 0.3, 0.5, 0.7, 0.9.
             # 0.1 > 0.09 OK. 0.9 < 0.91 OK.
             # y in [0.09, 0.91]. Grid y: 0.125, 0.375, 0.625, 0.875. OK.
             # 21st circle at 0.5, 0.5.
             # Dist to nearest (0.3, 0.375) is sqrt(0.2^2 + 0.125^2) = sqrt(0.04+0.0156) = sqrt(0.0556) = 0.235 > 0.18. OK.
             
    result = np.column_stack((x_best, y_best, r_best))
    return result

# EVOLVE-BLOCK-END
