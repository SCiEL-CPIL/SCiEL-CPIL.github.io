# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize

_N = 10
_TOL = 1e-7
_DEADLINE = 590.0

def _evaluate(centers, sides):
    """
    Returns (objective, violation).
    objective: sum of sides.
    violation: max(0, max_boundary_violation, max_overlap_violation)
    """
    cx = centers[:, 0]
    cy = centers[:, 1]
    obj = float(np.sum(sides))
    
    # Boundary check
    lo_x = cx - sides * 0.5
    hi_x = cx + sides * 0.5
    lo_y = cy - sides * 0.5
    hi_y = cy + sides * 0.5
    
    # Violations: how much outside [0, 1]
    bv_x = np.maximum(0.0, np.maximum(-lo_x, hi_x - 1.0))
    bv_y = np.maximum(0.0, np.maximum(-lo_y, hi_y - 1.0))
    max_bv = float(np.max(np.maximum(bv_x, bv_y)))
    
    # Overlap check
    # Pairwise
    dx = cx[:, None] - cx[None, :]
    dy = cy[:, None] - cy[None, :]
    t = 0.5 * (sides[:, None] + sides[None, :]) - _TOL
    
    # Overlap if |dx| < t and |dy| < t
    # Violation is max(0, t - |dx|) * max(0, t - |dy|) ? 
    # Actually, we just need to know if they overlap.
    # For a smooth penalty, we can use:
    # penalty = sum( max(0, t - |dx|) * max(0, t - |dy|) )
    # Or simpler: if |dx| < t and |dy| < t, violation = 1.
    # For local search, a continuous penalty is better.
    
    # Distance in x
    dx_abs = np.abs(dx)
    dy_abs = np.abs(dy)
    
    # Overlap in x: t - dx_abs if positive
    ov_x = np.maximum(0.0, t - dx_abs)
    ov_y = np.maximum(0.0, t - dy_abs)
    
    # Product is a measure of overlap area (approx)
    ov_area = ov_x * ov_y
    # Only consider i < j
    ov_area = np.triu(ov_area, 1)
    max_ov = float(np.max(ov_area))
    
    return obj, max(max_bv, max_ov)

def _initial_configurations():
    """
    Generate a list of diverse initial configurations.
    """
    configs = []
    
    # 1. Grid 2x5
    centers = np.array([
        [0.1, 0.1], [0.3, 0.1], [0.5, 0.1], [0.7, 0.1], [0.9, 0.1],
        [0.1, 0.9], [0.3, 0.9], [0.5, 0.9], [0.7, 0.9], [0.9, 0.9]
    ])
    sides = np.ones(10) * 0.18
    configs.append((centers, sides))
    
    # 2. 4 corners + 6 center
    centers = np.array([
        [0.25, 0.25], [0.75, 0.25], [0.25, 0.75], [0.75, 0.75],
        [0.5, 0.4], [0.5, 0.6], [0.4, 0.5], [0.6, 0.5], [0.5, 0.5], [0.5, 0.5]
    ])
    # This is messy. Let's do 4 large corners, 2 medium, 4 small.
    # Corners 0.3
    c1 = np.array([
        [0.15, 0.15], [0.85, 0.15], [0.15, 0.85], [0.85, 0.85]
    ])
    s1 = np.array([0.3, 0.3, 0.3, 0.3])
    
    # Center 0.4x0.4 space.
    # Place 2 squares of 0.2 at (0.4, 0.4) and (0.6, 0.6)? No.
    # Place 4 squares of 0.2 in 2x2 grid in center.
    c2 = np.array([
        [0.4, 0.4], [0.6, 0.4], [0.4, 0.6], [0.6, 0.6]
    ])
    s2 = np.array([0.2, 0.2, 0.2, 0.2])
    
    # Remaining 2 squares.
    # Space is tight. Maybe shrink corners to 0.25.
    # Then center is 0.5x0.5.
    # Can fit 4 squares of 0.25?
    # Let's try 4 squares of 0.25 at corners.
    c3 = np.array([
        [0.125, 0.125], [0.875, 0.125], [0.125, 0.875], [0.875, 0.875]
    ])
    s3 = np.array([0.25]*4)
    
    # Center 0.5x0.5.
    # Fit 6 squares of 0.15?
    # Or 2 squares of 0.25 and 4 squares of 0.1?
    # Let's just add random small ones in the center.
    c4 = np.array([
        [0.35, 0.35], [0.65, 0.35], [0.35, 0.65], [0.65, 0.65],
        [0.5, 0.5], [0.5, 0.5]
    ])
    s4 = np.array([0.2]*6)
    
    centers = np.vstack([c3, c4])
    sides = np.concatenate([s3, s4])
    configs.append((centers, sides))
    
    # 3. Random feasible
    rng = np.random.default_rng(42)
    centers = rng.uniform(0.1, 0.9, size=(10, 2))
    sides = rng.uniform(0.1, 0.2, size=10)
    configs.append((centers, sides))
    
    return configs

def _local_search(centers, sides, start_time, deadline, rng, max_iter=5000):
    """
    Local search with penalty method.
    """
    obj, viol = _evaluate(centers, sides)
    best_obj = obj
    best_centers = centers.copy()
    best_sides = sides.copy()
    
    penalty = 10.0
    step = 0.05
    
    for i in range(max_iter):
        if time.time() > deadline:
            break
        
        # Perturb
        idx = rng.integers(0, 20)
        new_centers = centers.copy()
        new_sides = sides.copy()
        
        if idx < 10:
            # Move center
            new_centers[idx, 0] += rng.normal(0, step)
            new_centers[idx, 1] += rng.normal(0, step)
        else:
            # Change size
            new_sides[idx - 10] += rng.normal(0, step * 0.5)
            new_sides[idx - 10] = max(new_sides[idx - 10], 1e-4)
            
        nobj, nviol = _evaluate(new_centers, new_sides)
        score = nobj - penalty * nviol
        cur_score = obj - penalty * viol
        
        if score > cur_score:
            centers = new_centers
            sides = new_sides
            obj = nobj
            viol = nviol
            step *= 1.02
            if viol < 1e-9 and obj > best_obj:
                best_obj = obj
                best_centers = centers.copy()
                best_sides = sides.copy()
        else:
            step *= 0.98
            if step < 1e-6:
                step = 1e-6
        
        # Increase penalty if we are infeasible
        if viol > 1e-9:
            penalty *= 1.01
        else:
            penalty = max(10.0, penalty * 0.99)
            
    return best_centers, best_sides, best_obj

def squares_in_square10() -> tuple:
    start_time = time.time()
    deadline = start_time + _DEADLINE
    
    configs = _initial_configurations()
    
    best_centers = None
    best_sides = None
    best_obj = -1.0
    
    for idx, (c, s) in enumerate(configs):
        if time.time() > deadline - 10:
            break
        rng = np.random.default_rng(100 + idx)
        c, s, obj = _local_search(c, s, time.time(), deadline, rng)
        if obj > best_obj:
            best_obj = obj
            best_centers = c
            best_sides = s
            
    # Final polish with SLSQP if time permits
    if time.time() < deadline - 20 and best_centers is not None:
        try:
            def objective(x):
                c = x[:20].reshape(10, 2)
                s = x[20:]
                o, v = _evaluate(c, s)
                return -(o - 1000*v)
                
            def constraint(x):
                c = x[:20].reshape(10, 2)
                s = x[20:]
                # Boundary constraints
                cons = []
                for i in range(10):
                    cons.append(c[i, 0] - s[i]/2)
                    cons.append(1.0 - c[i, 0] - s[i]/2)
                    cons.append(c[i, 1] - s[i]/2)
                    cons.append(1.0 - c[i, 1] - s[i]/2)
                # Overlap constraints (linearized? No, non-linear)
                # SLSQP handles non-linear.
                for i in range(10):
                    for j in range(i+1, 10):
                        # |cx_i - cx_j| >= s_i/2 + s_j/2 - tol
                        # This is non-linear.
                        # We can use the penalty in the objective instead.
                        pass
                return cons
                
            x0 = np.concatenate([best_centers.flatten(), best_sides])
            bounds = [(1e-4, None) for _ in range(10)] # Sides > 0
            # Centers bounds are handled by constraints
            res = minimize(objective, x0, method='SLSQP', bounds=bounds, 
                           options={'maxiter': 50, 'ftol': 1e-8})
            
            if res.success:
                c_new = res.x[:20].reshape(10, 2)
                s_new = res.x[20:]
                o_new, v_new = _evaluate(c_new, s_new)
                if v_new < 1e-7 and o_new > best_obj:
                    best_centers = c_new
                    best_sides = s_new
                    best_obj = o_new
        except Exception:
            pass
            
    # Final validation
    if best_centers is None:
        # Fallback
        best_centers = np.zeros((10, 2))
        best_sides = np.ones(10) * 0.1
        
    # Clamp to ensure strict feasibility
    sides = np.clip(best_sides, 1e-5, None)
    cx = np.clip(best_centers[:, 0], sides * 0.5, 1.0 - sides * 0.5)
    cy = np.clip(best_centers[:, 1], sides * 0.5, 1.0 - sides * 0.5)
    centers = np.stack([cx, cy], axis=1)
    
    obj, viol = _evaluate(centers, sides)
    if viol > 1e-7:
        # Fallback to a known good config
        centers = np.array([
            (0.125, 0.125), (0.875, 0.125), (0.125, 0.875), (0.875, 0.875),
            (0.35, 0.35), (0.65, 0.35), (0.35, 0.65), (0.65, 0.65),
            (0.5, 0.5), (0.5, 0.5)
        ])
        sides = np.array([0.25]*4 + [0.2]*4 + [0.1]*2)
        
    return centers, sides

# EVOLVE-BLOCK-END
