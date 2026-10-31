"""Rewritten program: multi-start L-BFGS-B with smooth overlap penalties and perimeter constraint."""
import numpy as np
from scipy.optimize import minimize
import time

N = 21

# EVOLVE-BLOCK-START
def _penalty_and_grad(x, W, H):
    """
    x: (2N,) vector of circle centers.
    W, H: rectangle dimensions with 2*(W+H)=4 => W+H=2.
    Returns (loss, grad) where loss penalizes overlaps and boundary violations.
    """
    n = N
    cx = x[0::2]
    cy = x[1::2]
    
    # Radii: we want to maximize sum of radii.
    # We use a variable radius approach: r_i = base_r * (1 + alpha * something)?
    # Simpler: fixed radii? No, we need to optimize radii too.
    # Let's parameterize: x = [cx_1, cy_1, r_1, ..., cx_n, cy_n, r_n]
    # But L-BFGS-B works better with fewer variables or simple bounds.
    # Let's stick to positions and use a uniform radius? 
    # No, non-uniform is better for packing.
    # However, the interface returns (n, 3) with [x, y, r].
    # If we optimize positions only, we can set r_i = min(dist_to_boundary, dist_to_other/2).
    # This is a max-min problem.
    # Alternative: optimize x, y, r directly with bounds and penalty.
    
    pass

def _solve_pack(seed_offset=0):
    """
    Solve the packing problem for a single random initialization.
    Returns (value, positions) where value is sum of radii.
    """
    # We optimize over centers (2N vars) and radii (N vars) -> 3N vars.
    # Bounds:
    # W + H = 2. Let W be a variable? Or fix aspect ratio?
    # The task says "variable rectangle of perimeter 4".
    # We can also optimize W. Let w be the width, then h = 2 - w.
    # Constraints: w > 0, h > 0 => 0 < w < 2.
    # Let's optimize w as well.
    # Variables: w, cx_1..cx_N, cy_1..cy_N, r_1..r_N
    # Total dims: 1 + 2N + N = 3N + 1 = 64.
    
    # Objective: Maximize sum(r_i)
    # Constraints:
    # 1. cx_i + r_i <= w
    # 2. cx_i - r_i >= 0
    # 3. cy_i + r_i <= h = 2 - w
    # 4. cy_i - r_i >= 0
    # 5. dist(i,j) >= r_i + r_j
    
    # We use a penalty method to handle inequality constraints softly.
    # Penalty term: sum of max(0, violation)^2
    
    def objective(params):
        w = params[0]
        if w <= 0.01 or w >= 1.99:
            return 1e6
        h = 2.0 - w
        
        cx = params[1::3]  # Wait, layout: w, then (cx, cy, r) for each?
        # Let's use layout: w, cx_1, cy_1, r_1, cx_2, cy_2, r_2 ...
        # So params[0] = w
        # params[1 + 3*i] = cx_i
        # params[2 + 3*i] = cy_i
        # params[3 + 3*i] = r_i
        
        cx = params[1::3]
        cy = params[2::3]
        r = params[3::3]
        
        # Boundary violations
        pen = 0.0
        for i in range(N):
            # Left
            viol = r[i] - cx[i]
            if viol > 0: pen += viol * viol
            # Right
            viol = (w - cx[i]) - r[i]
            if viol > 0: pen += viol * viol
            # Bottom
            viol = r[i] - cy[i]
            if viol > 0: pen += viol * viol
            # Top
            viol = (h - cy[i]) - r[i]
            if viol > 0: pen += viol * viol
        
        # Overlap violations
        for i in range(N):
            for j in range(i+1, N):
                dx = cx[i] - cx[j]
                dy = cy[i] - cy[j]
                dist = np.sqrt(dx*dx + dy*dy)
                min_dist = r[i] + r[j]
                viol = min_dist - dist
                if viol > 0:
                    pen += viol * viol
        
        # We want to maximize sum(r), so minimize -sum(r) + penalty
        return -np.sum(r) + 10.0 * pen
    
    # Initial guess
    rng = np.random.RandomState(seed_offset)
    # Start with a grid-like arrangement or random
    w0 = 1.0
    h0 = 1.0
    # Place circles in a grid
    side = int(np.ceil(np.sqrt(N)))
    # Spread them out
    cx0 = np.zeros(N)
    cy0 = np.zeros(N)
    r0 = np.zeros(N)
    
    # Simple random init within [0.1, 0.9] box for 1x1 rect
    for i in range(N):
        cx0[i] = 0.2 + 0.6 * rng.rand()
        cy0[i] = 0.2 + 0.6 * rng.rand()
        r0[i] = 0.05 + 0.05 * rng.rand()
        
    params0 = np.concatenate(([w0], np.column_stack((cx0, cy0, r0)).flatten()))
    
    # Bounds
    # w in [0.1, 1.9]
    # cx in [0.01, 1.99] (loose, penalty handles it)
    # cy in [0.01, 1.99]
    # r in [0.01, 0.5]
    bounds = [(0.1, 1.9)]
    for i in range(N):
        bounds.append((0.0, 2.0))  # cx
        bounds.append((0.0, 2.0))  # cy
        bounds.append((0.01, 0.5))  # r
        
    res = minimize(objective, params0, method='L-BFGS-B', bounds=bounds, 
                   options={'maxiter': 200, 'ftol': 1e-10})
    
    if res.success or res.fun < 1e-3:
        w = res.x[0]
        cx = res.x[1::3]
        cy = res.x[2::3]
        r = res.x[3::3]
        
        # Verify validity and compute exact value
        h = 2.0 - w
        valid = True
        for i in range(N):
            if cx[i] - r[i] < -1e-9 or cx[i] + r[i] > w + 1e-9:
                valid = False
            if cy[i] - r[i] < -1e-9 or cy[i] + r[i] > h + 1e-9:
                valid = False
        for i in range(N):
            for j in range(i+1, N):
                dist = np.sqrt((cx[i]-cx[j])**2 + (cy[i]-cy[j])**2)
                if dist < r[i] + r[j] - 1e-9:
                    valid = False
                    
        if valid:
            return np.sum(r), np.column_stack((cx, cy, r))
        else:
            # Return 0 or best effort? If invalid, we should try to repair or return low value.
            # For now, return a very small value to indicate failure.
            return 0.0, None
    else:
        return 0.0, None

def construct():
    start_time = time.time()
    deadline = start_time + 100.0  # Leave 20s margin
    
    best_value = 0.0
    best_pos = None
    
    # Seed program value for reference:
    # Grid 5x5 (25 cells, take 21).
    # Side=5. Cell size 0.2. Radius 0.1 - 1e-3 = 0.099.
    # Sum = 21 * 0.099 = 2.079.
    # Wait, the seed code:
    # side = 5
    # r = 0.5/5 - 1e-3 = 0.1 - 0.001 = 0.099
    # Sum = 21 * 0.099 = 2.079
    # Let's try to beat this.
    
    # Run multiple starts
    for seed in range(20):
        if time.time() > deadline:
            break
            
        val, pos = _solve_pack(seed)
        
        if val > best_value:
            best_value = val
            best_pos = pos
            
        # Check if we beat the seed significantly
        if best_value > 2.1:
            # We can break early if we find a good solution, but maybe more is better?
            # Let's continue for a bit to see if we can improve more.
            pass

    # If no valid solution found, fallback to seed-like solution
    if best_pos is None:
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        best_pos = np.array(cells[:N])
    else:
        # Ensure the returned array is exactly shape (N, 3)
        best_pos = best_pos[:N]

    return best_pos
# EVOLVE-BLOCK-END
