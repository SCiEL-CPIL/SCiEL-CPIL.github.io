"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from scipy.optimize import minimize

N = 21

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.special import logsumexp

def _rect_constraints(x, n):
    """
    x: array of shape (3*n,) = [x1, y1, r1, x2, y2, r2, ...]
    Rectangle has perimeter 4. We assume width W, height H with 2(W+H)=4 => W+H=2.
    The rectangle is defined by the bounding box of the circles.
    Actually, the problem says "variable rectangle of perimeter 4".
    This usually means we can choose the aspect ratio.
    Let the rectangle be [0, W] x [0, H] with W+H=2.
    The circles must be inside.
    Constraints:
    1. xi - ri >= 0
    2. xi + ri <= W
    3. yi - ri >= 0
    4. yi + ri <= H
    5. (xi - xj)^2 + (yi - yj)^2 >= (ri + rj)^2
    
    To use scipy.optimize.minimize with SLSQP, we need equality and inequality constraints.
    Equality: W + H - 2 = 0.
    But W and H are not in x.
    We can define W and H from the circles? No, that's circular.
    
    Alternative: Fix W and H? No, "variable rectangle".
    We can add W and H as variables.
    Let z = [x1, y1, r1, ..., xn, yn, rn, W, H]
    Size 3*n + 2.
    
    Equality: W + H - 2 = 0
    Inequalities:
    - xi - ri >= 0  => xi - ri
    - W - (xi + ri) >= 0 => W - xi - ri
    - yi - ri >= 0  => yi - ri
    - H - (yi + ri) >= 0 => H - yi - ri
    - (xi - xj)^2 + (yi - yj)^2 - (ri + rj)^2 >= 0
    
    Objective: Maximize sum(ri) => Minimize -sum(ri)
    """
    n = N
    # x contains [cx, cy, r] * n, W, H
    # indices: 0..3n-1 are circles, 3n is W, 3n+1 is H
    W = x[3*n]
    H = x[3*n+1]
    
    # Equality constraint: W + H - 2 = 0
    eq = W + H - 2.0
    
    # Inequality constraints
    # We need a list of constraint functions or a vectorized one.
    # Vectorized is faster.
    # Let's build a vector of all inequality violations (should be >= 0)
    
    # Boundary constraints
    # xi - ri >= 0
    # W - xi - ri >= 0
    # yi - ri >= 0
    # H - yi - ri >= 0
    
    cx = x[0:3*n:3]
    cy = x[1:3*n:3]
    r  = x[2:3*n:3]
    
    ineq = np.concatenate([
        cx - r,
        W - cx - r,
        cy - r,
        H - cy - r
    ])
    
    # Pairwise constraints
    # (xi - xj)^2 + (yi - yj)^2 - (ri + rj)^2 >= 0
    # This is O(n^2). For n=21, 210 pairs.
    # We can compute this efficiently or just loop.
    # For SLSQP, providing a function that returns a vector is fine.
    
    pairwise_ineq = []
    for i in range(n):
        for j in range(i+1, n):
            dx = cx[i] - cx[j]
            dy = cy[i] - cy[j]
            dr = r[i] + r[j]
            pairwise_ineq.append(dx*dx + dy*dy - dr*dr)
    
    ineq = np.concatenate([ineq, pairwise_ineq])
    
    return eq, ineq

def _objective(x):
    n = N
    r = x[2:3*n:3]
    return -np.sum(r)

def _construct_initial_guess(W, H):
    """
    Create an initial guess.
    Simple grid or random.
    """
    n = N
    # Try a grid layout
    # Aspect ratio W/H
    # If W >= H, put more columns.
    # cols = round(sqrt(n * W/H))
    # rows = ceil(n / cols)
    
    # Let's just use a random jittered grid inside a slightly smaller box to ensure feasibility
    # Or use the seed's grid logic but scaled to W, H.
    
    # Let's try to fit n circles in W x H.
    # Approximate radius r = W/(cols) / 2 ?
    
    # A safer initial guess:
    # Place circles in a grid, with small radius.
    # Then let optimizer expand them.
    
    # Grid dimensions
    ratio = W / H
    cols = max(1, int(np.sqrt(n * ratio)))
    rows = max(1, int(np.ceil(n / cols)))
    
    # Adjust W, H to fit the grid? No, W, H are variables but fixed in initial guess structure.
    # Actually, we initialize W, H as part of x.
    
    # Let's pick a specific W, H for initialization.
    # e.g., W=1, H=1 (square) or W=1.5, H=0.5.
    # The optimizer will adjust W, H.
    
    # Initial x
    x0 = np.zeros(3*n + 2)
    
    # Set W, H
    x0[3*n] = W
    x0[3*n+1] = H
    
    # Place circles
    # Margin to stay inside
    margin = 0.1
    w_eff = W - 2*margin
    h_eff = H - 2*margin
    
    # If w_eff or h_eff <= 0, this init is bad.
    if w_eff <= 0 or h_eff <= 0:
        return None
    
    # Grid positions
    # Distribute n points in cols x rows grid
    # We might have empty cells.
    # We just take the first n positions.
    
    dx = w_eff / cols if cols > 0 else w_eff
    dy = h_eff / rows if rows > 0 else h_eff
    
    # Center of each cell
    # Add margin to start
    start_x = margin
    start_y = margin
    
    # Calculate radius
    # Max radius that fits in cell is min(dx, dy)/2
    # But we need disjoint interiors, so we start smaller.
    r0 = min(dx, dy) * 0.4
    
    # Ensure r0 is positive and small enough
    r0 = max(r0, 0.01)
    r0 = min(r0, min(W, H)/2 - 0.01)
    
    for i in range(n):
        row = i // cols
        col = i % cols
        
        cx = start_x + (col + 0.5) * dx
        cy = start_y + (row + 0.5) * dy
        r = r0
        
        # Jitter to break symmetry
        cx += np.random.uniform(-0.01, 0.01)
        cy += np.random.uniform(-0.01, 0.01)
        
        # Clamp to box
        cx = np.clip(cx, r, W - r)
        cy = np.clip(cy, r, H - r)
        
        x0[3*i] = cx
        x0[3*i+1] = cy
        x0[3*i+2] = r
        
    return x0

def _refine(x0, time_limit=10):
    """
    Run SLSQP optimization.
    """
    start_time = time.time()
    
    # Check initial feasibility roughly
    # If not feasible, SLSQP might fail or take long.
    
    res = minimize(
        _objective,
        x0,
        method='SLSQP',
        bounds=[(0, None)] * (3*N + 2), # Radii and coords positive. W, H positive.
        # Actually, coords can be 0. Radii > 0.
        # Let's be safe:
        # x, y >= 0
        # r > 0
        # W, H > 0
        # We'll handle bounds properly.
        options={'maxiter': 200, 'ftol': 1e-9}
    )
    
    return res

def construct():
    start_time = time.time()
    deadline = 140.0  # Leave 10s safety margin
    
    best_x = None
    best_val = -np.inf
    
    # Try multiple initial aspect ratios
    # W + H = 2
    # W in [0.5, 1.5] (since H = 2-W, H in [0.5, 1.5])
    
    aspect_ratios = [
        1.0,   # Square
        1.5,   # Wide
        0.5,   # Tall (symmetric to wide)
        2.0,   # Very wide (H=0) -> invalid, H must be >0.
        # Let's use W values: 1.0, 1.2, 0.8, 1.4, 0.6
    ]
    
    # Better: W in [0.2, 1.8]
    # But H = 2-W. If W=1.8, H=0.2.
    # Circles need radius < H/2 = 0.1.
    
    W_starts = [1.0, 1.2, 0.8, 1.4, 0.6, 1.1, 0.9]
    
    for W in W_starts:
        if time.time() > deadline - 20: # Need time for at least one more attempt?
            break
            
        H = 2.0 - W
        
        # Check if W, H are reasonable
        if W < 0.1 or H < 0.1:
            continue
            
        x0 = _construct_initial_guess(W, H)
        if x0 is None:
            continue
        
        # Run optimization
        try:
            res = minimize(
                _objective,
                x0,
                method='SLSQP',
                bounds=[(0, None)] * (3*N + 2),
                options={'maxiter': 300, 'ftol': 1e-10}
            )
            
            if res.success or res.fun < best_val:
                # Check constraints manually to be sure
                x = res.x
                W_opt = x[3*N]
                H_opt = x[3*N+1]
                
                # Check perimeter
                if abs(2*(W_opt + H_opt) - 4.0) > 1e-4:
                    continue
                    
                # Check bounds
                cx = x[0:3*N:3]
                cy = x[1:3*N:3]
                r = x[2:3*N:3]
                
                valid = True
                for i in range(N):
                    if cx[i] - r[i] < -1e-5 or cx[i] + r[i] - W_opt > 1e-5:
                        valid = False
                        break
                    if cy[i] - r[i] < -1e-5 or cy[i] + r[i] - H_opt > 1e-5:
                        valid = False
                        break
                        
                if valid:
                    # Check pairwise
                    for i in range(N):
                        for j in range(i+1, N):
                            dx = cx[i] - cx[j]
                            dy = cy[i] - cy[j]
                            dr = r[i] + r[j]
                            if dx*dx + dy*dy < dr*dr - 1e-5:
                                valid = False
                                break
                        if not valid:
                            break
                    
                    if valid:
                        val = np.sum(r)
                        if val > best_val:
                            best_val = val
                            best_x = x.copy()
                            
        except Exception as e:
            pass
        
        if time.time() > deadline - 10:
            break
        
    if best_x is None:
        # Fallback to seed-like solution
        # Square 1x1
        W = 1.0
        H = 1.0
        side = int(np.ceil(np.sqrt(N)))
        # 21 circles in 5x5 grid (25 cells)
        # Use 5x4 grid? 20 cells. Need 21.
        # 5x5 grid, use first 21.
        cells = []
        for i in range(5):
            for j in range(5):
                if len(cells) >= N:
                    break
                cx = (i + 0.5) / 5.0
                cy = (j + 0.5) / 5.0
                r = 0.5 / 5.0 - 1e-3
                cells.append([cx, cy, r])
        return np.array(cells[:N])
        
    # Extract result
    # The result should be an array of shape (N, 3)
    # x = [cx1, cy1, r1, ..., cxN, cyN, rN, W, H]
    result = best_x[0:3*N].reshape(N, 3)
    
    # Final polish?
    # The problem asks for raw value.
    # We return the array.
    
    return result
# EVOLVE-BLOCK-END
