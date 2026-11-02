"""Circle packing in perimeter-4 rectangle: Differential Evolution + Newton Radius Max."""
import numpy as np
import time
from scipy.optimize import differential_evolution, minimize

N = 21

def _lse_penalty(x, y, r, w, h, beta=50.0):
    """
    Smooth log-sum-exp overlap penalty (differentiable, multi-modal friendly).
    This converts the hard constraint 'distance >= r_i + r_j' into a soft cost.
    """
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    d_sq = dx * dx + dy * dy
    r_sum = r[:, None] + r[None, :]
    
    # Upper triangle only to avoid double counting and self-interference
    iu = np.triu_indices(N, k=1)
    d_sq_pairs = d_sq[iu]
    r_sum_pairs = r_sum[iu]
    
    d_pairs = np.sqrt(np.maximum(d_sq_pairs, 1e-12))
    overlap = np.maximum(r_sum_pairs - d_pairs, 0.0)
    
    # Log-sum-exp for smoothness
    if np.max(overlap) < 1e-12:
        return 0.0
    max_ov = np.max(overlap)
    lse = np.log(np.sum(np.exp(beta * (overlap - max_ov)))) + max_ov * beta
    return lse

def _de_objective(params):
    """
    Objective for Differential Evolution.
    params: [x_0...x_N-1, y_0...y_N-1, aspect_ratio]
    Returns a scalar penalty. Lower is better.
    """
    x = params[:N]
    y = params[N:2*N]
    ar = params[2*N]
    
    # Aspect ratio is log-scaled for better search distribution
    # ar_log in [-1, 1] -> ar in [e^-1, e^1] ~ [0.37, 2.72]
    aspect = np.exp(ar)
    
    # Perimeter 4 => 2(W+H)=4 => W+H=2
    W = 2.0 * aspect / (1.0 + aspect)
    H = 2.0 - W
    
    # Wall penalty: circles must be inside [0, W] x [0, H]
    # We don't know radii yet in DE, so we assume a "virtual" radius for wall checking?
    # No, better: DE optimizes positions for a *fixed* set of radii? 
    # Actually, standard approach: DE optimizes positions and aspect ratio.
    # The radii are then maximized in a post-processing step.
    # But to guide DE, we need a cost.
    # Let's use a "target" radius. If we pack 21 circles in area ~1, avg r ~ 0.15.
    # Let's assume a tentative radius r_tent = 0.15 for penalty calculation.
    # If circles overlap or go out of bounds with this tentative radius, penalty is high.
    # This drives DE to find positions where circles *could* be large.
    
    r_tent = 0.15
    
    # Wall violations
    wall_pen = 0.0
    for i in range(N):
        if x[i] < r_tent: wall_pen += (r_tent - x[i])**2
        if x[i] > W - r_tent: wall_pen += (x[i] - (W - r_tent))**2
        if y[i] < r_tent: wall_pen += (r_tent - y[i])**2
        if y[i] > H - r_tent: wall_pen += (y[i] - (H - r_tent))**2
    
    # Overlap violations
    ov_pen = _lse_penalty(x, y, np.full(N, r_tent), W, H, beta=20.0)
    
    return wall_pen + ov_pen

def _max_radius_given_others(x, y, W, H, i, r):
    """
    Given fixed positions x, y and radii of all other circles,
    calculate the max radius for circle i.
    """
    xi, yi = x[i], y[i]
    r_max = min(xi, W - xi, yi, H - yi)
    
    for j in range(N):
        if i == j:
            continue
        dx = xi - x[j]
        dy = yi - y[j]
        d = np.sqrt(dx*dx + dy*dy)
        r_allowed = d - r[j]
        if r_allowed < r_max:
            r_max = r_allowed
            
    return max(0.0, r_max)

def _newton_radius_max(x, y, W, H, max_iter=50, tol=1e-8):
    """
    Iteratively expand radii.
    Since r_i max depends on other r_j, we use a simple iterative relaxation.
    For fixed positions, maximizing sum(r_i) subject to constraints is a convex problem?
    Actually, the feasible region for radii is defined by linear inequalities:
    r_i + r_j <= d_ij
    r_i <= wall_dist_i
    Max sum(r) is a Linear Program!
    We can solve it exactly with scipy.optimize.linprog.
    """
    # Variables: r_0 ... r_{N-1}
    # Maximize sum(r) -> Minimize -sum(r)
    c = -np.ones(N)
    
    # Inequalities: A_ub @ r <= b_ub
    # 1. Wall constraints: r_i <= wall_dist_i
    # 2. Pairwise: r_i + r_j <= d_ij
    
    A_rows = []
    b_rows = []
    
    # Wall constraints
    for i in range(N):
        row = np.zeros(N)
        row[i] = 1.0
        A_rows.append(row)
        b_rows.append(min(x[i], W-x[i], y[i], H-y[i]))
        
    # Pairwise constraints
    for i in range(N):
        for j in range(i+1, N):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            d = np.sqrt(dx*dx + dy*dy)
            row = np.zeros(N)
            row[i] = 1.0
            row[j] = 1.0
            A_rows.append(row)
            b_rows.append(d)
            
    A_ub = np.array(A_rows)
    b_ub = np.array(b_rows)
    
    # Bounds: r_i >= 0
    bounds = [(0, None) for _ in range(N)]
    
    try:
        from scipy.optimize import linprog
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        if res.success:
            return -res.fun, -res.x # Return sum and radii
        else:
            # Fallback to iterative if LP fails (shouldn't happen for valid geometry)
            r = np.zeros(N)
            for _ in range(max_iter):
                improved = False
                for i in range(N):
                    r_max = _max_radius_given_others(x, y, W, H, i, r)
                    if r_max > r[i] + tol:
                        r[i] = r_max
                        improved = True
                if not improved:
                    break
            return np.sum(r), r
    except Exception:
        r = np.zeros(N)
        for _ in range(max_iter):
            improved = False
            for i in range(N):
                r_max = _max_radius_given_others(x, y, W, H, i, r)
                if r_max > r[i] + tol:
                    r[i] = r_max
                    improved = True
            if not improved:
                break
        return np.sum(r), r

def _is_valid(x, y, r, W, H, tol=1e-9):
    if np.any(x - r < -tol) or np.any(y - r < -tol):
        return False
    if np.any(x + r > W + tol) or np.any(y + r > H + tol):
        return False
    for i in range(N):
        for j in range(i+1, N):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            d_sq = dx*dx + dy*dy
            r_sum = r[i] + r[j]
            if d_sq < r_sum*r_sum - tol:
                return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 140.0 # 10s safety margin
    
    best_value = 0.0
    best_pack = None
    
    def _update_best(x, y, r, W, H):
        nonlocal best_value, best_pack
        if _is_valid(x, y, r, W, H):
            val = float(np.sum(r))
            if val > best_value:
                best_value = val
                best_pack = (x.copy(), y.copy(), r.copy(), W, H)
                
    # Initial guess: Grid
    # We will run DE to find better positions.
    
    # DE Bounds:
    # x, y in [0, 2] (since W+H=2, max dimension is 2)
    # aspect_log in [-1, 1]
    bounds = [(0, 2)] * (2*N) + [(-1, 1)]
    
    # Run Differential Evolution
    # We use a moderate population size and maxiter to fit in time budget.
    # DE can be slow, so we might need to limit it.
    # 21 circles -> 42 vars + 1 = 43 vars.
    # popsize=15 -> 645 individuals.
    # maxiter=50 -> 32250 evals.
    # Each eval is O(N^2) ~ 400 ops. Very fast.
    
    try:
        res = differential_evolution(
            _de_objective, 
            bounds, 
            seed=42, 
            maxiter=100, 
            popsize=20, 
            tol=1e-6, 
            polish=False, # We do our own polish
            update_interval=10
        )
    except Exception as e:
        print(f"DE failed: {e}")
        res = None

    if res is not None and res.fun < 1.0: # If DE found a good spot
        x_opt = res.x[:N]
        y_opt = res.x[N:2*N]
        ar_opt = res.x[2*N]
        aspect = np.exp(ar_opt)
        W = 2.0 * aspect / (1.0 + aspect)
        H = 2.0 - W
        
        # Now maximize radii using LP
        sum_r, r_opt = _newton_radius_max(x_opt, y_opt, W, H)
        _update_best(x_opt, y_opt, r_opt, W, H)
        
    # Fallback / Alternative: Try a few fixed aspect ratios with a simpler local search if DE fails or is slow
    # But let's also try a local search refinement on the DE result if we have time.
    
    if best_pack is not None:
        x0, y0, r0, W0, H0 = best_pack
        
        # Local search: Randomly perturb positions, re-solve LP for radii.
        # This is a "Simulated Annealing" like approach on positions,
        # with exact radius optimization for each position.
        
        current_val = best_value
        current_x, current_y, current_r, current_W, current_H = best_pack
        T = 0.1 # Temperature
        T_min = 0.001
        cooling = 0.95
        
        rng = np.random.default_rng(42)
        
        for i in range(2000): # Max iterations
            if time.time() > deadline - 5.0:
                break
                
            # Perturb
            new_x = current_x + rng.normal(0, 0.05, N)
            new_y = current_y + rng.normal(0, 0.05, N)
            
            # Keep within bounds [0, W], [0, H] roughly
            new_x = np.clip(new_x, 0, current_W)
            new_y = np.clip(new_y, 0, current_H)
            
            # Solve LP for radii
            new_sum, new_r = _newton_radius_max(new_x, new_y, current_W, current_H)
            
            # Check validity (LP should ensure it, but floating point...)
            if _is_valid(new_x, new_y, new_r, current_W, current_H):
                delta = new_sum - current_val
                if delta > 0 or np.random.rand() < np.exp(delta / T):
                    current_x, current_y, current_r, current_val = new_x, new_y, new_r, new_sum
                    _update_best(current_x, current_y, current_r, current_W, current_H)
            T *= cooling
            if T < T_min:
                break

    # Try other aspect ratios if we have time, using the same SA/LP loop
    if time.time() < deadline - 10.0:
        for aspect in [1.0, 1.5, 0.75, 2.0]:
            W = 2.0 * aspect / (1.0 + aspect)
            H = 2.0 - W
            
            # Start from grid
            cols, rows = 5, 5
            cw, ch = W/cols, H/rows
            x = np.zeros(N)
            y = np.zeros(N)
            for k in range(N):
                cx = k % cols
                cy = k // cols
                x[k] = (cx + 0.5) * cw
                y[k] = (cy + 0.5) * ch
            
            sum_r, r = _newton_radius_max(x, y, W, H)
            _update_best(x, y, r, W, H)
            
            # Quick SA on this aspect
            if time.time() < deadline - 5.0:
                current_val = best_value
                current_x, current_y, current_r, current_W, current_H = best_pack
                T = 0.05
                for i in range(500):
                    if time.time() > deadline - 2.0:
                        break
                    new_x = current_x + np.random.normal(0, 0.02, N)
                    new_y = current_y + np.random.normal(0, 0.02, N)
                    new_x = np.clip(new_x, 0, current_W)
                    new_y = np.clip(new_y, 0, current_H)
                    
                    new_sum, new_r = _newton_radius_max(new_x, new_y, current_W, current_H)
                    if _is_valid(new_x, new_y, new_r, current_W, current_H):
                        delta = new_sum - current_val
                        if delta > 0 or np.random.rand() < np.exp(delta / T):
                            current_x, current_y, current_r, current_val = new_x, new_y, new_r, new_sum
                            _update_best(current_x, current_y, current_r, current_W, current_H)
                    T *= 0.98
                    if T < 0.001:
                        break

    if best_pack is None:
        # Absolute fallback
        W, H = 1.0, 1.0
        cols, rows = 5, 5
        cw, ch = W/cols, H/rows
        x = np.array([(i + 0.5) * cw for i in range(cols) for j in range(rows)][:N])
        y = np.array([(j + 0.5) * ch for i in range(cols) for j in range(rows)][:N])
        r = np.full(N, min(cw, ch) * 0.49)
        return np.column_stack((x, y, r))

    x, y, r, W, H = best_pack
    # Safety shrink
    for _ in range(10):
        r_trial = r * (1 - 1e-10)
        if _is_valid(x, y, r_trial, W, H):
            r = r_trial
        else:
            break
            
    return np.column_stack((x, y, r))
