"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 26

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize, differential_evolution
import time

_DEADLINE = 145.0  # safety margin before 150s kill

def _vectorized_penalty(params, n, lam=100.0):
    """
    Vectorized penalty for constraint violations.
    Returns 0 if valid, positive value if invalid.
    """
    pts = params.reshape(n, 3)
    x, y, r = pts[:, 0], pts[:, 1], pts[:, 2]
    
    # Boundary constraints: circle must be inside [0,1]^2
    # r <= x, r <= 1-x, r <= y, r <= 1-y
    b_left = np.maximum(0, r - x)
    b_right = np.maximum(0, r - (1 - x))
    b_bottom = np.maximum(0, r - y)
    b_top = np.maximum(0, r - (1 - y))
    
    boundary_pen = np.sum(b_left**2) + np.sum(b_right**2) + np.sum(b_bottom**2) + np.sum(b_top**2)
    
    # Non-overlap: distance >= r_i + r_j
    # Compute distance matrix
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    dist2 = dx**2 + dy**2
    dist = np.sqrt(dist2)
    
    # Sum of radii matrix
    r_sum = r[:, None] + r[None, :]
    
    # Violation is where dist < r_sum
    # We only care about i < j, but summing all and dividing by 2 is equivalent for penalty magnitude
    # To be safe and precise, we zero out diagonal and take upper triangle
    overlap_violation = np.maximum(0, r_sum - dist)
    np.fill_diagonal(overlap_violation, 0)
    
    # Sum of squared violations for i < j
    # np.triu_indices
    i_idx, j_idx = np.triu_indices(n, k=1)
    overlap_pen = np.sum(overlap_violation[i_idx, j_idx]**2)
    
    # Radii must be positive
    neg_r_pen = np.sum(np.maximum(0, -r)**2)
    
    return lam * (boundary_pen + overlap_pen + neg_r_pen)

def _total_objective(params, n, lam=100.0):
    """
    Objective function: Minimize -sum(radii) + penalty.
    """
    pts = params.reshape(n, 3)
    r = pts[:, 2]
    return -np.sum(r) + _vectorized_penalty(params, n, lam)

def _is_valid(pts, tol=1e-5):
    """Check if configuration is strictly valid."""
    n = len(pts)
    x, y, r = pts[:, 0], pts[:, 1], pts[:, 2]
    if np.any(r < 0):
        return False
    # Boundary
    if np.any(x - r < -tol) or np.any(1 - x - r < -tol) or np.any(y - r < -tol) or np.any(1 - y - r < -tol):
        return False
    # Non-overlap
    # Vectorized check for speed
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    dist = np.sqrt(dx**2 + dy**2)
    r_sum = r[:, None] + r[None, :]
    
    # Check all pairs
    # dist[i,j] >= r[i] + r[j] - tol
    # We check if any dist < r_sum - tol
    # Exclude diagonal
    mask = ~np.eye(n, dtype=bool)
    if np.any(dist[mask] < r_sum[mask] - tol):
        return False
    return True

def _make_grid_initial(n):
    side = int(np.ceil(np.sqrt(n)))
    pts = []
    for i in range(side):
        for j in range(side):
            if len(pts) >= n:
                break
            x = (i + 0.5) / side
            y = (j + 0.5) / side
            r = 0.5 / side * 0.95
            pts.append([x, y, r])
    return np.array(pts[:n])

def _make_hex_initial(n):
    # Hexagonal packing pattern
    pts = []
    # Try different hex spacings
    for rows in range(4, 8):
        cols = int(np.ceil(n / rows))
        if cols < 3:
            continue
        dx = 1.0 / (cols + 0.5)
        dy = 1.0 / (rows + 0.5)
        r = min(dx, dy) * 0.48
        count = 0
        for row in range(rows):
            offset = 0.0 if row % 2 == 0 else dx / 2
            for col in range(cols):
                if count >= n:
                    break
                x = (col + 0.5 + offset) * dx
                y = (row + 0.5) * dy
                # Clamp to [0,1]
                x = np.clip(x, r, 1 - r)
                y = np.clip(y, r, 1 - r)
                pts.append([x, y, r])
                count += 1
        if len(pts) >= n:
            break
    if len(pts) < n:
        return _make_grid_initial(n)
    return np.array(pts[:n])

def _make_random_initial(n, seed=42):
    rng = np.random.RandomState(seed)
    pts = []
    for _ in range(n):
        r = rng.uniform(0.02, 0.12)
        x = rng.uniform(r, 1 - r)
        y = rng.uniform(r, 1 - r)
        pts.append([x, y, r])
    return np.array(pts)

def _post_process(pts, n, max_iters=20):
    """
    Enforce constraints by shrinking radii and adjusting positions.
    Returns a valid configuration or the best effort.
    """
    pts = pts.copy()
    for _ in range(max_iters):
        changed = False
        # Clip to boundaries
        for i in range(n):
            x, y, r = pts[i]
            r = min(r, x, 1 - x, y, 1 - y)
            if r < 0.001:
                r = 0.001
            # Adjust position if needed
            x = np.clip(x, r, 1 - r)
            y = np.clip(y, r, 1 - r)
            if abs(pts[i, 0] - x) > 1e-10 or abs(pts[i, 1] - y) > 1e-10 or abs(pts[i, 2] - r) > 1e-10:
                changed = True
            pts[i] = [x, y, r]
        
        # Resolve overlaps by shrinking
        for i in range(n):
            for j in range(i + 1, n):
                d = np.sqrt((pts[i, 0] - pts[j, 0]) ** 2 + (pts[i, 1] - pts[j, 1]) ** 2)
                min_d = pts[i, 2] + pts[j, 2]
                if d < min_d - 1e-7:
                    # Shrink both radii slightly
                    shrink = (min_d - d) / 2 + 1e-5
                    pts[i, 2] = max(0.001, pts[i, 2] - shrink)
                    pts[j, 2] = max(0.001, pts[j, 2] - shrink)
                    changed = True
        if not changed:
            break
    return pts

def _refine_local(params, n, time_limit=5.0):
    """
    Perform local refinement using L-BFGS-B on a flattened parameter vector.
    """
    t0 = time.time()
    try:
        # Use a lower lambda for refinement to allow some slack if needed, 
        # but high enough to keep constraints tight
        res = minimize(
            lambda p: _total_objective(p, n, lam=500.0),
            params,
            method='L-BFGS-B',
            options={'maxiter': 100, 'maxfun': 1000, 'ftol': 1e-12, 'gtol': 1e-8}
        )
        if time.time() - t0 < time_limit:
             return res.x
        else:
             return params
    except Exception:
        return params

def construct():
    n = N
    t_start = time.time()
    
    # Bounds for Differential Evolution
    # x, y in [0, 1], r in [0, 0.5]
    # Flatten: [x0..x25, y0..y25, r0..r25]
    bounds = [(0, 1)] * n + [(0, 1)] * n + [(0, 0.5)] * n
    
    best_pts = None
    best_score = -np.inf
    
    # Generate initial configurations to guide DE or use as fallbacks
    inits = [
        _make_grid_initial(n),
        _make_hex_initial(n),
        _make_random_initial(n, seed=42),
        _make_random_initial(n, seed=123),
    ]
    
    # Pre-process inits to be valid-ish for starting points if we use them directly
    # But DE doesn't take initial points in the same way, it samples.
    # We can use the inits to check if a simple local optimization from them is good.
    
    # Strategy:
    # 1. Try local optimization from good inits (fast, might find local optima)
    # 2. Run Differential Evolution (slower, global search)
    
    # Phase 1: Local optimization from inits
    for idx, init in enumerate(inits):
        if time.time() - t_start > _DEADLINE - 20: # Reserve 20s for DE and final checks
            break
            
        params = init.flatten().copy()
        
        # Refine
        refined_params = _refine_local(params, n, time_limit=3.0)
        opt_pts = refined_params.reshape(n, 3)
        opt_pts = _post_process(opt_pts, n)
        
        if _is_valid(opt_pts):
            score = np.sum(opt_pts[:, 2])
            if score > best_score:
                best_score = score
                best_pts = opt_pts.copy()
        
        # Check time
        if time.time() - t_start > _DEADLINE - 15:
            break

    # Phase 2: Differential Evolution
    # DE is stochastic and can be slow. We limit population size and generations.
    remaining_time = _DEADLINE - time.time() - 5
    
    if remaining_time > 10:
        try:
            # Update objective for DE
            def de_objective(params):
                return _total_objective(params, n, lam=1000.0)
            
            # DE parameters:
            # popsize: 15 (default is 15, maybe reduce to 10 for speed if needed, but 15 is okay)
            # maxiter: calculate based on time? DE doesn't have a direct time limit option.
            # We will run it with a fixed maxiter and check time in a wrapper? 
            # No, scipy DE runs to completion. 
            # We must estimate iterations.
            # One iteration = popsize * (n_params) function calls roughly? 
            # Actually, DE generates popsize new individuals per generation.
            # Let's assume ~50 function calls per generation.
            # If we have 10s, and 1 function call takes ~0.1ms (vectorized is fast), we can do 100,000 calls.
            # 100,000 / 50 = 2000 generations. That's a lot.
            # Let's cap maxiter to 100 to be safe.
            
            # However, DE might get stuck. 
            # We can use 'init' parameter to seed the initial population from our inits?
            # scipy DE 'init' can be a string or an array.
            # If we pass an array of shape (popsize, len(bounds)), we can control the start.
            
            popsize = 10
            # Create initial population for DE
            # We can mix inits and random points
            init_pop = np.zeros((popsize, len(bounds)))
            
            # Fill with some inits and randoms
            for i in range(popsize):
                if i < len(inits):
                    init_pop[i] = inits[i].flatten()
                else:
                    # Random within bounds
                    for j, (low, high) in enumerate(bounds):
                        init_pop[i, j] = np.random.uniform(low, high)
            
            # Ensure inits are valid-ish for DE start (DE handles invalids with penalty, but valid starts are better)
            # The penalty in de_objective will guide it.
            
            res = differential_evolution(
                de_objective,
                bounds,
                maxiter=50, # Keep low to ensure we finish in time
                popsize=popsize,
                init=init_pop,
                tol=1e-5,
                seed=42,
                polish=True, # Use L-BFGS-B to polish the best solution found by DE
                # Note: polish=True might add time, but it's usually fast.
                # If time is tight, we might want to disable polish and do it manually.
                # Let's disable polish in DE and do it manually to have better time control.
                polish=False 
            )
            
            if res.success or (not res.success and res.fun < best_score + 1000): # Fun is -score + penalty
                 # If penalty is 0, fun = -score. So lower fun is better.
                 # If best_score is 1.0, fun is -1.0.
                 # If res.fun is -1.1, it's better.
                 
                 de_params = res.x
                 de_pts = de_params.reshape(n, 3)
                 de_pts = _post_process(de_pts, n)
                 
                 if _is_valid(de_pts):
                     de_score = np.sum(de_pts[:, 2])
                     if de_score > best_score:
                         best_score = de_score
                         best_pts = de_pts.copy()
                         
        except Exception as e:
            # Fallback if DE fails
            pass

    # Final Fallback
    if best_pts is None or not _is_valid(best_pts):
        # Use the best local optimization result from Phase 1, or grid
        if best_pts is not None:
             # Try to fix it
             fixed = _post_process(best_pts, n)
             if _is_valid(fixed):
                 best_pts = fixed
        else:
             best_pts = _make_grid_initial(n)
             best_pts = _post_process(best_pts, n)
             if not _is_valid(best_pts):
                 # Last resort: very small circles
                 side = int(np.ceil(np.sqrt(n)))
                 pts = []
                 for i in range(side):
                     for j in range(side):
                         if len(pts) >= n: break
                         x = (i + 0.5) / side
                         y = (j + 0.5) / side
                         r = 0.5 / side * 0.5
                         pts.append([x, y, r])
                 best_pts = np.array(pts[:n])

    return best_pts
# EVOLVE-BLOCK-END
