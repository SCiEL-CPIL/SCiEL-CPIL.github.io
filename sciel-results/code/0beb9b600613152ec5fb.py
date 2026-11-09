# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _check_feasibility(centers, sides, tol=1e-7):
    """Check if the configuration is feasible (all inside unit square, no overlaps)."""
    n = len(sides)
    for i in range(n):
        cx, cy = centers[i]
        s = sides[i]
        if s <= 0:
            return False
        if not (np.isfinite(cx) and np.isfinite(cy) and np.isfinite(s)):
            return False
        if cx - s / 2 < -tol or cx + s / 2 > 1 + tol:
            return False
        if cy - s / 2 < -tol or cy + s / 2 > 1 + tol:
            return False
    for i in range(n):
        for j in range(i + 1, n):
            t = (sides[i] + sides[j]) / 2 - tol
            if abs(centers[i, 0] - centers[j, 0]) < t and abs(centers[i, 1] - centers[j, 1]) < t:
                return False
    return True

def _sum_sides(sides):
    return float(np.sum(sides))

def _smooth_penalty(centers, sides, tol=1e-7, beta=50.0):
    """
    Smooth penalty using soft-min for constraint violations.
    Uses log-sum-exp style smoothing for the max of violations.
    """
    n = len(sides)
    penalty = 0.0
    
    # Boundary violations: smooth max(0, violation)
    for i in range(n):
        cx, cy = centers[i]
        s = sides[i]
        # x-boundary violations
        vx_left = -tol - (cx - s / 2)   # >0 means violation
        vx_right = (cx + s / 2) - (1 + tol)  # >0 means violation
        vy_bottom = -tol - (cy - s / 2)
        vy_top = (cy + s / 2) - (1 + tol)
        
        # Smooth max(0, v) using log-sum-exp
        for v in [vx_left, vx_right, vy_bottom, vy_top]:
            if v > 0:
                penalty += v / beta * np.log(1 + np.exp(beta * v))
    
    # Overlap violations: smooth
    for i in range(n):
        for j in range(i + 1, n):
            t = (sides[i] + sides[j]) / 2 - tol
            dx = abs(centers[i, 0] - centers[j, 0])
            dy = abs(centers[i, 1] - centers[j, 1])
            
            # Violation occurs when dx < t AND dy < t
            # Define overlap depth in each direction
            ox = t - dx  # >0 means x-overlap
            oy = t - dy  # >0 means y-overlap
            
            # Smooth product of max(0,ox) * max(0,oy)
            # Use soft-max for each
            if ox > 0 and oy > 0:
                # Approximate smooth product
                soft_ox = ox / beta * np.log(1 + np.exp(beta * ox))
                soft_oy = oy / beta * np.log(1 + np.exp(beta * oy))
                penalty += soft_ox * soft_oy
            elif ox > 0:
                penalty += 0.1 * ox / beta * np.log(1 + np.exp(beta * ox))
            elif oy > 0:
                penalty += 0.1 * oy / beta * np.log(1 + np.exp(beta * oy))
    
    return penalty

def _objective(centers, sides):
    """Objective function: maximize sum of sides, penalize infeasibility."""
    return _sum_sides(sides) - 1000.0 * _smooth_penalty(centers, sides)

def _flatten(centers, sides):
    """Flatten centers and sides into a single vector."""
    return np.concatenate([centers.flatten(), sides])

def _unflatten(x):
    """Unflatten vector back to centers and sides."""
    n = 10
    centers = x[:2*n].reshape(n, 2)
    sides = x[2*n:]
    return centers, sides

def _objective_flat(x):
    """Objective for flat parameter vector."""
    centers, sides = _unflatten(x)
    return -_objective(centers, sides)  # Negate for minimization

def _make_2big_8small():
    """
    The classic configuration for 10 squares:
    2 squares of side 0.5, 8 squares of side 0.25.
    Sum = 2*0.5 + 8*0.25 = 3.0
    """
    centers = np.zeros((10, 2), dtype=np.float64)
    sides = np.zeros(10, dtype=np.float64)

    centers[0] = [0.25, 0.25]
    sides[0] = 0.5
    centers[1] = [0.75, 0.75]
    sides[1] = 0.5

    centers[2] = [0.125, 0.625]
    sides[2] = 0.25
    centers[3] = [0.375, 0.625]
    sides[3] = 0.25
    centers[4] = [0.125, 0.875]
    sides[4] = 0.25
    centers[5] = [0.375, 0.875]
    sides[5] = 0.25

    centers[6] = [0.625, 0.125]
    sides[6] = 0.25
    centers[7] = [0.875, 0.125]
    sides[7] = 0.25
    centers[8] = [0.625, 0.375]
    sides[8] = 0.25
    centers[9] = [0.875, 0.375]
    sides[9] = 0.25

    return centers, sides

def _make_3big_7small():
    """
    Try a configuration with 3 larger squares.
    Layout: 
    - Square of side ~0.42 at bottom-left
    - Square of side ~0.42 at top-right  
    - Square of side ~0.42 at top-left
    - 7 smaller squares filling remaining space
    """
    centers = np.zeros((10, 2), dtype=np.float64)
    sides = np.zeros(10, dtype=np.float64)
    
    # 3 large squares of side 0.4
    s_large = 0.4
    centers[0] = [s_large/2, s_large/2]
    sides[0] = s_large
    centers[1] = [1 - s_large/2, 1 - s_large/2]
    sides[1] = s_large
    centers[2] = [s_large/2, 1 - s_large/2]
    sides[2] = s_large
    
    # 7 smaller squares to fill gaps
    # Right column below top-right square
    s_small = 0.3
    centers[3] = [1 - s_small/2, s_large + s_small/2]
    sides[3] = s_small
    centers[4] = [1 - s_small/2, s_large + 1.5*s_small]
    sides[4] = s_small
    
    # Bottom row right of bottom-left square
    centers[5] = [s_large + s_small/2, s_small/2]
    sides[5] = s_small
    centers[6] = [s_large + 1.5*s_small, s_small/2]
    sides[6] = s_small
    
    # Top row between large squares
    centers[7] = [s_large + 0.15, 1 - 0.15]
    sides[7] = 0.3
    
    # Fill remaining
    centers[8] = [0.5, 0.5]
    sides[8] = 0.25
    centers[9] = [0.5, 0.2]
    sides[9] = 0.2
    
    return centers, sides

def _make_random_start(seed=42):
    """Generate a random feasible-ish start."""
    rng = np.random.default_rng(seed)
    centers = rng.uniform(0.1, 0.9, size=(10, 2))
    sides = rng.uniform(0.15, 0.35, size=10)
    return centers, sides

def _simulated_annealing(centers, sides, time_limit=15.0, seed=42):
    """
    Simulated Annealing to maximize sum of side lengths.
    Uses adaptive step sizes.
    """
    n = len(sides)
    best_centers = centers.copy()
    best_sides = sides.copy()
    best_obj = _objective(best_centers, best_sides)
    current_centers = centers.copy()
    current_sides = sides.copy()
    current_obj = _objective(current_centers, current_sides)
    
    start_time = time.time()
    rng = np.random.default_rng(seed)
    
    T = 0.2
    T_min = 1e-8
    alpha = 0.999
    
    # Adaptive step sizes
    step_center = 0.05
    step_side = 0.03
    
    while T > T_min:
        if time.time() - start_time > time_limit:
            break
            
        # Perturb with adaptive steps
        dc = rng.normal(0, step_center, size=(n, 2))
        ds = rng.normal(0, step_side, size=n)
        
        new_centers = current_centers + dc
        new_sides = current_sides + ds
        
        # Ensure sides are positive and reasonable
        new_sides = np.clip(new_sides, 1e-6, 1.0)
        
        new_obj = _objective(new_centers, new_sides)
        
        delta_obj = new_obj - current_obj
        
        accept = False
        if delta_obj > 0:
            accept = True
        else:
            # Accept with probability exp(delta/T)
            if T > 1e-10:
                prob = np.exp(delta_obj / T)
                if rng.random() < prob:
                    accept = True
        
        if accept:
            current_centers = new_centers
            current_sides = new_sides
            current_obj = new_obj
            
            if current_obj > best_obj:
                best_centers = current_centers.copy()
                best_sides = current_sides.copy()
                best_obj = current_obj
                
            # Occasionally increase steps if improving
            if rng.random() < 0.01:
                step_center *= 1.1
                step_side *= 1.1
        else:
            # Decrease steps if rejecting
            if rng.random() < 0.05:
                step_center *= 0.95
                step_side *= 0.95
        
        T *= alpha
        
    return best_centers, best_sides

def _polish_with_scipy(centers, sides, time_limit=30.0):
    """
    Final polish using scipy.optimize with the smooth objective.
    """
    x0 = _flatten(centers, sides)
    
    # Define bounds
    n = 10
    bounds = []
    for i in range(n):
        # Center x: must keep square inside [0,1]
        bounds.append((0.01, 0.99))
        bounds.append((0.01, 0.99))
    for i in range(n):
        bounds.append((0.05, 0.95))
    
    start_time = time.time()
    
    try:
        result = minimize(
            _objective_flat,
            x0,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10, 'maxfun': 10000}
        )
        
        if result.success or result.fun < _objective_flat(x0):
            new_centers, new_sides = _unflatten(result.x)
            if _check_feasibility(new_centers, new_sides):
                return new_centers, new_sides
    except Exception:
        pass
    
    # Try Nelder-Mead as fallback
    try:
        result = minimize(
            _objective_flat,
            x0,
            method='Nelder-Mead',
            options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-10}
        )
        
        if result.fun < _objective_flat(x0):
            new_centers, new_sides = _unflatten(result.x)
            if _check_feasibility(new_centers, new_sides):
                return new_centers, new_sides
    except Exception:
        pass
    
    return centers, sides

def squares_in_square10() -> tuple:
    """
    Erdos "squares in a square" problem. Place n=10 axis-aligned squares with
    pairwise disjoint interiors inside the unit square [0,1]^2 so as to
    MAXIMIZE the sum of their side lengths.

    Returns:
        centers: np.ndarray of shape (10, 2) with the (x, y) center of each square.
        sides:   np.ndarray of shape (10,) with the side length of each square.
    The i-th axis-aligned square occupies
        [cx - s/2, cx + s/2] x [cy - s/2, cy + s/2].

    Strategy:
      1. Build the well-known 2-big/8-small configuration (sum = 3.0).
      2. Try 3-big/7-small configuration.
      3. Run Simulated Annealing from multiple starts.
      4. Polish best candidates with scipy optimization.
      5. Return the best feasible configuration found.
    """
    start_time = time.time()
    deadline = start_time + 570  # Safety margin before 600s limit

    n = 10

    candidates = []
    
    # --- Candidate 1: 2-big / 8-small (sum = 3.0) ---
    c1, s1 = _make_2big_8small()
    if _check_feasibility(c1, s1):
        c1_sa, s1_sa = _simulated_annealing(c1, s1, time_limit=10.0, seed=42)
        if _check_feasibility(c1_sa, s1_sa):
            candidates.append((c1_sa, s1_sa, _sum_sides(s1_sa)))
        # Also keep original if SA didn't improve
        candidates.append((c1, s1, _sum_sides(s1)))

    # --- Candidate 2: 3-big / 7-small ---
    c2, s2 = _make_3big_7small()
    if _check_feasibility(c2, s2):
        c2_sa, s2_sa = _simulated_annealing(c2, s2, time_limit=10.0, seed=123)
        if _check_feasibility(c2_sa, s2_sa):
            candidates.append((c2_sa, s2_sa, _sum_sides(s2_sa)))

    # --- Candidate 3-6: Random starts with SA ---
    for i in range(4):
        if time.time() > deadline - 60:
            break
        c_r, s_r = _make_random_start(seed=100 + i)
        c_r_sa, s_r_sa = _simulated_annealing(c_r, s_r, time_limit=8.0, seed=200 + i)
        if _check_feasibility(c_r_sa, s_r_sa):
            candidates.append((c_r_sa, s_r_sa, _sum_sides(s_r_sa)))

    # Pick the best feasible candidate before polishing
    if not candidates:
        c1, s1 = _make_2big_8small()
        return c1, s1

    best_c, best_s, best_val = max(candidates, key=lambda x: x[2])
    
    # --- Polish the best candidates ---
    if time.time() < deadline - 40:
        # Sort candidates by value and polish top 3
        candidates.sort(key=lambda x: x[2], reverse=True)
        
        for c, s, val in candidates[:3]:
            if time.time() > deadline - 20:
                break
            c_pol, s_pol = _polish_with_scipy(c, s, time_limit=15.0)
            if _check_feasibility(c_pol, s_pol):
                new_val = _sum_sides(s_pol)
                if new_val > best_val:
                    best_c, best_s, best_val = c_pol, s_pol, new_val

    # Final feasibility check
    if not _check_feasibility(best_c, best_s):
        # Fallback to known good
        c1, s1 = _make_2big_8small()
        return c1, s1

    return best_c, best_s

# EVOLVE-BLOCK-END
