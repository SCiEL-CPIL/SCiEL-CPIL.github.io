# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from itertools import combinations

def _check_feasibility(centers, sides, tol=1e-7):
    """Fast vectorized feasibility check."""
    n = len(sides)
    if not np.all(np.isfinite(centers)) or not np.all(np.isfinite(sides)):
        return False
    if np.any(sides <= 0):
        return False
    # Boundary checks
    if np.any(centers[:, 0] - sides / 2 < -tol):
        return False
    if np.any(centers[:, 0] + sides / 2 > 1 + tol):
        return False
    if np.any(centers[:, 1] - sides / 2 < -tol):
        return False
    if np.any(centers[:, 1] + sides / 2 > 1 + tol):
        return False
    # Overlap checks (vectorized)
    i_idx, j_idx = np.triu_indices(n, k=1)
    t = (sides[i_idx] + sides[j_idx]) / 2 - tol
    dx = np.abs(centers[i_idx, 0] - centers[j_idx, 0])
    dy = np.abs(centers[i_idx, 1] - centers[j_idx, 1])
    if np.any((dx < t) & (dy < t)):
        return False
    return True

def _sum_sides(sides):
    return float(np.sum(sides))

def _smooth_penalty(centers, sides, tol=1e-7, beta=80.0):
    """
    Vectorized smooth penalty for constraint violations.
    """
    n = len(sides)
    
    # Boundary violations
    vx_left = -tol - (centers[:, 0] - sides / 2)
    vx_right = (centers[:, 0] + sides / 2) - (1 + tol)
    vy_bottom = -tol - (centers[:, 1] - sides / 2)
    vy_top = (centers[:, 1] + sides / 2) - (1 + tol)
    
    # Smooth max(0, v) = v/beta * log(1 + exp(beta*v)) for v>0, ~0 otherwise
    def _softmax_pos(v):
        # numerically stable: for large positive v, v*log(1+e^{beta*v})/beta ~ v
        # for v<=0, return 0
        v = np.maximum(v, 0)
        # clip to avoid overflow
        bv = np.clip(beta * v, 0, 500)
        return v * np.log1p(np.exp(bv)) / beta
    
    penalty = (_softmax_pos(vx_left) + _softmax_pos(vx_right) +
               _softmax_pos(vy_bottom) + _softmax_pos(vy_top)).sum()
    
    # Overlap violations
    i_idx, j_idx = np.triu_indices(n, k=1)
    t = (sides[i_idx] + sides[j_idx]) / 2 - tol
    dx = np.abs(centers[i_idx, 0] - centers[j_idx, 0])
    dy = np.abs(centers[i_idx, 1] - centers[j_idx, 1])
    
    ox = np.maximum(t - dx, 0)
    oy = np.maximum(t - dy, 0)
    
    soft_ox = _softmax_pos(ox)
    soft_oy = _softmax_pos(oy)
    
    # Product penalty for full overlap + linear penalty for partial
    penalty += (soft_ox * soft_oy + 0.1 * soft_ox + 0.1 * soft_oy).sum()
    
    return float(penalty)

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

def _project_feasible(centers, sides, tol=1e-7):
    """
    Project a configuration toward feasibility by shrinking sides and
    adjusting centers to stay in bounds. Returns (new_centers, new_sides).
    """
    new_sides = sides.copy()
    new_centers = centers.copy()
    
    # Clip sides so squares fit in [0,1]^2 given current centers
    for i in range(10):
        cx, cy = new_centers[i]
        max_s = min(cx - (-tol), (1 + tol) - cx, cy - (-tol), (1 + tol) - cy) * 2
        new_sides[i] = min(new_sides[i], max(max_s, 1e-6))
    
    # Iteratively separate overlapping squares
    for iteration in range(20):
        moved = False
        for i in range(10):
            for j in range(i + 1, 10):
                t = (new_sides[i] + new_sides[j]) / 2 - tol
                dx = new_centers[j, 0] - new_centers[i, 0]
                dy = new_centers[j, 1] - new_centers[i, 1]
                if abs(dx) < t and abs(dy) < t:
                    # Push apart along the axis of least separation
                    if abs(dx) <= abs(dy):
                        push = (t - abs(dx)) + tol
                        if dx >= 0:
                            new_centers[i, 0] -= push / 2
                            new_centers[j, 0] += push / 2
                        else:
                            new_centers[i, 0] += push / 2
                            new_centers[j, 0] -= push / 2
                    else:
                        push = (t - abs(dy)) + tol
                        if dy >= 0:
                            new_centers[i, 1] -= push / 2
                            new_centers[j, 1] += push / 2
                        else:
                            new_centers[i, 1] += push / 2
                            new_centers[j, 1] -= push / 2
                    moved = True
        # Re-clip centers to bounds
        for i in range(10):
            new_centers[i, 0] = np.clip(new_centers[i, 0], new_sides[i]/2 - tol, 1 - new_sides[i]/2 + tol)
            new_centers[i, 1] = np.clip(new_centers[i, 1], new_sides[i]/2 - tol, 1 - new_sides[i]/2 + tol)
        if not moved:
            break
    
    return new_centers, new_sides

def _coordinate_ascent(centers, sides, time_limit=20.0, seed=42):
    """
    1D coordinate ascent with feasibility projection.
    Cycles through each parameter (cx_i, cy_i, s_i) and tries to improve.
    Uses small steps and accepts only if feasibility is maintained or restored.
    """
    n = 10
    best_centers = centers.copy()
    best_sides = sides.copy()
    best_sum = float(np.sum(best_sides))
    
    start_time = time.time()
    rng = np.random.default_rng(seed)
    
    # Parameters: 0..9 = cx_i, 10..19 = cy_i, 20..29 = s_i
    total_params = 3 * n
    
    # Step sizes
    base_step_c = 0.02
    base_step_s = 0.01
    
    no_improve_count = 0
    max_no_improve = 200
    
    while time.time() - start_time < time_limit and no_improve_count < max_no_improve:
        improved = False
        # Randomize parameter order for diversity
        param_order = rng.permutation(total_params)
        
        for p in param_order:
            if time.time() - start_time > time_limit:
                break
            
            if p < n:
                # cx of square p
                idx = p
                coord = 'x'
                base = base_step_c
            elif p < 2 * n:
                # cy of square p
                idx = p - n
                coord = 'y'
                base = base_step_c
            else:
                # side of square p
                idx = p - 2 * n
                coord = 's'
                base = base_step_s
            
            # Try several step sizes (decreasing)
            for scale in [1.0, 0.5, 0.25, 0.1, 0.05]:
                step = base * scale
                # Try both directions
                for direction in [1, -1]:
                    new_centers = best_centers.copy()
                    new_sides = best_sides.copy()
                    
                    if coord == 'x':
                        new_centers[idx, 0] += direction * step
                    elif coord == 'y':
                        new_centers[idx, 1] += direction * step
                    else:
                        new_sides[idx] += direction * step
                        if new_sides[idx] <= 1e-6:
                            continue
                    
                    # Check if this single-parameter change maintains feasibility
                    if _check_feasibility(new_centers, new_sides):
                        new_sum = float(np.sum(new_sides))
                        if new_sum > best_sum + 1e-12:
                            best_centers = new_centers
                            best_sides = new_sides
                            best_sum = new_sum
                            improved = True
                        break  # Found a working step size, stop trying smaller
                else:
                    continue
                break
            else:
                # No step worked; try a projected version
                new_centers = best_centers.copy()
                new_sides = best_sides.copy()
                if coord == 'x':
                    new_centers[idx, 0] += direction * base
                elif coord == 'y':
                    new_centers[idx, 1] += direction * base
                else:
                    new_sides[idx] += direction * base
                    if new_sides[idx] <= 1e-6:
                        continue
                
                proj_c, proj_s = _project_feasible(new_centers, new_sides)
                if _check_feasibility(proj_c, proj_s):
                    proj_sum = float(np.sum(proj_s))
                    if proj_sum > best_sum + 1e-12:
                        best_centers = proj_c
                        best_sides = proj_s
                        best_sum = proj_sum
                        improved = True
        
        if not improved:
            no_improve_count += 1
            # Shrink base steps
            base_step_c *= 0.9
            base_step_s *= 0.9
            if base_step_c < 1e-5:
                break
        else:
            no_improve_count = 0
    
    return best_centers, best_sides

def _simulated_annealing(centers, sides, time_limit=15.0, seed=42):
    """
    Simulated Annealing to maximize sum of side lengths.
    Uses adaptive step sizes and feasibility projection.
    """
    n = len(sides)
    best_centers = centers.copy()
    best_sides = sides.copy()
    best_sum = float(np.sum(best_sides))
    current_centers = centers.copy()
    current_sides = sides.copy()
    current_sum = float(np.sum(current_sides))
    
    start_time = time.time()
    rng = np.random.default_rng(seed)
    
    T = 0.15
    T_min = 1e-9
    alpha = 0.9995
    
    step_center = 0.04
    step_side = 0.02
    
    while T > T_min:
        if time.time() - start_time > time_limit:
            break
            
        dc = rng.normal(0, step_center, size=(n, 2))
        ds = rng.normal(0, step_side, size=n)
        
        new_centers = current_centers + dc
        new_sides = np.clip(current_sides + ds, 1e-6, 1.0)
        
        # Project toward feasibility
        proj_c, proj_s = _project_feasible(new_centers, new_sides)
        
        if _check_feasibility(proj_c, proj_s):
            new_sum = float(np.sum(proj_s))
            delta = new_sum - current_sum
            if delta > 0 or rng.random() < np.exp(delta / max(T, 1e-12)):
                current_centers = proj_c
                current_sides = proj_s
                current_sum = new_sum
                if current_sum > best_sum:
                    best_centers = current_centers.copy()
                    best_sides = current_sides.copy()
                    best_sum = current_sum
                if rng.random() < 0.02:
                    step_center = min(step_center * 1.05, 0.15)
                    step_side = min(step_side * 1.05, 0.08)
            else:
                if rng.random() < 0.03:
                    step_center *= 0.97
                    step_side *= 0.97
        else:
            # Try smaller perturbation
            small_dc = rng.normal(0, step_center * 0.3, size=(n, 2))
            small_ds = rng.normal(0, step_side * 0.3, size=n)
            sc = current_centers + small_dc
            ss = np.clip(current_sides + small_ds, 1e-6, 1.0)
            proj_c2, proj_s2 = _project_feasible(sc, ss)
            if _check_feasibility(proj_c2, proj_s2):
                new_sum2 = float(np.sum(proj_s2))
                delta2 = new_sum2 - current_sum
                if delta2 > 0 or rng.random() < np.exp(delta2 / max(T, 1e-12)):
                    current_centers = proj_c2
                    current_sides = proj_s2
                    current_sum = new_sum2
                    if current_sum > best_sum:
                        best_centers = current_centers.copy()
                        best_sides = current_sides.copy()
                        best_sum = current_sum
        
        T *= alpha
        
    return best_centers, best_sides

def _polish_with_scipy(centers, sides, time_limit=30.0):
    """
    Final polish using scipy.optimize with the smooth objective.
    """
    x0 = _flatten(centers, sides)
    
    n = 10
    bounds = []
    for i in range(n):
        bounds.append((0.01, 0.99))
        bounds.append((0.01, 0.99))
    for i in range(n):
        bounds.append((0.05, 0.95))
    
    start_time = time.time()
    obj0 = _objective_flat(x0)
    
    try:
        result = minimize(
            _objective_flat,
            x0,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 800, 'ftol': 1e-14, 'gtol': 1e-12, 'maxfun': 20000}
        )
        
        if result.fun < obj0:
            new_centers, new_sides = _unflatten(result.x)
            if _check_feasibility(new_centers, new_sides):
                return new_centers, new_sides
    except Exception:
        pass
    
    # Try SLSQP
    try:
        result = minimize(
            _objective_flat,
            x0,
            method='SLSQP',
            bounds=bounds,
            options={'maxiter': 200, 'ftol': 1e-12}
        )
        
        if result.fun < obj0:
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
      3. Run coordinate ascent + SA from multiple starts.
      4. Polish best candidates with scipy optimization.
      5. Return the best feasible configuration found.
    """
    start_time = time.time()
    deadline = start_time + 560  # Safety margin before 630s limit

    n = 10

    candidates = []
    
    # --- Candidate 1: 2-big / 8-small (sum = 3.0) ---
    c1, s1 = _make_2big_8small()
    if _check_feasibility(c1, s1):
        candidates.append((c1.copy(), s1.copy(), _sum_sides(s1)))
        # Coordinate ascent from this start
        c1_ca, s1_ca = _coordinate_ascent(c1, s1, time_limit=15.0, seed=42)
        if _check_feasibility(c1_ca, s1_ca):
            candidates.append((c1_ca, s1_ca, _sum_sides(s1_ca)))
        # SA from this start
        if time.time() < deadline - 120:
            c1_sa, s1_sa = _simulated_annealing(c1, s1, time_limit=12.0, seed=42)
            if _check_feasibility(c1_sa, s1_sa):
                candidates.append((c1_sa, s1_sa, _sum_sides(s1_sa)))

    # --- Candidate 2: 3-big / 7-small ---
    c2, s2 = _make_3big_7small()
    if _check_feasibility(c2, s2):
        candidates.append((c2.copy(), s2.copy(), _sum_sides(s2)))
        c2_ca, s2_ca = _coordinate_ascent(c2, s2, time_limit=10.0, seed=123)
        if _check_feasibility(c2_ca, s2_ca):
            candidates.append((c2_ca, s2_ca, _sum_sides(s2_ca)))

    # --- Candidates 3-8: Random starts with CA + SA ---
    for i in range(6):
        if time.time() > deadline - 100:
            break
        c_r, s_r = _make_random_start(seed=100 + i)
        # Project to feasibility first
        pc, ps = _project_feasible(c_r, s_r)
        if _check_feasibility(pc, ps):
            candidates.append((pc.copy(), ps.copy(), _sum_sides(ps)))
            # Coordinate ascent
            c_r_ca, s_r_ca = _coordinate_ascent(pc, ps, time_limit=8.0, seed=200 + i)
            if _check_feasibility(c_r_ca, s_r_ca):
                candidates.append((c_r_ca, s_r_ca, _sum_sides(s_r_ca)))
            # SA
            if time.time() < deadline - 80:
                c_r_sa, s_r_sa = _simulated_annealing(pc, ps, time_limit=8.0, seed=300 + i)
                if _check_feasibility(c_r_sa, s_r_sa):
                    candidates.append((c_r_sa, s_r_sa, _sum_sides(s_r_sa)))

    # --- Additional structured starts: try non-uniform sizes ---
    if time.time() < deadline - 60:
        # Try: 1 large (0.45) + 9 small
        c_struct = np.zeros((10, 2))
        s_struct = np.zeros(10)
        s_big = 0.45
        c_struct[0] = [s_big/2, s_big/2]
        s_struct[0] = s_big
        # 3 small in top-right area
        s_sm = (1 - s_big) / 2
        c_struct[1] = [s_big + s_sm/2, s_big + s_sm/2]
        s_struct[1] = s_sm
        c_struct[2] = [1 - s_sm/2, s_big + s_sm/2]
        s_struct[2] = s_sm
        c_struct[3] = [s_big + s_sm/2, 1 - s_sm/2]
        s_struct[3] = s_sm
        c_struct[4] = [1 - s_sm/2, 1 - s_sm/2]
        s_struct[4] = s_sm
        # 5 more small in remaining space
        s_sm2 = (1 - s_big) / 2
        c_struct[5] = [s_sm2/2, s_big + s_sm2/2]
        s_struct[5] = s_sm2
        c_struct[6] = [s_sm2/2, 1 - s_sm2/2]
        s_struct[6] = s_sm2
        c_struct[7] = [s_big + s_sm2*0.5, s_sm2/2]
        s_struct[7] = s_sm2
        c_struct[8] = [s_big + s_sm2*1.5, s_sm2/2]
        s_struct[8] = s_sm2
        c_struct[9] = [1 - s_sm2/2, s_sm2/2]
        s_struct[9] = s_sm2
        
        pc, ps = _project_feasible(c_struct, s_struct)
        if _check_feasibility(pc, ps):
            candidates.append((pc.copy(), ps.copy(), _sum_sides(ps)))
            c_ca, s_ca = _coordinate_ascent(pc, ps, time_limit=10.0, seed=500)
            if _check_feasibility(c_ca, s_ca):
                candidates.append((c_ca, s_ca, _sum_sides(s_ca)))

    # Pick the best feasible candidate
    if not candidates:
        c1, s1 = _make_2big_8small()
        return c1, s1

    # Sort by value
    candidates.sort(key=lambda x: x[2], reverse=True)
    best_c, best_s, best_val = candidates[0]
    
    # --- Polish the top candidates ---
    if time.time() < deadline - 30:
        for c, s, val in candidates[:4]:
            if time.time() > deadline - 20:
                break
            c_pol, s_pol = _polish_with_scipy(c, s, time_limit=12.0)
            if _check_feasibility(c_pol, s_pol):
                new_val = _sum_sides(s_pol)
                if new_val > best_val:
                    best_c, best_s, best_val = c_pol, s_pol, new_val
    
    # Final coordinate ascent polish on the best
    if time.time() < deadline - 5:
        c_final, s_final = _coordinate_ascent(best_c, best_s, time_limit=5.0, seed=999)
        if _check_feasibility(c_final, s_final) and _sum_sides(s_final) > best_val:
            best_c, best_s = c_final, s_final

    # Final feasibility check
    if not _check_feasibility(best_c, best_s):
        c1, s1 = _make_2big_8small()
        return c1, s1

    return best_c, best_s

# EVOLVE-BLOCK-END
