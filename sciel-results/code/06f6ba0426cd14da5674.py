import time
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import pdist
from scipy.spatial import cKDTree

N = 50
SIDE = 1.0
HALF = 0.5
DEADLINE = 145.0  # Safety margin before 150s kill

def _score_from_centres(c):
    """Compute the exact radius score for given centres (N, 2)."""
    if c.shape != (N, 2):
        return 0.0
    if not np.all(np.isfinite(c)):
        return 0.0
    
    # Check inside square
    if np.any(c < -HALF + 1e-12) or np.any(c > HALF - 1e-12):
        return 0.0
    
    # Pairwise distances
    d = pdist(c)
    min_pair = d.min() if d.size > 0 else 0.0
    
    # Distance to walls
    dist_walls = np.min(np.stack([
        c[:, 0] + HALF,
        HALF - c[:, 0],
        c[:, 1] + HALF,
        HALF - c[:, 1]
    ]), axis=0)
    min_wall = dist_walls.min()
    
    r = min(min_pair / 2.0, min_wall)
    return r

def _find_active_set(c, r, tol=1e-8):
    """
    Find indices of active constraints.
    Returns:
    pair_indices: list of (i, j) where dist(c_i, c_j) <= 2*r + tol
    wall_indices: list of (i, wall_id) where dist(c_i, wall) <= r + tol
    """
    pair_indices = []
    wall_indices = []
    
    # Use KDTree for pairs to be faster than O(N^2) if N is large, but N=50 is small.
    # pdist is fast enough.
    d = pdist(c)
    # Map condensed distance indices to (i, j)
    # This is a bit slow for every iteration, so we might cache or use KDTree.
    # For N=50, 1225 pairs. pdist is very fast.
    
    # Threshold for "active"
    thresh_pair = 2.0 * r + tol
    thresh_wall = r + tol
    
    # Pairs
    # We need the indices.
    # Instead of mapping every time, let's just compute all distances in the optimization loop if needed.
    # For the active set finder, let's use KDTree query_pairs for speed in tight loops.
    tree = cKDTree(c)
    pairs = tree.query_pairs(thresh_pair)
    pair_indices = list(pairs)
    
    # Walls
    for i in range(N):
        x, y = c[i]
        d_left = x + HALF
        d_right = HALF - x
        d_bottom = y + HALF
        d_top = HALF - y
        
        if d_left <= thresh_wall:
            wall_indices.append((i, 0))
        if d_right <= thresh_wall:
            wall_indices.append((i, 1))
        if d_bottom <= thresh_wall:
            wall_indices.append((i, 2))
        if d_top <= thresh_wall:
            wall_indices.append((i, 3))
            
    return pair_indices, wall_indices

def _active_set_objective(x, pair_indices, wall_indices):
    """
    Objective for SLSQP: Minimize -r.
    x = [c_1, ..., c_N, r]
    """
    c = x[:2*N].reshape(N, 2)
    r = x[2*N]
    
    # Clip c to ensure validity during optimization steps
    c = np.clip(c, -HALF + 1e-12, HALF - 1e-12)
    
    # We want to maximize r, so minimize -r
    return -r

def _active_set_jacobian(x, pair_indices, wall_indices):
    """
    Jacobian of the objective (which is just -r).
    Gradient is 0 for c, -1 for r.
    """
    grad = np.zeros(2*N + 1)
    grad[2*N] = -1.0
    return grad

def _make_constraint_func(pair_indices, wall_indices):
    """
    Create constraint functions for SLSQP.
    Constraints are of the form h(x) >= 0.
    """
    def constraints(x):
        c = x[:2*N].reshape(N, 2)
        r = x[2*N]
        
        vals = []
        
        # Pair constraints: dist(c_i, c_j) - 2r >= 0
        for i, j in pair_indices:
            d = np.linalg.norm(c[i] - c[j])
            vals.append(d - 2.0 * r)
            
        # Wall constraints: dist(c_i, wall) - r >= 0
        for i, w in wall_indices:
            x_i, y_i = c[i]
            if w == 0: # Left
                vals.append(x_i + HALF - r)
            elif w == 1: # Right
                vals.append(HALF - x_i - r)
            elif w == 2: # Bottom
                vals.append(y_i + HALF - r)
            elif w == 3: # Top
                vals.append(HALF - y_i - r)
                
        return np.array(vals)
        
    def jacobian(x):
        c = x[:2*N].reshape(N, 2)
        r = x[2*N]
        
        # Number of constraints
        n_pairs = len(pair_indices)
        n_walls = len(wall_indices)
        n_cons = n_pairs + n_walls
        
        J = np.zeros((n_cons, 2*N + 1))
        
        # Pair constraints
        for k, (i, j) in enumerate(pair_indices):
            diff = c[i] - c[j]
            d = np.linalg.norm(diff)
            if d < 1e-12:
                d = 1e-12
                unit = np.array([1.0, 0.0])
            else:
                unit = diff / d
                
            # d/d(c_i) (dist - 2r) = unit
            J[k, 2*i] = unit[0]
            J[k, 2*i+1] = unit[1]
            
            # d/d(c_j) (dist - 2r) = -unit
            J[k, 2*j] = -unit[0]
            J[k, 2*j+1] = -unit[1]
            
            # d/dr (dist - 2r) = -2
            J[k, 2*N] = -2.0
            
        # Wall constraints
        idx = n_pairs
        for i, w in wall_indices:
            if w == 0: # Left: x + 0.5 - r
                J[idx, 2*i] = 1.0
            elif w == 1: # Right: 0.5 - x - r
                J[idx, 2*i] = -1.0
            elif w == 2: # Bottom: y + 0.5 - r
                J[idx, 2*i+1] = 1.0
            elif w == 3: # Top: 0.5 - y - r
                J[idx, 2*i+1] = -1.0
                
            J[idx, 2*N] = -1.0
            idx += 1
            
        return J
        
    return constraints, jacobian

def _polish_active_set(c, max_iter=50, tol=1e-10):
    """
    Iteratively refine the packing using SLSQP on the active set.
    """
    current_c = c.copy()
    current_r = _score_from_centres(current_c)
    
    for it in range(max_iter):
        # Check time
        if time.time() > DEADLINE - 5:
            break
            
        # Find active set
        # Use a slightly larger tolerance to include near-active constraints
        pair_indices, wall_indices = _find_active_set(current_c, current_r, tol=1e-6)
        
        # If no active set, we can't improve (shouldn't happen if r is min)
        if len(pair_indices) == 0 and len(wall_indices) == 0:
            break
            
        # Prepare SLSQP
        x0 = np.concatenate([current_c.ravel(), [current_r]])
        
        # Bounds for centers
        bounds_c = [(-HALF + 1e-9, HALF - 1e-9)] * (2 * N)
        # Bounds for r
        bounds_r = [(0.0, HALF)]
        bounds = bounds_c + bounds_r
        
        cons, jac = _make_constraint_func(pair_indices, wall_indices)
        
        try:
            res = minimize(
                _active_set_objective,
                x0,
                args=(pair_indices, wall_indices),
                method='SLSQP',
                bounds=bounds,
                jac=_active_set_jacobian,
                constraints=[{'type': 'ineq', 'fun': cons, 'jac': jac}],
                options={'maxiter': 200, 'ftol': 1e-12, 'disp': False}
            )
            
            if res.success:
                new_c = res.x[:2*N].reshape(N, 2)
                new_r = res.x[2*N]
                
                # Clip to be safe
                new_c = np.clip(new_c, -HALF + 1e-12, HALF - 1e-12)
                
                # Verify improvement
                actual_r = _score_from_centres(new_c)
                
                if actual_r > current_r + 1e-12:
                    current_c = new_c
                    current_r = actual_r
                else:
                    # If SLSQP didn't improve, try with a tighter tolerance on active set
                    # or break if no progress
                    # Let's try one more time with tighter tol to see if it was a numerical issue
                    pair_indices, wall_indices = _find_active_set(current_c, current_r, tol=1e-8)
                    if len(pair_indices) == 0 and len(wall_indices) == 0:
                        break
                    # If still no improvement, break to avoid infinite loop
                    break
            else:
                # If SLSQP fails, break
                break
                
        except Exception:
            break
            
    return current_c

def _smooth_min_objective(x, beta=1000.0):
    """
    Objective for L-BFGS-B using soft-min approximation.
    x = [c_1, ..., c_N]
    """
    c = x.reshape(N, 2)
    c = np.clip(c, -HALF + 1e-9, HALF - 1e-9)
    
    d = pdist(c) / 2.0
    wall_dists = np.stack([
        c[:, 0] + HALF,
        HALF - c[:, 0],
        c[:, 1] + HALF,
        HALF - c[:, 1]
    ], axis=1).flatten()
    
    all_dists = np.concatenate([d, wall_dists])
    
    min_d = all_dists.min()
    # Soft min: -1/beta * log(sum(exp(-beta * d)))
    # To avoid overflow, subtract min_d
    exp_d = np.exp(-beta * (all_dists - min_d))
    soft_min = min_d - np.log(exp_d.sum()) / beta
    
    return -soft_min

def _grid_init():
    """Generate a grid initial layout."""
    # 7x8 grid is 56 points, we take 50
    # Or 5x10?
    # Let's try to space them out
    rows, cols = 7, 8
    x = np.linspace(-HALF + 0.05, HALF - 0.05, cols)
    y = np.linspace(-HALF + 0.05, HALF - 0.05, rows)
    
    full_grid = np.array([(xi, yi) for yi in y for xi in x])
    # Keep the ones closest to center
    dists_to_center = np.sum(full_grid**2, axis=1)
    sorted_idx = np.argsort(dists_to_center)
    pts = full_grid[sorted_idx[:N]]
    return pts

def _hex_init():
    """Hexagonal packing initial layout."""
    points = []
    r_init = 0.07
    dx = 2 * r_init
    dy = r_init * np.sqrt(3)
    
    max_x = HALF - r_init
    max_y = HALF - r_init
    
    row = 0
    y = -max_y
    while y <= max_y:
        x_start = -max_x
        if row % 2 == 1:
            x_start += dx / 2
        x = x_start
        while x <= max_x:
            points.append([x, y])
            x += dx
        y += dy
        row += 1
        if len(points) > N * 3:
            break
            
    pts = np.array(points)
    dists_to_center = np.sum(pts**2, axis=1)
    sorted_idx = np.argsort(dists_to_center)
    pts = pts[sorted_idx[:N]]
    return pts

def _random_init(rng):
    return rng.uniform(-HALF + 0.01, HALF - 0.01, size=(N, 2))

def _polish_lbfgs(c, beta=1000.0, max_iter=500):
    """
    Polish using L-BFGS-B with soft-min.
    """
    x0 = c.ravel()
    bounds = [(-HALF + 1e-9, HALF - 1e-9)] * (2 * N)
    
    try:
        res = minimize(
            _smooth_min_objective,
            x0,
            args=(beta,),
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': max_iter, 'ftol': 1e-14, 'gtol': 1e-14, 'iprint': 0}
        )
        c_opt = np.clip(res.x.reshape(N, 2), -HALF + 1e-12, HALF - 1e-12)
        return c_opt
    except Exception:
        return c

def construct():
    t0 = time.time()
    rng = np.random.default_rng(42)
    
    # Initial configurations
    inits = [
        _grid_init(),
        _hex_init(),
        _random_init(rng)
    ]
    
    best_centres = inits[0]
    best_score = _score_from_centres(best_centres)
    
    # Phase 1: Global Exploration / Rough Polish
    # Use L-BFGS-B with moderate beta to get out of local minima
    for i in range(len(inits)):
        if time.time() - t0 > DEADLINE - 30:
            break
        c_init = inits[i]
        c_opt = _polish_lbfgs(c_init, beta=100.0, max_iter=200)
        s = _score_from_centres(c_opt)
        if s > best_score:
            best_score = s
            best_centres = c_opt
            
    # Phase 2: Iterative Active Set Refinement
    # This is the key improvement. We loop, refining the active set.
    # We do this for the best candidate found so far.
    # We might also try a few other candidates if time permits.
    
    candidates = [best_centres]
    # Add a few perturbed versions of the best
    for i in range(3):
        if time.time() - t0 > DEADLINE - 20:
            break
        c_pert = best_centres + rng.normal(0, 0.005, size=(N, 2))
        c_pert = np.clip(c_pert, -HALF + 1e-9, HALF - 1e-9)
        candidates.append(c_pert)
        
    for c_cand in candidates:
        if time.time() - t0 > DEADLINE - 10:
            break
            
        # Step 2a: High precision L-BFGS-B to get very close
        c_smooth = _polish_lbfgs(c_cand, beta=1000.0, max_iter=500)
        
        # Step 2b: Active Set SLSQP Polish
        c_final = _polish_active_set(c_smooth, max_iter=100)
        
        s_final = _score_from_centres(c_final)
        if s_final > best_score:
            best_score = s_final
            best_centres = c_final
            
    # Phase 3: Final Check and Validation
    # Ensure no points are outside
    best_centres = np.clip(best_centres, -HALF + 1e-12, HALF - 1e-12)
    
    # One last active set polish on the very best
    if time.time() - t0 < DEADLINE - 2:
        c_final = _polish_active_set(best_centres, max_iter=50)
        s_final = _score_from_centres(c_final)
        if s_final > best_score:
            best_score = s_final
            best_centres = c_final

    return best_centres
