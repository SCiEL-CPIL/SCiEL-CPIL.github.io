"""Circle packing in unit square: maximize sum of 26 radii with disjoint interiors."""
import numpy as np
import time

N = 26
TIME_LIMIT = 110.0  # safety margin before 120s kill

def _init_random(seed):
    rng = np.random.RandomState(seed)
    centers = rng.uniform(0.05, 0.95, size=(N, 2))
    radii = rng.uniform(0.01, 0.12, size=N)
    return centers, radii

def _overlap_penalty(centers, radii, softness=50.0):
    """Penalize pairwise overlaps with a smooth function."""
    n = len(centers)
    total = 0.0
    for i in range(n):
        for j in range(i+1, n):
            dx = centers[i,0] - centers[j,0]
            dy = centers[i,1] - centers[j,1]
            dist = np.sqrt(dx*dx + dy*dy + 1e-12)
            min_dist = radii[i] + radii[j]
            if dist < min_dist:
                # smooth penalty
                total += (min_dist - dist) ** 2
    return total

def _boundary_penalty(centers, radii):
    """Penalize circles going outside [0,1]^2."""
    total = 0.0
    for i in range(len(centers)):
        x, y = centers[i]
        r = radii[i]
        # distance to each wall
        d_left = x - r
        d_right = (1.0 - x) - r
        d_bottom = y - r
        d_top = (1.0 - y) - r
        for d in [d_left, d_right, d_bottom, d_top]:
            if d < 0:
                total += d * d
    return total

def _objective(params):
    """Objective to minimize: negative sum of radii + penalties."""
    centers = params[:2*N]
    radii = np.exp(params[2*N:])  # ensure positive
    # Clip radii to reasonable range
    radii = np.clip(radii, 1e-4, 0.5)
    
    neg_sum = -np.sum(radii)
    op = _overlap_penalty(centers, radii)
    bp = _boundary_penalty(centers, radii)
    return neg_sum + 100.0 * op + 100.0 * bp

def _is_valid(centers, radii, tol=1e-6):
    """Check if packing is valid: all circles inside [0,1]^2 and disjoint."""
    for i in range(len(centers)):
        x, y = centers[i]
        r = radii[i]
        if x - r < -tol or x + r > 1.0 + tol or y - r < -tol or y + r > 1.0 + tol:
            return False
    for i in range(len(centers)):
        for j in range(i+1, len(centers)):
            dx = centers[i,0] - centers[j,0]
            dy = centers[i,1] - centers[j,1]
            dist = np.sqrt(dx*dx + dy*dy)
            if dist < radii[i] + radii[j] - tol:
                return False
    return True

def _repair(centers, radii):
    """Try to fix small violations by shrinking radii slightly."""
    centers = centers.copy()
    radii = radii.copy()
    # Ensure inside boundary
    for i in range(N):
        x, y = centers[i]
        r = radii[i]
        max_r = min(x, 1.0-x, y, 1.0-y)
        if r > max_r:
            radii[i] = max(max_r - 1e-8, 1e-4)
    # Iteratively shrink to resolve overlaps
    for _ in range(50):
        changed = False
        for i in range(N):
            for j in range(i+1, N):
                dx = centers[i,0] - centers[j,0]
                dy = centers[i,1] - centers[j,1]
                dist = np.sqrt(dx*dx + dy*dy + 1e-12)
                min_dist = radii[i] + radii[j]
                if dist < min_dist:
                    excess = min_dist - dist
                    # shrink both
                    shrink = excess * 0.5 + 1e-6
                    radii[i] = max(radii[i] - shrink, 1e-4)
                    radii[j] = max(radii[j] - shrink, 1e-4)
                    changed = True
        if not changed:
            break
    return centers, radii

def _try_improve(centers, radii, max_iters=2000, timeout=None):
    """Use L-BFGS-B to optimize the packing."""
    from scipy.optimize import minimize
    
    # Parameterize: centers directly, radii as log
    params0 = np.concatenate([centers, np.log(radii)])
    
    def obj(p):
        c = p[:2*N]
        r = np.exp(p[2*N:])
        r = np.clip(r, 1e-4, 0.5)
        neg_sum = -np.sum(r)
        op = _overlap_penalty(c, r)
        bp = _boundary_penalty(c, r)
        return neg_sum + 100.0 * op + 100.0 * bp
    
    best_val = obj(params0)
    best_params = params0.copy()
    
    try:
        res = minimize(obj, params0, method='L-BFGS-B', 
                       options={'maxiter': max_iters, 'maxfun': max_iters*2})
        if res.fun < best_val:
            best_val = res.fun
            best_params = res.x
    except Exception:
        pass
    
    c_best = best_params[:2*N]
    r_best = np.exp(best_params[2*N:])
    r_best = np.clip(r_best, 1e-4, 0.5)
    
    # Try to repair
    c_rep, r_rep = _repair(c_best, r_best)
    if _is_valid(c_rep, r_rep):
        return c_rep, r_rep, np.sum(r_rep)
    
    # Try original
    if _is_valid(c_best, r_best):
        return c_best, r_best, np.sum(r_best)
    
    return None, None, 0.0

def construct():
    start_time = time.time()
    
    best_centers = None
    best_radii = None
    best_score = 0.0
    
    # Try multiple random restarts
    for seed in range(200):
        elapsed = time.time() - start_time
        if elapsed > TIME_LIMIT:
            break
        
        centers, radii = _init_random(seed)
        
        # Quick local optimization
        c, r, score = _try_improve(centers, radii, max_iters=1000)
        
        if c is not None and score > best_score:
            best_score = score
            best_centers = c.copy()
            best_radii = r.copy()
        
        # Check time
        elapsed = time.time() - start_time
        if elapsed > TIME_LIMIT - 10:
            break
    
    # If no valid packing found, fall back to grid
    if best_centers is None:
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) 
                 for i in range(side) for j in range(side)]
        return np.array(cells[:N])
    
    # Final validation
    if not _is_valid(best_centers, best_radii):
        # Try harder repair
        best_centers, best_radii = _repair(best_centers, best_radii)
        if not _is_valid(best_centers, best_radii):
            side = int(np.ceil(np.sqrt(N)))
            cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) 
                     for i in range(side) for j in range(side)]
            return np.array(cells[:N])
    
    result = np.column_stack([best_centers, best_radii])
    return result

# EVOLVE-BLOCK-END
