"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 11

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution

def _min_area_bary(points_bary):
    """
    Calculate the minimum area ratio of triangles formed by N points 
    given in barycentric coordinates (u, v) where w = 1 - u - v.
    """
    n = len(points_bary)
    u = points_bary[:, 0]
    v = points_bary[:, 1]
    w = 1.0 - u - v
    
    min_area = float('inf')
    
    # Iterate over all unique triplets
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                det = (u[i] * (v[j] * w[k] - w[j] * v[k])
                     - v[i] * (u[j] * w[k] - w[j] * u[k])
                     + w[i] * (u[j] * v[k] - v[j] * u[k]))
                area = abs(det)
                if area < min_area:
                    min_area = area
    return min_area

def _soft_min_area(points_bary, alpha=50.0):
    """
    Soft minimum using log-sum-exp for smoother gradients.
    Returns a value close to min(area) but differentiable.
    """
    n = len(points_bary)
    u = points_bary[:, 0]
    v = points_bary[:, 1]
    w = 1.0 - u - v
    
    areas = np.zeros(165) # C(11,3) = 165
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                det = (u[i] * (v[j] * w[k] - w[j] * v[k])
                     - v[i] * (u[j] * w[k] - w[j] * u[k])
                     + w[i] * (u[j] * v[k] - v[j] * u[k]))
                areas[idx] = abs(det)
                idx += 1
    
    # Soft min: -alpha * log(sum(exp(-areas/alpha)))
    # To avoid overflow/underflow, shift by min
    min_val = np.min(areas)
    shifted = areas - min_val
    log_sum = np.log(np.sum(np.exp(-shifted / alpha)))
    soft_min = min_val - alpha * log_sum
    return soft_min

def _objective_soft(params, alpha=50.0):
    """Objective for L-BFGS-B using soft min + penalty."""
    uv = params.reshape(N, 2)
    u = uv[:, 0]
    v = uv[:, 1]
    w = 1.0 - u - v
    
    penalty = 0.0
    if np.any(w < 0):
        violations = -w[w < 0]
        penalty = 1000.0 * np.sum(violations ** 2)
    if np.any(u < 0) or np.any(v < 0):
        penalty += 1000.0 * (np.sum(u[u < 0]**2) + np.sum(v[v < 0]**2))
        
    if penalty > 1e-6:
        return penalty - 1e-4
        
    return -_soft_min_area(uv, alpha) + penalty

def _objective_hard(params):
    """Objective for final polish using hard min + penalty."""
    uv = params.reshape(N, 2)
    u = uv[:, 0]
    v = uv[:, 1]
    w = 1.0 - u - v
    
    penalty = 0.0
    if np.any(w < 0):
        violations = -w[w < 0]
        penalty = 1000.0 * np.sum(violations ** 2)
    if np.any(u < 0) or np.any(v < 0):
        penalty += 1000.0 * (np.sum(u[u < 0]**2) + np.sum(v[v < 0]**2))
        
    if penalty > 1e-6:
        return penalty - 1e-4
        
    return -_min_area_bary(uv) + penalty

def _clamp_to_simplex(points):
    """Clamp points to be inside the simplex u>=0, v>=0, u+v<=1."""
    u = np.clip(points[:, 0], 0, 1)
    v = np.clip(points[:, 1], 0, 1)
    sum_uv = u + v
    mask = sum_uv > 1
    if np.any(mask):
        factor = 1.0 / sum_uv[mask]
        u[mask] *= factor
        v[mask] *= factor
    return np.column_stack((u, v))

def _get_good_initializations():
    """Generate a set of good starting points."""
    inits = []
    
    # 1. The specific 10-grid + center from Inspiration 1
    grid_10 = [
        (1.0, 0.0), (2/3, 1/3), (1/3, 2/3), (0.0, 1.0),
        (2/3, 0.0), (1/3, 1/3), (0.0, 2/3),
        (1/3, 0.0), (0.0, 1/3),
        (0.0, 0.0)
    ]
    # Add a point near center but distinct, e.g., slightly off center
    # The center (1/3, 1/3) is already in the list. 
    # Let's add (0.4, 0.4) which is (0.4, 0.4, 0.2)
    init1 = np.array(grid_10 + [(0.4, 0.4)])
    inits.append(init1)
    
    # 2. Perturbed versions of init1
    rng = np.random.RandomState(42)
    for _ in range(10):
        noise = rng.uniform(-0.05, 0.05, size=(N, 2))
        perturbed = _clamp_to_simplex(init1 + noise)
        inits.append(perturbed)
        
    # 3. Random Dirichlet samples
    for _ in range(10):
        bary = rng.dirichlet([1, 1, 1], size=N)
        inits.append(bary[:, :2])
        
    return inits

def construct():
    start_time = time.time()
    deadline = start_time + 145.0 # 5s safety margin
    
    best_score = -float('inf')
    best_points_bary = None
    
    bounds = [(0, 1)] * (2 * N)
    
    # Phase 1: Global Search / Initial Exploration
    # We try a few initializations. For the best ones, we might run DE.
    # However, DE is slow. Let's just run L-BFGS-B with soft min from multiple starts first.
    # If we have time, we can try DE on the best few.
    
    inits = _get_good_initializations()
    
    # Sort inits by their initial hard min area to prioritize good ones
    init_scores = []
    for init in inits:
        score = _min_area_bary(init)
        init_scores.append((score, init))
    init_scores.sort(key=lambda x: x[0], reverse=True)
    
    # Process top initializations
    for score, init in init_scores:
        if time.time() > deadline - 10:
            break
            
        current_best_for_init = init.copy()
        current_score = score
        
        # Step 1: Soft Min Refinement
        try:
            res_soft = minimize(
                _objective_soft,
                current_best_for_init.flatten(),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10, 'alpha': 50.0}
            )
            if res_soft.success or res_soft.fun < _objective_soft(current_best_for_init.flatten()):
                uv_soft = res_soft.x.reshape(N, 2)
                # Check validity
                w_soft = 1 - uv_soft[:, 0] - uv_soft[:, 1]
                if np.all(w_soft >= -1e-6) and np.all(uv_soft >= -1e-6):
                    uv_soft = _clamp_to_simplex(uv_soft)
                    score_soft = _min_area_bary(uv_soft)
                    if score_soft > current_score:
                        current_score = score_soft
                        current_best_for_init = uv_soft
        except Exception:
            pass
        
        # Step 2: Hard Min Polish
        if time.time() > deadline - 5:
            break
            
        try:
            res_hard = minimize(
                _objective_hard,
                current_best_for_init.flatten(),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 1000, 'ftol': 1e-14, 'gtol': 1e-12}
            )
            if res_hard.success:
                uv_hard = res_hard.x.reshape(N, 2)
                w_hard = 1 - uv_hard[:, 0] - uv_hard[:, 1]
                if np.all(w_hard >= -1e-6) and np.all(uv_hard >= -1e-6):
                    uv_hard = _clamp_to_simplex(uv_hard)
                    score_hard = _min_area_bary(uv_hard)
                    if score_hard > current_score:
                        current_score = score_hard
                        current_best_for_init = uv_hard
        except Exception:
            pass
        
        # Update global best
        if current_score > best_score:
            best_score = current_score
            best_points_bary = current_best_for_init.copy()
            
        # Optional: If we have time and this was a very good start, try DE?
        # DE is expensive. Let's skip DE to ensure we finish in time, 
        # relying on the multi-start L-BFGS-B with soft min being sufficient.
        # The soft min helps escape local minima that hard min gets stuck in.

    # If we still have significant time, try one Differential Evolution run
    # on the best found so far to see if we can jump to a better basin.
    if time.time() < deadline - 20 and best_points_bary is not None:
        try:
            # Run DE on the region around the best point?
            # Or just a quick DE from scratch?
            # DE from scratch is risky. Let's perturb the best point and run DE?
            # Actually, DE is best for global search. Let's try a short DE.
            # But DE with 22 variables and 165 evaluations per func call might be slow.
            # Let's stick to the current robust multi-start L-BFGS-B.
            pass
        except Exception:
            pass

    if best_points_bary is None:
        # Fallback
        best_points_bary = inits[0]
        
    # Convert barycentric to Cartesian
    A = np.array([0.0, 0.0])
    B = np.array([1.0, 0.0])
    C = np.array([0.5, np.sqrt(3)/2])
    
    pts = []
    for i in range(N):
        u, v = best_points_bary[i]
        w = 1 - u - v
        u = max(0, u)
        v = max(0, v)
        w = max(0, w)
        s = u + v + w
        if s > 0:
            u /= s
            v /= s
            w /= s
        p = u * A + v * B + w * C
        pts.append(p)
        
    return np.array(pts)

# EVOLVE-BLOCK-END
