"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 11

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import itertools
import time

V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, np.sqrt(3) / 2])
TRI_AREA = np.sqrt(3) / 4.0

def _clamp_bary(bary):
    """Clamp barycentric coords to valid region: u>=0, v>=0, u+v<=1."""
    b = np.clip(bary, 0, 1)
    s = b[:, 0] + b[:, 1]
    mask = s > 1
    if np.any(mask):
        b[mask] /= s[mask, np.newaxis]
    return b

def _bary_to_cart(bary):
    """Vectorized barycentric to Cartesian conversion."""
    return bary[:, 0:1] * V1 + bary[:, 1:2] * V2

def _min_triangle_area_vectorized(points):
    """Compute min triangle area among all triples, vectorized."""
    n = len(points)
    min_area = float('inf')
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                ax, ay = points[i]
                bx, by = points[j]
                cx, cy = points[k]
                area = 0.5 * abs((bx - ax) * (cy - ay) - (cx - ax) * (by - ay))
                if area < min_area:
                    min_area = area
    return min_area

def _softmin_objective(x, sharpness=50.0):
    """Soft-min surrogate: -log-sum-exp of negative areas, scaled by sharpness.
    This is smooth and differentiable, suitable for L-BFGS-B."""
    bary = _clamp_bary(x.reshape(-1, 2))
    points = _bary_to_cart(bary)
    n = len(points)
    
    # Compute all triangle areas
    areas = []
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                ax, ay = points[i]
                bx, by = points[j]
                cx, cy = points[k]
                area = 0.5 * abs((bx - ax) * (cy - ay) - (cx - ax) * (by - ay))
                areas.append(area)
    
    areas = np.array(areas)
    # Soft-min: -1/sharpness * log(sum(exp(-sharpness * areas)))
    # This approximates the minimum area as sharpness -> inf
    neg_areas = -sharpness * areas
    max_neg = np.max(neg_areas)
    log_sum = max_neg + np.log(np.sum(np.exp(neg_areas - max_neg)))
    soft_min = -log_sum / sharpness
    return soft_min

def _make_initial_bary_grid(level, seed=42):
    """Generate grid initial points in barycentric coordinates."""
    rng = np.random.RandomState(seed)
    bary_list = []
    for i in range(level + 1):
        for j in range(level + 1 - i):
            u = i / level
            v = j / level
            if u + v <= 1.0:
                ju = rng.uniform(-0.02, 0.02)
                jv = rng.uniform(-0.02, 0.02)
                bary_list.append([u + ju, v + jv])
    
    if len(bary_list) > N:
        indices = rng.choice(len(bary_list), size=N, replace=False)
        bary_list = [bary_list[i] for i in indices]
    elif len(bary_list) < N:
        while len(bary_list) < N:
            u = rng.uniform(0, 1)
            v = rng.uniform(0, 1 - u)
            bary_list.append([u, v])
    return np.array(bary_list)

def _make_initial_bary_random(seed):
    """Generate random initial points in barycentric coordinates."""
    rng = np.random.RandomState(seed)
    bary_list = []
    for _ in range(N):
        u = rng.uniform(0, 1)
        v = rng.uniform(0, 1 - u)
        bary_list.append([u, v])
    return np.array(bary_list)

def _make_initial_bary_perturb(base_bary, scale=0.1, seed=42):
    """Perturb a base configuration."""
    rng = np.random.RandomState(seed)
    bary = base_bary + rng.uniform(-scale, scale, size=base_bary.shape)
    return _clamp_bary(bary)

def _nelder_mead_optimize(x0, maxiter=2000):
    """Nelder-Mead optimization with true min area."""
    def obj(x):
        bary = _clamp_bary(x.reshape(-1, 2))
        points = _bary_to_cart(bary)
        return -_min_triangle_area_vectorized(points)
    
    try:
        result = minimize(
            obj, x0, method='Nelder-Mead',
            options={'maxiter': maxiter, 'xatol': 1e-8, 'fatol': 1e-8, 'adaptive': True}
        )
        return result.x
    except Exception:
        return x0

def _lbfgsb_optimize(x0, maxiter=500, sharpness=50.0):
    """L-BFGS-B optimization with soft-min surrogate."""
    def obj(x):
        return _softmin_objective(x, sharpness=sharpness)
    
    try:
        result = minimize(
            obj, x0, method='L-BFGS-B',
            options={'maxiter': maxiter, 'gtol': 1e-6}
        )
        return result.x
    except Exception:
        return x0

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    best_points = None
    best_score = -float('inf')
    
    def update_best(points):
        nonlocal best_points, best_score
        score = _min_triangle_area_vectorized(points)
        if score > best_score:
            best_score = score
            best_points = points.copy()
    
    # Generate diverse initializations
    initializations = []
    
    # Grid initializations
    for level in [2, 3, 4, 5]:
        for seed in [42, 123, 7, 999]:
            initializations.append(_make_initial_bary_grid(level, seed))
    
    # Random initializations
    for seed in [42, 123, 7, 999, 2024, 31415, 27182]:
        initializations.append(_make_initial_bary_random(seed))
    
    # Perturbed versions of some initializations
    for idx in [0, 3, 6]:
        if idx < len(initializations):
            for scale in [0.05, 0.1, 0.2]:
                initializations.append(_make_initial_bary_perturb(initializations[idx], scale, seed=idx*10+int(scale*100)))
    
    # Optimize each initialization
    for idx, bary_init in enumerate(initializations):
        if time.time() > deadline - 20.0:
            break
        
        x0 = bary_init.flatten()
        
        # Try L-BFGS-B with soft-min first (faster convergence)
        if time.time() < deadline - 25.0:
            x_lbfgsb = _lbfgsb_optimize(x0, maxiter=300, sharpness=50.0)
            bary_opt = _clamp_bary(x_lbfgsb.reshape(-1, 2))
            points = _bary_to_cart(bary_opt)
            update_best(points)
            x0 = x_lbfgsb  # Use as starting point for Nelder-Mead
        
        # Refine with Nelder-Mead on true objective
        if time.time() < deadline - 10.0:
            x_nm = _nelder_mead_optimize(x0, maxiter=1500)
            bary_opt = _clamp_bary(x_nm.reshape(-1, 2))
            points = _bary_to_cart(bary_opt)
            update_best(points)
    
    # If we still have time, do additional refinement on best solution
    if best_points is not None and time.time() < deadline - 5.0:
        bary_best = np.array([
            [best_points[i][0] * 0.5 + best_points[i][1] / (np.sqrt(3)/2) * 0.5,
             best_points[i][1] / (np.sqrt(3)/2)]
            for i in range(N)
        ])
        # Convert to barycentric properly
        bary_best = np.zeros((N, 2))
        for i in range(N):
            px, py = best_points[i]
            v = py / (np.sqrt(3) / 2)
            u = px - v * 0.5
            bary_best[i] = [u, v]
        bary_best = _clamp_bary(bary_best)
        
        x_refined = _nelder_mead_optimize(bary_best.flatten(), maxiter=2000)
        bary_opt = _clamp_bary(x_refined.reshape(-1, 2))
        points = _bary_to_cart(bary_opt)
        update_best(points)
    
    if best_points is None:
        # Fallback
        bary_fallback = _make_initial_bary_grid(3, seed=42)
        best_points = _bary_to_cart(bary_fallback)
    
    return best_points
# EVOLVE-BLOCK-END
