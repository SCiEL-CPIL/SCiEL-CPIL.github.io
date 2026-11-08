import time
import numpy as np
from scipy.optimize import minimize, differential_evolution
from scipy.spatial import ConvexHull

N = 14
DEADLINE = 145.0  # Safety margin before 150s kill

def _hull_area(pts):
    """Compute convex hull area."""
    try:
        if len(pts) < 3:
            return 0.0
        # Check for collinearity or degeneracy
        if np.allclose(pts[:, 0], pts[0, 0]) or np.allclose(pts[:, 1], pts[0, 1]):
            return 0.0
        hull = ConvexHull(pts)
        return hull.volume
    except Exception:
        return 0.0

def _all_triangle_areas(points):
    """
    Return array of all triangle areas (N choose 3).
    Vectorized calculation for speed.
    """
    n = len(points)
    x = points[:, 0]
    y = points[:, 1]
    
    # Number of triangles
    num_tris = n * (n - 1) * (n - 2) // 6
    areas = np.empty(num_tris)
    idx = 0
    for i in range(n - 2):
        for j in range(i + 1, n - 1):
            # Vectorize over k > j
            dx_ki = x[j+1:] - x[i]
            dy_ki = y[j+1:] - y[i]
            # Cross product (x_j - x_i) * (y_k - y_i) - (x_k - x_i) * (y_j - y_i)
            # Wait, the formula for area of triangle (i, j, k) is:
            # 0.5 * | (x_j - x_i)(y_k - y_i) - (x_k - x_i)(y_j - y_i) |
            # My vectorization:
            # For fixed i, j:
            # x[j] - x[i] is constant for all k
            # y[j] - y[i] is constant for all k
            # x[j+1:] - x[i] is dx_ki
            # y[j+1:] - y[i] is dy_ki
            # cross = (x_j - x_i) * dy_ki - dx_ki * (y_j - y_i)
            cross = (x[j] - x[i]) * dy_ki - dx_ki * (y[j] - y[i])
            tri_areas = 0.5 * np.abs(cross)
            areas[idx:idx+len(tri_areas)] = tri_areas
            idx += len(tri_areas)
            
    return areas

def _min_triangle_area_vec(points):
    """Compute the minimum triangle area."""
    areas = _all_triangle_areas(points)
    if len(areas) == 0:
        return 0.0
    return np.min(areas)

def _score(pts):
    """Compute min triangle area / hull area."""
    hull_a = _hull_area(pts)
    if hull_a <= 1e-10:
        return 0.0
    min_area = _min_triangle_area_vec(pts)
    return min_area / hull_a

def _soft_min(areas, beta):
    """
    Compute the soft minimum of an array of values.
    soft_min(x) = -1/beta * log(sum(exp(-beta*x)))
    """
    if len(areas) == 0:
        return 0.0
    min_val = np.min(areas)
    if min_val < 1e-12:
        return 0.0
    diff = areas - min_val
    # Prevent overflow in exp
    exp_terms = np.exp(-beta * diff)
    log_sum = np.log(np.sum(exp_terms))
    return min_val - log_sum / beta

def _objective_smooth(pts_flat, beta):
    """
    Objective function to minimize: hull_area / soft_min(areas, beta).
    """
    points = pts_flat.reshape(-1, 2)
    if np.any(~np.isfinite(points)):
        return 1e10
        
    areas = _all_triangle_areas(points)
    if len(areas) == 0 or np.min(areas) < 1e-12:
        return 1e10
        
    hull_area = _hull_area(points)
    if hull_area < 1e-10:
        return 1e10
        
    sm = _soft_min(areas, beta)
    if sm < 1e-12:
        return 1e10
        
    return hull_area / sm

def _objective_exact(pts_flat):
    """
    Objective function for final polishing: hull_area / min_area.
    """
    points = pts_flat.reshape(-1, 2)
    if np.any(~np.isfinite(points)):
        return 1e10
        
    areas = _all_triangle_areas(points)
    if len(areas) == 0 or np.min(areas) < 1e-12:
        return 1e10
        
    hull_area = _hull_area(points)
    if hull_area < 1e-10:
        return 1e10
        
    min_a = np.min(areas)
    return hull_area / min_a

def _normalize_unit_area(pts):
    """Scale points so convex hull area is 1."""
    hull_a = _hull_area(pts)
    if hull_a > 0:
        return pts / np.sqrt(hull_a)
    return pts

def _generate_initial_configs():
    """Generate diverse initial configurations."""
    configs = []
    
    # 1. Regular 14-gon
    angles = 2 * np.pi * np.arange(N) / N
    p_reg = np.column_stack([np.cos(angles), np.sin(angles)])
    configs.append(p_reg)
    
    # 2. Regular 14-gon with noise
    rng = np.random.RandomState(42)
    for seed in range(5):
        noise = rng.normal(0, 0.05, size=(N, 2))
        p_noisy = p_reg + noise
        configs.append(p_noisy)
        
    # 3. Two concentric rings (7 and 7)
    ang1 = 2 * np.pi * np.arange(7) / 7
    r1 = 0.6
    p1 = np.column_stack([r1 * np.cos(ang1), r1 * np.sin(ang1)])
    
    ang2 = 2 * np.pi * np.arange(7) / 7 + np.pi/7
    r2 = 1.0
    p2 = np.column_stack([r2 * np.cos(ang2), r2 * np.sin(ang2)])
    
    p_rings = np.vstack([p1, p2])
    configs.append(p_rings)
    
    # 4. Three concentric rings (5, 5, 4)
    ang_a = 2 * np.pi * np.arange(5) / 5
    p_a = np.column_stack([0.4 * np.cos(ang_a), 0.4 * np.sin(ang_a)])
    ang_b = 2 * np.pi * np.arange(5) / 5 + np.pi/5
    p_b = np.column_stack([0.7 * np.cos(ang_b), 0.7 * np.sin(ang_b)])
    ang_c = 2 * np.pi * np.arange(4) / 4 + np.pi/4
    p_c = np.column_stack([1.0 * np.cos(ang_c), 1.0 * np.sin(ang_c)])
    p_3rings = np.vstack([p_a, p_b, p_c])
    configs.append(p_3rings)
    
    # 5. Distorted regular polygon (3-fold symmetry)
    mod = 1.0 + 0.1 * np.cos(3 * angles)
    p_mod = np.column_stack([mod * np.cos(angles), mod * np.sin(angles)])
    configs.append(p_mod)
    
    # 6. Elliptical distribution
    t = np.linspace(0, 2*np.pi, N, endpoint=False)
    p_ell = np.column_stack([1.2 * np.cos(t), 0.8 * np.sin(t)])
    configs.append(p_ell)
    
    # 7. Random points in disk
    for seed in [1, 2]:
        rng2 = np.random.RandomState(seed)
        r = rng2.uniform(0.2, 1.0, N)
        th = rng2.uniform(0, 2*np.pi, N)
        p_rand = np.column_stack([r * np.cos(th), r * np.sin(th)])
        configs.append(p_rand)
    
    # Normalize all to unit area for consistency
    normalized_configs = [_normalize_unit_area(c) for c in configs]
    return normalized_configs

def _polish_candidate(x0, start_time, deadline):
    """
    Polish a single candidate using beta annealing and Nelder-Mead.
    """
    current_best_x = x0.copy()
    current_best_obj = _objective_exact(x0)
    
    # Phase 1: Beta Annealing with L-BFGS-B
    # Start with low beta (smooth) and increase
    betas = [20.0, 50.0, 100.0, 200.0, 400.0]
    
    for beta in betas:
        if time.time() - start_time > deadline - 10:
            break
            
        def obj_func(x, b=beta):
            return _objective_smooth(x, b)
            
        try:
            res = minimize(obj_func, current_best_x, method='L-BFGS-B',
                           options={'maxiter': 50, 'ftol': 1e-12, 'gtol': 1e-10})
            
            if res.success and res.fun < current_best_obj:
                current_best_obj = res.fun
                current_best_x = res.x
        except Exception:
            pass
    
    # Phase 2: Exact Objective with Nelder-Mead
    if time.time() - start_time < deadline - 5:
        def obj_exact(x):
            return _objective_exact(x)
            
        try:
            res = minimize(obj_exact, current_best_x, method='Nelder-Mead',
                           options={'maxiter': 100, 'xatol': 1e-10, 'fatol': 1e-12})
            
            if res.fun < current_best_obj:
                current_best_obj = res.fun
                current_best_x = res.x
        except Exception:
            pass
            
    return current_best_x

def construct():
    start_time = time.time()
    
    # Get initial configs
    inits = _generate_initial_configs()
    
    # Evaluate initial scores
    best_score = -np.inf
    best_pts = inits[0]
    
    for init in inits:
        s = _score(init)
        if s > best_score:
            best_score = s
            best_pts = init
            
    # Sort inits by score (best first)
    init_scores = [_score(c) for c in inits]
    sorted_indices = np.argsort(init_scores)[::-1]
    inits = [inits[i] for i in sorted_indices]
    
    # Phase 1: Differential Evolution in Cartesian coordinates
    # This explores the global landscape
    if time.time() - start_time < DEADLINE - 60:
        try:
            bounds = [(-2, 2), (-2, 2)] * N
            
            def objective_de(params):
                if time.time() > DEADLINE - 40:
                    raise TimeoutError()
                pts = params.reshape(N, 2)
                return -_score(pts)
            
            # Use moderate population
            result = differential_evolution(
                objective_de,
                bounds,
                maxiter=30,
                popsize=12,
                tol=1e-4,
                seed=42,
                polish=False
            )
            
            if result.fun < 0:
                points_de = result.x.reshape(N, 2)
                score_de = _score(points_de)
                if score_de > best_score:
                    best_score = score_de
                    best_pts = points_de
        except Exception:
            pass
    
    # Phase 2: Polish top candidates
    # Candidates: Best from DE, Best from Inits, and a few other good inits
    candidates = [best_pts]
    for init in inits[:4]:
        candidates.append(init)
    
    # If DE found a good point, add it
    # We already have best_pts which might be from DE or init
    
    for cand in candidates:
        if time.time() - start_time > DEADLINE - 20:
            break
        
        x0 = cand.flatten()
        
        # Polish
        polished_x = _polish_candidate(x0, start_time, DEADLINE)
        
        # Normalize and score
        pts = polished_x.reshape(N, 2)
        pts = _normalize_unit_area(pts)
        s = _score(pts)
        
        if s > best_score:
            best_score = s
            best_pts = pts
    
    # Phase 3: Final high-precision polish on the best solution
    if time.time() - start_time < DEADLINE - 5:
        polished_x = _polish_candidate(best_pts.flatten(), start_time, DEADLINE)
        pts = polished_x.reshape(N, 2)
        pts = _normalize_unit_area(pts)
        s = _score(pts)
        if s > best_score:
            best_score = s
            best_pts = pts
    
    # Final Nelder-Mead polish with very tight tolerances
    if time.time() - start_time < DEADLINE - 2:
        def obj_final(x):
            return _objective_exact(x)
        
        try:
            res = minimize(obj_final, best_pts.flatten(), method='Nelder-Mead',
                           options={'maxiter': 200, 'xatol': 1e-12, 'fatol': 1e-14})
            if res.fun < _objective_exact(best_pts.flatten()):
                best_pts = res.x.reshape(N, 2)
        except Exception:
            pass
        
        best_pts = _normalize_unit_area(best_pts)
    
    return best_pts

# The entry point must be construct()
if __name__ == "__main__":
    pts = construct()
    print(_score(pts))
