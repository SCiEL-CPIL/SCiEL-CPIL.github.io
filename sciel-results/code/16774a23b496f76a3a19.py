"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 14

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
import cma

DEADLINE = 135.0  # Safety margin before 150s kill

def _hull_area(pts):
    """Compute convex hull area."""
    try:
        hull = ConvexHull(pts)
        return hull.volume
    except Exception:
        return 0.0

def _min_triangle_area_vec(points):
    """Vectorized minimum triangle area for N points."""
    n = len(points)
    min_area = float('inf')
    x = points[:, 0]
    y = points[:, 1]
    for i in range(n - 2):
        for j in range(i + 1, n - 1):
            dx_ki = x[j+1:] - x[i]
            dy_ki = y[j+1:] - y[i]
            cross = (x[j] - x[i]) * dy_ki - dx_ki * (y[j] - y[i])
            local_min = 0.5 * np.min(np.abs(cross))
            if local_min < min_area:
                min_area = local_min
                if min_area < 1e-14:
                    return min_area
    return min_area

def _score(pts):
    """Compute min triangle area / hull area."""
    hull_a = _hull_area(pts)
    if hull_a <= 0:
        return 0.0
    min_area = _min_triangle_area_vec(pts)
    return min_area / hull_a

def _neg_score(pts_flat):
    """Negative score for minimization."""
    pts = pts_flat.reshape(-1, 2)
    s = _score(pts)
    return -s

def _all_triangle_areas(points):
    """Return array of all triangle areas (N choose 3)."""
    n = len(points)
    x = points[:, 0]
    y = points[:, 1]
    areas_list = []
    for i in range(n - 2):
        for j in range(i + 1, n - 1):
            dx_ki = x[j+1:] - x[i]
            dy_ki = y[j+1:] - y[i]
            cross = (x[j] - x[i]) * dy_ki - dx_ki * (y[j] - y[i])
            areas_list.append(0.5 * np.abs(cross))
    if not areas_list:
        return np.array([0.0])
    return np.concatenate(areas_list)

def _normalize_unit_area(pts):
    """Scale points so convex hull area is 1."""
    hull_a = _hull_area(pts)
    if hull_a > 0:
        return pts / np.sqrt(hull_a)
    return pts

def _initial_regular_polygon(n):
    """Points on a regular n-gon inscribed in unit circle, scaled to unit area."""
    angles = 2 * np.pi * np.arange(n) / n
    pts = np.column_stack([np.cos(angles), np.sin(angles)])
    return _normalize_unit_area(pts)

def _simulated_annealing_step(pts, temp, rng):
    """
    Perform a simulated annealing step.
    Perturb the configuration based on critical triangles.
    """
    n = len(pts)
    areas = _all_triangle_areas(pts)
    min_area = np.min(areas)
    hull_area = _hull_area(pts)
    if hull_area <= 0 or min_area <= 0:
        return pts
    
    # Identify critical triangles (those with area close to min)
    threshold = min_area * 1.1
    critical_mask = areas < threshold
    critical_indices = np.where(critical_mask)[0]
    
    if len(critical_indices) == 0:
        return pts
    
    # Pick a random critical triangle
    idx = critical_indices[rng.randint(len(critical_indices))]
    
    # Map linear index to (i, j, k)
    count = 0
    i, j, k = 0, 0, 0
    for ii in range(n - 2):
        for jj in range(ii + 1, n - 1):
            for kk in range(jj + 1, n):
                if count == idx:
                    i, j, k = ii, jj, kk
                    break
                count += 1
            if i > 0: break
        if i > 0: break
    
    # Perturb the points of the critical triangle
    # Move them slightly away from each other to increase area
    p_i = pts[i].copy()
    p_j = pts[j].copy()
    p_k = pts[k].copy()
    
    # Calculate centroid
    centroid = (p_i + p_j + p_k) / 3.0
    
    # Move each point away from centroid
    scale = 0.01 * temp
    p_i += (p_i - centroid) * scale
    p_j += (p_j - centroid) * scale
    p_k += (p_k - centroid) * scale
    
    new_pts = pts.copy()
    new_pts[i] = p_i
    new_pts[j] = p_j
    new_pts[k] = p_k
    
    return new_pts

def _cmaes_optimize(x0, max_evals=5000, seed=42):
    """
    Optimize using CMA-ES.
    """
    start_time = time.time()
    options = {
        'maxiter': 500,
        'verbose': -9,
        'seed': seed,
        'bounds': [-5, 5],  # Loose bounds, we normalize anyway
        'CMA_stopsamefit': 100,
        'CMA_verb_log': 0
    }
    
    def objective(x):
        if time.time() - start_time > DEADLINE - 5:
            return 1e10
        pts = x.reshape(-1, 2)
        s = _score(pts)
        return -s
    
    es = cma.CMAEvolutionStrategy(x0, 0.5, options)
    while not es.stop():
        if time.time() - start_time > DEADLINE - 5:
            break
        solutions = es.ask()
        fitnesses = [objective(sol) for sol in solutions]
        es.tell(solutions, fitnesses)
    
    best_x = es.result.xbest
    return best_x

def construct():
    start_time = time.time()
    
    # Initial configurations
    inits = [
        _initial_regular_polygon(N),
    ]
    
    # Add some random initializations
    rng = np.random.RandomState(42)
    for _ in range(5):
        base = _initial_regular_polygon(N)
        noise = rng.normal(0, 0.1, size=base.shape)
        perturbed = base + noise
        perturbed = _normalize_unit_area(perturbed)
        inits.append(perturbed)
    
    best_pts = None
    best_score = 0.0
    
    # Phase 1: CMA-ES from multiple starts
    for i, init in enumerate(inits):
        if time.time() - start_time > DEADLINE - 20:
            break
        
        x0 = init.flatten()
        try:
            # Limit evaluations based on remaining time
            remaining = DEADLINE - (time.time() - start_time)
            max_evals = int(remaining * 50)  # Rough estimate
            max_evals = max(500, min(max_evals, 5000))
            
            best_x = _cmaes_optimize(x0, max_evals=max_evals, seed=42+i)
            if best_x is not None:
                cand = best_x.reshape(-1, 2)
                s = _score(cand)
                if s > best_score:
                    best_score = s
                    best_pts = cand
        except Exception:
            pass
    
    # Phase 2: Simulated Annealing + Local Polish
    if best_pts is not None and time.time() - start_time < DEADLINE - 10:
        current_pts = best_pts.copy()
        current_score = best_score
        temp = 0.1
        cooling = 0.99
        rng = np.random.RandomState(123)
        
        sa_start = time.time()
        max_sa_time = DEADLINE - (time.time() - start_time) - 15
        
        while time.time() - sa_start < max_sa_time:
            new_pts = _simulated_annealing_step(current_pts, temp, rng)
            new_score = _score(new_pts)
            
            # Accept if better or with probability exp(-(new-old)/T)
            if new_score > current_score:
                current_pts = new_pts
                current_score = new_score
                if current_score > best_score:
                    best_score = current_score
                    best_pts = new_pts.copy()
            else:
                delta = new_score - current_score
                if delta < 0:
                    prob = np.exp(delta / temp)
                    if rng.random() < prob:
                        current_pts = new_pts
                        current_score = new_score
            
            temp *= cooling
            
            # Periodic local polish
            if rng.random() < 0.1 and time.time() - start_time < DEADLINE - 15:
                try:
                    x0 = current_pts.flatten()
                    res = minimize(_neg_score, x0, method='Nelder-Mead',
                                  options={'maxiter': 200, 'xatol': 1e-6, 'fatol': 1e-8,
                                          'adaptive': True})
                    if res.x is not None:
                        cand = res.x.reshape(-1, 2)
                        s = _score(cand)
                        if s > current_score:
                            current_pts = cand
                            current_score = s
                            if s > best_score:
                                best_score = s
                                best_pts = cand.copy()
                except Exception:
                    pass
    
    # Final polish
    if best_pts is not None and time.time() - start_time < DEADLINE - 5:
        try:
            x0 = best_pts.flatten()
            res = minimize(_neg_score, x0, method='Nelder-Mead',
                          options={'maxiter': 1000, 'xatol': 1e-9, 'fatol': 1e-12,
                                  'adaptive': True})
            if res.x is not None:
                cand = res.x.reshape(-1, 2)
                s = _score(cand)
                if s > best_score:
                    best_score = s
                    best_pts = cand
        except Exception:
            pass
    
    if best_pts is None:
        best_pts = _initial_regular_polygon(N)
    
    return best_pts
# EVOLVE-BLOCK-END
