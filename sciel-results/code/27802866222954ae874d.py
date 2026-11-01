import numpy as np
import scipy.spatial as spat
import time
import itertools
from scipy.optimize import minimize, differential_evolution

N = 14
DEADLINE = 140  # safety margin before 150s kill

def _calc_min_tri_area(points):
    """
    Compute the minimum triangle area over all triples.
    Vectorized for speed.
    """
    n = len(points)
    if n < 3:
        return 0.0
    
    # Create index arrays
    i_idx, j_idx, k_idx = np.indices((n, n, n))
    # Only keep unique triples i < j < k
    mask = (i_idx < j_idx) & (j_idx < k_idx)
    i_idx = i_idx[mask]
    j_idx = j_idx[mask]
    k_idx = k_idx[mask]
    
    # Get points
    p_i = points[i_idx]
    p_j = points[j_idx]
    p_k = points[k_idx]
    
    # Calculate cross product area
    v1 = p_j - p_i
    v2 = p_k - p_i
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    
    return np.min(areas)

def _hull_area(points):
    """Compute convex hull area."""
    try:
        hull = spat.ConvexHull(points)
        return hull.volume
    except Exception:
        return 1e-10

def _score(points):
    """Score = min triangle area / hull area. Higher is better."""
    hull_a = _hull_area(points)
    if hull_a < 1e-10:
        return 0.0
    min_tri = _calc_min_tri_area(points)
    return min_tri / hull_a

def _soft_min_score(points, beta=50.0):
    """
    Compute a smooth approximation of the minimum triangle area divided by hull area.
    Uses log-sum-exp for the soft minimum.
    """
    n = len(points)
    if n < 3:
        return 0.0
    
    # Create index arrays for unique triples i < j < k
    i_idx, j_idx, k_idx = np.indices((n, n, n))
    mask = (i_idx < j_idx) & (j_idx < k_idx)
    i_idx = i_idx[mask]
    j_idx = j_idx[mask]
    k_idx = k_idx[mask]
    
    p_i = points[i_idx]
    p_j = points[j_idx]
    p_k = points[k_idx]
    
    v1 = p_j - p_i
    v2 = p_k - p_i
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    
    # Soft min using log-sum-exp trick
    z = -beta * areas
    z_max = np.max(z)
    if not np.isfinite(z_max):
        return 0.0
        
    exp_z_minus_max = np.exp(z - z_max)
    sum_exp = np.sum(exp_z_minus_max)
    if sum_exp == 0:
        return 0.0
        
    log_sum_exp = z_max + np.log(sum_exp)
    soft_min_area = -1.0 / beta * log_sum_exp
    
    hull_a = _hull_area(points)
    if hull_a < 1e-10:
        return 0.0
        
    return soft_min_area / hull_a

def _regular_polygon(n, radius=1.0, center=(0.0, 0.0)):
    """Generate n points on a regular polygon."""
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    xs = center[0] + radius * np.cos(angles)
    ys = center[1] + radius * np.sin(angles)
    return np.column_stack([xs, ys])

def construct():
    start_time = time.time()
    
    best_points = None
    best_score = -np.inf
    
    def update_best(pts, score):
        nonlocal best_points, best_score
        if score > best_score:
            best_score = score
            best_points = pts.copy()
    
    # Initial configurations
    initial_configs = []
    
    # 1. Regular polygon
    initial_configs.append(_regular_polygon(N))
    
    # 2. Two nested regular heptagons with different parameters
    for ratio in [0.5, 0.6, 0.7, 0.8]:
        for rot in [0.0, np.pi/N, 2*np.pi/N, 3*np.pi/N]:
            angles_inner = np.linspace(rot, rot + 2*np.pi, 7, endpoint=False)
            angles_outer = np.linspace(0, 2*np.pi, 7, endpoint=False)
            xs = np.concatenate([ratio * np.cos(angles_inner), np.cos(angles_outer)])
            ys = np.concatenate([ratio * np.sin(angles_inner), np.sin(angles_outer)])
            initial_configs.append(np.column_stack([xs, ys]))
    
    # 3. Random points in a circle
    rng_init = np.random.RandomState(42)
    for _ in range(10):
        r = np.sqrt(rng_init.random(N))
        theta = rng_init.uniform(0, 2 * np.pi, N)
        pts = np.column_stack([r * np.cos(theta), r * np.sin(theta)])
        initial_configs.append(pts)
    
    # Evaluate initial configurations
    for init_pts in initial_configs:
        elapsed = time.time() - start_time
        if elapsed > DEADLINE - 30:
            break
        
        hull_a = _hull_area(init_pts)
        if hull_a < 1e-10:
            continue
        init_pts = init_pts / np.sqrt(hull_a)
        score = _score(init_pts)
        update_best(init_pts, score)
    
    # Phase 1: Differential Evolution (global search)
    elapsed = time.time() - start_time
    if elapsed < DEADLINE - 60:
        # Use the best initial config as the center for DE
        center_pts = best_points if best_points is not None else initial_configs[0]
        center_pts = center_pts / np.sqrt(_hull_area(center_pts))
        
        def de_objective(x):
            if time.time() - start_time > DEADLINE - 5:
                return 1e10  # Return bad score to stop
            
            pts = x.reshape(-1, 2)
            hull_a = _hull_area(pts)
            if hull_a < 1e-10:
                return 1e10
            
            # Normalize
            pts = pts / np.sqrt(hull_a)
            
            # Use soft min for smoothness
            return -_soft_min_score(pts, beta=30.0)
        
        # Bounds: roughly within [-1, 1] after normalization
        bounds = [(-1.5, 1.5)] * (2 * N)
        
        try:
            de_result = differential_evolution(
                de_objective,
                bounds,
                seed=42,
                maxiter=50,
                tol=1e-6,
                popsize=10,
                init='sobol',
                updating='immediate'
            )
            
            if de_result.success or de_result.fun < 1e9:
                de_pts = de_result.x.reshape(-1, 2)
                hull_a = _hull_area(de_pts)
                if hull_a > 1e-10:
                    de_pts = de_pts / np.sqrt(hull_a)
                    de_score = _score(de_pts)
                    update_best(de_pts, de_score)
                    
                    # Also try soft min result
                    soft_score = _soft_min_score(de_pts, beta=30.0)
                    # The actual score is what we care about
                    if de_score > best_score:
                        best_score = de_score
                        best_points = de_pts.copy()
        except Exception as e:
            pass
    
    # Phase 2: L-BFGS-B local refinement from best point
    elapsed = time.time() - start_time
    if elapsed < DEADLINE - 10 and best_points is not None:
        current_pts = best_points.copy()
        current_pts = current_pts / np.sqrt(_hull_area(current_pts))
        current_score = _score(current_pts)
        
        def lbfgs_objective(x):
            if time.time() - start_time > DEADLINE - 5:
                return 1e10
            
            pts = x.reshape(-1, 2)
            hull_a = _hull_area(pts)
            if hull_a < 1e-10:
                return 1e10
            
            pts = pts / np.sqrt(hull_a)
            return -_soft_min_score(pts, beta=50.0)
        
        try:
            res = minimize(
                lbfgs_objective,
                current_pts.flatten(),
                method='L-BFGS-B',
                options={'maxiter': 1000, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            
            if res.fun < 1e9:
                res_pts = res.x.reshape(-1, 2)
                hull_a = _hull_area(res_pts)
                if hull_a > 1e-10:
                    res_pts = res_pts / np.sqrt(hull_a)
                    res_score = _score(res_pts)
                    update_best(res_pts, res_score)
        except Exception as e:
            pass
    
    # Phase 3: Simulated Annealing fine-tuning
    elapsed = time.time() - start_time
    if elapsed < DEADLINE - 5 and best_points is not None:
        current_pts = best_points.copy()
        current_pts = current_pts / np.sqrt(_hull_area(current_pts))
        current_score = _score(current_pts)
        
        T = 0.01
        cooling_rate = 0.995
        rng_sa = np.random.RandomState(123)
        
        for i in range(2000):
            elapsed = time.time() - start_time
            if elapsed > DEADLINE - 3:
                break
            
            new_pts = current_pts.copy()
            idx = rng_sa.randint(N)
            noise = rng_sa.normal(0, 0.005, 2)
            new_pts[idx] += noise
            
            new_hull_a = _hull_area(new_pts)
            if new_hull_a < 1e-10:
                T *= cooling_rate
                continue
            
            new_pts = new_pts / np.sqrt(new_hull_a)
            new_score = _score(new_pts)
            
            if new_score > current_score:
                current_pts = new_pts
                current_score = new_score
                update_best(current_pts, current_score)
            else:
                delta = new_score - current_score
                if T > 1e-8:
                    accept_prob = np.exp(delta / T)
                    if rng_sa.random() < accept_prob:
                        current_pts = new_pts
                        current_score = new_score
            
            T *= cooling_rate
    
    if best_points is None:
        best_points = initial_configs[0]
        
    return np.array(best_points)
