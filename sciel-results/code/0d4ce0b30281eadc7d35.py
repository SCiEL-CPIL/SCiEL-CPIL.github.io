import numpy as np
import scipy.optimize as opt
import scipy.spatial as spat
import time

N = 14

def _hull_area(points):
    try:
        hull = spat.ConvexHull(points)
        return hull.volume
    except Exception:
        return 1.0

def _score(points):
    """Exact score: min triangle area / hull area."""
    n = len(points)
    if n < 3:
        return 0.0
    hull_area = _hull_area(points)
    if hull_area < 1e-12:
        return 0.0
    
    # Compute all triangle areas
    i_idx, j_idx, k_idx = [], [], []
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                i_idx.append(i)
                j_idx.append(j)
                k_idx.append(k)
    
    i_idx = np.array(i_idx)
    j_idx = np.array(j_idx)
    k_idx = np.array(k_idx)
    
    pi = points[i_idx]
    pj = points[j_idx]
    pk = points[k_idx]
    
    v1 = pj - pi
    v2 = pk - pi
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    
    min_area = np.min(areas)
    return min_area / hull_area

def _smooth_obj(params, temp):
    """
    Objective to minimize: -logsumexp(-areas/temp) / hull_area
    As temp -> 0, this approaches -min(areas) / hull_area
    """
    points = params.reshape(N, 2)
    
    # Quick check for degeneracy
    if np.any(np.isinf(points)) or np.any(np.isnan(points)):
        return 1e10
        
    hull_area = _hull_area(points)
    if hull_area < 1e-12:
        return 1e10
        
    # Compute areas
    i_idx, j_idx, k_idx = [], [], []
    for i in range(N):
        for j in range(i + 1, N):
            for k in range(j + 1, N):
                i_idx.append(i)
                j_idx.append(j)
                k_idx.append(k)
                
    i_idx = np.array(i_idx)
    j_idx = np.array(j_idx)
    k_idx = np.array(k_idx)
    
    pi = points[i_idx]
    pj = points[j_idx]
    pk = points[k_idx]
    
    v1 = pj - pi
    v2 = pk - pi
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    
    # Log-sum-exp for stability
    # We want - (1/temp) * log(sum(exp(-areas/temp)))
    # Let z = -areas / temp
    # logsumexp(z) = max_z + log(sum(exp(z - max_z)))
    # Objective = - (1/temp) * logsumexp(z) / hull_area
    # Wait, standard softmin is -T log(sum exp(-x/T)).
    # We want to maximize min_area / hull_area.
    # So minimize - (softmin(areas) / hull_area).
    
    z = -areas / temp
    max_z = np.max(z)
    if max_z > 700: # overflow protection, though areas are bounded
        log_sum_exp = max_z + np.log(np.sum(np.exp(z - max_z)))
    else:
        log_sum_exp = max_z + np.log(np.sum(np.exp(z - max_z)))
        
    soft_min = -temp * log_sum_exp
    val = soft_min / hull_area
    
    return -val

def construct():
    start_time = time.time()
    deadline = start_time + 145.0 # Safety margin
    
    best_points = None
    best_score = 0.0
    
    # 1. Generate multiple initial candidates
    candidates = []
    
    # A: Regular 14-gon
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    p1 = np.column_stack((np.cos(angles), np.sin(angles)))
    candidates.append(p1)
    
    # B: Two nested heptagons (Optimized geometry from inspirations)
    # Inner ring: 7 pts, Outer ring: 7 pts
    # Optimal ratio often around 0.5-0.6, rotation offset pi/7
    for ratio in [0.5, 0.6, 0.7]:
        for offset in [0, np.pi/7, np.pi/14]:
            angles_in = np.linspace(0, 2 * np.pi, 7, endpoint=False) + offset
            angles_out = np.linspace(0, 2 * np.pi, 7, endpoint=False)
            
            x_in = ratio * np.cos(angles_in)
            y_in = ratio * np.sin(angles_in)
            x_out = np.cos(angles_out)
            y_out = np.sin(angles_out)
            
            p = np.vstack([
                np.column_stack((x_in, y_in)),
                np.column_stack((x_out, y_out))
            ])
            candidates.append(p)
            
    # C: Random points in unit circle
    np.random.seed(42)
    for _ in range(5):
        r = np.sqrt(np.random.rand(N))
        theta = np.random.rand(N) * 2 * np.pi
        p = np.column_stack((r * np.cos(theta), r * np.sin(theta)))
        candidates.append(p)
        
    # 2. Evaluate candidates and pick best few for refinement
    scored_candidates = []
    for p in candidates:
        s = _score(p)
        if s > best_score:
            best_score = s
            best_points = p.copy()
        scored_candidates.append((s, p))
        
    # Sort by score, take top 3 for optimization
    scored_candidates.sort(key=lambda x: x[0], reverse=True)
    top_candidates = [p for _, p in scored_candidates[:3]]
    
    # 3. Smooth optimization phase
    # We will use L-BFGS-B with a decreasing temperature schedule
    # Start with high temp (smooth), end with low temp (sharp)
    
    temperatures = [0.1, 0.05, 0.02, 0.01, 0.005]
    
    for init_p in top_candidates:
        if time.time() > deadline - 30:
            break
            
        current_params = init_p.flatten()
        
        for temp in temperatures:
            if time.time() > deadline - 10:
                break
                
            try:
                # L-BFGS-B
                # No bounds, but we can add loose bounds to prevent points flying away
                # Actually, scale invariance means we can fix one point or constrain centroid to 0?
                # Simpler: just let it run. If it diverges, score drops, we keep best.
                
                result = opt.minimize(
                    _smooth_obj,
                    current_params,
                    args=(temp,),
                    method='L-BFGS-B',
                    options={'maxiter': 100, 'ftol': 1e-10, 'gtol': 1e-8}
                )
                
                if result.fun < _smooth_obj(current_params, temp):
                    current_params = result.x
                    
            except Exception:
                pass
                
        # Evaluate final result of this chain
        final_p = current_params.reshape(N, 2)
        final_s = _score(final_p)
        
        if final_s > best_score:
            best_score = final_s
            best_points = final_p.copy()
            
    # 4. Nelder-Mead Polish on the best point
    # NM is derivative-free and good for final local search on non-smooth functions
    if time.time() < deadline - 5 and best_points is not None:
        try:
            # Use the exact score (negative) as objective for NM
            def nm_obj(params):
                p = params.reshape(N, 2)
                return -_score(p)
                
            # Initial simplex: perturb best_points
            # NM requires initial simplex of size N+1
            # We can generate it by perturbing each dimension
            x0 = best_points.flatten()
            
            # Construct initial simplex
            simplex = [x0]
            step = 0.01
            for i in range(N * 2):
                x_pert = x0.copy()
                x_pert[i] += step
                simplex.append(x_pert)
                
            result = opt.minimize(
                nm_obj,
                x0,
                method='Nelder-Mead',
                options={
                    'xatol': 1e-8, 
                    'fatol': 1e-10, 
                    'maxiter': 500, 
                    'initial_simplex': simplex
                }
            )
            
            if result.fun < -best_score: # i.e., score improved
                nm_p = result.x.reshape(N, 2)
                nm_s = _score(nm_p)
                if nm_s > best_score:
                    best_score = nm_s
                    best_points = nm_p.copy()
                    
        except Exception:
            pass
            
    return best_points
