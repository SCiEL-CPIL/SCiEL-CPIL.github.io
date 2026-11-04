"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 14

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize, differential_evolution

def _score(pts):
    """
    Compute the score: min triangle area / convex hull area.
    """
    if len(pts) < 3:
        return 0.0
    
    try:
        hull = ConvexHull(pts)
        hull_area = hull.volume
    except Exception:
        return 0.0
        
    if hull_area < 1e-12:
        return 0.0
        
    n = len(pts)
    min_area = float('inf')
    
    # Efficient vectorized computation
    for i in range(n - 2):
        pi = pts[i]
        for j in range(i + 1, n - 1):
            pj = pts[j]
            pk = pts[j+1:]
            
            v1x = pj[0] - pi[0]
            v1y = pj[1] - pi[1]
            v2x = pk[:, 0] - pi[0]
            v2y = pk[:, 1] - pi[1]
            
            crosses = np.abs(v1x * v2y - v1y * v2x)
            areas = 0.5 * crosses
            
            local_min = np.min(areas)
            if local_min < min_area:
                min_area = local_min
                
    return min_area / hull_area

def _soft_min_score(pts, temp=0.05):
    """
    Differentiable approximation of the score using soft-min.
    This allows gradient-based optimization.
    """
    if len(pts) < 3:
        return 0.0
    
    try:
        hull = ConvexHull(pts)
        hull_area = hull.volume
    except Exception:
        return -1e9
        
    if hull_area < 1e-12:
        return -1e9
        
    n = len(pts)
    areas = []
    
    # Compute all triangle areas
    for i in range(n - 2):
        pi = pts[i]
        for j in range(i + 1, n - 1):
            pj = pts[j]
            pk = pts[j+1:]
            
            v1x = pj[0] - pi[0]
            v1y = pj[1] - pi[1]
            v2x = pk[:, 0] - pi[0]
            v2y = pk[:, 1] - pi[1]
            
            crosses = np.abs(v1x * v2y - v1y * v2x)
            areas.extend(0.5 * crosses)
            
    areas = np.array(areas)
    
    # Soft min: -log(sum(exp(-area/temp))) / temp
    # To avoid overflow, subtract max
    min_a = np.min(areas)
    exp_areas = np.exp(-(areas - min_a) / temp)
    soft_min_val = min_a - temp * np.log(np.sum(exp_areas))
    
    return soft_min_val / hull_area

def _generate_regular_polygon(n):
    angles = np.linspace(0, 2*np.pi, n, endpoint=False)
    return np.column_stack([np.cos(angles), np.sin(angles)])

def _generate_two_ring_points(r, theta):
    """
    Generate 14 points: 7 on unit circle (outer), 7 on circle of radius r (inner).
    """
    angles_outer = np.linspace(0, 2*np.pi, 7, endpoint=False)
    angles_inner = np.linspace(0, 2*np.pi, 7, endpoint=False) + theta
    
    outer_pts = np.column_stack([np.cos(angles_outer), np.sin(angles_outer)])
    inner_pts = r * np.column_stack([np.cos(angles_inner), np.sin(angles_inner)])
    
    pts = np.vstack([outer_pts, inner_pts])
    return pts

def _optimize_general(deadline):
    """
    Optimize the general 14-point configuration using Differential Evolution
    on the 28 coordinates, followed by a local polish.
    """
    best_score = -float('inf')
    best_pts = None
    
    # Initial seeds
    seeds = [
        _generate_regular_polygon(14),
        _generate_two_ring_points(0.6, np.pi/7),
        _generate_two_ring_points(0.7, 0.0),
        _generate_two_ring_points(0.5, np.pi/14)
    ]
    
    # Add some random perturbations to regular polygon
    for _ in range(3):
        base = _generate_regular_polygon(14)
        noise = np.random.normal(0, 0.1, base.shape)
        seeds.append(base + noise)
        
    # Differential Evolution on 28D space
    # Bounds: keep points within a reasonable box, e.g., [-1.5, 1.5]
    bounds = [(-1.5, 1.5)] * (2 * N)
    
    def objective_flat(params):
        pts = params.reshape(N, 2)
        # Normalize to unit area? 
        # The score function handles normalization by hull area.
        # But DE works better if the scale is consistent.
        # We can normalize inside the objective to help DE?
        # Actually, the score is scale-invariant (ratio). 
        # So we can just return the negative soft-min score.
        # However, if points collapse, hull area -> 0, score -> -inf or error.
        # _soft_min_score returns -1e9 on error.
        return -_soft_min_score(pts, temp=0.05)
        
    try:
        # Use DE for global search
        res = differential_evolution(
            objective_flat, 
            bounds, 
            maxiter=30, 
            popsize=10, 
            tol=1e-5, 
            seed=42, 
            polish=False,
            init='sobol'
        )
        
        if res.fun < 1e8:
            cand_pts = res.x.reshape(N, 2)
            cand_score = _score(cand_pts)
            if cand_score > best_score:
                best_score = cand_score
                best_pts = cand_pts
    except Exception:
        pass
        
    # Also try optimizing the seeds directly with Nelder-Mead
    # This is often faster and more effective for local improvements
    for seed in seeds:
        if time.time() > deadline - 10.0:
            break
            
        def neg_soft_min_flat(params):
            pts = params.reshape(N, 2)
            return -_soft_min_score(pts, temp=0.05)
            
        try:
            res = minimize(
                neg_soft_min_flat, 
                seed.flatten(), 
                method='Nelder-Mead',
                options={'xatol': 1e-6, 'fatol': 1e-6, 'maxiter': 200, 'adaptive': True}
            )
            
            if res.fun < 1e8:
                cand_pts = res.x.reshape(N, 2)
                cand_score = _score(cand_pts)
                if cand_score > best_score:
                    best_score = cand_score
                    best_pts = cand_pts
        except Exception:
            pass
            
    # Final Polish: Coordinate Descent on the best point
    if best_pts is not None and time.time() < deadline - 2.0:
        best_pts = _polish_points(best_pts, deadline)
        final_score = _score(best_pts)
        if final_score > best_score:
            best_score = final_score
            # best_pts is already updated by _polish_points
            
    return best_pts, best_score

def _polish_points(pts, deadline):
    """
    Refine points using coordinate descent with Nelder-Mead on small neighborhoods.
    """
    current_pts = pts.copy()
    current_score = _score(current_pts)
    
    # Try small perturbations for each point
    for _ in range(10): # Limited iterations
        if time.time() > deadline - 2.0:
            break
            
        improved = False
        for i in range(N):
            if time.time() > deadline - 2.0:
                break
                
            # Define a function to optimize point i
            def obj_i(x):
                trial_pts = current_pts.copy()
                trial_pts[i] = x
                return -_score(trial_pts)
                
            try:
                res = minimize(
                    obj_i, 
                    current_pts[i], 
                    method='Nelder-Mead',
                    options={'xatol': 1e-6, 'fatol': 1e-6, 'maxiter': 50}
                )
                
                new_score = -res.fun
                if new_score > current_score + 1e-9:
                    current_pts[i] = res.x
                    current_score = new_score
                    improved = True
            except Exception:
                pass
                
        if not improved:
            break
            
    return current_pts

def construct():
    start_time = time.time()
    deadline = start_time + 140.0 # 10s safety margin
    
    # Run the general optimization
    best_pts, best_score = _optimize_general(deadline)
    
    # Fallback if optimization failed
    if best_pts is None:
        best_pts = _generate_regular_polygon(N)
        
    return best_pts

# EVOLVE-BLOCK-END
