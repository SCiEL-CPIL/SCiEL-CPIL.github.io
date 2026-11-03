"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 14

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull

def _min_triangle_area(points):
    """
    Compute the minimum area of any triangle formed by 3 points.
    Vectorized for speed where possible, but N=14 is small enough for simple loops.
    """
    min_area = float('inf')
    x = points[:, 0]
    y = points[:, 1]
    
    # Iterate over all pairs (i, j) and compute cross product with all k > j
    for i in range(N):
        for j in range(i + 1, N):
            if j + 1 >= N:
                continue
            
            dx_ji = x[j] - x[i]
            dy_ji = y[j] - y[i]
            
            # Vectors from i to all k > j
            dx_ki = x[j+1:] - x[i]
            dy_ki = y[j+1:] - y[i]
            
            # Cross product magnitude gives 2 * area
            cross = dx_ji * dy_ki - dx_ki * dy_ji
            areas = 0.5 * np.abs(cross)
            
            if len(areas) > 0:
                local_min = np.min(areas)
                if local_min < min_area:
                    min_area = local_min
                    
    return min_area

def _hull_area(points):
    """
    Compute the area of the convex hull.
    """
    try:
        hull = ConvexHull(points)
        return hull.volume
    except Exception:
        return 0.0

def _score(points):
    """
    Compute the minimum triangle area divided by the convex hull area.
    """
    hull_area = _hull_area(points)
    if hull_area < 1e-10:
        return 0.0
        
    min_area = _min_triangle_area(points)
    return min_area / hull_area

def _soft_min_obj(points_flat, beta=100.0):
    """
    Smooth objective function.
    Minimizes - (log-sum-exp of negative areas) / hull_area.
    Using a large beta makes this a smooth approximation of -min(area)/hull_area.
    """
    points = points_flat.reshape(N, 2)
    
    if np.any(np.isnan(points)) or np.any(np.isinf(points)):
        return 1e10
        
    x = points[:, 0]
    y = points[:, 1]
    
    # Collect all triangle areas
    areas = []
    for i in range(N):
        for j in range(i + 1, N):
            if j + 1 >= N:
                continue
            dx_ji = x[j] - x[i]
            dy_ji = y[j] - y[i]
            dx_ki = x[j+1:] - x[i]
            dy_ki = y[j+1:] - y[i]
            cross = dx_ji * dy_ki - dx_ki * dy_ji
            a = 0.5 * np.abs(cross)
            areas.append(a)
            
    if not areas:
        return 1e10
        
    areas_arr = np.concatenate(areas)
    
    # Log-sum-exp trick for stability: sum(exp(x - max(x)))
    max_a = np.max(areas_arr)
    if max_a < 1e-10:
        return 1e10
        
    # Soft min is approximated by -log-sum-exp(-beta * areas) / beta
    # We want to maximize min(areas), so we minimize -min(areas).
    # -min(areas) ~ - (1/beta) * log(sum(exp(-beta * areas)))
    # To avoid underflow, we can compute log(sum(exp(-beta * (areas - max_a)))) + beta * max_a
    shifted = areas_arr - max_a
    log_sum = np.log(np.sum(np.exp(-beta * shifted)))
    soft_min_val = (max_a - log_sum / beta) # This is approx min
    # Actually, soft_min = (1/beta) * log(sum(exp(beta * areas)))
    # Let's use the standard soft min formula:
    # S(x) = (1/beta) * log(sum(exp(beta * x_i)))
    # We want to maximize S(areas). So minimize -S(areas).
    
    # Recalculate properly
    # To maximize min, we can use soft-min of areas.
    # soft_min(areas) = (1/beta) * log(sum(exp(beta * areas)))
    # This is a lower bound on min? No, soft-min is a lower bound on min?
    # Wait. max is log-sum-exp. min is -log-sum-exp(-x).
    # soft_min(x) = - (1/beta) * log(sum(exp(-beta * x)))
    
    # Let's stick to the robust version:
    # We want to maximize min(area).
    # Approximation: (1/beta) * log(sum(exp(beta * area))) is a smooth approximation of max? No.
    # max(x) = lim beta->inf (1/beta) log(sum exp(beta x)).
    # min(x) = -lim beta->inf (1/beta) log(sum exp(-beta x)).
    
    # So soft_min(areas) = - (1/beta) * log(sum(exp(-beta * areas)))
    
    # We want to maximize soft_min(areas) / hull_area.
    # So minimize -soft_min(areas) / hull_area.
    
    hull_area = _hull_area(points)
    if hull_area < 1e-10:
        return 1e10
        
    # Compute soft min
    # To avoid overflow/underflow in exp:
    # sum(exp(-beta * a)) = exp(-beta * max_a) * sum(exp(-beta * (a - max_a)))
    # log(sum(exp(-beta * a))) = -beta * max_a + log(sum(exp(-beta * (a - max_a))))
    
    # shifted = areas - max_a
    # log_sum_exp_neg = -beta * max_a + np.log(np.sum(np.exp(-beta * shifted)))
    # soft_min = - (1/beta) * log_sum_exp_neg
    
    # Let's just use the raw values if N is small enough and beta is moderate.
    # N=14, ~364 triangles.
    # If areas are ~0.1, beta=100, exp(-10) is small. exp(0) is 1.
    # It should be stable enough with the shift.
    
    log_sum_exp_neg = -beta * max_a + np.log(np.sum(np.exp(-beta * (areas_arr - max_a))))
    soft_min_val = - (1.0 / beta) * log_sum_exp_neg
    
    score = soft_min_val / hull_area
    if score < 1e-6:
        return 1e10
        
    return -score

def _raw_obj(points_flat):
    """
    Raw objective for final polish.
    """
    points = points_flat.reshape(N, 2)
    if np.any(np.isnan(points)) or np.any(np.isinf(points)):
        return 1e10
    score = _score(points)
    if score < 1e-6:
        return 1e10
    return -score

def _initial_regular_polygon():
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    return np.column_stack([np.cos(angles), np.sin(angles)])

def _initial_concentric_rings():
    # 7 on outer ring, 7 on inner ring? Or 6 and 8?
    # Let's try 7 and 7.
    n1 = 7
    n2 = N - n1
    r1 = 1.0
    r2 = 0.5
    a1 = np.linspace(0, 2 * np.pi, n1, endpoint=False)
    a2 = np.linspace(0, 2 * np.pi, n2, endpoint=False) + np.pi/n2 # offset
    
    p1 = np.column_stack([r1 * np.cos(a1), r1 * np.sin(a1)])
    p2 = np.column_stack([r2 * np.cos(a2), r2 * np.sin(a2)])
    
    return np.vstack([p1, p2])

def _initial_jittered_grid():
    # 4x4 grid with one removed and jittered?
    # 14 points. 4x4 is 16. Remove 2 corners?
    # Or just random points in a square?
    # Let's do a structured jitter.
    x = np.linspace(-1, 1, 4)
    y = np.linspace(-1, 1, 4)
    xx, yy = np.meshgrid(x, y)
    pts = np.column_stack([xx.ravel(), yy.ravel()])
    # Remove 2 points (e.g., last two)
    pts = pts[:N]
    # Add small random jitter
    np.random.seed(42)
    pts += np.random.uniform(-0.1, 0.1, pts.shape)
    return pts

def construct():
    """
    Generate 14 points in a planar convex region normalized to unit area
    to maximize the minimum area over all triples.
    
    Strategy:
    1. Multiple initializations: Regular polygon, Concentric rings, Jittered grid.
    2. Optimization Phase 1: Nelder-Mead with Soft-Min Objective (Smooth surrogate).
       This helps navigate the landscape before the 'min' operator creates sharp ridges.
    3. Optimization Phase 2: Nelder-Mead with Raw Objective (Exact min).
       Refines the solution using the true metric.
    4. Optimization Phase 3: Powell's Method Polish with Raw Objective.
       Squeezes out final precision.
    5. Normalize to unit area.
    """
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    initializers = [
        _initial_regular_polygon,
        _initial_concentric_rings,
        _initial_jittered_grid
    ]
    
    best_result = None
    best_score_val = -1e10
    
    for init_func in initializers:
        if time.time() > deadline - 40:
            break
            
        try:
            x0 = init_func().flatten()
        except Exception:
            continue
            
        current_best = x0.copy()
        current_score = _score(x0.reshape(N, 2))
        
        # Phase 1: Soft-Min Nelder-Mead
        if time.time() < deadline - 30:
            try:
                # Beta controls smoothness. Higher is closer to raw min.
                # Start with moderate beta, maybe increase in later iterations?
                # For simplicity, fixed beta=50 or 100.
                res_soft = minimize(
                    _soft_min_obj, 
                    current_best, 
                    args=(50.0,),
                    method='Nelder-Mead', 
                    options={
                        'maxiter': 2000, 
                        'xatol': 1e-6, 
                        'fatol': 1e-8,
                        'adaptive': True
                    }
                )
                # Evaluate raw score of the soft-min result
                soft_score = _score(res_soft.x.reshape(N, 2))
                if soft_score > current_score:
                    current_score = soft_score
                    current_best = res_soft.x
            except Exception:
                pass
        
        # Phase 2: Raw Nelder-Mead
        if time.time() < deadline - 20:
            try:
                res_raw = minimize(
                    _raw_obj, 
                    current_best, 
                    method='Nelder-Mead', 
                    options={
                        'maxiter': 2000, 
                        'xatol': 1e-8, 
                        'fatol': 1e-10,
                        'adaptive': True
                    }
                )
                raw_score = _score(res_raw.x.reshape(N, 2))
                if raw_score > current_score:
                    current_score = raw_score
                    current_best = res_raw.x
            except Exception:
                pass
        
        # Phase 3: Powell Polish
        if time.time() < deadline - 10:
            try:
                res_powell = minimize(
                    _raw_obj, 
                    current_best, 
                    method='Powell', 
                    options={
                        'maxiter': 500, 
                        'xtol': 1e-10, 
                        'ftol': 1e-12
                    }
                )
                powell_score = _score(res_powell.x.reshape(N, 2))
                if powell_score > current_score:
                    current_score = powell_score
                    current_best = res_powell.x
            except Exception:
                pass
        
        # Update global best
        if current_score > best_score_val:
            best_score_val = current_score
            best_result = current_best.copy()
            
    if best_result is None:
        # Fallback
        best_result = _initial_regular_polygon().flatten()
        
    final_points = best_result.reshape(N, 2)
    
    # Final Normalization
    try:
        hull = ConvexHull(final_points)
        hull_area = hull.volume
    except Exception:
        final_points = _initial_regular_polygon()
        hull = ConvexHull(final_points)
        hull_area = hull.volume
        
    if hull_area > 1e-10:
        scale = np.sqrt(1.0 / hull_area)
        final_points = final_points * scale
    else:
        final_points = _initial_regular_polygon()
        hull = ConvexHull(final_points)
        hull_area = hull.volume
        scale = np.sqrt(1.0 / hull_area)
        final_points = final_points * scale

    return final_points
# EVOLVE-BLOCK-END
