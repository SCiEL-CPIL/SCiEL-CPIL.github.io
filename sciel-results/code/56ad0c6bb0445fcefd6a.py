import numpy as np
import time
from scipy.optimize import minimize

N = 11
SQRT3 = np.sqrt(3)

# Precompute indices for all triangles (i, j, k) with i < j < k
triangle_indices = np.array([(i, j, k) for i in range(N) for j in range(i+1, N) for k in range(j+1, N)])
I_IDX, J_IDX, K_IDX = triangle_indices[:, 0], triangle_indices[:, 1], triangle_indices[:, 2]

def get_cartesian(params):
    """
    Convert barycentric-like parameters (s, t) to Cartesian coordinates.
    s, t in [0, 1].
    u = (1-t)(1-s), v = t, w = s(1-t)
    x = v + 0.5 w
    y = SQRT3/2 * w
    """
    s = params[0::2]
    t = params[1::2]
    w = s * (1 - t)
    v = t
    x = v + 0.5 * w
    y = (SQRT3 / 2) * w
    return np.column_stack((x, y))

def compute_min_area(points):
    """
    Compute the minimum area of any triangle formed by 3 points.
    points: shape (N, 2)
    """
    Pi = points[I_IDX]
    Pj = points[J_IDX]
    Pk = points[K_IDX]
    v1 = Pj - Pi
    v2 = Pk - Pi
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    return np.min(areas)

def soft_min_objective(params, beta):
    """
    Objective function for optimization.
    Returns negative soft-min of triangle areas.
    """
    points = get_cartesian(params)
    Pi = points[I_IDX]
    Pj = points[J_IDX]
    Pk = points[K_IDX]
    v1 = Pj - Pi
    v2 = Pk - Pi
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    
    # Soft min: -1/beta * log(sum(exp(-beta * areas)))
    # Stable log-sum-exp
    x = -beta * areas
    x_max = np.max(x)
    # If x_max is very large positive, exp(x - x_max) is fine.
    # If x_max is very large negative, sum(exp(...)) might underflow?
    # exp(x - x_max) is in [exp(-range), 1].
    log_sum_exp = x_max + np.log(np.sum(np.exp(x - x_max)))
    soft_min = -log_sum_exp / beta
    return -soft_min

def construct():
    start_time = time.time()
    deadline = 145.0
    
    best_params = None
    best_score = -np.inf
    
    # Set seed for reproducibility in initial attempts, but allow some randomness
    # Actually, random restarts benefit from non-determinism, but we must ensure
    # the final result is deterministic if required? The task doesn't specify.
    # Using a fixed seed for the *first* few starts ensures a baseline.
    np.random.seed(42)
    
    # Generate a pool of start points
    # 1. Random
    # 2. Grid-like (low discrepancy)
    # 3. Symmetry-based
    
    starts = []
    
    # 20 Random starts
    for _ in range(20):
        starts.append(np.random.rand(2 * N))
        
    # 10 "Structured" starts
    # Try to place points roughly evenly
    # Simple grid in barycentric space is hard.
    # Just use Halton sequence or similar for better coverage?
    # For simplicity, more random restarts is better with fast optimization.
    
    for _ in range(50):
        starts.append(np.random.rand(2 * N))
        
    # Optimization Loop
    for idx, start_params in enumerate(starts):
        if time.time() - start_time > deadline - 10:
            break
            
        current_params = start_params.copy()
        
        # Continuation method on beta
        # Start with low beta (smooth), increase to sharp (approx min)
        betas = [10, 50, 100, 200, 400]
        
        for beta in betas:
            if time.time() - start_time > deadline - 5:
                break
                
            def obj(p):
                return soft_min_objective(p, beta)
                
            try:
                # L-BFGS-B is good for bounded problems
                res = minimize(obj, current_params, method='L-BFGS-B', 
                               bounds=[(0, 1)] * (2 * N), 
                               options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-8})
                
                if res.success:
                    current_params = res.x
                else:
                    # If it fails, we might still have a good point, but 
                    # usually res.x is the last iteration.
                    current_params = res.x
                    
            except Exception:
                # If it crashes, skip
                pass
        
        # Evaluate the true min area
        points = get_cartesian(current_params)
        score = compute_min_area(points)
        
        if score > best_score:
            best_score = score
            best_params = current_params.copy()
            
        # Optional: If this was a very good score, try to polish it further
        # with a higher beta?
        # The loop above already did that.
        
    # Final Polish: Run a few iterations of Nelder-Mead on the best candidate
    # with a very high beta to fine-tune the minimum.
    if best_params is not None and time.time() - start_time < deadline - 2:
        final_beta = 1000
        def final_obj(p):
            return soft_min_objective(p, final_beta)
            
        try:
            res = minimize(final_obj, best_params, method='Nelder-Mead', 
                           options={'maxiter': 200, 'fatol': 1e-12})
            if res.success:
                points = get_cartesian(res.x)
                score = compute_min_area(points)
                if score > best_score:
                    best_score = score
                    best_params = res.x
        except Exception:
            pass

    if best_params is None:
        # Fallback
        best_params = np.random.rand(2 * N)
        
    return get_cartesian(best_params)
