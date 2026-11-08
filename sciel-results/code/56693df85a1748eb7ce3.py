"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 14

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import differential_evolution, minimize

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

def _soft_min_triangle_area(points, beta=50.0):
    """
    Approximate the minimum triangle area using log-sum-exp (soft-min).
    This makes the objective differentiable for L-BFGS-B.
    """
    n = len(points)
    x = points[:, 0]
    y = points[:, 1]
    
    # Collect all triangle areas
    areas = []
    for i in range(n - 2):
        for j in range(i + 1, n - 1):
            dx_ki = x[j+1:] - x[i]
            dy_ki = y[j+1:] - y[i]
            cross = (x[j] - x[i]) * dy_ki - dx_ki * (y[j] - y[i])
            abs_cross = np.abs(cross)
            areas.extend(abs_cross * 0.5)
            
    areas = np.array(areas)
    if len(areas) == 0:
        return 0.0
    
    # Calculate soft min
    min_A = np.min(areas)
    shifted = areas - min_A
    # Numerical stability: log(sum(exp(-beta * shifted)))
    # If beta * shifted is large, exp underflows to 0, which is fine.
    # The max term is exp(0)=1, so log_sum >= 0.
    log_sum = np.log(np.sum(np.exp(-beta * shifted)))
    soft_min = min_A - (1.0 / beta) * log_sum
    
    return soft_min

def _soft_score(params, beta=50.0):
    """
    Objective for optimization.
    params: flattened array of 2N coordinates.
    Returns negative soft score (to be minimized).
    """
    pts = params.reshape(N, 2)
    
    # Basic sanity check: points shouldn't be exactly coincident or degenerate
    # We rely on the convex hull area check to handle degeneracy.
    
    hull_a = _hull_area(pts)
    if hull_a <= 1e-10:
        return 1e10
        
    # Normalize to unit area for consistent scaling during optimization
    # Note: The final score is invariant to scaling, but the optimization landscape
    # is better conditioned if we work in a normalized space or if we penalize scaling.
    # However, since the objective is scale-invariant (min_area / hull_area),
    # the gradient with respect to scaling is zero at the optimum.
    # We can just compute the score directly.
    
    # To avoid numerical issues with very large/small coordinates, 
    # we can normalize the points inside the objective before computing the soft min.
    # Let's normalize to unit hull area.
    pts_norm = pts / np.sqrt(hull_a)
    
    soft_min_area = _soft_min_triangle_area(pts_norm, beta=beta)
    score = soft_min_area / 1.0
    
    return -score

def _initial_random():
    """Generate a random initial guess."""
    # Random points in a unit circle
    r = np.sqrt(np.random.rand(N))
    theta = np.random.rand(N) * 2 * np.pi
    pts = np.column_stack([r * np.cos(theta), r * np.sin(theta)])
    return pts.flatten()

def construct():
    start_time = time.time()
    
    # Strategy:
    # 1. Use Differential Evolution for global exploration (robust to local minima).
    # 2. Use L-BFGS-B with a high-beta soft-min objective for local refinement.
    
    # Define bounds for DE. 
    # Since the problem is scale-invariant, we can restrict points to a reasonable box, 
    # e.g., [-1, 1] x [-1, 1]. The normalization in the score handles the scale.
    # However, if all points are too close, hull area is small. 
    # DE will explore this space.
    bounds = [(-1.5, 1.5)] * (2 * N)
    
    best_pts = None
    best_score = 0.0
    
    # Phase 1: Differential Evolution
    # DE is stochastic and robust. 
    # We limit maxiter to ensure it finishes in time.
    # popsize=10, maxiter=50 -> 500 evaluations. 
    # Each eval is fast (vectorized area calc).
    
    try:
        # Use a callback to check time
        def de_callback(xk, convergence):
            if time.time() > DEADLINE - 30:
                return True # Stop DE
            return False
        
        de_result = differential_evolution(
            _soft_score, 
            bounds, 
            args=(50.0,), # beta=50 for DE phase
            maxiter=100, 
            popsize=15, 
            tol=1e-4, 
            seed=42,
            callback=de_callback,
            polish=False # We will do our own polish
        )
        
        if de_result.success or de_result.x is not None:
            cand_pts = de_result.x.reshape(N, 2)
            cand_score = _score(cand_pts)
            if cand_score > best_score:
                best_score = cand_score
                best_pts = cand_pts
    except Exception as e:
        print(f"DE failed: {e}")
    
    # If DE failed or didn't produce a good result, use a structured initial guess
    if best_pts is None or best_score < 1.0:
        # Two interleaved heptagons
        angles0 = 2 * np.pi * np.arange(7) / 7.0
        angles1 = angles0 + np.pi / 7.0
        pts0 = np.column_stack([0.5 * np.cos(angles0), 0.5 * np.sin(angles0)])
        pts1 = np.column_stack([1.0 * np.cos(angles1), 1.0 * np.sin(angles1)])
        best_pts = np.vstack([pts0, pts1])
        best_score = _score(best_pts)

    # Phase 2: Local Refinement with L-BFGS-B
    # Use a higher beta to sharpen the soft-min approximation to the true min.
    # L-BFGS-B is gradient-based and very fast for smooth problems.
    
    if time.time() < DEADLINE - 15:
        try:
            # Increase beta for sharper optimization
            beta_opt = 100.0
            
            res = minimize(
                _soft_score, 
                best_pts.flatten(), 
                args=(beta_opt,),
                method='L-BFGS-B',
                bounds=[(-5, 5)] * (2 * N),
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            
            if res.x is not None:
                cand_pts = res.x.reshape(N, 2)
                cand_score = _score(cand_pts)
                if cand_score > best_score:
                    best_score = cand_score
                    best_pts = cand_pts
        except Exception as e:
            print(f"L-BFGS-B failed: {e}")
            
    # Phase 3: Optional fine-tuning with even higher beta or Nelder-Mead if time permits
    if time.time() < DEADLINE - 5:
        try:
            beta_final = 200.0
            res = minimize(
                _soft_score, 
                best_pts.flatten(), 
                args=(beta_final,),
                method='L-BFGS-B',
                bounds=[(-5, 5)] * (2 * N),
                options={'maxiter': 100, 'ftol': 1e-14, 'gtol': 1e-10}
            )
            if res.x is not None:
                cand_pts = res.x.reshape(N, 2)
                cand_score = _score(cand_pts)
                if cand_score > best_score:
                    best_score = cand_score
                    best_pts = cand_pts
        except Exception:
            pass

    return best_pts

# EVOLVE-BLOCK-END
