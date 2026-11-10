# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist
from scipy.optimize import differential_evolution, minimize
import time

def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions in order to maximize the ratio of minimum to maximum distance.
    
    Uses Differential Evolution for global search and L-BFGS-B with a smooth 
    surrogate objective for local refinement. The points are constrained to the 
    unit sphere via normalization in the objective function, as this is the 
    optimal manifold for maximizing minimum distance.
    
    Returns
        points: np.ndarray of shape (14,3) containing the (x,y,z) coordinates of the 14 points.
    """
    
    n = 14
    d = 3
    start_time = time.time()
    time_limit = 600  # Safety margin before 630s kill

    def normalize_points(pts):
        """Normalize points to the unit sphere."""
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        # Handle zero norms
        norms[norms == 0] = 1.0
        return pts / norms

    def get_true_score(pts):
        """Compute the actual metric (d_min / d_max)^2."""
        if not np.all(np.isfinite(pts)):
            return 0.0
        pts = pts.reshape((n, d))
        pts = normalize_points(pts)
        dists = pdist(pts, metric='euclidean')
        if len(dists) == 0:
            return 0.0
        d_min = np.min(dists)
        d_max = np.max(dists)
        if d_max == 0:
            return 0.0
        return (d_min / d_max) ** 2

    def get_smooth_objective(x, tau=0.1):
        """
        Smooth surrogate for maximizing (d_min/d_max)^2.
        Returns the negative log ratio to be minimized.
        """
        if time.time() - start_time > time_limit - 10:
            return 1e10
            
        p = x.reshape((n, d))
        if not np.all(np.isfinite(p)):
            return 1e10
            
        p = normalize_points(p)
        
        dists = pdist(p, metric='euclidean')
        if len(dists) == 0:
            return 1e10
            
        # Soft min: -tau * log(sum(exp(-d_i/tau)) / n)
        # Soft max: tau * log(sum(exp(d_i/tau)) / n)
        
        # Numerical stability for log-sum-exp
        neg_d_over_t = -dists / tau
        max_neg = np.max(neg_d_over_t)
        log_sum_exp_neg = max_neg + np.log(np.sum(np.exp(neg_d_over_t - max_neg)) / n)
        soft_min = -tau * log_sum_exp_neg
        
        d_over_t = dists / tau
        max_pos = np.max(d_over_t)
        log_sum_exp_pos = max_pos + np.log(np.sum(np.exp(d_over_t - max_pos)) / n)
        soft_max = tau * log_sum_exp_pos
        
        if soft_min <= 1e-9 or soft_max <= 1e-9:
            return 1e10
        
        # Minimize log(soft_max) - log(soft_min)
        return np.log(soft_max) - np.log(soft_min)

    # Initial bounds for DE
    # Since we normalize, the magnitude doesn't matter, but we need bounds for DE.
    # [-1, 1] is sufficient.
    bounds = [(-1.0, 1.0)] * (n * d)

    best_points = None
    best_score = 0.0

    # Phase 1: Differential Evolution
    # DE is robust for global optimization.
    # We use a moderate population size and iteration count.
    try:
        de_result = differential_evolution(
            get_smooth_objective,
            bounds,
            args=(0.1,), # Moderate tau for DE
            maxiter=200,
            popsize=20,
            tol=1e-6,
            seed=42,
            polish=False, # We will do our own polish
            callback=None
        )
        
        if de_result.success or de_result.fun < 1e10:
            best_points = de_result.x
            best_score = get_true_score(best_points)
    except Exception as e:
        print(f"DE failed: {e}")

    # If DE failed or didn't run, use a random start
    if best_points is None:
        np.random.seed(42)
        best_points = np.random.randn(n, d)
        best_score = get_true_score(best_points)

    # Phase 2: Local Refinement with L-BFGS-B
    # We refine the points found by DE using decreasing tau values.
    # This allows the optimizer to first navigate the broad landscape 
    # and then sharpen the specific distances.
    
    current_points = best_points.copy()
    current_score = best_score
    
    taus = [0.1, 0.05, 0.01, 0.005]
    
    for tau in taus:
        if time.time() - start_time > time_limit - 20:
            break
            
        try:
            res = minimize(
                get_smooth_objective,
                current_points.ravel(),
                args=(tau,),
                method='L-BFGS-B',
                options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            
            if res.fun < 1e10:
                opt_points = res.x.reshape((n, d))
                opt_score = get_true_score(opt_points)
                
                if opt_score > current_score:
                    current_points = opt_points
                    current_score = opt_score
        except Exception as e:
            print(f"L-BFGS failed with tau={tau}: {e}")
            continue

    best_points = current_points
    best_score = current_score

    # Final sanity check
    if not np.all(np.isfinite(best_points)):
        # Fallback to icosahedron + poles if something went wrong
        phi = (1 + np.sqrt(5)) / 2
        ico_verts = np.array([
            [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
            [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
            [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
        ])
        norms_ico = np.linalg.norm(ico_verts, axis=1)
        ico_verts_norm = ico_verts / norms_ico[:, np.newaxis]
        best_points = np.vstack([ico_verts_norm, np.array([[0, 0, 1.0], [0, 0, -1.0]])])
        
    return best_points

# EVOLVE-BLOCK-END
