# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist
from scipy.optimize import minimize
import time

def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions in order to maximize the ratio of minimum to maximum distance.
    
    Uses a hybrid approach:
    1. Simulated Annealing (SA) with a smooth surrogate to explore the landscape.
    2. L-BFGS-B optimization with the same smooth surrogate for precise local refinement.
    
    Returns
        points: np.ndarray of shape (14,3) containing the (x,y,z) coordinates of the 14 points.
    """
    
    n = 14
    d = 3
    start_time = time.time()
    time_limit = 600  # Safety margin before 630s kill

    def get_score(pts):
        """Compute the actual metric (d_min / d_max)^2."""
        if not np.all(np.isfinite(pts)):
            return 0.0
        dists = pdist(pts, metric='euclidean')
        if len(dists) == 0:
            return 0.0
        d_min = np.min(dists)
        d_max = np.max(dists)
        if d_max == 0:
            return 0.0
        return (d_min / d_max) ** 2

    def get_smooth_objective(x, scale=0.1):
        """
        Smooth surrogate for maximizing (d_min/d_max)^2.
        Returns the negative log ratio to be minimized.
        Objective: minimize log(soft_max) - log(soft_min)
        """
        if time.time() - start_time > time_limit - 10:
            return 1e10
            
        p = x.reshape((n, d))
        if not np.all(np.isfinite(p)):
            return 1e10
            
        dists = pdist(p, metric='euclidean')
        if len(dists) == 0:
            return 1e10
            
        # Soft min: -scale * log(sum(exp(-d_i/scale)) / n)
        # Soft max: scale * log(sum(exp(d_i/scale)) / n)
        
        # Numerical stability for log-sum-exp
        neg_d_over_s = -dists / scale
        max_neg = np.max(neg_d_over_s)
        log_sum_exp_neg = max_neg + np.log(np.sum(np.exp(neg_d_over_s - max_neg)) / n)
        soft_min = -scale * log_sum_exp_neg
        
        d_over_s = dists / scale
        max_pos = np.max(d_over_s)
        log_sum_exp_pos = max_pos + np.log(np.sum(np.exp(d_over_s - max_pos)) / n)
        soft_max = scale * log_sum_exp_pos
        
        if soft_min <= 1e-9 or soft_max <= 1e-9:
            return 1e10
        
        # We want to maximize soft_min / soft_max
        # Equivalent to minimizing log(soft_max) - log(soft_min)
        return np.log(soft_max) - np.log(soft_min)

    def sa_optimize(init_pts, time_budget):
        """
        Run Simulated Annealing to optimize the points using the smooth surrogate for acceptance.
        """
        current_pts = init_pts.copy()
        current_score = get_score(current_pts)
        best_pts = current_pts.copy()
        best_score = current_score
        
        # Use a fixed scale for SA acceptance to keep it stable
        sa_scale = 0.5 
        current_smooth = get_smooth_objective(current_pts.ravel(), sa_scale)
        
        T = 0.1  # Temperature for smooth objective (range is roughly [-2, 2])
        T_min = 1e-5
        cooling_rate = 0.9995
        move_sigma = 0.2
        
        no_improve_steps = 0
        max_no_improve = 3000
        
        while time.time() - start_time < time_limit - 30:
            if no_improve_steps > max_no_improve:
                current_pts = best_pts.copy()
                current_score = best_score
                current_smooth = get_smooth_objective(current_pts.ravel(), sa_scale)
                T = 0.1
                no_improve_steps = 0
                move_sigma = 0.2
                continue

            idx = np.random.randint(n)
            move = np.random.randn(d) * move_sigma
            new_pts = current_pts.copy()
            new_pts[idx] += move
            
            new_score = get_score(new_pts)
            new_smooth = get_smooth_objective(new_pts.ravel(), sa_scale)
            
            delta = new_smooth - current_smooth
            if delta < 0:
                accept_prob = np.exp(delta / T) if delta > -10 * T else 0
                accept = np.random.rand() < accept_prob
            else:
                accept = True
            
            if accept:
                current_pts = new_pts
                current_score = new_score
                current_smooth = new_smooth
                no_improve_steps = 0
                if current_score > best_score:
                    best_score = current_score
                    best_pts = current_pts.copy()
            else:
                no_improve_steps += 1
            
            T *= cooling_rate
            if T < T_min:
                T = T_min
                move_sigma *= 0.98
                if move_sigma < 1e-3:
                    # Restart
                    T = 0.05
                    move_sigma = 0.05
                    no_improve_steps = 0

        return best_pts, best_score

    def l_bfgs_polish(pts, time_budget):
        """
        Use L-BFGS-B to polish the points using the smooth surrogate.
        """
        if time.time() - start_time > time_limit - 5:
            return pts, get_score(pts)
            
        try:
            # Start with a relatively small scale for sharper approximation
            # but not too small to avoid numerical issues
            best_local_pts = pts.copy()
            best_local_score = get_score(pts)
            
            scales = [0.2, 0.1, 0.05]
            
            for scale in scales:
                if time.time() - start_time > time_limit - 5:
                    break
                    
                res = minimize(
                    get_smooth_objective, 
                    best_local_pts.ravel(), 
                    args=(scale,),
                    method='L-BFGS-B', 
                    options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-10, 'maxfun': 5000}
                )
                
                if res.fun < 1e10:
                    opt_pts = res.x.reshape((n, d))
                    opt_score = get_score(opt_pts)
                    if opt_score > best_local_score:
                        best_local_pts = opt_pts
                        best_local_score = opt_score
                        
        except Exception:
            pass
            
        return best_local_pts, best_local_score

    # Initializations
    initializations = []
    
    # 1. Icosahedron + Poles (normalized)
    phi = (1 + np.sqrt(5)) / 2
    ico_verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ])
    norms_ico = np.linalg.norm(ico_verts, axis=1)
    ico_verts_norm = ico_verts / norms_ico[:, np.newaxis]
    base_points = np.vstack([ico_verts_norm, np.array([[0, 0, 1.0], [0, 0, -1.0]])])
    initializations.append(base_points)
    
    # 2. Random points on sphere
    np.random.seed(42)
    for i in range(15):
        rand_pts = np.random.randn(n, d)
        norms = np.linalg.norm(rand_pts, axis=1)
        rand_pts = rand_pts / norms[:, np.newaxis]
        initializations.append(rand_pts)
        
        # 3. Random points in cube (scaled)
        if i % 3 == 0:
            rand_pts_cube = np.random.rand(n, d) * 2 - 1
            initializations.append(rand_pts_cube)

    best_points_overall = initializations[0].copy()
    best_score_overall = get_score(best_points_overall)
    
    # Phase 1: SA Exploration
    # We will run SA on a subset of initializations, or cycle through them if time permits.
    # Given the time limit, we'll prioritize the best initializations.
    
    # First, quickly score all initializations to pick the top few for SA
    scored_inits = [(get_score(init), init) for init in initializations]
    scored_inits.sort(key=lambda x: x[0], reverse=True)
    
    # Take top 5 initializations for detailed SA search
    top_inits = [init for score, init in scored_inits[:5]]
    
    for init_pts in top_inits:
        elapsed = time.time() - start_time
        if elapsed > time_limit - 60: # Reserve 60s for polishing
            break
            
        local_best_pts, local_best_score = sa_optimize(init_pts, time_limit - 60)
        
        if local_best_score > best_score_overall:
            best_score_overall = local_best_score
            best_points_overall = local_best_pts.copy()
            
    # Phase 2: L-BFGS-B Polish
    # Apply L-BFGS-B to the best point found so far
    elapsed = time.time() - start_time
    if elapsed < time_limit - 10:
        best_points_overall, best_score_overall = l_bfgs_polish(best_points_overall, time_limit - 10)
        
    # Final check for validity
    if not np.all(np.isfinite(best_points_overall)):
        best_points_overall = initializations[0].copy()
        
    return best_points_overall

# EVOLVE-BLOCK-END
