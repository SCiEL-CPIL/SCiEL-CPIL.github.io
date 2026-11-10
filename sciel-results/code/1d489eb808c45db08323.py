import numpy as np
import time
import scipy.optimize
import scipy.spatial.distance

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the (x,y) points.

    """
    n = 16
    d = 2
    start_time = time.time()
    deadline = 600 - 30  # 570 seconds to be safe

    def get_distances(points):
        """Calculate pairwise distances for a (N, 2) array."""
        return scipy.spatial.distance.pdist(points)

    def raw_objective(x):
        """
        Objective function to minimize.
        We want to maximize R = (d_min / d_max)^2.
        So we minimize -sqrt(R) = -d_min/d_max.
        """
        points = x.reshape(n, d)
        dists = get_distances(points)
        d_min = np.min(dists)
        d_max = np.max(dists)
        
        if d_max == 0:
            return 0.0
        return -(d_min / d_max)

    def smoothed_objective(x, beta=10.0):
        """
        Smoothed surrogate objective.
        Uses softmin and softmax to approximate min and max in a differentiable way.
        This helps gradient-based or local search methods navigate the landscape.
        """
        points = x.reshape(n, d)
        dists = get_distances(points)
        
        # Softmin: -logsumexp(-beta * dists) / beta
        # To avoid overflow, shift by max
        d_min_soft = -np.log(np.sum(np.exp(-beta * (dists - np.max(dists)))) * np.exp(-beta * np.max(dists))) / beta
        # Wait, standard softmin is: -1/beta * log(sum(exp(-beta * x)))
        # Let's implement carefully.
        # min(x) ~ -1/beta * log(sum(exp(-beta * x)))
        # max(x) ~ 1/beta * log(sum(exp(beta * x)))
        
        # For stability:
        max_d = np.max(dists)
        min_d = np.min(dists)
        
        # Softmin
        # -1/beta * log( sum( exp(-beta * d) ) )
        # = -1/beta * log( exp(-beta*max_d) * sum( exp(-beta * (d - max_d)) ) )
        # = -1/beta * ( -beta*max_d + log( sum( exp(-beta * (d - max_d)) ) ) )
        # = max_d - 1/beta * log( sum( exp(-beta * (d - max_d)) ) )
        
        term_min = np.exp(-beta * (dists - max_d))
        sum_min = np.sum(term_min)
        d_min_soft = max_d - (1.0 / beta) * np.log(sum_min)
        
        # Softmax
        # 1/beta * log( sum( exp(beta * d) ) )
        # = 1/beta * log( exp(beta*max_d) * sum( exp(beta * (d - max_d)) ) )
        # = max_d + 1/beta * log( sum( exp(beta * (d - max_d)) ) )
        
        term_max = np.exp(beta * (dists - max_d))
        sum_max = np.sum(term_max)
        d_max_soft = max_d + (1.0 / beta) * np.log(sum_max)
        
        if d_max_soft == 0:
            return 0.0
            
        ratio = d_min_soft / d_max_soft
        return -ratio

    # 1. Generate diverse initial candidates
    # Candidate 1: 4x4 Grid
    grid_x, grid_y = np.meshgrid(np.linspace(-1, 1, 4), np.linspace(-1, 1, 4))
    grid_pts = np.column_stack((grid_x.ravel(), grid_y.ravel()))
    
    # Candidate 2: Circle (regular polygon)
    angles = np.linspace(0, 2*np.pi, n, endpoint=False)
    circle_pts = np.column_stack((np.cos(angles), np.sin(angles)))
    
    # Candidate 3: Random
    np.random.seed(123)
    rand_pts = np.random.randn(n, d)
    
    # Candidate 4: Two clusters (squares)
    # Top-left square
    sq1 = np.array([[-1, -1], [-1, 1], [1, -1], [1, 1]])
    # Bottom-right square
    sq2 = sq1 + np.array([2, 2])
    # Need 16 points. Let's make 4 squares of 4 points each.
    # This might be a good topology for high ratio.
    sq3 = sq1 + np.array([-2, 2])
    sq4 = sq1 + np.array([-2, -2])
    # Scale down to fit
    all_sq = np.vstack([sq1, sq2, sq3, sq4])
    # Center and scale
    all_sq = (all_sq - np.mean(all_sq, axis=0)) / np.std(all_sq)
    
    candidates = [grid_pts, circle_pts, rand_pts, all_sq]
    
    best_points = None
    best_score = -np.inf
    
    # 2. Optimize using Simulated Annealing with Smoothed Objective
    # SA is good for non-smooth problems, but using a smoothed objective allows for larger steps
    # and better acceptance criteria based on the surrogate landscape.
    
    def simulated_annealing(init_pts, max_iters=5000, T_start=0.1, T_end=0.001, cooling=0.999):
        """
        Run Simulated Annealing starting from init_pts.
        """
        current_pts = init_pts.copy()
        current_x = current_pts.flatten()
        
        # Calculate initial objective
        # We use the smoothed objective for the search steps to get a smoother landscape
        # but we track the true objective for the final score.
        current_obj_smooth = smoothed_objective(current_x, beta=5.0)
        current_obj_true = raw_objective(current_x)
        
        best_local_x = current_x.copy()
        best_local_true = current_obj_true
        
        T = T_start
        step_size = 0.1
        
        for i in range(max_iters):
            if time.time() > deadline:
                break
                
            # Perturb
            # Use a random direction
            direction = np.random.randn(n * d)
            new_x = current_x + step_size * direction
            
            # Evaluate smoothed objective
            new_obj_smooth = smoothed_objective(new_x, beta=5.0)
            
            # Acceptance criterion
            delta = new_obj_smooth - current_obj_smooth
            if delta < 0 or np.random.rand() < np.exp(-delta / T):
                current_x = new_x
                current_obj_smooth = new_obj_smooth
                current_pts = current_x.reshape(n, d)
                
                # Check true objective
                new_obj_true = raw_objective(current_x)
                if new_obj_true < best_local_true:
                    best_local_true = new_obj_true
                    best_local_x = current_x.copy()
                
                # Occasionally adjust step size based on acceptance
                # If many moves are rejected, decrease step size
                # If many are accepted, maybe increase it slightly
                # Simple heuristic: decrease step size as T decreases
                step_size *= 0.9995
            else:
                # Rejected, maybe decrease step size slightly more?
                step_size *= 0.9999
                
            # Cool
            T *= cooling
            
            # If T is too low, we might get stuck. 
            # If we haven't improved in a long time, we could restart locally, 
            # but for now we just let it run.
            
        return best_local_x, -best_local_true  # Return points and score (negative of min obj)

    for i, init_pts in enumerate(candidates):
        if time.time() > deadline:
            break
            
        # Run SA
        try:
            sa_x, sa_score = simulated_annealing(init_pts, max_iters=2000)
            
            if sa_score > best_score:
                best_score = sa_score
                best_points = sa_x.reshape(n, d)
        except Exception:
            pass
        
        # Also try a local refinement with Nelder-Mead on the smoothed objective
        # starting from the best SA result
        if best_points is not None:
            try:
                res_local = scipy.optimize.minimize(
                    smoothed_objective, 
                    best_points.flatten(), 
                    args=(20.0,), # Higher beta for sharper approximation during refinement
                    method='Nelder-Mead',
                    options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-8}
                )
                if res_local.fun < smoothed_objective(best_points.flatten(), beta=20.0):
                    # Check true score
                    true_score = -raw_objective(res_local.x)
                    if true_score > best_score:
                        best_score = true_score
                        best_points = res_local.x.reshape(n, d)
            except Exception:
                pass

        # Check initial candidate
        init_score = -raw_objective(init_pts.flatten())
        if init_score > best_score:
            best_score = init_score
            best_points = init_pts

    # Fallback
    if best_points is None:
        best_points = grid_pts

    # Final refinement: One last Nelder-Mead on the raw objective to squeeze out precision
    # The raw objective is non-differentiable, but NM can still handle it if we are close.
    if time.time() < deadline - 10:
        try:
            res_final = scipy.optimize.minimize(
                raw_objective, 
                best_points.flatten(), 
                method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-10}
            )
            if res_final.fun < raw_objective(best_points.flatten()):
                best_points = res_final.x.reshape(n, d)
        except Exception:
            pass

    return best_points
