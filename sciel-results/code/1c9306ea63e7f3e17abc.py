import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial.distance import pdist

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.

    """
    n = 16
    d = 2
    start_time = time.time()
    time_limit = 580  # Safety margin before 600s kill

    # Helper to calculate the true score
    def score(points):
        if points.shape != (n, d):
            return 0.0
        dists = pdist(points)
        if len(dists) == 0:
            return 0.0
        d_min = np.min(dists)
        d_max = np.max(dists)
        if d_max == 0:
            return 0.0
        ratio = d_min / d_max
        return ratio * ratio

    # Smooth surrogate objective to MAXIMIZE
    # We minimize the negative of the objective
    # Objective: log(d_min) - log(d_max)
    # Surrogate: -1/beta * log(sum exp(-beta d)) - 1/beta * log(sum exp(beta d))
    # Plus penalties for centroid != 0 and scale != n
    
    beta = 100.0
    
    def objective_flat(flat_points):
        p = flat_points.reshape(n, d)
        
        # Constraints as penalties
        centroid = np.mean(p, axis=0)
        penalty_centroid = np.sum(centroid**2)
        
        norm_sq_sum = np.sum(p**2)
        target_norm_sq = n # Average squared dist from origin is 1
        penalty_scale = (norm_sq_sum - target_norm_sq)**2
        
        # Pairwise distances
        dists = pdist(p)
        
        # Soft min and max
        # Clip distances to avoid overflow in exp
        d_safe = np.clip(dists, 1e-9, 1e9)
        
        # log-sum-exp for stability
        # soft_min = -1/beta * log(sum(exp(-beta * d)))
        # soft_max =  1/beta * log(sum(exp(  beta * d)))
        
        # To prevent overflow, subtract max/min before exp
        # For soft_min: max of -beta*d is -beta*min(d). 
        # Let m = min(d). sum(exp(-beta*d)) = exp(-beta*m) * sum(exp(-beta*(d-m)))
        # log(sum(exp(-beta*d))) = -beta*m + log(sum(exp(-beta*(d-m))))
        
        m_min = np.min(d_safe)
        m_max = np.max(d_safe)
        
        # Calculate soft_min
        exp_terms_min = np.exp(-beta * (d_safe - m_min))
        log_sum_exp_min = np.log(np.sum(exp_terms_min))
        soft_min = -1.0 / beta * (-beta * m_min + log_sum_exp_min)
        
        # Calculate soft_max
        exp_terms_max = np.exp(beta * (d_safe - m_max))
        log_sum_exp_max = np.log(np.sum(exp_terms_max))
        soft_max = 1.0 / beta * (beta * m_max + log_sum_exp_max)
        
        # We want to maximize soft_min - soft_max
        # So we minimize -(soft_min - soft_max)
        val = -(soft_min - soft_max)
        
        # Add penalties (minimize these)
        val += 100.0 * penalty_centroid
        val += 100.0 * penalty_scale
        
        return val

    # Helper to normalize points (center and scale)
    def normalize_points(p):
        p = p - np.mean(p, axis=0)
        norm_sq = np.sum(p**2)
        if norm_sq < 1e-9:
            return p
        scale = np.sqrt(target_norm_sq / norm_sq)
        return p * scale

    # Initial guesses
    initial_guesses = []
    
    # 1. 4x4 Grid
    x = np.linspace(-1.5, 1.5, 4)
    y = np.linspace(-1.5, 1.5, 4)
    xx, yy = np.meshgrid(x, y)
    grid = np.column_stack((xx.ravel(), yy.ravel()))
    initial_guesses.append(normalize_points(grid))
    
    # 2. Hexagonal Lattice Subset
    # Generate a larger hex grid and pick 16 closest to center
    hex_points = []
    R = 3 # Radius
    for i in range(-R, R+1):
        for j in range(-R, R+1):
            # Hex grid coords to cartesian
            x_h = i + 0.5 * j
            y_h = np.sqrt(3) / 2 * j
            hex_points.append([x_h, y_h])
    hex_points = np.array(hex_points)
    # Sort by distance to origin
    dists_to_origin = np.linalg.norm(hex_points, axis=1)
    indices = np.argsort(dists_to_origin)
    hex_subset = hex_points[indices[:n]]
    initial_guesses.append(normalize_points(hex_subset))
    
    # 3. Random points
    np.random.seed(42)
    for _ in range(5):
        rand_pts = np.random.randn(n, d)
        initial_guesses.append(normalize_points(rand_pts))
        
    # 4. Circle points
    angles = np.linspace(0, 2*np.pi, n, endpoint=False)
    circle_pts = np.column_stack((np.cos(angles), np.sin(angles)))
    initial_guesses.append(normalize_points(circle_pts))

    best_score = 0.0
    best_points = initial_guesses[0]

    for guess in initial_guesses:
        if time.time() - start_time > time_limit:
            break
            
        flat_guess = guess.ravel()
        
        try:
            # Use L-BFGS-B for speed
            res = minimize(
                objective_flat, 
                flat_guess, 
                method='L-BFGS-B', 
                options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            
            if res.success or res.fun < 1e6: # Check if it didn't explode
                res_points = res.x.reshape(n, d)
                res_points = normalize_points(res_points)
                s = score(res_points)
                
                if s > best_score:
                    best_score = s
                    best_points = res_points
                    
        except Exception as e:
            pass

    # Final Polish: Use a smaller beta or exact objective with a local optimizer?
    # The smooth surrogate with beta=100 is very close to the exact objective.
    # We can do a few more iterations on the best point found with a tighter tolerance.
    
    if time.time() - start_time < time_limit - 10:
        flat_best = best_points.ravel()
        # Refine with higher beta? Or just more iterations?
        # Let's try SLSQP with the exact objective? Exact objective is non-smooth.
        # Stick to smooth surrogate but increase beta slightly and run again?
        # Or just run L-BFGS-B again from the best point with more iterations.
        
        try:
            res = minimize(
                objective_flat, 
                flat_best, 
                method='L-BFGS-B', 
                options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            if res.success:
                res_points = res.x.reshape(n, d)
                res_points = normalize_points(res_points)
                s = score(res_points)
                if s > best_score:
                    best_score = s
                    best_points = res_points
        except:
            pass

    return best_points

# EVOLVE-BLOCK-END
