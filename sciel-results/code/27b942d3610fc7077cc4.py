# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist
from scipy.optimize import minimize, differential_evolution
import time

def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (14,3) containing the (x,y,z) coordinates of the 14 points.
    """
    
    n = 14
    d = 3
    start_time = time.time()
    time_limit = 600  # seconds
    
    def get_score(pts):
        dists = pdist(pts, metric='euclidean')
        if len(dists) == 0:
            return 0.0
        d_min = np.min(dists)
        d_max = np.max(dists)
        if d_max == 0:
            return 0.0
        return (d_min / d_max) ** 2

    def smooth_objective(x, beta=0.05):
        """
        Minimizes (d_max / d_min)^2 using soft min/max approximations.
        x: flat array of length 42
        """
        p = x.reshape((n, d))
        
        # Check for NaNs or Infs
        if not np.all(np.isfinite(p)):
            return 1e10

        dists = pdist(p, metric='euclidean')
        
        # Soft min: -log(sum(exp(-d/beta))) * beta
        # Soft max: log(sum(exp(d/beta))) * beta
        # We want to minimize (soft_max / soft_min)^2
        
        # Numerical stability for exp
        a_neg = -dists / beta
        max_a_neg = np.max(a_neg)
        log_sum_neg = max_a_neg + np.log(np.sum(np.exp(a_neg - max_a_neg)))
        soft_min = -beta * log_sum_neg
        
        a_pos = dists / beta
        max_a_pos = np.max(a_pos)
        log_sum_pos = max_a_pos + np.log(np.sum(np.exp(a_pos - max_a_pos)))
        soft_max = beta * log_sum_pos
        
        if soft_min <= 0 or soft_max <= 0:
            return 1e10
            
        ratio_sq = (soft_max / soft_min) ** 2
        return ratio_sq
    
    def smooth_objective_wrapped(x, beta):
        """Wrapper to pass beta to smooth_objective."""
        return smooth_objective(x, beta=beta)

    # Initial guess: Icosahedron vertices + 2 poles, normalized
    phi = (1 + np.sqrt(5)) / 2
    ico_verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ])
    norms_ico = np.linalg.norm(ico_verts, axis=1)
    ico_verts_norm = ico_verts / norms_ico[:, np.newaxis]
    
    # Add two points to make 14. Poles are a natural choice for symmetry.
    points_init = np.vstack([ico_verts_norm, np.array([[0, 0, 1.0], [0, 0, -1.0]])])
    
    # Center the points to remove translation invariance (helps optimizer)
    # But ratio is translation invariant, so centering is good for numerical stability.
    points_init = points_init - np.mean(points_init, axis=0)
    
    # Try multiple initial configurations
    initial_configs = [points_init.copy()]
    
    # Alternative: 14 points on sphere with better distribution
    # Use Fibonacci sphere for 14 points
    def fibonacci_sphere(n_points):
        points = np.zeros((n_points, 3))
        golden_angle = np.pi * (3.0 - np.sqrt(5.0))
        for i in range(n_points):
            z = 1.0 - (i / (n_points - 1)) * 2.0
            radius = np.sqrt(1.0 - z**2)
            theta = golden_angle * i
            points[i] = [radius * np.cos(theta), radius * np.sin(theta), z]
        return points
    
    fib_points = fibonacci_sphere(n)
    initial_configs.append(fib_points - np.mean(fib_points, axis=0))
    
    # Another: icosahedron with poles at different positions
    for pole_dist in [0.5, 0.8, 1.0]:
        alt_points = np.vstack([ico_verts_norm, np.array([[0, 0, pole_dist], [0, 0, -pole_dist]])])
        alt_points = alt_points - np.mean(alt_points, axis=0)
        initial_configs.append(alt_points)
    
    # Pick the best initial configuration
    best_points = initial_configs[0]
    best_score = get_score(best_points)
    for cfg in initial_configs[1:]:
        cfg_score = get_score(cfg)
        if cfg_score > best_score:
            best_points = cfg
            best_score = cfg_score
    
    # 1. Global Optimization using Differential Evolution
    # Use larger beta for smoother landscape during DE
    
    # Check time
    if time.time() - start_time > time_limit - 30:
        return best_points

    try:
        # Differential Evolution with larger beta for smoother landscape
        beta_de = 0.15
        
        def de_objective(x):
            return smooth_objective(x, beta=beta_de)
        
        res_de = differential_evolution(
            de_objective, 
            bounds=[(-3, 3)] * (n * d), 
            maxiter=80, 
            popsize=12, 
            tol=1e-4, 
            seed=42,
            init='latinhypercube'
        )
        
        if res_de.fun < 1e9:
            cand_points = res_de.x.reshape((n, d))
            cand_score = get_score(cand_points)
            if cand_score > best_score:
                best_points = cand_points
                best_score = cand_score
                
    except Exception as e:
        print(f"DE failed: {e}")

    # 2. Direct pairwise-gradient refinement: equalize critical distances
    
    def pairwise_gradient_refine(pts, n_iter=200, lr=0.01, tol=1e-10):
        """
        Directly adjust points to equalize the minimum and maximum distances.
        For the pair with min distance, push them apart.
        For the pair with max distance, pull them together.
        """
        pts = pts.copy()
        for it in range(n_iter):
            if time.time() - start_time > time_limit - 5:
                break
                
            dists = pdist(pts, metric='euclidean')
            if len(dists) == 0:
                break
            d_min = np.min(dists)
            d_max = np.max(dists)
            
            if d_max == 0:
                break
                
            # Get indices of min and max distance pairs
            min_idx = np.argmin(dists)
            max_idx = np.argmax(dists)
            
            # Convert flat index to pair indices
            def flat_to_pair(idx):
                i = 0
                for j in range(n):
                    if idx < j:
                        break
                    i = j + 1
                    idx -= j
                return i - 1, i
            
            min_i, min_j = flat_to_pair(min_idx)
            max_i, max_j = flat_to_pair(max_idx)
            
            # Gradients: for min pair, push apart; for max pair, pull together
            grad = np.zeros_like(pts)
            
            # Min distance pair: increase distance
            diff_min = pts[min_i] - pts[min_j]
            norm_min = np.linalg.norm(diff_min)
            if norm_min > 1e-12:
                unit_min = diff_min / norm_min
                grad[min_i] += unit_min
                grad[min_j] -= unit_min
            
            # Max distance pair: decrease distance
            diff_max = pts[max_i] - pts[max_j]
            norm_max = np.linalg.norm(diff_max)
            if norm_max > 1e-12:
                unit_max = diff_max / norm_max
                grad[max_i] -= unit_max
                grad[max_j] += unit_max
            
            # Normalize gradient
            grad_norm = np.linalg.norm(grad)
            if grad_norm < 1e-12:
                break
            grad = grad / grad_norm
            
            # Adaptive learning rate
            effective_lr = lr * min(1.0, 1.0 / (1.0 + abs(d_max - d_min)))
            
            new_pts = pts + effective_lr * grad
            
            # Check if this improves the score
            new_score = get_score(new_pts)
            if new_score > get_score(pts) - 1e-10:
                pts = new_pts
            else:
                # Try smaller step
                new_pts = pts + effective_lr * 0.1 * grad
                new_score = get_score(new_pts)
                if new_score > get_score(pts) - 1e-10:
                    pts = new_pts
                else:
                    break
            
            # Convergence check
            if abs(d_max - d_min) < tol * max(1.0, d_max):
                break
        
        return pts

    # Apply pairwise gradient refinement
    if time.time() - start_time < time_limit - 20:
        refined = pairwise_gradient_refine(best_points, n_iter=300, lr=0.05, tol=1e-12)
        refined_score = get_score(refined)
        if refined_score > best_score:
            best_points = refined
            best_score = refined_score

    # 3. Local Refinement using L-BFGS-B
    # Use progressively smaller beta for sharper objective
    
    candidates_to_polish = [
        best_points.copy(),
    ]
    
    # Add perturbations of the best points
    np.random.seed(123)
    for scale in [0.05, 0.1, 0.2]:
        perturb = best_points + np.random.randn(n, d) * scale
        candidates_to_polish.append(perturb)

    # Use adaptive beta: start moderate, then tighten
    beta_schedule = [0.08, 0.04, 0.02, 0.01]
    
    for beta_local in beta_schedule:
        if time.time() - start_time > time_limit - 15:
            break
            
        for init_pts in candidates_to_polish:
            if time.time() - start_time > time_limit - 10:
                break
                
            try:
                # Use the smooth objective with current beta
                def local_obj(x):
                    return smooth_objective(x, beta=beta_local)
                
                res_local = minimize(
                    local_obj, 
                    init_pts.ravel(), 
                    method='L-BFGS-B', 
                    options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-10}
                )
                
                if res_local.fun < 1e9:
                    opt_points = res_local.x.reshape((n, d))
                    opt_score = get_score(opt_points)
                    
                    if opt_score > best_score:
                        best_points = opt_points
                        best_score = opt_score
                        # Update candidates for next beta
                        candidates_to_polish = [opt_points.copy()]
                        for scale in [0.02, 0.05]:
                            perturb = opt_points + np.random.randn(n, d) * scale
                            candidates_to_polish.append(perturb)
            except Exception as e:
                pass

    # Apply pairwise gradient refinement again after L-BFGS
    if time.time() - start_time < time_limit - 10:
        refined2 = pairwise_gradient_refine(best_points, n_iter=200, lr=0.03, tol=1e-12)
        refined2_score = get_score(refined2)
        if refined2_score > best_score:
            best_points = refined2
            best_score = refined2_score

    # Final high-precision polish with very tight beta
    if time.time() - start_time < time_limit - 5:
        try:
            def final_obj(x):
                return smooth_objective(x, beta=0.005)
            
            res_final = minimize(
                final_obj, 
                best_points.ravel(), 
                method='L-BFGS-B', 
                options={'maxiter': 1000, 'ftol': 1e-14, 'gtol': 1e-12}
            )
            
            if res_final.fun < 1e9:
                final_points = res_final.x.reshape((n, d))
                final_score = get_score(final_points)
                if final_score > best_score:
                    best_points = final_points
                    best_score = final_score
        except Exception as e:
            pass

    # Final pairwise gradient polish
    if time.time() - start_time < time_limit - 2:
        refined3 = pairwise_gradient_refine(best_points, n_iter=100, lr=0.01, tol=1e-14)
        refined3_score = get_score(refined3)
        if refined3_score > best_score:
            best_points = refined3
            best_score = refined3_score

    # Final validity check
    if not np.all(np.isfinite(best_points)):
        return initial_configs[0]

    return best_points

# EVOLVE-BLOCK-END
