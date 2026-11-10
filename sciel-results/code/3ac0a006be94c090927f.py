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
    safety_margin = 30  # seconds before limit to stop

    def get_score(pts):
        dists = pdist(pts, metric='euclidean')
        if len(dists) == 0:
            return 0.0
        d_min = np.min(dists)
        d_max = np.max(dists)
        if d_max == 0:
            return 0.0
        return (d_min / d_max) ** 2

    def get_min_max_info(pts):
        """
        Returns d_min, d_max, and the indices of the pairs achieving them.
        """
        dists = pdist(pts, metric='euclidean')
        d_min = np.min(dists)
        d_max = np.max(dists)
        
        # Get the pair indices for min and max
        # pdist returns distances in a specific order. We need to map back to (i, j)
        # scipy.spatial.distance.squareform can do this, but it's expensive.
        # We can compute it manually or use the fact that pdist order is known.
        # However, for n=14, squareform is fast enough.
        dist_matrix = squareform(dists)
        
        min_idx = np.unravel_index(np.argmin(dist_matrix), dist_matrix.shape)
        max_idx = np.unravel_index(np.argmax(dist_matrix), dist_matrix.shape)
        
        return d_min, d_max, min_idx, max_idx

    def compute_subgradient(pts):
        """
        Computes the subgradient of the objective f(x) = d_max / d_min.
        We want to minimize this ratio.
        """
        d_min, d_max, min_idx, max_idx = get_min_max_info(pts)
        
        if d_min < 1e-10:
            # Degenerate case
            return np.zeros_like(pts)

        # Gradient of d_max with respect to points
        # d_max = ||p_i - p_j||
        # grad d_max = (p_i - p_j) / d_max for i, and -(p_i - p_j) / d_max for j
        grad_d_max = np.zeros_like(pts)
        i, j = max_idx
        diff_max = pts[i] - pts[j]
        grad_d_max[i] += diff_max / d_max
        grad_d_max[j] -= diff_max / d_max
        
        # Gradient of d_min with respect to points
        k, l = min_idx
        diff_min = pts[k] - pts[l]
        grad_d_min = np.zeros_like(pts)
        grad_d_min[k] += diff_min / d_min
        grad_d_min[l] -= diff_min / d_min
        
        # f = d_max / d_min
        # grad f = (1/d_min) * grad d_max - (d_max / d_min^2) * grad d_min
        grad_f = (1.0 / d_min) * grad_d_max - (d_max / (d_min ** 2)) * grad_d_min
        
        return grad_f

    def subgradient_descent(pts, max_iter=200, step_size=0.05, decay=0.95):
        """
        Custom subgradient descent to minimize d_max / d_min.
        """
        current_pts = pts.copy()
        best_pts = current_pts.copy()
        best_ratio = np.inf
        current_ratio = d_max / d_min if (d_min, d_max, _, _) != (0,0,0,0) else np.inf
        
        # Estimate optimal value for Polyak step size
        # f* is unknown, use a running average or a lower bound.
        # Since we want to minimize ratio, and ratio >= 1 (for n points in 3D, actually can be < 1? No, d_min <= d_max so ratio >= 1).
        # Actually, the ratio d_max/d_min is >= 1.
        # Let's use a simple decaying step size as it's more stable without knowing f*.
        
        step = step_size
        
        for it in range(max_iter):
            grad = compute_subgradient(current_pts)
            
            # Normalize gradient to control step size magnitude
            grad_norm = np.linalg.norm(grad)
            if grad_norm > 1e-10:
                grad = grad / grad_norm
            
            # Update
            current_pts = current_pts - step * grad
            
            # Center the points to remove translation invariance
            current_pts = current_pts - np.mean(current_pts, axis=0)
            
            # Check validity
            if not np.all(np.isfinite(current_pts)):
                break
                
            d_min, d_max, _, _ = get_min_max_info(current_pts)
            if d_min <= 0 or d_max <= 0:
                break
                
            current_ratio = d_max / d_min
            
            if current_ratio < best_ratio:
                best_ratio = current_ratio
                best_pts = current_pts.copy()
                
            # Decay step size
            step *= decay
            
            # Early stop if step is tiny
            if step < 1e-6:
                break
                
        return best_pts

    def smooth_objective(x, beta=0.05):
        """
        Soft min/max approximation for L-BFGS polish.
        """
        p = x.reshape((n, d))
        if not np.all(np.isfinite(p)):
            return 1e10

        dists = pdist(p, metric='euclidean')
        
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
    points_init = points_init - np.mean(points_init, axis=0)
    
    best_points = points_init.copy()
    best_score = get_score(best_points)
    
    # 1. Global Optimization using Differential Evolution
    if time.time() - start_time < time_limit - safety_margin:
        try:
            def de_objective(x):
                p = x.reshape((n, d))
                if not np.all(np.isfinite(p)):
                    return 1e10
                d_min, d_max, _, _ = get_min_max_info(p)
                if d_min <= 0 or d_max <= 0:
                    return 1e10
                return d_max / d_min
            
            res_de = differential_evolution(
                de_objective, 
                bounds=[(-5, 5)] * (n * d), 
                maxiter=50, 
                popsize=15, 
                tol=1e-5, 
                seed=42,
                init='latinhypercube'
            )
            
            if res_de.fun < 1e9:
                cand_points = res_de.x.reshape((n, d))
                cand_score = get_score(cand_points)
                if cand_score > best_score:
                    best_points = cand_points
                    best_score = cand_score
                    
        except Exception:
            pass

    # 2. Local Refinement using Subgradient Descent
    # This is the structural change: using exact subgradients instead of soft-min approximation for the main local step.
    candidates_to_polish = [best_points.copy(), points_init.copy()]
    
    # Add random perturbations
    np.random.seed(123)
    for _ in range(5):
        perturb = best_points + np.random.randn(n, d) * 0.1
        candidates_to_polish.append(perturb)
        
    # Fibonacci sphere initial guesses
    def fibonacci_sphere(n_points, seed=0):
        rng = np.random.RandomState(seed)
        points = []
        for i in range(n_points):
            z = 1 - 2 * (i + 0.5) / n_points
            r = np.sqrt(1 - z**2)
            theta = 2 * np.pi * i / n_points
            x = r * np.cos(theta)
            y = r * np.sin(theta)
            points.append([x, y, z])
        pts = np.array(points)
        pts = pts - np.mean(pts, axis=0)
        return pts
    
    for s in [1, 2, 3]:
        candidates_to_polish.append(fibonacci_sphere(n, seed=s))

    for init_pts in candidates_to_polish:
        if time.time() - start_time > time_limit - safety_margin:
            break
            
        try:
            # Run subgradient descent
            opt_points = subgradient_descent(init_pts, max_iter=300, step_size=0.1, decay=0.98)
            
            opt_score = get_score(opt_points)
            
            if opt_score > best_score:
                best_points = opt_points
                best_score = opt_score
                
        except Exception:
            pass

    # 3. Final high-precision polish with L-BFGS on soft objective
    # Use a very small beta to make the soft objective closely approximate the true metric
    if time.time() - start_time < time_limit - safety_margin:
        try:
            # Use the best points found so far
            res_final = minimize(
                lambda x: smooth_objective(x, beta=0.005), 
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
        except Exception:
            pass

    # 4. One more subgradient pass on the L-BFGS result to ensure we haven't drifted
    if time.time() - start_time < time_limit - safety_margin:
        try:
            opt_points = subgradient_descent(best_points, max_iter=200, step_size=0.05, decay=0.98)
            opt_score = get_score(opt_points)
            if opt_score > best_score:
                best_points = opt_points
                best_score = opt_score
        except Exception:
            pass

    # Final validity check
    if not np.all(np.isfinite(best_points)):
        return points_init

    return best_points

# EVOLVE-BLOCK-END
