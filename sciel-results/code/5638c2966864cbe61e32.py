import numpy as np
import time
import scipy.spatial
import scipy.optimize

def heilbronn_convex13() -> np.ndarray:
    """
    Construct an arrangement of 13 points to maximize the ratio of the smallest triangle area
    to the convex hull area using a combination of structured initialization,
    gradient-based optimization with soft-min surrogate, and local search.
    """
    n = 13
    start_time = time.time()
    time_limit = 580  # Safety margin before 600s limit

    # Precompute indices for triangles
    idx = np.arange(n)
    i_idx, j_idx, k_idx = np.meshgrid(idx, idx, idx, indexing='ij')
    mask = (i_idx < j_idx) & (j_idx < k_idx)
    tri_i = i_idx[mask]
    tri_j = j_idx[mask]
    tri_k = k_idx[mask]
    # tri_i, tri_j, tri_k are arrays of length 286

    def get_points_from_angles(angles):
        """Convert 13 angles to 13 (x,y) points on unit circle."""
        x = np.cos(angles)
        y = np.sin(angles)
        return np.column_stack((x, y))

    def calculate_triangle_areas(points):
        """Calculate areas of all 286 triangles."""
        p_i = points[tri_i]
        p_j = points[tri_j]
        p_k = points[tri_k]
        
        v1 = p_j - p_i
        v2 = p_k - p_i
        cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
        areas = 0.5 * np.abs(cross)
        return areas

    def calculate_metric(points, beta=None):
        """
        Calculate the metric. If beta is None, use hard min.
        Otherwise use soft-min approximation.
        """
        try:
            hull = scipy.spatial.ConvexHull(points)
            A_hull = hull.volume
            if A_hull <= 1e-9:
                return 0.0
        except Exception:
            return 0.0
            
        areas = calculate_triangle_areas(points)
        
        if np.min(areas) < 1e-9:
            return 0.0
            
        if beta is None:
            return np.min(areas) / A_hull
        else:
            # Stable soft-min approximation
            m = np.min(areas)
            shifted = areas - m
            exp_terms = np.exp(-beta * shifted)
            sum_exp = np.sum(exp_terms)
            
            if sum_exp == 0:
                soft_min = m
            else:
                soft_min = m - (1.0 / beta) * np.log(sum_exp)
                
            return soft_min / A_hull

    def neg_objective_smooth(angles, beta):
        points = get_points_from_angles(angles)
        val = calculate_metric(points, beta)
        return -val

    def neg_objective_hard(angles):
        points = get_points_from_angles(angles)
        return -calculate_metric(points, beta=None)

    def compute_gradient(angles, beta):
        """
        Numerical gradient of the smooth objective with respect to angles.
        """
        eps = 1e-6
        grad = np.zeros(n)
        base_val = neg_objective_smooth(angles, beta)
        
        for i in range(n):
            angles_plus = angles.copy()
            angles_plus[i] += eps
            val_plus = neg_objective_smooth(angles_plus, beta)
            
            angles_minus = angles.copy()
            angles_minus[i] -= eps
            val_minus = neg_objective_smooth(angles_minus, beta)
            
            grad[i] = (val_plus - val_minus) / (2 * eps)
            
        return grad

    def gradient_descent(angles, beta, n_steps=500, lr_init=0.01, lr_final=0.0001):
        """
        Gradient descent with adaptive learning rate.
        """
        angles = angles.copy()
        best_val = neg_objective_smooth(angles, beta)
        best_angles = angles.copy()
        
        for step in range(n_steps):
            t = step / max(n_steps - 1, 1)
            lr = lr_init * (1 - t) + lr_final * t
            
            grad = compute_gradient(angles, beta)
            
            # Normalize gradient
            grad_norm = np.linalg.norm(grad)
            if grad_norm > 1e-12:
                grad = grad / grad_norm
            
            angles -= lr * grad
            angles = angles % (2 * np.pi)
            
            # Check if improved
            val = neg_objective_smooth(angles, beta)
            if val < best_val:
                best_val = val
                best_angles = angles.copy()
            
            # Early stopping if gradient is small
            if grad_norm < 1e-8:
                break
            
            # Time check
            if time.time() - start_time > time_limit - 30:
                break
        
        return best_angles

    def simulated_annealing(angles, beta, n_iterations=10000, T_init=0.1, T_final=0.001):
        """
        Simulated annealing in angular space.
        """
        angles = angles.copy()
        current_val = neg_objective_smooth(angles, beta)
        best_val = current_val
        best_angles = angles.copy()
        
        rng = np.random.default_rng(42)
        
        for i in range(n_iterations):
            T = T_init * (T_final / T_init) ** (i / n_iterations)
            
            # Random perturbation
            perturbation = rng.normal(0, 0.1, n)
            new_angles = (angles + perturbation) % (2 * np.pi)
            
            new_val = neg_objective_smooth(new_angles, beta)
            delta = new_val - current_val
            
            if delta < 0 or rng.random() < np.exp(-delta / T):
                angles = new_angles
                current_val = new_val
                
                if current_val < best_val:
                    best_val = current_val
                    best_angles = angles.copy()
            
            # Time check
            if time.time() - start_time > time_limit - 30:
                break
        
        return best_angles

    # Initial Guess: Regular 13-gon
    theta0 = np.linspace(0, 2 * np.pi, n, endpoint=False)
    
    best_score = -1.0
    best_angles = theta0.copy()
    
    # Evaluate initial
    init_points = get_points_from_angles(theta0)
    init_score = calculate_metric(init_points, beta=None)
    best_score = init_score
    
    # Phase 1: Gradient descent with increasing beta
    betas = [10, 50, 100, 200, 500, 1000, 2000, 5000]
    current_angles = theta0.copy()
    
    for beta in betas:
        elapsed = time.time() - start_time
        if elapsed > time_limit - 60:
            break
        
        # Gradient descent
        gd_result = gradient_descent(current_angles, beta, n_steps=300, lr_init=0.005, lr_final=0.0001)
        
        # Evaluate hard metric
        gd_points = get_points_from_angles(gd_result)
        gd_score = calculate_metric(gd_points, beta=None)
        
        if gd_score > best_score:
            best_score = gd_score
            best_angles = gd_result.copy()
            current_angles = gd_result.copy()
        
        # Simulated annealing
        sa_result = simulated_annealing(current_angles, beta, n_iterations=2000, T_init=0.05, T_final=0.001)
        
        sa_points = get_points_from_angles(sa_result)
        sa_score = calculate_metric(sa_points, beta=None)
        
        if sa_score > best_score:
            best_score = sa_score
            best_angles = sa_result.copy()
            current_angles = sa_result.copy()

    # Phase 2: Nelder-Mead on hard objective
    elapsed = time.time() - start_time
    if elapsed < time_limit - 30:
        try:
            res_final = scipy.optimize.minimize(
                neg_objective_hard, 
                best_angles, 
                method='Nelder-Mead',
                options={'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-12}
            )
            
            final_hard_val = -res_final.fun
            if final_hard_val > best_score:
                best_score = final_hard_val
                best_angles = res_final.x.copy()
                
        except Exception:
            pass

    # Phase 3: Final local search with multiple restarts
    elapsed = time.time() - start_time
    if elapsed < time_limit - 10:
        rng = np.random.default_rng(123)
        
        for _ in range(5):
            if time.time() - start_time > time_limit - 10:
                break
            
            # Perturb best angles
            perturbed = best_angles + rng.normal(0, 0.05, n)
            perturbed = perturbed % (2 * np.pi)
            
            try:
                res = scipy.optimize.minimize(
                    neg_objective_hard, 
                    perturbed, 
                    method='Nelder-Mead',
                    options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-12}
                )
                
                val = -res.fun
                if val > best_score:
                    best_score = val
                    best_angles = res.x.copy()
                    
            except Exception:
                pass

    final_points = get_points_from_angles(best_angles)
    
    # Ensure validity
    if not np.all(np.isfinite(final_points)):
        # Fallback to regular polygon
        final_points = get_points_from_angles(theta0)

    return final_points
