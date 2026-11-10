# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist
from scipy.optimize import minimize
import time

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the (x,y) coordinates.

    """
    start_time = time.time()
    deadline = 600  # seconds
    safety_margin = 30  # seconds
    
    n = 16
    d = 2
    
    def compute_R(points):
        """Compute the R metric directly."""
        dists = pdist(points, 'euclidean')
        if len(dists) == 0:
            return 0.0
        d_min = np.min(dists)
        d_max = np.max(dists)
        if d_max == 0:
            return 0.0
        return (d_min / d_max) ** 2

    def objective_smooth(params, p_min=100, p_max=100):
        """
        Smooth surrogate objective using log-sum-exp for soft-min and soft-max.
        This makes the objective differentiable, enabling gradient-based optimization.
        """
        if time.time() > start_time + deadline - safety_margin:
            raise TimeoutError()
        
        points = params.reshape(n, d)
        dists = pdist(points, 'euclidean')
        if len(dists) == 0:
            return 0.0
        
        # Soft min of dists
        d_scaled_min = -dists / p_min
        max_d_min = np.max(d_scaled_min)
        soft_min = -p_min * (max_d_min + np.log(np.sum(np.exp(d_scaled_min - max_d_min))))
        
        # Soft max of dists
        d_scaled_max = dists / p_max
        max_d_max = np.max(d_scaled_max)
        soft_max = p_max * (max_d_max + np.log(np.sum(np.exp(d_scaled_max - max_d_max))))
        
        if soft_max <= 0:
            return 0.0
        
        ratio = soft_min / soft_max
        return -(ratio ** 2)

    def objective_exact(params):
        """Exact objective for final evaluation."""
        if time.time() > start_time + deadline - safety_margin:
            raise TimeoutError()
            
        points = params.reshape(n, d)
        R = compute_R(points)
        return -R  # minimize -R

    def bisection_feasibility(points, d_max_target, max_iter=100, tol=1e-8):
        """
        Bisection on r (minimum distance) to find max feasible r given d_max constraint.
        Inner problem: maximize r subject to ||p_i - p_j||^2 >= r^2 for all pairs,
        and ||p_i - p_j||^2 <= d_max_target^2 for all pairs.
        Uses SLSQP for the inner constraint optimization.
        """
        if time.time() > start_time + deadline - safety_margin:
            return points, 0.0
        
        # Get initial d_min and d_max
        dists = pdist(points, 'euclidean')
        d_min_init = np.min(dists)
        d_max_init = np.max(dists)
        
        # Scale points so d_max = 1 for the bisection
        if d_max_init > 0:
            points_scaled = points / d_max_init
        else:
            points_scaled = points.copy()
        
        d_min_scaled = d_min_init / d_max_init if d_max_init > 0 else 0.0
        
        # Bisection: find max r such that we can place points with min dist >= r and max dist <= 1
        r_lo = 0.0
        r_hi = 1.0  # max possible is 1 (all on boundary of unit sphere)
        
        best_points = points_scaled.copy()
        best_r = d_min_scaled
        
        for it in range(max_iter):
            if time.time() > start_time + deadline - safety_margin:
                break
            
            r_mid = (r_lo + r_hi) / 2.0
            
            # Try to find configuration with min dist >= r_mid and max dist <= 1
            # Objective: maximize min dist (or equivalently, minimize violation)
            # Subject to: max dist <= 1
            
            def inner_obj(params):
                """Minimize negative of minimum distance, with penalty for max distance violation."""
                pts = params.reshape(n, d)
                d = pdist(pts, 'euclidean')
                min_d = np.min(d)
                max_d = np.max(d)
                # Penalize if max_d > 1
                penalty = max(0, max_d - 1.0) ** 2 * 1000
                return -(min_d - r_mid) + penalty
            
            def inner_grad(params):
                """Gradient of inner objective."""
                pts = params.reshape(n, d)
                d = pdist(pts, 'euclidean')
                min_d = np.min(d)
                max_d = np.max(d)
                
                # Gradient of -min_d: find the pair with min distance
                min_idx = np.argmin(d)
                # Map back to point indices
                # pdist ordering: for n points, pair (i,j) where i<j
                # Index k corresponds to (i, j) where:
                # k = i*n - i*(i+1)/2 + (j - i - 1)
                # We need inverse mapping
                ii = 0
                jj = 1
                for i in range(n):
                    for j in range(i+1, n):
                        if (i*n - i*(i+1)//2 + (j - i - 1)) == min_idx:
                            ii = i
                            jj = j
                            break
                    else:
                        continue
                    break
                
                # Gradient of min distance w.r.t. points
                diff = pts[ii] - pts[jj]
                dist_val = np.linalg.norm(diff)
                if dist_val < 1e-10:
                    grad = np.zeros((n, d))
                else:
                    grad = np.zeros((n, d))
                    grad[ii] = diff / dist_val
                    grad[jj] = -diff / dist_val
                
                # Gradient of penalty term
                if max_d > 1.0:
                    max_idx = np.argmax(d)
                    ii2 = 0
                    jj2 = 1
                    for i in range(n):
                        for j in range(i+1, n):
                            if (i*n - i*(i+1)//2 + (j - i - 1)) == max_idx:
                                ii2 = i
                                jj2 = j
                                break
                        else:
                            continue
                        break
                    
                    diff2 = pts[ii2] - pts[jj2]
                    dist2_val = np.linalg.norm(diff2)
                    if dist2_val > 1e-10:
                        penalty_grad = 2 * (max_d - 1.0) * 1000 * (diff2 / dist2_val)
                        grad[ii2] += penalty_grad
                        grad[jj2] -= penalty_grad
                
                return -grad.flatten()
            
            # Use SLSQP with bounds
            try:
                result = minimize(
                    inner_obj,
                    points_scaled.flatten(),
                    jac=inner_grad,
                    method='SLSQP',
                    bounds=[(-1.5, 1.5)] * (n * d),
                    options={'maxiter': 200, 'ftol': 1e-8}
                )
                
                new_points = result.x.reshape(n, d)
                new_dists = pdist(new_points, 'euclidean')
                new_min_d = np.min(new_dists)
                new_max_d = np.max(new_dists)
                
                if new_min_d >= r_mid - 1e-6 and new_max_d <= 1.0 + 1e-6:
                    # Feasible: increase r_lo
                    r_lo = r_mid
                    best_points = new_points.copy()
                    best_r = new_min_d
                else:
                    # Not feasible: decrease r_hi
                    r_hi = r_mid
                    
            except Exception:
                r_hi = r_mid
            
            if r_hi - r_lo < tol:
                break
        
        # Scale back
        if d_max_init > 0:
            final_points = best_points * d_max_init
        else:
            final_points = best_points
        
        return final_points, best_r

    def simulated_annealing(initial_points, n_iter=50000, t_start=1.0, t_end=0.001, seed=42):
        """
        Simulated Annealing to explore the global landscape.
        """
        rng = np.random.RandomState(seed)
        current_points = initial_points.copy()
        current_R = compute_R(current_points)
        best_points = current_points.copy()
        best_R = current_R
        
        n_iter = min(n_iter, 200000)  # Cap iterations
        
        for i in range(n_iter):
            # Check deadline
            if time.time() > start_time + deadline - safety_margin:
                break
            
            # Cooling schedule
            alpha = i / n_iter
            temperature = t_start * (t_end / t_start) ** alpha
            
            # Generate perturbation: move one or two points
            n_moves = 1 if rng.rand() < 0.7 else 2
            indices = rng.choice(n, size=n_moves, replace=False)
            
            # Perturbation size decreases with temperature
            scale = 0.5 * temperature + 0.01
            new_points = current_points.copy()
            for idx in indices:
                new_points[idx] += rng.normal(0, scale, size=d)
            
            new_R = compute_R(new_points)
            
            # Acceptance criterion
            if new_R > current_R or rng.rand() < np.exp((new_R - current_R) / max(temperature, 1e-10)):
                current_points = new_points
                current_R = new_R
                if current_R > best_R:
                    best_R = current_R
                    best_points = current_points.copy()
        
        return best_points, best_R

    def local_refine(points):
        """
        Local refinement using smooth surrogate first, then exact objective.
        """
        if time.time() > start_time + deadline - safety_margin:
            return points
        
        params = points.flatten()
        
        # Step 1: Optimize smooth surrogate with L-BFGS-B
        best_params = params.copy()
        best_smooth_val = objective_smooth(best_params, p_min=50, p_max=50)
        
        try:
            result = minimize(
                lambda x: objective_smooth(x, p_min=50, p_max=50),
                best_params, 
                method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-10}
            )
            if result.success:
                smooth_candidate = result.x
                if objective_smooth(smooth_candidate, p_min=50, p_max=50) < best_smooth_val:
                    best_params = smooth_candidate
                    best_smooth_val = objective_smooth(best_params, p_min=50, p_max=50)
        except Exception:
            pass
        
        # Step 2: Refine with finer smooth surrogate
        if time.time() > start_time + deadline - safety_margin:
            return best_params.reshape(n, d)
        
        try:
            result = minimize(
                lambda x: objective_smooth(x, p_min=20, p_max=20),
                best_params, 
                method='L-BFGS-B',
                options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-12}
            )
            if result.success:
                smooth_candidate = result.x
                if objective_smooth(smooth_candidate, p_min=20, p_max=20) < best_smooth_val:
                    best_params = smooth_candidate
        except Exception:
            pass
        
        # Step 3: Powell on exact objective
        if time.time() > start_time + deadline - safety_margin:
            return best_params.reshape(n, d)
        
        best_val = objective_exact(best_params)
        try:
            result = minimize(
                objective_exact, 
                best_params, 
                method='Powell', 
                options={'maxiter': 1000, 'xatol': 1e-10, 'fatol': 1e-10}
            )
            if result.fun < best_val:
                best_params = result.x
                best_val = result.fun
        except Exception:
            pass
        
        # Step 4: Nelder-Mead polish on exact objective
        if time.time() > start_time + deadline - safety_margin:
            return best_params.reshape(n, d)
        
        try:
            result = minimize(
                objective_exact, 
                best_params, 
                method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-12, 'fatol': 1e-12, 'adaptive': True}
            )
            if result.fun < best_val:
                best_params = result.x
                best_val = result.fun
        except Exception:
            pass
        
        return best_params.reshape(n, d)

    # Generate diverse initial points for SA
    def make_initial_points(seed):
        """Create various initial configurations."""
        rng = np.random.RandomState(seed)
        config_type = seed % 4
        
        if config_type == 0:
            # 4x4 grid
            x = np.linspace(-1.5, 1.5, 4)
            y = np.linspace(-1.5, 1.5, 4)
            X, Y = np.meshgrid(x, y)
            points = np.column_stack((X.ravel(), Y.ravel()))
        elif config_type == 1:
            # Random uniform in unit square
            points = rng.uniform(-1, 1, size=(n, d))
        elif config_type == 2:
            # Regular hexagonal packing approximation
            points = np.zeros((n, d))
            for i in range(n):
                row = i // 4
                col = i % 4
                points[i, 0] = col * 1.0 + (row % 2) * 0.5
                points[i, 1] = row * 0.866
            points -= points.mean(axis=0)
        else:
            # Random on circle
            angles = np.linspace(0, 2*np.pi, n, endpoint=False)
            radii = rng.uniform(0.5, 1.5, size=n)
            points = np.column_stack((radii * np.cos(angles), radii * np.sin(angles)))
        
        return points

    # Run multiple SA restarts with different initial conditions
    best_points = None
    best_R = 0.0
    
    n_restarts = 8
    for restart in range(n_restarts):
        if time.time() > start_time + deadline - safety_margin:
            break
        
        init_points = make_initial_points(seed=42 + restart)
        
        # Run SA with this initial condition
        sa_points, sa_R = simulated_annealing(init_points, n_iter=30000, t_start=1.0, t_end=0.001, seed=42 + restart)
        
        if sa_R > best_R:
            best_R = sa_R
            best_points = sa_points.copy()
        
        # Local refinement
        refined_points = local_refine(sa_points)
        refined_R = compute_R(refined_points)
        
        if refined_R > best_R:
            best_R = refined_R
            best_points = refined_points.copy()
    
    # Bisection feasibility search on the best result
    if best_points is not None and time.time() < start_time + deadline - safety_margin:
        bisection_points, bisection_r = bisection_feasibility(best_points, 1.0, max_iter=60, tol=1e-7)
        bisection_R = compute_R(bisection_points)
        
        if bisection_R > best_R:
            best_R = bisection_R
            best_points = bisection_points.copy()
    
    # Final high-precision polish on the best result
    if best_points is not None:
        if time.time() < start_time + deadline - safety_margin:
            final_points = local_refine(best_points)
            final_R = compute_R(final_points)
            if final_R > best_R:
                best_points = final_points
                best_R = final_R
    
    # Fallback if something went wrong
    if best_points is None or best_points.shape != (n, d):
        x = np.linspace(-1.5, 1.5, 4)
        y = np.linspace(-1.5, 1.5, 4)
        X, Y = np.meshgrid(x, y)
        best_points = np.column_stack((X.ravel(), Y.ravel()))
    
    return best_points

# EVOLVE-BLOCK-END
