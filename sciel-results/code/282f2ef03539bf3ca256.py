import numpy as np
import time
from scipy.optimize import differential_evolution, minimize, basinhopping
import sys

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.
    """
    start_time = time.time()
    TIME_LIMIT = 630.0 # Evaluator limit
    SAFETY_MARGIN = 10.0 # Seconds to spare for cleanup
    N = 16

    def get_ratio(params):
        """
        Calculate the actual ratio R = (d_min / d_max)^2.
        """
        pts = params.reshape(N, 2)
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dist_sq = np.sum(diff * diff, axis=-1)
        np.fill_diagonal(dist_sq, np.inf)
        d_max_sq = np.max(dist_sq)
        d_min_sq = np.min(dist_sq)
        if d_min_sq <= 1e-12 or d_max_sq <= 1e-12:
            return 0.0
        return (d_min_sq / d_max_sq)

    def smooth_objective(params):
        """
        Smooth surrogate: Minimize the variance of the pairwise squared distances.
        This encourages all distances to be equal, which maximizes the min/max ratio.
        We normalize the points to have centroid 0 and unit variance to fix scale.
        """
        pts = params.reshape(N, 2)
        # Center
        pts = pts - np.mean(pts, axis=0)
        # Scale to unit variance (trace of covariance = 1)
        # If variance is 0, return large penalty
        var = np.mean(np.sum(pts**2, axis=1))
        if var < 1e-12:
            return 1e10
        pts = pts / np.sqrt(var)
        
        # Calculate pairwise squared distances
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dist_sq = np.sum(diff * diff, axis=-1)
        np.fill_diagonal(dist_sq, 0)
        
        # Get upper triangle
        i, j = np.triu_indices(N, k=1)
        d_sq = dist_sq[i, j]
        
        # Minimize variance of distances
        # This is smooth
        mean_d = np.mean(d_sq)
        var_d = np.var(d_sq)
        
        # We want to minimize var_d.
        # To help convergence, we can also penalize deviation from a target, but variance alone is a good start.
        return var_d

    def smooth_objective_with_grad(params):
        """
        Placeholder for gradient if needed, but we'll use Powell/BFGS without explicit grad first.
        """
        return smooth_objective(params)

    best_result = None
    best_score = 0.0

    # Initial guess: Regular 16-gon
    angles = np.linspace(0, 2*np.pi, N, endpoint=False)
    init_pts = np.column_stack([np.cos(angles), np.sin(angles)])
    # Flatten
    init_params = init_pts.flatten()

    # Try Basin Hopping with the smooth objective
    # Basin hopping uses a local minimizer (e.g., L-BFGS-B or Powell) and random perturbations.
    
    def local_minimizer(x):
        # Use L-BFGS-B for speed and smoothness
        res = minimize(smooth_objective, x, method='L-BFGS-B', options={'maxiter': 100, 'ftol': 1e-12})
        return {'x': res.x, 'fun': res.fun, 'nit': res.nit, 'nfev': res.nfev}

    try:
        # Run basin hopping
        # niter: number of iterations. Each iteration involves a local minimization.
        # We have ~600s. Local minimization is fast.
        # Let's try 100 iterations.
        
        # Define step size
        stepsize = 0.5
        
        bh_res = basinhopping(
            smooth_objective, 
            init_params, 
            minimizer_kwargs={'method': 'L-BFGS-B'}, # Actually, let's use the wrapper or just pass method
            niter=200,
            stepsize=0.5,
            minimizer_kwargs={'method': 'L-BFGS-B', 'options': {'maxiter': 50, 'ftol': 1e-10}},
            seed=42
        )
        
        if bh_res.fun < 1e9:
            best_result = bh_res.x.copy()
            # Evaluate true score
            score = get_ratio(best_result)
            if score > best_score:
                best_score = score

    except Exception as e:
        print(f"Basin Hopping failed: {e}", file=sys.stderr)

    # Refine with Differential Evolution on the true objective (nonsmooth) but starting from good points
    # This combines the global exploration of DE with the local refinement of the smooth method.
    
    if time.time() - start_time < TIME_LIMIT - 60:
        try:
            # Use DE with the true objective, but initialize with the best result from BH
            # and some random points.
            
            def true_cost(params):
                # Minimize -ratio
                return -get_ratio(params)
            
            # We need to be careful with DE bounds.
            # Since the objective is scale invariant, we can bound the coordinates.
            # However, DE might find solutions with large scale.
            # Let's normalize inside the cost function for DE? No, DE works on raw params.
            # Let's just use bounds [-2, 2] which is large enough for unit variance points.
            
            bounds = [(-2.0, 2.0)] * (N * 2)
            
            # Use the best result from BH as one of the initial points?
            # DE doesn't directly accept a single initial point, it uses a population.
            # We can use `init` to specify a custom population.
            
            # Let's just run DE for a shorter time to polish.
            de_res = differential_evolution(
                true_cost, 
                bounds, 
                maxiter=100, 
                popsize=20, 
                tol=1e-8,
                seed=123,
                polish=True, # Use L-BFGS-B for polish
                polish_kwargs={'method': 'L-BFGS-B', 'options': {'maxiter': 50}}
            )
            
            if de_res.fun < 0: # Since we minimized -ratio, fun should be negative if ratio > 0
                candidate = de_res.x
                score = -de_res.fun
                if score > best_score:
                    best_score = score
                    best_result = candidate

        except Exception as e:
            print(f"DE failed: {e}", file=sys.stderr)

    # Final Polish with Nelder-Mead on the true objective
    if best_result is not None and time.time() - start_time < TIME_LIMIT - 10:
        try:
            def true_cost(params):
                return -get_ratio(params)
            
            nm_res = minimize(
                true_cost, 
                best_result, 
                method='Nelder-Mead',
                options={'maxiter': 5000, 'xatol': 1e-12, 'fatol': 1e-12, 'adaptive': True}
            )
            
            if nm_res.fun < 0:
                score = -nm_res.fun
                if score > best_score:
                    best_score = score
                    best_result = nm_res.x

        except Exception as e:
            print(f"NM failed: {e}", file=sys.stderr)

    # Fallback
    if best_result is None or not np.all(np.isfinite(best_result)):
        best_result = init_params

    return best_result.reshape(16, 2)
