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
    start_time = time.time()
    TIME_LIMIT = 600.0
    SAFETY_MARGIN = 10.0
    
    N = 16
    
    def get_metric(pts):
        """Calculate the actual score R = (d_min / d_max)^2. Higher is better."""
        d = pdist(pts)
        if len(d) == 0:
            return 0.0
        d_min = np.min(d)
        d_max = np.max(d)
        if d_max < 1e-12:
            return 0.0
        return (d_min / d_max) ** 2

    def get_energy(pts, power=20):
        """
        Smooth surrogate objective to minimize.
        Minimizing sum(1/d^power) pushes points apart, maximizing d_min.
        We use a high power to make the minimum distance the dominant factor.
        """
        # Compute pairwise distances
        # For N=16, full matrix is fine
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dist_sq = np.sum(diff * diff, axis=-1)
        
        # Mask diagonal to avoid division by zero
        mask = ~np.eye(N, dtype=bool)
        dist_sq = np.where(mask, dist_sq, 1.0) # Avoid 0 in calculation, will be masked later
        
        # Calculate inverse power
        # dist = sqrt(dist_sq)
        # 1/dist^power = dist_sq^(-power/2)
        # Using log-sum-exp style stability if needed, but for power=20 and reasonable scales, direct is often ok
        # However, if dist is very small, this blows up.
        
        # To be safe against overflow/underflow and singularities:
        # We want to minimize sum(1/d^p).
        # Let's compute d = sqrt(dist_sq)
        d = np.sqrt(dist_sq)
        
        # Mask diagonal
        d_masked = np.where(mask, d, np.inf)
        
        if np.any(d_masked < 1e-10):
            return 1e10 # Penalty for coincident points
            
        # Sum of inverse powers
        # power=20 is quite stiff. power=10 is softer.
        # Let's try power=10 for smoother gradients, or 20 for sharper.
        # The prompt suggests smooth surrogate.
        energy = np.sum(1.0 / (d_masked ** power))
        
        return energy

    def get_energy_grad(pts, power=20):
        """
        Gradient of the energy function.
        E = sum_{i<j} d_{ij}^{-p}
        dE/dd_{ij} = -p * d_{ij}^{-p-1}
        dd_{ij}/dx_k = (x_{ki} - x_{kj}) / d_{ij}  (for i != j)
        """
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dist_sq = np.sum(diff * diff, axis=-1)
        mask = ~np.eye(N, dtype=bool)
        
        # Avoid division by zero
        dist_sq_safe = np.where(mask, dist_sq, 1.0)
        d = np.sqrt(dist_sq_safe)
        
        # Gradient of d_{ij} wrt x_k
        # grad_x_k d_{ij} = (x_{ki} - x_{kj}) / d_{ij}
        # Note: diff[i,j] = x_i - x_j
        # So term is diff[i,j] / d[i,j]
        
        # We need sum_{j != i} (x_{ij} - x_{ji})/d_{ij} * (-p) * d_{ij}^{-p-1}
        # Note x_{ji} = -x_{ij} in vector diff? No.
        # diff[i,j] = pts[i] - pts[j]
        # grad_{pts[i]} d_{ij} = (pts[i] - pts[j]) / d_{ij} = diff[i,j] / d_{ij}
        # grad_{pts[j]} d_{ij} = (pts[j] - pts[i]) / d_{ij} = -diff[i,j] / d_{ij}
        
        # Energy term for pair (i,j) is d_{ij}^{-p}
        # dE/dpts[i] += -p * d_{ij}^{-p-1} * (pts[i] - pts[j]) / d_{ij}
        #             = -p * d_{ij}^{-p-2} * (pts[i] - pts[j])
        
        # Let's compute the coefficient for each pair
        # coeff[i,j] = -p * d_{ij}^{-p-2}
        # Then grad[i] = sum_{j != i} coeff[i,j] * (pts[i] - pts[j])
        
        # Calculate d^{-p-2}
        # d = sqrt(dist_sq)
        # d^{-p-2} = (dist_sq)^{-(p+2)/2}
        
        exponent = -(power + 2) / 2.0
        # dist_sq_safe is 1.0 on diagonal, but we will zero out diagonal in coeff
        coeff = np.where(mask, -power * (dist_sq_safe ** exponent), 0.0)
        
        # grad[i] = sum_j coeff[i,j] * (pts[i] - pts[j])
        # This is equivalent to:
        # grad[i] = pts[i] * sum_j coeff[i,j] - sum_j coeff[i,j] * pts[j]
        
        sum_coeff = np.sum(coeff, axis=1, keepdims=True) # (N, 1)
        
        # Matrix multiplication: coeff (N,N) @ pts (N,2)
        # But coeff is symmetric? No, coeff[i,j] = coeff[j,i] because d is symmetric.
        # coeff[i,j] * pts[j]
        term2 = coeff @ pts
        
        grad = pts * sum_coeff - term2
        
        return grad

    def objective_and_grad(params, power=20):
        pts = params.reshape(N, 2)
        val = get_energy(pts, power)
        grad = get_energy_grad(pts, power)
        return val, grad

    def nelder_mead_optimize(initial_pts, time_budget, power=20):
        """
        Run Nelder-Mead optimization on a single set of initial points.
        """
        current_time = time.time()
        elapsed = current_time - start_time
        remaining = TIME_LIMIT - elapsed - SAFETY_MARGIN
        
        if remaining <= 0:
            return initial_pts, get_metric(initial_pts)

        initial_params = initial_pts.flatten()
        
        # Nelder-Mead is derivative-free but robust.
        # We can also use L-BFGS-B with the gradient if we want to be faster per step.
        # Given the small dimension (32), Nelder-Mead is often sufficient and robust to non-smoothness (though our surrogate is smooth).
        # Let's try L-BFGS-B with gradient as it's faster to converge locally.
        
        try:
            # Use L-BFGS-B
            res = minimize(
                objective_and_grad, 
                initial_params, 
                method='L-BFGS-B', 
                jac=True,
                options={
                    'maxiter': 2000,
                    'ftol': 1e-12,
                    'gtol': 1e-8,
                    'maxls': 50
                }
            )
            if res.success or np.isfinite(res.fun):
                res_pts = res.x.reshape(N, 2)
                # Normalize/center? The metric is scale invariant.
                # Centering helps numerical stability.
                res_pts = res_pts - np.mean(res_pts, axis=0)
                return res_pts, get_metric(res_pts)
        except Exception:
            pass

        # Fallback to Nelder-Mead if L-BFGS fails
        try:
            res = minimize(
                lambda x: get_energy(x.reshape(N, 2), power), 
                initial_params, 
                method='Nelder-Mead', 
                options={
                    'maxiter': 5000,
                    'xatol': 1e-10,
                    'fatol': 1e-12,
                    'adaptive': True
                }
            )
            if np.isfinite(res.fun):
                res_pts = res.x.reshape(N, 2)
                res_pts = res_pts - np.mean(res_pts, axis=0)
                return res_pts, get_metric(res_pts)
        except Exception:
            pass

        return initial_pts, get_metric(initial_pts)

    # Initial Candidates
    # 1. Random
    np.random.seed(42)
    candidates = []
    
    # Random normal
    candidates.append(np.random.randn(N, 2))
    
    # Hexagonal lattice (good initial structure)
    # Generate a hex lattice and pick the best 16
    R = 5
    pts_hex = []
    for i in range(-R, R+1):
        for j in range(-R, R+1):
            x = i + 0.5 * j
            y = np.sqrt(3) / 2 * j
            pts_hex.append([x, y])
    pts_hex = np.array(pts_hex)
    # Pick 16 closest to centroid
    centroid = np.mean(pts_hex, axis=0)
    dists = np.linalg.norm(pts_hex - centroid, axis=1)
    idx = np.argsort(dists)
    candidates.append(pts_hex[idx[:N]])
    
    # Square grid
    g = np.array([(i, j) for i in range(4) for j in range(4)], dtype=float)
    candidates.append(g)
    
    # Best known configuration (approximation of 16-point packing)
    # We can start with a circle + center + perturbations
    # But random restarts are more reliable.
    
    best_pts = None
    best_score = 0.0
    
    # Run multiple restarts
    # We have ~600s.
    # L-BFGS-B is fast. We can run many restarts.
    
    # Strategy:
    # 1. Run L-BFGS-B on each initial candidate.
    # 2. Keep the best.
    # 3. Run more random restarts with perturbations of the best.
    
    num_restarts = 50 # Can adjust based on time
    
    for i in range(num_restarts):
        current_time = time.time()
        if current_time - start_time > TIME_LIMIT - SAFETY_MARGIN:
            break
        
        # Determine initial point
        if i < len(candidates):
            init_pts = candidates[i].copy()
        else:
            # Perturb the best found so far
            if best_pts is not None:
                # Add noise proportional to the scale
                scale = np.std(best_pts)
                noise = np.random.randn(N, 2) * scale * 0.1
                init_pts = best_pts + noise
            else:
                init_pts = np.random.randn(N, 2)
        
        # Center it
        init_pts = init_pts - np.mean(init_pts, axis=0)
        
        # Optimize
        res_pts, score = nelder_mead_optimize(init_pts, TIME_LIMIT - (current_time - start_time), power=20)
        
        if score > best_score:
            best_score = score
            best_pts = res_pts
            
        # Also try with a different power for the next few iterations?
        # Higher power (e.g., 40) might find tighter packings.
        # Lower power (e.g., 10) might be smoother.
        # Let's stick to 20 for consistency, but maybe vary it in later restarts.
        if i > 10 and i % 5 == 0:
             # Try a higher power
             res_pts2, score2 = nelder_mead_optimize(init_pts, TIME_LIMIT - (current_time - start_time), power=40)
             if score2 > best_score:
                 best_score = score2
                 best_pts = res_pts2

    # Final Polish:
    # Use the best found configuration and run a very tight Nelder-Mead or L-BFGS-B.
    if best_pts is not None and time.time() - start_time < TIME_LIMIT - SAFETY_MARGIN:
         # Normalize best_pts to unit variance for stability
         std = np.std(best_pts)
         if std > 1e-10:
             best_pts_norm = best_pts / std
         else:
             best_pts_norm = best_pts
             
         # Run a final high-precision optimization
         final_pts, final_score = nelder_mead_optimize(best_pts_norm, TIME_LIMIT - (time.time() - start_time), power=20)
         
         if final_score > best_score:
             best_pts = final_pts
             best_score = final_score
             
         # One more try with power=50
         final_pts2, final_score2 = nelder_mead_optimize(best_pts_norm, TIME_LIMIT - (time.time() - start_time), power=50)
         if final_score2 > best_score:
             best_pts = final_pts2
             best_score = final_score2

    if best_pts is None:
        best_pts = np.random.randn(N, 2)
        
    # Ensure finite
    if not np.all(np.isfinite(best_pts)):
        best_pts = candidates[0]

    return best_pts
