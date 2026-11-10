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
    n = 16
    d = 2
    start_time = time.time()
    time_limit = 500  # Keep well under 600s

    # Helper to normalize points: center at 0, scale such that sum of squares is 1
    def normalize(points):
        pts = points.copy()
        pts -= np.mean(pts, axis=0)
        norm_sq = np.sum(pts**2)
        if norm_sq > 1e-12:
            pts /= np.sqrt(norm_sq)
        return pts

    # Objective function: minimize negative soft-min of distances
    # Soft-min approximates the minimum distance
    beta = 50.0  # Temperature parameter for soft-min
    
    def objective_and_grad(flat_vec):
        pts = flat_vec.reshape(n, d)
        pts = normalize(pts)
        
        # Compute pairwise distances
        dists = pdist(pts)
        
        # Soft min: -1/beta * log(sum(exp(-beta * d)))
        # For numerical stability, subtract max
        # We want to maximize min(d), so we minimize -soft_min
        # soft_min = -1/beta * log(sum(exp(-beta * d)))
        # -soft_min = 1/beta * log(sum(exp(-beta * d)))
        
        # To avoid underflow/overflow, use log-sum-exp trick
        # max_d = np.max(dists)
        # log_sum = max_d + log(sum(exp(-beta * (dists - max_d))))
        # This is for sum(exp(beta * d)). We have sum(exp(-beta * d)).
        # Let's just use the direct formula if beta is not too large, or shift.
        
        # Actually, for maximizing the minimum, the weights w_ij = exp(-beta * d_ij) / sum(...)
        # are largest for the smallest distances.
        
        # Gradient of objective f = 1/beta * log(sum(exp(-beta * d_ij)))
        # df/dx = 1/beta * (sum(exp(-beta * d_ij) * (-beta) * dd_ij/dx)) / sum(exp(-beta * d_ij))
        # df/dx = - sum(w_ij * dd_ij/dx)
        # where w_ij = exp(-beta * d_ij) / sum(exp(-beta * d_ij))
        
        # Calculate weights
        # Use log-sum-exp for stability
        # log_sum_exp(-beta * dists)
        max_val = np.max(-beta * dists)
        log_sum = max_val + np.log(np.sum(np.exp(-beta * dists - max_val)))
        
        # Weights w_ij
        log_w = -beta * dists - log_sum
        w = np.exp(log_w)
        
        # Objective value
        obj_val = log_sum / beta
        
        # Gradient
        # We need d(d_ij)/dx for all pairs
        # This is a bit expensive to do in a loop for every pair, but N=16 is small.
        # We can vectorize.
        # dists is of shape (120,). We need to map back to pairs.
        # pdist returns distances in a specific order: (0,1), (0,2), ..., (0,15), (1,2), ...
        
        grad = np.zeros((n, d))
        
        # Efficient gradient calculation
        # For each pair (i, j), grad_i += w_ij * (p_j - p_i) / d_ij
        # grad_j += w_ij * (p_i - p_j) / d_ij
        
        # We can use a loop over the 120 pairs.
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                w_ij = w[idx]
                if w_ij < 1e-10:
                    idx += 1
                    continue
                p_i = pts[i]
                p_j = pts[j]
                d_ij = dists[idx]
                if d_ij < 1e-10:
                    # Handle coincident points
                    diff = p_i - p_j
                    if np.linalg.norm(diff) < 1e-10:
                        # Random small perturbation? Or just skip?
                        # If coincident, gradient is undefined. 
                        # In practice, with continuous optimization, they won't be exactly 0.
                        pass
                    else:
                        grad_vec = diff / d_ij
                        grad[i] += w_ij * grad_vec
                        grad[j] -= w_ij * grad_vec
                else:
                    diff = p_i - p_j
                    grad_vec = diff / d_ij
                    grad[i] += w_ij * grad_vec
                    grad[j] -= w_ij * grad_vec
                idx += 1
                
        # The objective is f = log_sum / beta
        # df/dx = - sum(w_ij * dd_ij/dx)
        # Wait, let's re-verify the sign.
        # f = 1/beta * log( sum exp(-beta d) )
        # df/dx = 1/beta * 1/sum * sum( exp(-beta d) * (-beta) * dd/dx )
        # df/dx = - sum( w_ij * dd/dx )
        # So we minimize f, so we subtract the gradient.
        # The gradient of f is - sum(w_ij * dd/dx).
        # My loop calculates sum(w_ij * dd/dx) into `grad`.
        # So the gradient of the objective is -grad.
        
        return obj_val, -grad

    def score(points):
        pts = normalize(points)
        dists = pdist(pts)
        d_min = np.min(dists)
        d_max = np.max(dists)
        if d_max == 0:
            return 0.0
        return (d_min / d_max)**2

    best_points = None
    best_score = -1.0

    # Generate initial guesses
    inits = []
    
    # 1. 4x4 Grid
    x = np.linspace(-1.5, 1.5, 4)
    y = np.linspace(-1.5, 1.5, 4)
    xx, yy = np.meshgrid(x, y)
    grid_pts = np.column_stack((xx.ravel(), yy.ravel()))
    inits.append(grid_pts)

    # 2. Hexagonal-like
    # Rows of 4, 4, 4, 4 with offsets
    hex_pts = np.zeros((16, 2))
    row_height = np.sqrt(3) / 2
    for r in range(4):
        offset = 0.5 if r % 2 == 1 else 0.0
        for c in range(4):
            hex_pts[r*4 + c, 0] = c + offset
            hex_pts[r*4 + c, 1] = r * row_height
    inits.append(hex_pts)

    # 3. Random
    np.random.seed(42)
    rand_pts = np.random.randn(16, 2)
    inits.append(rand_pts)

    # 4. Concentric circles
    circ_pts = np.zeros((16, 2))
    # 4 on inner, 12 on outer? Or 8 and 8?
    # Let's try 4 inner, 12 outer
    angles_inner = np.linspace(0, 2*np.pi, 4, endpoint=False)
    for i in range(4):
        circ_pts[i, 0] = 0.5 * np.cos(angles_inner[i])
        circ_pts[i, 1] = 0.5 * np.sin(angles_inner[i])
    angles_outer = np.linspace(0, 2*np.pi, 12, endpoint=False)
    for i in range(12):
        circ_pts[4+i, 0] = 1.0 * np.cos(angles_outer[i])
        circ_pts[4+i, 1] = 1.0 * np.sin(angles_outer[i])
    inits.append(circ_pts)

    for init in inits:
        if time.time() - start_time > time_limit:
            break
            
        init_norm = normalize(init)
        flat_init = init_norm.ravel()
        
        try:
            res = minimize(
                objective_and_grad,
                flat_init,
                method='L-BFGS-B',
                jac=True,
                options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            
            res_pts = normalize(res.x.reshape(n, d))
            res_score = score(res_pts)
            
            if res_score > best_score:
                best_score = res_score
                best_points = res_pts
                
        except Exception as e:
            pass

    # If no optimization succeeded, return the best initial guess
    if best_points is None:
        for init in inits:
            init_norm = normalize(init)
            s = score(init_norm)
            if s > best_score:
                best_score = s
                best_points = init_norm

    return best_points

# EVOLVE-BLOCK-END
