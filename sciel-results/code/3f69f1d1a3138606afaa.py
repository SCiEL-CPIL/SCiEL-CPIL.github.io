import numpy as np
import time
from scipy.optimize import minimize

N = 50

def _compute_radius(centers):
    """Compute the exact maximum common radius."""
    # Distance to nearest face
    face_dist = 0.5 - np.max(np.abs(centers), axis=1)
    min_face = np.min(face_dist)
    
    # Minimum pairwise distance
    if N < 2:
        return min_face
    
    # Compute all pairwise distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    # Zero out diagonal
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists)
    
    return min(min_face, min_pair / 2.0)

def _soft_min_objective_and_grad(x, beta):
    """
    Compute the negative soft-min of the minimum distance constraints and its gradient.
    We want to maximize min(v_i), where v_i are distances to boundaries and half-distances between pairs.
    soft_min(v) = -1/beta * log(sum(exp(-beta * v_i)))
    We minimize -soft_min(v) = 1/beta * log(sum(exp(-beta * v_i)))
    
    Let S = sum(exp(-beta * v_i)).
    obj = (1/beta) * log(S)
    grad_obj = (1/beta) * (1/S) * dS/dx
    dS/dx = sum(-beta * exp(-beta * v_i) * grad_v_i)
    So grad_obj = - sum(exp(-beta * v_i) / S * grad_v_i)
    Let w_i = exp(-beta * v_i) / S.
    grad_obj = - sum(w_i * grad_v_i)
    
    The variables x are the flattened centers (N*3).
    v_i are:
    1. For each point k and axis d: dist to face = 0.5 - |x_{k,d}|. 
       Note: The problem defines distance to boundary as min(0.5-x, 0.5+x).
       This is equivalent to 0.5 - |x|.
       grad_x (0.5 - |x|) = -sign(x).
    2. For each pair (i, j): dist/2 = 0.5 * ||x_i - x_j||.
       grad_{x_i} = 0.5 * (x_i - x_j) / ||x_i - x_j||
       grad_{x_j} = 0.5 * (x_j - x_i) / ||x_i - x_j||
    """
    centers = x.reshape(N, 3)
    
    # 1. Boundary terms
    # v_b[k, d] = 0.5 - |x[k, d]|
    abs_centers = np.abs(centers)
    v_bounds = 0.5 - abs_centers  # shape (N, 3)
    v_bounds_flat = v_bounds.flatten()
    
    # 2. Pairwise terms
    idx_i, idx_j = np.triu_indices(N, k=1)
    diffs = centers[idx_i] - centers[idx_j]
    dists = np.linalg.norm(diffs, axis=1)
    # Avoid division by zero
    safe_dists = np.where(dists > 1e-12, dists, 1.0)
    v_pairs = 0.5 * dists
    
    # Combine all v values
    v_all = np.concatenate([v_pairs, v_bounds_flat])
    
    # Calculate weights w_i = exp(-beta * v_i) / sum(exp(-beta * v_j))
    # Use log-sum-exp trick for stability
    max_v = np.max(v_all)
    exp_terms = np.exp(-beta * (v_all - max_v))
    sum_exp = np.sum(exp_terms)
    weights = exp_terms / sum_exp
    
    # Objective value: (1/beta) * log(sum_exp) + (-max_v) 
    # Wait: log(sum(exp(-beta(v_i - max_v)))) = log(sum(exp(-beta v_i + beta max_v)))
    # = log(exp(beta max_v) * sum(exp(-beta v_i)))
    # = beta max_v + log(sum(exp(-beta v_i)))
    # So sum(exp(-beta v_i)) = exp(log_sum - beta max_v) ? No.
    # Let L = log(sum(exp(-beta v_i))).
    # L = beta max_v + log(sum(exp(-beta (v_i - max_v))))
    # obj = L / beta = max_v + (1/beta) * log(sum(exp(-beta (v_i - max_v))))
    # sum_exp is sum(exp(-beta (v_i - max_v))).
    obj = max_v + (1.0 / beta) * np.log(sum_exp)
    
    # Gradient
    # grad_obj = - sum(w_i * grad_v_i)
    grad = np.zeros_like(centers)
    
    # 1. Pairwise gradients
    w_pairs = weights[:len(idx_i)]
    unit_vecs = diffs / safe_dists[:, np.newaxis]
    # grad_{x_i} v_pair = 0.5 * unit_vec
    # grad_{x_j} v_pair = -0.5 * unit_vec
    forces = 0.5 * w_pairs[:, np.newaxis] * unit_vecs
    np.add.at(grad, idx_i, forces)
    np.add.at(grad, idx_j, -forces)
    
    # 2. Boundary gradients
    w_bounds = weights[len(idx_i):].reshape(N, 3)
    signs = np.sign(centers)
    # grad_{x} (0.5 - |x|) = -sign(x)
    # So term is w * (-sign(x))
    grad += -w_bounds * signs
    
    return obj, grad.flatten()

def _optimize_from_init(initial_centers, deadline):
    """
    Run L-BFGS-B with a continuation schedule for beta.
    """
    current_centers = initial_centers.copy()
    beta = 5.0
    beta_end = 1000.0
    steps = 100 # Iterations per stage? L-BFGS-B handles iterations internally.
    # We will run minimize for a few stages.
    
    best_centers = current_centers.copy()
    best_radius = _compute_radius(current_centers)
    
    # Continuation stages
    # Start with low beta (smooth), end with high beta (sharp min)
    betas = [5.0, 10.0, 20.0, 50.0, 100.0, 200.0, 500.0, 1000.0]
    
    for b in betas:
        if time.time() > deadline - 5.0:
            break
            
        # Define objective and gradient for this beta
        def fun(x):
            return _soft_min_objective_and_grad(x, b)[0]
        
        def jac(x):
            return _soft_min_objective_and_grad(x, b)[1]
        
        # Bounds: [-0.5, 0.5] for each coordinate
        bounds = [(-0.5, 0.5)] * (N * 3)
        
        # Run L-BFGS-B
        try:
            res = minimize(fun, current_centers.flatten(), jac=jac, method='L-BFGS-B', bounds=bounds, 
                           options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8})
            if res.success or res.status == 0:
                current_centers = res.x.reshape(N, 3)
            else:
                # If it fails, keep current
                pass
                
            # Evaluate exact radius
            r = _compute_radius(current_centers)
            if r > best_radius:
                best_radius = r
                best_centers = current_centers.copy()
                
        except Exception:
            pass
            
    return best_centers

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s margin
    
    # Initial guess: 4x4x4 grid
    # 4x4x4 = 64 points. We need 50.
    # We can take a random subset of 50 to break symmetry.
    m = 4
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    
    # Randomly select 50 points from the 64 grid points
    np.random.seed(42)
    indices = np.random.choice(len(grid), N, replace=False)
    initial_centers = grid[indices].copy()
    
    # Add small noise to avoid exact symmetries that might cause issues or get stuck
    initial_centers += np.random.normal(0, 0.01, initial_centers.shape)
    initial_centers = np.clip(initial_centers, -0.49, 0.49) # Keep strictly inside initially
    
    # Optimization
    best_centers = _optimize_from_init(initial_centers, deadline)
    
    # Final polish: Run one more stage with very high beta if time permits
    if time.time() < deadline - 2.0:
         # Try a very high beta to sharpen the min
         beta = 2000.0
         def fun(x):
             return _soft_min_objective_and_grad(x, beta)[0]
         def jac(x):
             return _soft_min_objective_and_grad(x, beta)[1]
         bounds = [(-0.5, 0.5)] * (N * 3)
         try:
             res = minimize(fun, best_centers.flatten(), jac=jac, method='L-BFGS-B', bounds=bounds, 
                            options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-10})
             if res.success:
                 cand = res.x.reshape(N, 3)
                 r = _compute_radius(cand)
                 r_best = _compute_radius(best_centers)
                 if r > r_best:
                     best_centers = cand
         except Exception:
             pass

    return best_centers
