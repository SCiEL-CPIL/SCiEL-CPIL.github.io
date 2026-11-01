import numpy as np
import time
from scipy.optimize import minimize

N = 50

def _softmin_obj_and_grad(c_flat, beta):
    """
    Compute the negative soft-min objective and its gradient.
    c_flat: shape (2*N,)
    beta: float
    Returns: objective value, gradient
    """
    c = c_flat.reshape(N, 2)
    
    # Compute all pairwise distances / 2
    # d[i, j] = ||c_i - c_j|| / 2
    # We only need i < j
    idx_i, idx_j = np.triu_indices(N, k=1)
    diffs = c[idx_i] - c[idx_j]
    dists = np.linalg.norm(diffs, axis=1) / 2.0
    
    # Compute distances to boundary: 1 - ||c_i||
    norms = np.linalg.norm(c, axis=1)
    boundary_dists = 1.0 - norms
    
    # Combine all values into a single vector y
    y = np.concatenate([dists, boundary_dists])
    
    # Soft-min: -1/beta * log(sum(exp(-beta * y)))
    # For numerical stability, subtract max(y)
    max_y = np.max(y)
    exp_terms = np.exp(-beta * (y - max_y))
    sum_exp = np.sum(exp_terms)
    
    if sum_exp == 0:
        # Fallback, shouldn't happen with reasonable beta and y
        obj = -max_y
        grad = np.zeros_like(c_flat)
        return obj, grad
        
    # Weights w_k = exp(-beta * (y_k - max_y)) / sum_exp
    w = exp_terms / sum_exp
    
    # Objective value
    obj = -(max_y - (1.0 / beta) * np.log(sum_exp))
    # Note: softmin(y) = max_y - 1/beta * log(sum(exp(-beta(y-max_y))))
    # We want to MINIMIZE -softmin(y)
    # So obj = -softmin(y)
    
    # Gradient
    # d(obj)/dc_m = sum_k w_k * d(-y_k)/dc_m
    # d(y_k)/dc_m is the gradient of the distance term.
    
    grad_c = np.zeros((N, 2))
    
    # Pairwise terms
    # For each pair (i, j) with weight w_pair:
    # d(y_ij)/dc_i = 0.5 * (c_i - c_j) / ||c_i - c_j||
    # d(y_ij)/dc_j = 0.5 * (c_j - c_i) / ||c_i - c_j||
    
    # We need to accumulate gradients.
    # Vectorized accumulation is tricky with variable indices, 
    # but N=50 is small, so a loop is fine, or we can use np.add.at
    
    # Let's use np.add.at for efficiency
    # w_pair corresponds to w[0 : len(dists)]
    w_pair = w[:len(dists)]
    
    # Normalize diffs to get unit vectors
    # Avoid division by zero
    norms_diff = np.linalg.norm(diffs, axis=1)
    mask = norms_diff > 1e-12
    unit_vecs = np.zeros_like(diffs)
    unit_vecs[mask] = diffs[mask] / norms_diff[mask, np.newaxis]
    
    # Gradient contribution to c_i: 0.5 * w * unit_vec
    # Gradient contribution to c_j: 0.5 * w * (-unit_vec)
    
    # np.add.at(grad_c, idx_i, 0.5 * w_pair[:, np.newaxis] * unit_vecs)
    # np.add.at(grad_c, idx_j, -0.5 * w_pair[:, np.newaxis] * unit_vecs)
    
    # Since we are minimizing -softmin, and softmin gradient is sum(w * grad_y),
    # the gradient of the objective (which is -softmin) is -sum(w * grad_y).
    # Wait.
    # obj = -softmin(y)
    # grad_obj = -grad_softmin(y)
    # grad_softmin(y) = sum(w_k * grad_y_k)
    # So grad_obj = -sum(w_k * grad_y_k)
    
    # Let's re-verify the sign.
    # We want to MAXIMIZE min(y).
    # We MINIMIZE -min(y).
    # softmin approximates min.
    # So we MINIMIZE -softmin(y).
    
    # grad(-softmin(y)) = - grad(softmin(y))
    # grad(softmin(y)) = sum(w_k * grad(y_k))
    
    # So we subtract the weighted gradients.
    
    contrib_i = 0.5 * w_pair[:, np.newaxis] * unit_vecs
    contrib_j = -0.5 * w_pair[:, np.newaxis] * unit_vecs
    
    np.add.at(grad_c, idx_i, -contrib_i)
    np.add.at(grad_c, idx_j, -contrib_j)
    
    # Boundary terms
    # y_i0 = 1 - ||c_i||
    # grad(y_i0) = -c_i / ||c_i||
    # Weight w_boundary = w[len(dists):]
    w_bound = w[len(dists):]
    
    # grad_obj += - w_bound * grad(y_i0)
    # grad_obj += - w_bound * (-c_i / ||c_i||)
    # grad_obj += w_bound * (c_i / ||c_i||)
    
    norms_safe = np.where(norms > 1e-12, norms, 1e-12)
    unit_norms = c / norms_safe[:, np.newaxis]
    
    grad_c += w_bound[:, np.newaxis] * unit_norms
    
    return obj, grad_c.flatten()

def _ring_init(perturb=0.01):
    """
    Initialize centers using a ring structure.
    """
    centers = np.zeros((N, 2))
    
    # Structure: 1, 6, 12, 18, 13
    # Approximate radii
    # r_circle ~ 0.125
    # Ring 1: 6 circles. Radius ~ 0.25
    # Ring 2: 12 circles. Radius ~ 0.45
    # Ring 3: 18 circles. Radius ~ 0.65
    # Ring 4: 13 circles. Radius ~ 0.85
    
    counts = [1, 6, 12, 18, 13]
    radii = [0.0, 0.25, 0.45, 0.65, 0.85]
    
    idx = 0
    for r_ring, count in zip(radii, counts):
        if count == 1:
            centers[idx] = [0.0, 0.0]
            idx += 1
        else:
            angles = np.linspace(0, 2 * np.pi, count, endpoint=False)
            # Add perturbation
            angles += np.random.normal(0, perturb, count)
            r_pert = r_ring + np.random.normal(0, perturb, count)
            
            x = r_pert * np.cos(angles)
            y = r_pert * np.sin(angles)
            
            centers[idx:idx+count, 0] = x
            centers[idx:idx+count, 1] = y
            idx += count
            
    return centers.flatten()

def _validate_and_score(centers):
    """
    Check validity and compute the raw score.
    """
    c = centers.reshape(N, 2)
    
    # Check finite
    if not np.all(np.isfinite(c)):
        return -1.0, False
        
    # Check inside container
    norms = np.linalg.norm(c, axis=1)
    if np.any(norms > 1.0):
        return -1.0, False
        
    # Check overlaps
    min_dist = 1.0
    for i in range(N):
        for j in range(i+1, N):
            d = np.linalg.norm(c[i] - c[j])
            if d < min_dist:
                min_dist = d
                
    r = min(min_dist / 2.0, 1.0 - np.max(norms))
    
    if r <= 0:
        return -1.0, False
        
    return r, True

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    best_score = 0.0
    best_centers = None
    
    # Beta schedule
    betas = [10, 50, 100, 200, 500]
    
    # Multiple restarts
    n_restarts = 5
    
    for restart in range(n_restarts):
        if time.time() > deadline:
            break
            
        # Initialize
        c0 = _ring_init(perturb=0.01 * (restart + 1))
        
        # Optimize with increasing beta
        current_c = c0
        for beta in betas:
            if time.time() > deadline:
                break
                
            try:
                res = minimize(
                    _softmin_obj_and_grad,
                    current_c,
                    args=(beta,),
                    method='L-BFGS-B',
                    jac=True,
                    options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-8}
                )
                current_c = res.x
            except Exception:
                # If optimization fails, try next beta or restart
                pass
        
        # Validate and score
        score, valid = _validate_and_score(current_c)
        if valid and score > best_score:
            best_score = score
            best_centers = current_c.copy()
            
        # Also check the initial guess just in case
        score0, valid0 = _validate_and_score(c0)
        if valid0 and score0 > best_score:
            best_score = score0
            best_centers = c0.copy()

    # If no valid solution found, return a simple valid one (e.g. grid)
    if best_centers is None:
        # Fallback: simple grid
        # This should not happen with ring init
        best_centers = _ring_init(perturb=0.0)
        score, valid = _validate_and_score(best_centers)
        if not valid:
            # Last resort: single point? No, N=50.
            # Just return random points inside unit circle, they will overlap, score 0 or invalid.
            # But we need to return SOMETHING.
            # Let's try to space them out roughly.
            angles = np.linspace(0, 2*np.pi, N, endpoint=False)
            radii = np.sqrt(np.random.uniform(0, 1, N))
            best_centers = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
            
    return best_centers.reshape(N, 2)
