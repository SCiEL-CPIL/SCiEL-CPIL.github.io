import numpy as np
import time

N = 50

def objective_and_grad(x_flat):
    x = x_flat.reshape(N, 3)
    # Pairwise distances
    # Diff matrix: (N, N, 3)
    diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    # Avoid division by zero
    np.fill_diagonal(dists, 1.0) # Diagonal doesn't matter for sum, but for grad calc we need to be careful
    
    # Wall distances
    # x is (N, 3)
    # W_i_k = 0.5 - |x_i_k|
    abs_x = np.abs(x)
    W = 0.5 - abs_x # (N, 3)
    
    lambda_val = 15.0
    
    # Objective
    # Pairwise term: sum_{i<j} exp(-lambda * dists[i,j])
    # We can use the full matrix and divide by 2, or use triu
    upper_idx = np.triu_indices(N, k=1)
    pair_dists = dists[upper_idx]
    pair_terms = np.exp(-lambda_val * pair_dists)
    obj_pair = np.sum(pair_terms)
    
    # Wall term: sum_{i,k} exp(-lambda * W[i,k])
    wall_terms = np.exp(-lambda_val * W)
    obj_wall = np.sum(wall_terms)
    
    obj = obj_pair + obj_wall
    
    # Gradient
    grad = np.zeros_like(x)
    
    # Pairwise gradient
    # d/dx_a sum_{j != a} exp(-lambda * D_aj)
    # = sum_{j != a} exp(-lambda * D_aj) * (-lambda) * (x_a - x_j) / D_aj
    # Vectorized:
    # For each i, sum over j
    # Let's compute the force on each point i
    # Force from j on i: -lambda * exp(-lambda * D_ij) * (x_i - x_j) / D_ij
    # Sum over all j != i
    
    # dists is (N, N)
    # diff is (N, N, 3)
    # We want to compute G_i = sum_{j != i} -lambda * exp(-lambda * dists[i,j]) * diff[i,j] / dists[i,j]
    
    # Mask diagonal
    mask = ~np.eye(N, dtype=bool)
    
    # Calculate coefficients for each pair (i,j)
    # coeff[i,j] = -lambda * exp(-lambda * dists[i,j]) / dists[i,j]
    # Set diagonal to 0
    coeff = np.zeros_like(dists)
    coeff[mask] = -lambda_val * np.exp(-lambda_val * dists[mask]) / dists[mask]
    
    # grad_pair[i] = sum_j coeff[i,j] * diff[i,j]
    # diff[i,j] = x_i - x_j
    # So grad_pair = coeff @ diff ? No.
    # grad_pair[i] = sum_j coeff[i,j] * (x_i - x_j)
    # = (sum_j coeff[i,j]) * x_i - sum_j (coeff[i,j] * x_j)
    
    # Let C = coeff (N, N)
    # Sum_C = C.sum(axis=1) (N,)
    # Term1 = Sum_C[:, np.newaxis] * x (N, 3)
    # Term2 = C @ x (N, 3)
    # grad_pair = Term1 - Term2
    
    Sum_C = coeff.sum(axis=1)
    Term1 = Sum_C[:, np.newaxis] * x
    Term2 = coeff @ x
    grad_pair = Term1 - Term2
    
    # Wall gradient
    # d/dx_i_k exp(-lambda * (0.5 - |x_i_k|))
    # = exp(-lambda * W) * (-lambda) * (-sign(x_i_k))
    # = lambda * sign(x_i_k) * exp(-lambda * W)
    # sign(0) is 0, which is fine (gradient is 0 at 0 for |x|? No, |x| has kink. But numerically it's okay, or we can use sign with 0->1? 
    # Actually, if x is close to 0, the force should push it away from the wall? 
    # Wait, W = 0.5 - |x|. Max W means min |x|? No.
    # We want to MAXIMIZE W. So we MINIMIZE exp(-lambda W).
    # If x > 0, W = 0.5 - x. dW/dx = -1. dObj/dx = -lambda exp(-lambda W) * (-1) = lambda exp(-lambda W).
    # This force is positive, pushing x to the right?
    # If x > 0, increasing x decreases W. We want to INCREASE W. So we want to DECREASE x.
    # So the gradient of the objective (which we are minimizing) should be positive?
    # If obj increases when x increases, grad is positive.
    # Let's check: x=0.4, W=0.1. x=0.41, W=0.09.
    # exp(-15*0.1) vs exp(-15*0.09).
    # exp(-1.5) = 0.22. exp(-1.35) = 0.26.
    # Obj increases. So grad is positive.
    # So we push x to decrease? No, gradient descent moves AGAINST the gradient.
    # If grad is positive, we decrease x.
    # Decreasing x (from 0.4 to 0.39) increases W (to 0.11).
    # So yes, it pushes x towards 0?
    # Wait. We want to maximize W. W is max when x=0.
    # So the optimizer will try to push all x coordinates to 0?
    # That would collapse all spheres to the center!
    # Ah. The wall constraint is $W_{ik} \ge r$.
    # If we just minimize sum of exp(-lambda W), we maximize the "average" or "soft" W.
    # But the pairwise term pushes them apart.
    # The balance is: push apart (pairwise) vs push to center (walls)?
    # No. The wall term $W = 0.5 - |x|$.
    # If x is near wall (0.5), W is small. exp(-lambda W) is large. We want to reduce this.
    # To reduce exp(-lambda W), we need to INCREASE W.
    # To increase W, we need to DECREASE |x|.
    # So yes, the wall term pulls points towards the center (0).
    # The pairwise term pushes them apart.
    # This is correct. The equilibrium is a balance.
    
    sign_x = np.sign(x)
    # Handle 0: sign(0)=0. Gradient is 0. This is a saddle point for |x|. 
    # But in practice, points won't stay exactly at 0 due to pairwise repulsion.
    grad_wall = lambda_val * sign_x * wall_terms
    
    grad = grad_pair + grad_wall
    
    return obj, grad.flatten()

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit
    
    # Initial guess: random points inside the cube
    # We want to maximize the minimum distance. 
    # A good initial guess for packing is a grid or random with some repulsion.
    # Let's start with a simple random initialization inside [-0.4, 0.4] to avoid immediate wall clashes.
    np.random.seed(42)
    x = np.random.uniform(-0.4, 0.4, (N, 3))
    
    best_x = x.copy()
    best_score = -1.0
    
    # Simple gradient descent with line search or fixed step
    # The objective is smooth (sum of exps).
    # We minimize the objective to maximize the "soft" minimum distance.
    
    step_size = 0.01
    momentum = 0.0
    v = np.zeros((N, 3))
    
    # We will iterate until time runs out
    while time.time() < deadline:
        obj, grad_flat = objective_and_grad(x.flatten())
        grad = grad_flat.reshape(N, 3)
        
        # Momentum update
        v = 0.9 * v - 0.001 * grad
        x_new = x + v
        
        # Project back to valid range? 
        # The objective penalizes going outside, but hard constraints are better.
        # The problem statement says "inside the cube". 
        # If x goes outside, W becomes negative, exp(-lambda * W) becomes huge.
        # So the gradient will push it back.
        # However, to be safe and efficient, we can clip x to [-0.5, 0.5].
        # But clipping changes the gradient direction if we just clip.
        # Let's rely on the soft penalty for now, but clip to prevent extreme values.
        x_new = np.clip(x_new, -0.5, 0.5)
        
        # Evaluate raw score of x_new to track best
        # Score is min(min_pair_dist/2, min_wall_dist)
        diff = x_new[:, np.newaxis, :] - x_new[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dists, np.inf)
        min_pair_dist = np.min(dists)
        
        abs_x = np.abs(x_new)
        wall_dists = 0.5 - abs_x
        min_wall_dist = np.min(wall_dists)
        
        current_score = min(min_pair_dist / 2.0, min_wall_dist)
        
        if current_score > best_score:
            best_score = current_score
            best_x = x_new.copy()
            
        x = x_new
        
    return best_x
