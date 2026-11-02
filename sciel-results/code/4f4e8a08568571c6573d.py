import numpy as np
import time
from scipy.optimize import minimize

N = 50

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    best_radius = -np.inf
    best_centers = None
    
    # Soft-min parameter beta. Higher is more accurate to min, lower is smoother.
    # We can try a few values or anneal. Let's start with a fixed reasonable value.
    beta = 20.0
    
    # Number of restarts to attempt
    max_restarts = 20
    
    for restart in range(max_restarts):
        if time.time() > deadline:
            break
            
        # Random initialization
        # Scale initial positions to be within [-0.4, 0.4] to avoid immediate wall clipping issues in gradients?
        # No, L-BFGS-B handles bounds. Random in [-0.5, 0.5] is fine.
        centers = np.random.uniform(-0.5, 0.5, (N, 3))
        
        # Define objective and gradient
        def objective_and_grad(flat_centers):
            c = flat_centers.reshape(N, 3)
            
            # Calculate all pairwise distances
            # c: (N, 3)
            # diff: (N, N, 3)
            diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
            dists = np.linalg.norm(diff, axis=2)
            
            # Mask diagonal (dist=0)
            mask = ~np.eye(N, dtype=bool)
            # We only need upper triangle for i<j to avoid double counting?
            # The soft-min sums over all terms. If we include i,j and j,i, it's just 2x the term.
            # It doesn't change the argmin, just scales the weight.
            # Let's just use all i != j.
            
            v_pair = 0.5 * dists
            v_pair[~mask] = np.inf # Exclude self
            
            # Wall distances
            # For each center, distance to 6 walls? Or just min of 3 axes?
            # d_wall = min(0.5 - |cx|, 0.5 - |cy|, 0.5 - |cz|)
            # This is the min of 3 values per sphere.
            # Let's include all 3 axis distances in the sum.
            # v_wall[i, 0] = 0.5 - abs(c[i,0])
            # v_wall[i, 1] = 0.5 - abs(c[i,1])
            # v_wall[i, 2] = 0.5 - abs(c[i,2])
            
            abs_c = np.abs(c)
            v_wall = 0.5 - abs_c # Shape (N, 3)
            
            # Combine all values
            # Flatten v_pair (mask out inf) and v_wall
            # To compute gradient, we need to know which term contributed to which coordinate.
            
            # Let's compute the weights for the soft-min
            # LSE = -1/beta * ln( sum exp(-beta * v) )
            # We want to maximize LSE, so minimize -LSE.
            
            # Create a list of all values and their gradients
            # This might be slow in Python loops. Let's vectorize.
            
            # All values:
            # Pairwise: shape (N, N), exclude diagonal.
            # Wall: shape (N, 3).
            
            # Let's flatten them into a single vector of values V
            # And a matrix of gradients? No, gradient is w.r.t flat_centers.
            
            # Let's compute the unnormalized weights w_k = exp(-beta * v_k)
            # Sum W = sum(w_k)
            # Soft-min value S = -1/beta * ln(W)
            # Grad S wrt v_k = w_k / W
            # Grad S wrt c = sum_k (w_k / W) * (Grad v_k wrt c)
            
            # 1. Pairwise terms
            # v_ij = 0.5 * dist_ij
            # Grad v_ij wrt c_i = 0.5 * (c_i - c_j) / dist_ij
            # Grad v_ij wrt c_j = 0.5 * (c_j - c_i) / dist_ij
            
            # We need to handle dist_ij == 0.
            safe_dists = np.where(dists > 1e-8, dists, 1.0)
            
            # Weights for pairwise terms
            # Only for i != j
            exp_pair = np.exp(-beta * v_pair)
            exp_pair[~mask] = 0.0
            
            # 2. Wall terms
            # v_ik = 0.5 - |c_ik|
            # Grad v_ik wrt c_ik = -sign(c_ik)
            
            exp_wall = np.exp(-beta * v_wall)
            
            # Total Sum
            W = np.sum(exp_pair) + np.sum(exp_wall)
            
            if W == 0:
                return 0, np.zeros_like(flat_centers)
                
            # Objective: -SoftMin = 1/beta * ln(W)
            obj = (1.0 / beta) * np.log(W)
            
            # Gradient
            grad = np.zeros((N, 3))
            
            # Pairwise contribution
            # grad[i] += sum_{j != i} (exp_pair[i,j] / W) * (0.5 * (c_i - c_j) / dist_ij)
            # Note: exp_pair[i,j] is the weight for the term v_ij.
            
            # Vectorized pairwise gradient:
            # We need to sum over j for each i.
            # coeff[i, j] = exp_pair[i, j] / W * 0.5 / safe_dists[i, j]
            # grad[i] = sum_j coeff[i, j] * (c[i] - c[j])
            
            coeff = exp_pair / W * 0.5 / safe_dists
            # Set diagonal to 0 to be safe, though exp_pair is 0 there
            np.fill_diagonal(coeff, 0.0)
            
            # grad_pair[i] = sum_j coeff[i,j] * c[i] - sum_j coeff[i,j] * c[j]
            # = c[i] * sum_j coeff[i,j] - (coeff @ c)[i]
            
            sum_coeff = np.sum(coeff, axis=1) # Shape (N,)
            # coeff @ c: (N, N) @ (N, 3) -> (N, 3)
            grad_pair = c * sum_coeff[:, np.newaxis] - (coeff @ c)
            
            # Wall contribution
            # grad_wall[i, k] = (exp_wall[i, k] / W) * (-sign(c[i, k]))
            
            grad_wall = (exp_wall / W) * (-np.sign(c))
            
            grad = grad_pair + grad_wall
            
            return obj, grad.reshape(-1)
        
        # Run optimization
        # Bounds
        bounds = [(-0.5, 0.5)] * (N * 3)
        
        try:
            res = minimize(
                objective_and_grad, 
                centers.flatten(), 
                method='L-BFGS-B', 
                jac=True, 
                bounds=bounds,
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8}
            )
        except Exception:
            continue
            
        opt_centers = res.x.reshape(N, 3)
        
        # Evaluate true radius
        # Min pairwise dist / 2
        diff = opt_centers[:, np.newaxis, :] - opt_centers[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dists, np.inf)
        min_pair_dist = np.min(dists) / 2.0
        
        # Min wall dist
        abs_c = np.abs(opt_centers)
        wall_dists = 0.5 - abs_c
        min_wall_dist = np.min(wall_dists)
        
        true_radius = min(min_pair_dist, min_wall_dist)
        
        if true_radius > best_radius:
            best_radius = true_radius
            best_centers = opt_centers.copy()
            
    return best_centers
