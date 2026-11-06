"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 300

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit

    # --- Stage 1: Initialization ---
    # Use a Fermat spiral for a dense, symmetric start.
    # We scale it to fit comfortably inside the unit circle with a target radius.
    # Target radius r_target ~ 0.053.
    # The spiral radius goes from 0 to 1-r_target.
    
    i = np.arange(N)
    # Golden angle
    phi = np.pi * (3.0 - np.sqrt(5.0))
    # Distribute points by area
    r_spiral = np.sqrt(i / N) * (1.0 - 0.053)
    theta = i * phi
    
    points = np.column_stack([r_spiral * np.cos(theta), r_spiral * np.sin(theta)])
    
    # --- Stage 2: Optimization ---
    # We want to maximize r.
    # Constraints:
    # 1. ||p_i - p_j|| >= 2r  for all i < j
    # 2. ||p_i|| <= 1 - r
    
    # This is a non-smooth problem. We smooth it using a Log-Sum-Exp (Soft-Min) approximation.
    # Let D_ij = ||p_i - p_j||. We want min D_ij >= 2r.
    # Equivalent to maximizing min( D_ij/2, 1-||p_i|| ).
    
    # We define a smooth objective function to maximize:
    # f(x, r) = - (1/T) * log( sum_{i<j} exp(- (D_ij - 2r) / T) + sum_i exp(- (1-||p_i|| - r) / T) )
    # Wait, standard soft-min is: min(x) approx -1/T log(sum exp(-x/T)).
    # We want to maximize the minimum clearance.
    # Let clearance C = min( D_ij/2, 1-||p_i|| ).
    # Maximize C.
    # Smooth approximation: C_smooth = -T * log( sum exp(-C_k / T) ) where C_k are the clearances.
    
    # Variables: x (2N coordinates) and r (scalar).
    # We can optimize over x and r jointly.
    # Objective: Minimize - C_smooth(x, r)
    # Subject to: C_smooth(x, r) >= r (actually we want to find the max r such that constraints hold).
    
    # A better formulation:
    # Maximize r
    # Subject to: D_ij >= 2r + eps, ||p_i|| <= 1 - r - eps
    # This is hard for L-BFGS-B directly with many constraints.
    
    # Alternative: Penalized Objective
    # Maximize: r - lambda * Penalty(x, r)
    # Where Penalty = sum of violations.
    # Violation_ij = max(0, 2r - D_ij)
    # Violation_i = max(0, r + ||p_i|| - 1)
    
    # To make it differentiable, use soft-plus:
    # SoftPlus(z) = log(1 + exp(z)) ~ max(0, z) for large z.
    # Penalty = sum SoftPlus((2r - D_ij)/sigma) + sum SoftPlus((r + ||p_i|| - 1)/sigma)
    
    # We will use a smooth approximation for the "minimum distance" directly.
    # Let's define the objective to maximize:
    # Obj = r - mu * ( 1/T * log( sum_{i<j} exp(-(D_ij/2 - r)/T) + sum_i exp(-(1-||p_i|| - r)/T) ) )
    # This is getting complex. Let's stick to a robust penalty method on the constraints.
    
    # Actually, the "Soft-Min" of the clearances is a very good surrogate for the max-min.
    # Let C_k be the clearances:
    # C_pair_ij = D_ij / 2
    # C_bound_i = 1 - ||p_i||
    # We want to maximize min(C_k).
    # Smooth min: S = -T * log( sum_k exp(-C_k / T) )
    # As T -> 0, S -> min(C_k).
    # We can simply maximize S. The value of S will be close to the optimal r.
    # We don't even need r as a variable! We just maximize the smooth min of the clearances.
    
    def smooth_min_clearance(x, T):
        p = x.reshape(N, 2)
        # Pairwise distances
        # To save memory/time, we can compute in chunks or use cdist
        # N=300 is small enough for full matrix.
        diff = p[:, np.newaxis, :] - p[np.newaxis, :, :]
        dist = np.sqrt(np.sum(diff**2, axis=-1))
        
        # Boundary distances
        norms = np.sqrt(np.sum(p**2, axis=1))
        bound_dist = 1.0 - norms
        
        # Clearances
        pair_clearances = dist / 2.0
        # We only care about i < j
        i_idx, j_idx = np.triu_indices(N, k=1)
        pc = pair_clearances[i_idx, j_idx]
        
        # Combine all clearances
        # We want to avoid the diagonal (distance 0) in pair_clearances, but we masked it.
        # We also need to handle the boundary.
        
        # Soft min: -T * log( sum exp(-c / T) )
        # To prevent overflow in exp, we can shift by min_c.
        # c_min = min(min(pc), min(bound_dist))
        # sum exp(-(c - c_min)/T) = exp(c_min/T) * sum exp(-c/T)
        # log(sum exp(-c/T)) = c_min/T + log(sum exp(-(c-c_min)/T))
        # S = -T * (c_min/T + log(...)) = -c_min - T*log(...)
        # This is stable.
        
        all_clearances = np.concatenate([pc, bound_dist])
        
        # Clip clearances to a small positive number to avoid log(0) or issues if points coincide
        # If points coincide, clearance is 0. exp(-0/T) = 1.
        # However, if clearance is very negative (overlap), exp(-neg/T) = exp(pos) -> overflow.
        # We should clip the argument of exp.
        # Actually, if there is overlap, the penalty should be huge.
        # Let's just compute the sum of exps carefully.
        
        c_min = np.min(all_clearances)
        
        # Shift
        shifted = all_clearances - c_min
        # shifted >= 0
        # exp(-shifted/T) <= 1
        # Sum is at most N_clearances
        log_sum = np.log(np.sum(np.exp(-shifted / T)))
        
        S = -c_min - T * log_sum
        
        # Gradient calculation is tricky if we do it analytically inside the function.
        # L-BFGS-B can use numerical gradients, but that's slow (2N+1 evaluations).
        # Let's provide an analytical gradient.
        
        # dS/dx = - d(c_min)/dx - T * d(log_sum)/dx
        # This is complex because c_min changes which element is the min.
        # However, for the soft-min, the gradient is:
        # dS/dx = - ( sum_k (1/T * exp(-c_k/T)) * dc_k/dx ) / ( sum_k exp(-c_k/T) )
        # Let w_k = exp(-c_k/T) / sum(exp(-c_j/T)). These are softmax weights.
        # dS/dx = - sum_k w_k * (1/T) * dc_k/dx ? 
        # Let's re-derive.
        # S = -T log( sum exp(-c_k/T) )
        # dS/dx = -T * (1 / sum exp(-c_k/T)) * sum ( exp(-c_k/T) * (-1/T) * dc_k/dx )
        # dS/dx = (1 / sum exp(-c_k/T)) * sum ( exp(-c_k/T) * dc_k/dx )
        # dS/dx = sum ( w_k * dc_k/dx ) where w_k = exp(-c_k/T) / sum exp(-c_j/T)
        
        # So the gradient of S is the weighted average of the gradients of the individual clearances.
        # dc_pair_ij/dx_i = (p_i - p_j) / D_ij
        # dc_pair_ij/dx_j = (p_j - p_i) / D_ij
        # dc_bound_i/dx_i = -p_i / ||p_i||
        
        # We can compute this efficiently.
        
        # Weights for pairs
        # We need to map back to the full matrix or accumulate.
        
        # Let's compute the weights first.
        # exp_terms = exp(-all_clearances / T)
        # To avoid overflow/underflow, use the shifted version.
        # exp(-shifted/T) is safe.
        exp_terms = np.exp(-shifted / T)
        sum_exp = np.sum(exp_terms)
        weights = exp_terms / sum_exp
        
        # Split weights back into pair and boundary parts
        n_pairs = len(pc)
        w_pairs = weights[:n_pairs]
        w_bounds = weights[n_pairs:]
        
        # Gradient of pair clearances
        # grad_x[i] += sum_{j>i} w_ij * (p_i - p_j)/D_ij
        # grad_x[i] += sum_{j<i} w_ji * (p_i - p_j)/D_ij  (Note: w_ji corresponds to pair (j,i) which is same as (i,j) in our list? 
        # Our list is i<j. So for a specific pair (i,j) with i<j, the weight is w_pairs[idx].
        # dc/dx_i = (p_i - p_j)/D
        # dc/dx_j = (p_j - p_i)/D = - dc/dx_i
        
        grad_x = np.zeros_like(x)
        
        # We need the distance matrix again for normalization
        # We already have dist matrix.
        # But we need to be careful with zeros.
        
        # Iterate over pairs? N=300 -> 45k pairs. Python loop might be slow.
        # Let's use vectorized operations.
        
        # Create a matrix of weights for the full NxN distance matrix?
        # W_full[i,j] = w_pairs[idx] if i<j else 0.
        
        # Actually, let's just compute the contribution.
        # grad_x = sum_{i<j} w_ij * ( dc_ij/dx_i * e_i + dc_ij/dx_j * e_j )
        # dc_ij/dx_i = (p_i - p_j) / D_ij
        # dc_ij/dx_j = (p_j - p_i) / D_ij
        
        # Let V_ij = (p_i - p_j) / D_ij.
        # grad_x[i] += w_ij * V_ij
        # grad_x[j] += w_ij * (-V_ij)
        
        # We can use scatter-add.
        
        # Get indices
        ii, jj = i_idx, j_idx
        
        # Compute vectors V
        # diff is p[:,None,:] - p[None,:,:]
        # V[i,j] = diff[i,j] / dist[i,j]
        
        # Handle dist=0
        dist_safe = np.where(dist < 1e-9, 1.0, dist)
        V = diff / dist_safe[:, :, np.newaxis]
        
        # Get V for the pairs
        V_pairs = V[ii, jj] # Shape (n_pairs, 2)
        
        # Contributions to i and j
        # contrib_i = w_pairs * V_pairs
        # contrib_j = -w_pairs * V_pairs
        
        # Scatter add
        # np.add.at(grad_x, ii, contrib_i)
        # np.add.at(grad_x, jj, -contrib_i)
        
        # np.add.at is slow. Let's try to use bincount or matrix ops if possible, or just accept the speed for N=300.
        # 45k additions * 2 coords. It's okay.
        
        np.add.at(grad_x, ii, w_pairs[:, np.newaxis] * V_pairs)
        np.add.at(grad_x, jj, -w_pairs[:, np.newaxis] * V_pairs)
        
        # Boundary gradient
        # dc_bound_i/dx_i = -p_i / ||p_i||
        # grad_x[i] += w_bounds[i] * (-p_i / ||p_i||)
        
        norms_safe = np.where(norms < 1e-9, 1.0, norms)
        V_bound = -p / norms_safe[:, np.newaxis]
        
        grad_x += w_bounds[:, np.newaxis] * V_bound
        
        return -S, -grad_x # We want to maximize S, so minimize -S. Gradient of -S is -grad_S.

    # Annealing T
    # Start with large T (smooth) and decrease T (sharper).
    
    best_x = points.copy()
    best_score = -np.inf
    
    T = 0.05 # Start smooth
    # Decrease T over time
    # We will run several rounds of L-BFGS-B with decreasing T.
    
    rounds = 5
    for round_idx in range(rounds):
        if time.time() > deadline - 10:
            break
            
        # Define objective for this T
        def obj(x):
            val, grad = smooth_min_clearance(x, T)
            return val, grad
            
        # L-BFGS-B
        try:
            res = minimize(obj, points, method='L-BFGS-B', jac=True, options={'maxiter': 100, 'ftol': 1e-10, 'gtol': 1e-8})
            points = res.x
            
            # Evaluate true score (hard min)
            p = points.reshape(N, 2)
            diff = p[:, np.newaxis, :] - p[np.newaxis, :, :]
            dist = np.sqrt(np.sum(diff**2, axis=-1))
            norms = np.sqrt(np.sum(p**2, axis=1))
            
            i_idx, j_idx = np.triu_indices(N, k=1)
            min_pair = np.min(dist[i_idx, j_idx]) / 2.0
            min_bound = np.min(1.0 - norms)
            true_score = min(min_pair, min_bound)
            
            if true_score > best_score:
                best_score = true_score
                best_x = points.copy()
                
            # Decrease T
            T *= 0.5
            
        except Exception as e:
            # If optimization fails, keep previous points
            pass

    # Final check and return
    return best_x

# EVOLVE-BLOCK-END
