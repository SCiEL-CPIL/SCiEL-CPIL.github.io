import numpy as np
import time
import scipy.optimize as opt

N = 200

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    time_limit = 145.0  # Safety margin before 150s kill
    
    best_score = -1.0
    best_centers = None
    
    # Helper to compute score
    def get_score(centers):
        # centers shape (N, 2)
        # Compute pairwise distances
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dists, np.inf)
        min_pair_dist = np.min(dists) / 2.0
        
        # Compute boundary distances
        # Distance to left/right walls: 0.5 - |x|
        # Distance to top/bottom walls: 0.5 - |y|
        min_bound_dist = np.min(np.hstack([
            0.5 - np.abs(centers[:, 0]),
            0.5 - np.abs(centers[:, 1])
        ]))
        
        return min(min_pair_dist, min_bound_dist)

    # Objective function for optimization
    # We minimize -softmin(distances)
    # distances include pairwise/2 and boundary distances
    def objective_and_grad(flat_centers, beta):
        centers = flat_centers.reshape(N, 2)
        
        # Compute pairwise distances
        # diff[i, j, k] = centers[i, k] - centers[j, k]
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        # We want min of dists/2, so let's work with r_pairs = dists / 2
        r_pairs = dists / 2.0
        np.fill_diagonal(r_pairs, np.inf)
        
        # Compute boundary distances
        # r_bounds[i, 0] = 0.5 - |x_i|
        # r_bounds[i, 1] = 0.5 - |y_i|
        r_bounds = np.hstack([
            0.5 - np.abs(centers[:, 0]),
            0.5 - np.abs(centers[:, 1])
        ]).reshape(-1)
        
        # Combine all radii into a single vector
        # We exclude inf values (diagonal)
        valid_pairs = np.isfinite(r_pairs)
        r_flat = np.concatenate([r_pairs[valid_pairs], r_bounds])
        
        # Soft-min: -1/beta * log(sum(exp(-beta * r)))
        # To avoid overflow/underflow, shift by max r
        max_r = np.max(r_flat)
        # If max_r is very small, exp(-beta*r) might be huge. 
        # Actually, we want to maximize min(r). 
        # Let's use the standard log-sum-exp trick.
        # S = sum(exp(-beta * r))
        # log(S) = -beta * max_r + log(sum(exp(-beta * (r - max_r))))
        # softmin = -1/beta * log(S) = max_r - 1/beta * log(sum(exp(-beta * (r - max_r))))
        
        shifted = -beta * (r_flat - max_r)
        # Clip shifted to avoid underflow issues in exp, though exp(-large) -> 0 is fine.
        # exp(large) is bad, but here argument is <= 0.
        exp_shifted = np.exp(shifted)
        sum_exp = np.sum(exp_shifted)
        
        if sum_exp == 0:
            val = -max_r # Fallback
            grad = np.zeros_like(flat_centers)
            return val, grad
            
        log_sum = np.log(sum_exp)
        val = -(max_r - log_sum / beta)
        
        # Gradient
        # d(val)/d(r_k) = -1/beta * d(log_sum)/d(r_k)
        # d(log_sum)/d(r_k) = (1/sum_exp) * d(sum_exp)/d(r_k)
        # d(sum_exp)/d(r_k) = sum_exp_k * (-beta)
        # So d(log_sum)/d(r_k) = -beta * exp_shifted[k] / sum_exp
        # d(val)/d(r_k) = exp_shifted[k] / sum_exp
        # This is a probability distribution over the terms.
        weights = exp_shifted / sum_exp
        
        # Now chain rule: d(val)/d(centers)
        # val depends on r_pairs and r_bounds
        # d(val)/d(centers) = sum_k weights[k] * d(r_k)/d(centers)
        
        grad_centers = np.zeros_like(centers)
        
        # Contribution from pairs
        # r_pairs[i, j] = 0.5 * ||c_i - c_j||
        # d(r_pairs[i, j])/d(c_i) = 0.5 * (c_i - c_j) / ||c_i - c_j||
        # Note: r_pairs is symmetric, but we flattened it.
        # We need to map weights back to pairs.
        
        # Create a matrix of weights for pairs
        w_pairs = np.zeros((N, N))
        # The valid_pairs mask corresponds to the flattened r_pairs (excluding diagonal)
        # It's easier to recompute or use a structured approach.
        
        # Let's reconstruct the weight matrix for pairs
        # The flattened r_pairs came from r_pairs[valid_pairs].
        # valid_pairs is True for i != j.
        # The order in np.concatenate is row-major for the flattened array.
        # r_pairs is (N, N). valid_pairs is (N, N).
        # r_pairs[valid_pairs] flattens in C-order (row by row).
        
        # We have weights for all terms. 
        # First part of weights: w_pair_flat = weights[:N*(N-1)]
        # Second part: w_bound_flat = weights[N*(N-1):]
        
        w_pair_flat = weights[:N*(N-1)]
        w_bound_flat = weights[N*(N-1):]
        
        # Reshape w_pair_flat to (N, N) with diagonal 0
        w_pairs = np.zeros((N, N))
        # Fill in the upper triangle? No, valid_pairs includes both (i,j) and (j,i).
        # The flattening of r_pairs[valid_pairs] includes all i != j.
        # So we can just reshape if we handle the diagonal carefully.
        # Actually, np.flatnonzero(valid_pairs) gives the indices.
        # It's safer to use the mask directly.
        
        # Let's use a more direct gradient calculation for pairs to avoid reshaping errors.
        # For each pair (i, j), the weight is w_ij.
        # The force on i is w_ij * 0.5 * (c_i - c_j) / dist_ij
        # The force on j is w_ji * 0.5 * (c_j - c_i) / dist_ji
        # Since w_ij = w_ji (symmetric in r_flat? No, r_flat contains both i,j and j,i as separate entries).
        # Yes, r_flat contains r_{ij} and r_{ji}. They are equal. Their weights will be equal.
        
        # Let's compute the gradient by iterating or vectorizing carefully.
        # Vectorized:
        # grad_from_pairs[i] = sum_{j != i} w_{ij} * 0.5 * (c_i - c_j) / dist_{ij}
        
        # We need w_{ij} for all i, j.
        # The flattened array order of r_pairs[valid_pairs]:
        # Row 0: (0,1), (0,2), ..., (0,N-1)
        # Row 1: (1,0), (1,2), ..., (1,N-1)
        # ...
        # This is tricky to reshape directly to a matrix because of the missing diagonal.
        
        # Alternative: Compute gradient via autograd-like logic or explicit loops if N is small? N=200 is too big for pure Python loops.
        # Let's use the fact that we can compute the sum of forces.
        
        # Let's create a full matrix of weights for pairs.
        # We know the order of valid_pairs.
        # Let's just use the mask to assign weights.
        w_pairs_mat = np.zeros((N, N))
        # We need to map w_pair_flat back to w_pairs_mat.
        # The indices in w_pair_flat correspond to the True indices in valid_pairs.
        # This is a bit slow if done with indices.
        
        # Better approach:
        # Compute the gradient contribution directly.
        # d(val)/d(c_i) = sum_{j} w_{ij} * d(r_{ij})/d(c_i) + w_{bound} terms
        # d(r_{ij})/d(c_i) = 0.5 * (c_i - c_j) / ||c_i - c_j||
        
        # Let's compute the vector of unit vectors u_{ij} = (c_i - c_j) / dist_{ij}
        # and the distances dist_{ij}.
        
        # dists is already computed (N, N).
        # u = diff / dists[..., np.newaxis]
        # Note: dists[i,i] is 0, so division by zero. We masked it with inf earlier for r_pairs, 
        # but here we need to handle it.
        # Let's recompute dists safely.
        dists_safe = np.where(dists > 1e-12, dists, 1.0) # Avoid div by 0
        u = diff / dists_safe[..., np.newaxis]
        
        # Now we need to sum w_{ij} * 0.5 * u_{ij} for each i.
        # w_{ij} is in w_pair_flat.
        # We need to construct w_pairs_mat from w_pair_flat.
        
        # Let's use np.put or similar.
        # Actually, we can just compute the gradient using the weights directly on the flattened array
        # and then unflatten, but we need to know the mapping.
        
        # Let's try a different strategy for the gradient of the pair part:
        # The term in the sum is sum_{i<j} w_{ij} * (r_{ij} + r_{ji}) ... no, they are separate terms.
        # Sum over all i != j.
        
        # Let's reconstruct w_pairs_mat.
        # The valid pairs are all i != j.
        # We can create an array of indices.
        idx_i, idx_j = np.where(valid_pairs)
        w_pairs_mat[idx_i, idx_j] = w_pair_flat
        
        # Now compute gradient from pairs
        # grad_pair[i] = sum_j w_pairs_mat[i, j] * 0.5 * u[i, j]
        # Note: u[i, j] = (c_i - c_j) / dist.
        # So grad_pair = 0.5 * (w_pairs_mat * u).sum(axis=1) ?
        # w_pairs_mat is (N, N). u is (N, N, 2).
        # We need to multiply w_pairs_mat[i, j] with u[i, j, k].
        # This is a batched dot product.
        # grad_pair = 0.5 * np.einsum('ij,ijk->ik', w_pairs_mat, u)
        
        grad_pair = 0.5 * np.einsum('ij,ijk->ik', w_pairs_mat, u)
        
        # Gradient from boundaries
        # r_bound[i, 0] = 0.5 - |x_i|
        # d(r_bound[i, 0])/d(x_i) = -sign(x_i)
        # r_bound[i, 1] = 0.5 - |y_i|
        # d(r_bound[i, 1])/d(y_i) = -sign(y_i)
        
        # w_bound_flat is interleaved? 
        # r_bounds was np.hstack([0.5 - |x|, 0.5 - |y|])
        # So for circle i, index 2*i is x-bound, 2*i+1 is y-bound.
        
        grad_bound = np.zeros_like(centers)
        # x-component
        signs_x = np.sign(centers[:, 0])
        signs_x[centers[:, 0] == 0] = 1.0 # Arbitrary, gradient is 0 at 0 anyway for |x|, but let's be safe
        # Actually, derivative of |x| at 0 is undefined. Subgradient is [-1, 1]. 
        # For optimization, we can pick 0 or 1. Let's pick 0 to avoid pushing away from 0 unnecessarily?
        # Or better, use a smooth approximation? No, let's just use sign, it's usually fine.
        
        # Contribution to grad_x from x-bound: w_bound[2*i] * (-sign(x_i))
        # Contribution to grad_y from y-bound: w_bound[2*i+1] * (-sign(y_i))
        
        w_x = w_bound_flat[0::2]
        w_y = w_bound_flat[1::2]
        
        grad_bound[:, 0] = -w_x * signs_x
        grad_bound[:, 1] = -w_y * np.sign(centers[:, 1])
        
        total_grad = grad_pair + grad_bound
        
        # Flatten gradient
        grad_flat = total_grad.flatten()
        
        return val, grad_flat

    # Optimization loop with annealing
    betas = [5, 10, 20, 50, 100, 200]
    n_restarts = 5
    
    for restart in range(n_restarts):
        if time.time() - start_time > time_limit - 10:
            break
            
        # Initialize
        if restart == 0:
            # Grid init
            m = int(np.ceil(np.sqrt(N)))
            ticks = (np.arange(m) + 0.5) / m - 0.5
            grid = np.array([(x, y) for y in ticks for x in ticks])
            centers = grid[:N].copy()
        else:
            # Jittered grid or random
            m = int(np.ceil(np.sqrt(N)))
            ticks = (np.arange(m) + 0.5) / m - 0.5
            grid = np.array([(x, y) for y in ticks for x in ticks])
            centers = grid[:N].copy()
            # Add small random jitter
            centers += np.random.normal(0, 0.01, centers.shape)
            # Clip to box
            centers = np.clip(centers, -0.49, 0.49)
            
        # Optimize for each beta
        for beta in betas:
            if time.time() - start_time > time_limit - 5:
                break
                
            # Reshape centers to flat
            flat_centers = centers.flatten()
            
            # Use L-BFGS-B
            try:
                res = opt.minimize(
                    objective_and_grad,
                    flat_centers,
                    args=(beta,),
                    method='L-BFGS-B',
                    jac=True, # We return (val, grad)
                    options={'maxiter': 100, 'ftol': 1e-10, 'gtol': 1e-8}
                )
                centers = res.x.reshape(N, 2)
            except Exception:
                pass
            
            # Evaluate score
            current_score = get_score(centers)
            if current_score > best_score:
                best_score = current_score
                best_centers = centers.copy()
                
    # Final polish with high beta if time permits
    if time.time() - start_time < time_limit - 2 and best_centers is not None:
        for beta in [500, 1000]:
            if time.time() - start_time > time_limit - 2:
                break
            flat_centers = best_centers.flatten()
            try:
                res = opt.minimize(
                    objective_and_grad,
                    flat_centers,
                    args=(beta,),
                    method='L-BFGS-B',
                    jac=True,
                    options={'maxiter': 50, 'ftol': 1e-12, 'gtol': 1e-10}
                )
                centers = res.x.reshape(N, 2)
                current_score = get_score(centers)
                if current_score > best_score:
                    best_score = current_score
                    best_centers = centers.copy()
            except Exception:
                pass

    if best_centers is None:
        # Fallback to grid
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        return grid[:N]
        
    return best_centers
# EVOLVE-BLOCK-END
