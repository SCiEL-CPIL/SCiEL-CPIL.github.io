import numpy as np
import time

N = 50

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Leave 10s margin
    
    # 1. Initialization: Perturbed Grid
    # 4x4x4 grid is 64 points. We need 50.
    # We can use a 4x4x4 grid and remove 14 points, or just use a 50-point subset.
    # Better: Start with a 4x4x4 grid, take the first 50, add noise.
    m = 4
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    # Shuffle and take 50 to ensure spatial distribution? 
    # The grid is already spatially distributed.
    # Taking first 50 of a lexicographically ordered grid might be biased (e.g., all z=0 and z=0.25 layers full, z=0.5 partial).
    # Let's randomize the selection slightly or just use the first 50.
    # Actually, a uniform random initialization followed by a few relaxation steps might be better to avoid grid traps.
    # But grid is close to optimal. Let's stick to grid + noise.
    
    np.random.seed(42)
    centers = grid[:N].copy()
    centers += np.random.normal(0, 0.01, centers.shape)
    
    # Ensure inside cube
    centers = np.clip(centers, -0.5, 0.5)
    
    # Precompute pairs
    idx_i, idx_j = np.triu_indices(N, k=1)
    pairs = np.column_stack((idx_i, idx_j))
    
    # Soft-min parameters
    beta = 1.0
    beta_end = 2000.0
    alpha = 0.005
    alpha_min = 1e-6
    total_steps = 5000
    
    best_score = -np.inf
    best_centers = centers.copy()
    
    for step in range(total_steps):
        # Time check
        if time.time() > deadline:
            break
        
        # Calculate values v_m
        # 1. Pairwise distances / 2
        diffs = centers[idx_i] - centers[idx_j]
        dists = np.linalg.norm(diffs, axis=1)
        v_pairs = dists / 2.0
        
        # 2. Boundary distances
        # For each point i and axis k: 0.5 - |x_ik|
        abs_centers = np.abs(centers)
        v_bounds = 0.5 - abs_centers # shape (N, 3)
        v_bounds_flat = v_bounds.flatten()
        
        # Combine all values
        v_all = np.concatenate([v_pairs, v_bounds_flat])
        
        # Calculate weights w_m
        # w_m = exp(-beta * v_m) / sum(exp(-beta * v_j))
        # Use log-sum-exp trick for stability
        max_v = np.max(v_all)
        exp_terms = np.exp(-beta * (v_all - max_v))
        sum_exp = np.sum(exp_terms)
        weights = exp_terms / sum_exp
        
        # Calculate gradients
        # grad_S = sum w_m * grad_v_m
        
        grad = np.zeros_like(centers)
        
        # 1. Pairwise gradients
        # grad_{x_i} (dist/2) = 0.5 * (x_i - x_j) / dist
        # grad_{x_j} (dist/2) = 0.5 * (x_j - x_i) / dist
        # dists are already computed.
        # Avoid division by zero
        safe_dists = np.where(dists > 1e-9, dists, 1.0)
        unit_vecs = diffs / safe_dists[:, np.newaxis]
        
        # Force on i is +0.5 * unit_vec * w_i
        # Force on j is -0.5 * unit_vec * w_i
        # weights for pairs are weights[:len(pairs)]
        w_pairs = weights[:len(pairs)]
        
        # Accumulate forces
        # np.add.at is slow, use bincount or scatter?
        # N is small (50), simple loop or vectorized accumulation is fine.
        # Vectorized:
        # For each pair, add 0.5 * w * unit_vec to grad[i]
        # and -0.5 * w * unit_vec to grad[j]
        
        # Using np.add.at for simplicity and correctness
        forces = 0.5 * w_pairs[:, np.newaxis] * unit_vecs
        np.add.at(grad, idx_i, forces)
        np.add.at(grad, idx_j, -forces)
        
        # 2. Boundary gradients
        # v_ik = 0.5 - |x_ik|
        # grad_{x_i} v_ik = -sign(x_ik) * e_k
        # weights for bounds are weights[len(pairs):]
        w_bounds = weights[len(pairs):]
        w_bounds_reshaped = w_bounds.reshape(N, 3)
        
        signs = np.sign(centers)
        # If center is 0, sign is 0, gradient is 0? 
        # Technically |x| is not differentiable at 0. 
        # But 0 is measure zero. If x is very close to 0, sign is stable.
        # Let's treat sign(0) as 0 to be safe, or 1. 
        # If x=0, moving in either direction increases distance to one wall and decreases to other.
        # The soft-min weight will handle the balance if we consider both walls?
        # No, v_ik is distance to wall k+. v_i,k+3 is distance to wall k-.
        # Wait, my v_bounds is just 0.5 - |x|. This combines + and - walls.
        # The gradient of 0.5 - |x| is -sign(x).
        # This pushes x towards 0.
        # This is correct: it pushes the point to the center of the face.
        
        # grad contribution for point i, axis k is w_ik * (-sign(x_ik)) * e_k
        # So grad[i, k] += -w_ik * sign(x_ik)
        
        grad += -w_bounds_reshaped * signs
        
        # Update
        centers = centers + alpha * grad
        
        # Project to cube
        centers = np.clip(centers, -0.5, 0.5)
        
        # Calculate actual score
        # min(v_all) is a lower bound on the soft-min, but close for high beta.
        # Let's compute exact min for tracking.
        current_score = np.min(v_all)
        
        if current_score > best_score:
            best_score = current_score
            best_centers = centers.copy()
            
        # Annealing
        # Increase beta
        beta *= 1.005
        # Decrease alpha
        alpha = max(alpha * 0.999, alpha_min)
        
    # Final polish: High beta, small steps
    if time.time() < deadline - 2.0:
        beta = 5000.0
        alpha = 1e-5
        for step in range(1000):
            if time.time() > deadline - 1.0:
                break
            diffs = centers[idx_i] - centers[idx_j]
            dists = np.linalg.norm(diffs, axis=1)
            v_pairs = dists / 2.0
            abs_centers = np.abs(centers)
            v_bounds = 0.5 - abs_centers
            v_bounds_flat = v_bounds.flatten()
            v_all = np.concatenate([v_pairs, v_bounds_flat])
            
            max_v = np.max(v_all)
            exp_terms = np.exp(-beta * (v_all - max_v))
            sum_exp = np.sum(exp_terms)
            weights = exp_terms / sum_exp
            
            grad = np.zeros_like(centers)
            safe_dists = np.where(dists > 1e-9, dists, 1.0)
            unit_vecs = diffs / safe_dists[:, np.newaxis]
            w_pairs = weights[:len(pairs)]
            forces = 0.5 * w_pairs[:, np.newaxis] * unit_vecs
            np.add.at(grad, idx_i, forces)
            np.add.at(grad, idx_j, -forces)
            
            w_bounds = weights[len(pairs):].reshape(N, 3)
            signs = np.sign(centers)
            grad += -w_bounds * signs
            
            centers = centers + alpha * grad
            centers = np.clip(centers, -0.5, 0.5)
            
            current_score = np.min(v_all)
            if current_score > best_score:
                best_score = current_score
                best_centers = centers.copy()
                
            alpha *= 0.99
            
    return best_centers
