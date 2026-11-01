"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

N = 300
TIME_LIMIT = 110.0  # Safety margin from 120s

def construct():
    start_time = time.time()
    
    # 1. Initialization: Hexagonal lattice
    # Approximate radius estimate for 300 circles in unit circle
    # Area per circle ~ pi/N. r ~ 0.05
    # Spacing ~ 2r ~ 0.1
    points = _init_hex_lattice(N)
    
    # Objective: Minimize g(P) = (1/beta) * log( sum(exp(-beta * r_k)) )
    # where r_k are the candidate radii (pair distances / 2 and boundary distances).
    # This approximates -min(r_k). Minimizing this maximizes min(r_k).
    
    # Parameters
    beta = 50.0
    step_size = 0.01
    iterations = 0
    best_score = _calc_score(points)
    best_points = points.copy()
    
    # Precompute indices for pairs? N=300, pairs=44850. 
    # Computing all pair distances every iteration is O(N^2) = 90k ops. 
    # 10k iterations -> 900M ops. Might be slow in pure Python/Numpy without optimization.
    # We can optimize by only considering neighbors or using a spatial hash, 
    # but for N=300, full matrix might be okay if vectorized well.
    # Let's try vectorized computation.
    
    # We will run a loop with adaptive beta and step size.
    # Schedule:
    # Phase 1: beta=20, steps=2000, step_size=0.02
    # Phase 2: beta=100, steps=2000, step_size=0.005
    # Phase 3: beta=500, steps=1000, step_size=0.001
    
    phases = [
        (20.0, 2000, 0.02),
        (100.0, 2000, 0.005),
        (500.0, 2000, 0.001),
        (2000.0, 1000, 0.0002)
    ]
    
    for beta, max_steps, init_step in phases:
        step = init_step
        for i in range(max_steps):
            if time.time() - start_time > TIME_LIMIT:
                break
            
            # Compute distances
            # d_ij for i<j
            # d_i for boundary
            
            # Vectorized distance calculation
            # To save memory and time, we can compute the gradient directly.
            
            # 1. Pairwise distances
            # diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
            # dist = np.linalg.norm(diff, axis=2)
            # This creates a 300x300x2 array, then 300x300. 
            # 300x300 is small.
            
            diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
            dist_sq = np.sum(diff**2, axis=2)
            # Avoid division by zero
            dist_sq[dist_sq == 0] = 1e-12
            dist = np.sqrt(dist_sq)
            
            # Radii from pairs: r_ij = dist / 2
            r_pairs = dist / 2.0
            # Only upper triangle
            mask = np.triu(np.ones((N, N), dtype=bool), k=1)
            r_pairs_flat = r_pairs[mask]
            
            # Boundary radii
            norms = np.linalg.norm(points, axis=1)
            r_bound = 1.0 - norms
            
            # Combine all radii
            # We need weights w_k = exp(-beta * r_k) / sum(exp(-beta * r_j))
            # And gradient: grad_g = - sum( w_k * grad_r_k )
            
            # To avoid overflow in exp, subtract max
            all_r = np.concatenate([r_pairs_flat, r_bound])
            max_r = np.max(all_r)
            # Actually, we want to focus on the MINIMUM r.
            # The soft-min formula: m = (1/beta) log(sum(exp(-beta r)))
            # This is numerically stable if we do:
            # log(sum(exp(-beta (r - r_min)))) - beta * r_min ... no.
            # Let's use:
            # exp_terms = exp(-beta * (all_r - min_r))
            # sum_exp = sum(exp_terms)
            # weights = exp_terms / sum_exp
            # grad = - sum(weights * grad_r)
            
            min_r = np.min(all_r)
            # Shift to prevent underflow
            shifted_r = all_r - min_r
            exp_terms = np.exp(-beta * shifted_r)
            sum_exp = np.sum(exp_terms)
            weights = exp_terms / sum_exp
            
            # Now compute gradient of g(P) = (1/beta) log(sum_exp)
            # dg/dP = (1/sum_exp) * sum( d(exp(-beta r_k))/dP )
            # d(exp(-beta r_k))/dP = exp(-beta r_k) * (-beta) * dr_k/dP
            # So dg/dP = (1/sum_exp) * sum( exp(-beta r_k) * (-beta) * dr_k/dP )
            #            = -beta * sum( weights_k * dr_k/dP )
            
            grad = np.zeros_like(points)
            
            # Contribution from pairs
            # We have weights for the flattened pairs.
            # We need to map them back to the 2D structure or accumulate directly.
            # It's easier to compute the gradient contribution for each pair (i,j)
            # and add to grad[i] and grad[j].
            
            # Let's re-compute weights in 2D form for easier accumulation?
            # Or just loop? N=300, pairs ~45k. Looping 45k times in Python is slow.
            # Vectorized accumulation is better.
            
            # Let's create a 2D weight matrix W
            W = np.zeros((N, N))
            # Fill upper triangle with weights
            # We need to map the flat weights back to indices
            # This is tricky. Let's compute weights in 2D directly.
            
            # Recompute exp_terms in 2D
            # dist is 2D. r_pairs is 2D.
            # We only care about i < j.
            
            # Let's just compute the gradient using the 2D arrays.
            # g = (1/beta) log( sum_{i<j} exp(-beta d_ij/2) + sum_i exp(-beta (1-norm_i)) )
            
            # Term 1: Pairs
            # exp_pairs = exp(-beta * dist/2)
            # Mask upper triangle
            exp_pairs = np.exp(-beta * dist / 2.0)
            exp_pairs[~mask] = 0
            sum_pairs = np.sum(exp_pairs)
            
            # Term 2: Boundary
            exp_bound = np.exp(-beta * r_bound)
            sum_bound = np.sum(exp_bound)
            
            total_sum = sum_pairs + sum_bound
            
            # Weights
            w_pairs = exp_pairs / total_sum
            w_bound = exp_bound / total_sum
            
            # Gradient of log(sum) is sum( (1/term) * d(term) )
            # d(exp(-beta r))/dP = exp(-beta r) * (-beta) * dr/dP
            
            # Grad from pairs:
            # For each pair (i,j), contribution to grad[i] is:
            # w_pairs[i,j] * (-beta) * d(dist_ij/2)/dP_i
            # d(dist_ij)/dP_i = (P_i - P_j) / dist_ij
            # So d(dist_ij/2)/dP_i = (P_i - P_j) / (2 * dist_ij)
            
            # Vectorized:
            # grad_pair[i] = sum_j ( w_pairs[i,j] * (-beta) * (P_i - P_j) / (2 * dist_ij) )
            
            # Note: w_pairs is 0 for i>=j.
            
            # Let's compute this.
            # factor = w_pairs * (-beta) / (2 * dist)
            # Avoid div by zero
            safe_dist = np.where(dist > 1e-12, dist, 1.0)
            factor = w_pairs * (-beta) / (2.0 * safe_dist)
            
            # grad_i = sum_j ( factor[i,j] * (P_i - P_j) )
            # = P_i * sum_j(factor[i,j]) - sum_j( factor[i,j] * P_j )
            
            sum_factor = np.sum(factor, axis=1)
            # weighted_sum_P = factor @ P ?
            # factor is (N,N), P is (N,2).
            # (factor @ P)[i] = sum_j factor[i,j] P_j
            weighted_P = factor @ points
            
            grad_pairs = points * sum_factor[:, np.newaxis] - weighted_P
            
            # Grad from boundary:
            # w_bound[i] * (-beta) * d(1 - norm_i)/dP_i
            # d(1 - norm_i)/dP_i = - P_i / norm_i
            # So term is w_bound[i] * (-beta) * (-P_i / norm_i) = w_bound[i] * beta * P_i / norm_i
            
            safe_norms = np.where(norms > 1e-12, norms, 1.0)
            grad_bound = w_bound * beta * points / safe_norms[:, np.newaxis]
            
            total_grad = grad_pairs + grad_bound
            
            # We are minimizing g(P). So we move in direction -grad.
            # points = points - step * total_grad
            
            # Adaptive step size? 
            # Let's just use the fixed step for the phase, maybe decay it.
            # Decay step size within phase
            current_step = init_step * (0.99 ** (i / max_steps))
            
            new_points = points - current_step * total_grad
            
            # Check validity
            # If any point goes out of bounds significantly, we might need to project.
            # The objective function handles it via penalty, but if it flies out, 
            # the gradient might be unstable.
            # Simple projection: clip to unit circle? 
            # Actually, if it goes out, r_bound becomes negative, exp(-beta * negative) is huge.
            # The gradient will push it back strongly.
            
            # Update
            points = new_points
            
            # Calculate score periodically
            if i % 100 == 0:
                score = _calc_score(points)
                if score > best_score:
                    best_score = score
                    best_points = points.copy()
                    
            iterations += 1
            
        if time.time() - start_time > TIME_LIMIT:
            break
            
    # Final polish: Check best found
    return best_points

def _init_hex_lattice(N):
    # Generate points in a hex lattice inside unit circle
    # Estimate spacing
    # Area ~ pi. N circles. r ~ sqrt(pi/N). spacing ~ 2r.
    # N=300, r~0.05, spacing~0.1
    spacing = 0.1
    
    points = []
    # Range to cover unit circle
    # Max radius 1. 
    # Hex grid:
    # y = j * spacing * sqrt(3)/2
    # x = (i + 0.5*(j%2)) * spacing
    
    k = int(1.0 / spacing) + 2
    for j in range(-k, k + 1):
        for i in range(-k, k + 1):
            x = (i + 0.5 * (j % 2)) * spacing
            y = j * spacing * np.sqrt(3.0) / 2.0
            if x**2 + y**2 <= 1.0:
                points.append([x, y])
                
    points = np.array(points)
    
    # If we have too many, we need to select N.
    # If we have too few, we need to add more or adjust spacing.
    # For N=300, spacing 0.1 usually gives ~300 points.
    # Let's check count.
    if len(points) >= N:
        # Select N points. Randomly? Or closest to center?
        # Random selection from the lattice is a good start.
        idx = np.random.choice(len(points), N, replace=False)
        return points[idx]
    else:
        # If not enough, reduce spacing
        # This is a fallback, shouldn't happen for N=300 with spacing 0.1
        # Just return what we have or pad?
        # Better to generate with smaller spacing.
        spacing = 0.05
        points = []
        k = int(1.0 / spacing) + 2
        for j in range(-k, k + 1):
            for i in range(-k, k + 1):
                x = (i + 0.5 * (j % 2)) * spacing
                y = j * spacing * np.sqrt(3.0) / 2.0
                if x**2 + y**2 <= 1.0:
                    points.append([x, y])
        points = np.array(points)
        if len(points) >= N:
            idx = np.random.choice(len(points), N, replace=False)
            return points[idx]
        else:
            # Should not reach here
            return points[:N] if len(points) >= N else np.zeros((N,2))

def _calc_score(points):
    # Calculate the radius
    # min pair dist / 2
    # min boundary dist
    
    if len(points) < 2:
        return 0.0
        
    # Pair distances
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dist_sq = np.sum(diff**2, axis=2)
    dist = np.sqrt(dist_sq)
    
    # Mask upper triangle
    mask = np.triu(np.ones((len(points), len(points)), dtype=bool), k=1)
    min_pair_dist = np.min(dist[mask])
    
    # Boundary
    norms = np.linalg.norm(points, axis=1)
    min_bound_dist = np.min(1.0 - norms)
    
    r = min(min_pair_dist / 2.0, min_bound_dist)
    
    # Check validity
    if r <= 0:
        return 0.0
    if not np.isfinite(r):
        return 0.0
        
    return r

# EVOLVE-BLOCK-END
