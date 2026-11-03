"""
Strategy: Gradient-Based Optimization with Soft-Min Surrogate.
Instead of identifying discrete active constraints and applying discrete forces (which is
nonsmooth and prone to getting stuck in local minima of the active set topology),
this strategy optimizes a smooth surrogate objective: the soft-min of all pairwise
distances and wall distances.

The soft-min is computed via the log-sum-exp trick:
soft_min(d_1, ..., d_M) = -log( sum( exp(-d_i / T) ) ) * T

As T -> 0, this approaches the true min. We use a cooling schedule for T to gradually
transition from a smooth, easy-to-optimize landscape to the sharp, true objective.

We use projected gradient descent with momentum. The gradient of the soft-min with
respect to point positions is smooth and well-defined everywhere, allowing the optimizer
to "see" the global structure of the packing rather than just the immediate bottleneck.
"""
import numpy as np
import time
import math

N = 200

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill

    def compute_radius(centers):
        """
        Computes the exact maximum radius.
        """
        # Wall distances
        dx_min = centers[:, 0] + 0.5
        dx_max = 0.5 - centers[:, 0]
        dy_min = centers[:, 1] + 0.5
        dy_max = 0.5 - centers[:, 1]
        
        min_wall_dist = np.min([np.min(dx_min), np.min(dx_max), np.min(dy_min), np.min(dy_max)])
        
        # Pairwise distances
        # Using cdist is fast, but for gradient we need to be careful.
        # For evaluation, cdist is fine.
        dist_matrix = np.sqrt(np.sum((centers[:, np.newaxis, :] - centers[np.newaxis, :, :])**2, axis=2))
        np.fill_diagonal(dist_matrix, np.inf)
        min_pair_dist = np.min(dist_matrix)
        
        return min(min_pair_dist / 2.0, min_wall_dist)

    def initialize_hex_grid():
        """
        Generates a high-quality initial hexagonal packing.
        """
        d = math.sqrt(2.0 / (N * math.sqrt(3)))
        
        pts = []
        row = 0
        y = -0.5
        while y <= 0.5:
            x_offset = 0.0 if row % 2 == 0 else d / 2.0
            x = -0.5 + d / 2.0 + x_offset
            while x <= 0.5:
                if -0.5 < x < 0.5 and -0.5 < y < 0.5:
                    pts.append([x, y])
                x += d
            y += d * math.sqrt(3) / 2.0
            row += 1
            
        pts = np.array(pts)
        
        if len(pts) < N:
            m = int(math.ceil(math.sqrt(N)))
            ticks = (np.arange(m) + 0.5) / m - 0.5
            grid = np.array([(x, y) for y in ticks for x in ticks])
            return grid[:N]
        
        # If we have more than N points, select the ones that form the best packing.
        # Sort by distance to center, take N closest.
        dists_to_origin = np.linalg.norm(pts, axis=1)
        sorted_idx = np.argsort(dists_to_origin)
        pts = pts[sorted_idx[:N]]
        
        return pts

    def objective_and_grad(centers, T):
        """
        Computes the soft-min objective and its gradient.
        Objective: soft_min(min_pair_dist/2, min_wall_dist)
        We want to MAXIMIZE this, so we MINIMIZE -objective.
        
        Let d_pairs be the vector of pairwise distances / 2.
        Let d_walls be the vector of wall distances.
        Let D = concat(d_pairs, d_walls).
        
        soft_min(D) = -T * log( sum( exp(-d_i / T) ) )
        
        grad_{c_i} soft_min(D) = - sum_j ( exp(-d_j/T) / sum_k exp(-d_k/T) ) * grad_{c_i} d_j
        
        Note: The gradient of the soft-min points in the direction that DECREASES the soft-min.
        We want to INCREASE the soft-min. So we use -grad for ascent.
        
        However, calculating the full pairwise gradient is O(N^2).
        We can vectorize this.
        """
        # 1. Pairwise Distances
        # diff[i, j] = centers[i] - centers[j]
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dists, np.inf)
        
        # We consider d_pair = dists / 2.0
        # d_pair[i, j] = dists[i, j] / 2.0
        # grad_{c_i} d_pair[i, j] = (c_i - c_j) / dists[i, j]  (for i != j)
        # Note: d_pair[i, j] = d_pair[j, i].
        # The contribution to the gradient of c_i from pair (i, j) is:
        # - w_ij * (c_i - c_j) / dists[i, j]  where w_ij is the softmax weight.
        # Wait, the gradient of d_pair[i, j] w.r.t c_i is (c_i - c_j) / dists[i, j].
        # We want to maximize the soft-min.
        # grad_{c_i} soft_min = - sum_j w_ij * (c_i - c_j) / dists[i, j]
        
        # 2. Wall Distances
        # d_walls for point i:
        # x_i + 0.5, 0.5 - x_i, y_i + 0.5, 0.5 - y_i
        # Let's create a matrix D_walls of shape (N, 4)
        D_walls = np.zeros((N, 4))
        D_walls[:, 0] = centers[:, 0] + 0.5
        D_walls[:, 1] = 0.5 - centers[:, 0]
        D_walls[:, 2] = centers[:, 1] + 0.5
        D_walls[:, 3] = 0.5 - centers[:, 1]
        
        # Combine all distances into one big vector for the log-sum-exp
        # This might be memory heavy: N^2 + 4N ~ 40k + 800 = 41k floats. That's fine.
        # But we need to compute weights for each.
        
        # To avoid overflow in exp(-d/T), we subtract the min.
        # Let's compute the pairwise part and wall part separately, then combine?
        # No, the soft-min of the union is not the soft-min of the soft-mins.
        # We must treat all constraints together.
        
        # Let's flatten the pairwise distances (upper triangle to avoid double counting? 
        # No, the gradient calculation is symmetric. If we include both (i,j) and (j,i),
        # the forces on i from j and on j from i are handled correctly by the sum.
        # Actually, d_pair[i,j] is the same constraint. 
        # If we sum over all i!=j, we count each pair twice.
        # This just scales the gradient by a factor of 2 for pairs.
        # Since we use gradient descent, the scale can be absorbed into the learning rate.
        # It is simpler to just sum over all i!=j.
        
        d_pairs = dists / 2.0
        d_pairs_flat = d_pairs.flatten()
        
        # Wall distances are already per-point.
        # We need to associate each wall distance with the point it belongs to.
        # d_walls[i, k] is the distance of point i to wall k.
        
        # Let's compute the weights for pairs and walls separately, then combine?
        # No. The denominator is sum over ALL constraints.
        
        # Let's compute the log-sum-exp for all constraints.
        # Total constraints M = N*(N-1) + 4*N
        
        # This is too large to flatten easily in a way that maps back to gradients easily without indexing.
        # Let's compute the pairwise gradient contribution and wall gradient contribution separately,
        # but they share the same normalization constant Z = sum(exp(-d/T)).
        
        # 1. Compute exp(-d/T) for pairs
        # To prevent underflow/overflow, we can just compute it. T is small, d is ~0.03.
        # -d/T is large negative. exp is small.
        # It's safer to compute weights using a stable log-sum-exp if we were doing log-likelihood,
        # but here we just need the weights w_i = exp(-d_i/T) / Z.
        
        # Let's compute the sum of exps for pairs and walls.
        # Note: d_pairs has inf on diagonal. exp(-inf/T) = 0. So diagonal contributes 0.
        
        exp_pairs = np.exp(-d_pairs / T)
        exp_walls = np.exp(-D_walls / T)
        
        Z = np.sum(exp_pairs) + np.sum(exp_walls)
        
        # Weights
        w_pairs = exp_pairs / Z
        w_walls = exp_walls / Z
        
        # Objective value
        # soft_min = -T * log(Z)  (since log(sum(exp(-d/T))) = log(Z))
        # Wait. soft_min(d) = -T * log( sum( exp(-d_i/T) ) ).
        # Yes.
        obj = -T * math.log(Z)
        
        # Gradient
        # grad_{c_i} obj = -T * (1/Z) * sum_j ( exp(-d_j/T) * grad_{c_i} (-d_j/T) )
        # grad_{c_i} obj = -T * (1/Z) * sum_j ( exp(-d_j/T) * (-1/T) * grad_{c_i} d_j )
        # grad_{c_i} obj = (1/Z) * sum_j ( exp(-d_j/T) * grad_{c_i} d_j )
        # grad_{c_i} obj = sum_j w_j * grad_{c_i} d_j
        
        # So the gradient of the soft-min is the weighted average of the gradients of the individual distances.
        # We want to ASCEND this gradient.
        
        grad = np.zeros_like(centers)
        
        # Pairwise contribution
        # grad_{c_i} d_pair[i, j] = (c_i - c_j) / dists[i, j]
        # We need to sum over j: w_pairs[i, j] * (c_i - c_j) / dists[i, j]
        # Note: w_pairs[i, j] is symmetric.
        
        # Vectorized:
        # term[i, j] = w_pairs[i, j] * (c_i - c_j) / dists[i, j]
        # We need to sum over j for each i.
        
        # Create a matrix of unit vectors
        # unit_vec[i, j] = (c_i - c_j) / dists[i, j]
        # Handle zero distance (shouldn't happen if initialized well, but safe)
        with np.errstate(divide='ignore', invalid='ignore'):
            unit_vec = diff / dists[:, :, np.newaxis]
        unit_vec[~np.isfinite(unit_vec)] = 0.0
        
        # Weighted sum
        # grad_pairs[i] = sum_j w_pairs[i, j] * unit_vec[i, j]
        grad_pairs = np.sum(w_pairs[:, :, np.newaxis] * unit_vec, axis=1)
        
        # Wall contribution
        # grad_{c_i} d_wall[i, k]
        # Wall 0 (x = -0.5): d = x_i + 0.5. grad = (1, 0)
        # Wall 1 (x = 0.5): d = 0.5 - x_i. grad = (-1, 0)
        # Wall 2 (y = -0.5): d = y_i + 0.5. grad = (0, 1)
        # Wall 3 (y = 0.5): d = 0.5 - y_i. grad = (0, -1)
        
        # w_walls[i, k] is the weight.
        # grad_walls[i] = w_walls[i, 0]*(1,0) + w_walls[i, 1]*(-1,0) + w_walls[i, 2]*(0,1) + w_walls[i, 3]*(0,-1)
        
        grad_walls = np.zeros_like(centers)
        grad_walls[:, 0] = w_walls[:, 0] - w_walls[:, 1]
        grad_walls[:, 1] = w_walls[:, 2] - w_walls[:, 3]
        
        grad = grad_pairs + grad_walls
        
        return obj, grad

    # --- Main Optimization Loop ---
    
    # 1. Initialization
    centers = initialize_hex_grid()
    best_centers = centers.copy()
    best_r = compute_radius(centers)
    
    # 2. Optimization Parameters
    # Start with a larger T (smoother landscape) and cool down.
    # T controls the "softness". 
    # Typical distances are ~0.03. 
    # If T is too large, the objective is too flat. If T is too small, it's too sharp.
    # Start T = 0.01, end T = 0.001.
    
    start_T = 0.02
    end_T = 0.001
    start_lr = 0.005
    end_lr = 0.0001
    
    total_iterations = 5000
    momentum = 0.9
    
    v = np.zeros_like(centers)
    
    # We will run iterations until time limit
    stuck_counter = 0
    max_stuck = 50
    
    for iter in range(total_iterations):
        if time.time() > deadline - 5.0:
            break
            
        # Cool down T and adjust learning rate
        progress = iter / total_iterations
        # Exponential cooling for T
        current_T = start_T * (end_T / start_T) ** progress
        # Linear or exponential decay for LR
        current_lr = start_lr * (end_lr / start_lr) ** progress
        
        # Get current state
        obj, grad = objective_and_grad(centers, current_T)
        
        # Update velocity (Momentum)
        v = momentum * v + current_lr * grad
        
        # Update centers
        new_centers = centers + v
        
        # Project back into square
        new_centers = np.clip(new_centers, -0.5 + 1e-9, 0.5 - 1e-9)
        
        # Evaluate exact radius for tracking best
        new_r = compute_radius(new_centers)
        
        # Accept if better (in terms of exact radius)
        # Note: The soft-min objective and exact radius are correlated but not identical.
        # We should accept moves that improve the exact radius, or if we are in the "exploration" phase (high T),
        # we might accept moves that improve the soft-min even if exact radius drops slightly, to escape local minima.
        # However, to be safe and ensure monotonic improvement in the final phase,
        # let's use a hybrid acceptance criterion.
        
        if new_r >= best_r:
            centers = new_centers
            stuck_counter = 0
            if new_r > best_r:
                best_r = new_r
                best_centers = new_centers.copy()
        else:
            # Check if the soft-min objective improved significantly?
            # If T is large, the landscape is smooth. We should follow the gradient.
            # If T is small, we should be more strict.
            
            # Simple heuristic: If we are in the early phase (high T), accept with some probability
            # or if the move is small.
            # For now, let's just reject if exact radius drops, unless we are stuck.
            stuck_counter += 1
            
            if stuck_counter > max_stuck:
                # Perturb to escape local minimum
                # Add small random noise
                noise_scale = 0.005 * (start_T / current_T) # Larger noise when T is larger?
                # Actually, when T is large, the gradient is less specific.
                # Let's just add fixed noise when stuck.
                centers += np.random.normal(0, 0.005, size=centers.shape)
                centers = np.clip(centers, -0.5 + 1e-9, 0.5 - 1e-9)
                v = np.zeros_like(centers) # Reset momentum
                stuck_counter = 0

    # Final polish: Very small T, small steps
    centers = best_centers.copy()
    for _ in range(1000):
        if time.time() > deadline - 2.0:
            break
        
        # Use a very small T for high precision
        current_T = 0.0005
        current_lr = 0.00005
        
        obj, grad = objective_and_grad(centers, current_T)
        
        # Simple gradient step without momentum for stability in final phase
        new_centers = centers + current_lr * grad
        new_centers = np.clip(new_centers, -0.5 + 1e-9, 0.5 - 1e-9)
        
        new_r = compute_radius(new_centers)
        
        if new_r > best_r:
            centers = new_centers
            best_r = new_r
            best_centers = new_centers.copy()
        else:
            # If no improvement, try a smaller step or break
            # Since we are polishing, if we don't improve, we are likely at a local max of the soft-min
            # which should be close to the local max of the exact radius.
            # Let's try a few more steps with even smaller LR
            if _ > 500:
                break

    # Ensure validity
    if not np.all(np.isfinite(best_centers)):
        m = int(math.ceil(math.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        best_centers = grid[:N]
        
    return best_centers

# EVOLVE-BLOCK-END
