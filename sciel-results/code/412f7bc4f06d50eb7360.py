"""
Strategy: Differentiable Soft-Min Optimization with Hexagonal Guidance.
Instead of discrete force-directed moves, we treat the packing as a smooth optimization
problem. We define a differentiable surrogate for the minimum distance constraint using
the log-sum-exp function. This allows us to use powerful gradient-based solvers (L-BFGS-B)
to navigate the landscape, effectively "feeling" all constraints simultaneously rather
than just the active ones. We initialize with a hexagonal grid to stay in the basin of
attraction of the optimal packing structure. A final high-precision active-set polish
squeezes out the last bits of radius.
"""
import numpy as np
import time
import math
from scipy.optimize import minimize
from scipy.spatial.distance import cdist

N = 200

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill

    def compute_radius_and_bottlenecks(centers):
        """
        Computes the exact maximum radius and identifies the bottleneck indices.
        """
        if len(centers) < 2:
            return 0.5, [], []
        
        dx_min = centers[:, 0] + 0.5
        dx_max = 0.5 - centers[:, 0]
        dy_min = centers[:, 1] + 0.5
        dy_max = 0.5 - centers[:, 1]
        
        dist_matrix = cdist(centers, centers)
        np.fill_diagonal(dist_matrix, np.inf)
        min_pair_dist = np.min(dist_matrix)
        
        min_wall_dist = np.min([np.min(dx_min), np.min(dx_max), np.min(dy_min), np.min(dy_max)])
        
        r = min(min_pair_dist / 2.0, min_wall_dist)
        
        active_pairs = []
        active_walls = []
        
        tol = 1e-8
        
        if min_pair_dist / 2.0 <= r + tol:
            i_idx, j_idx = np.triu_indices(N, k=1)
            d_pairs = dist_matrix[i_idx, j_idx]
            mask = d_pairs <= min_pair_dist + tol
            if np.any(mask):
                active_pairs = list(zip(i_idx[mask], j_idx[mask]))
                
        if min_wall_dist <= r + tol:
            if np.min(dx_min) <= r + tol:
                idxs = np.where(dx_min <= r + tol)[0]
                active_walls.extend([(i, 0) for i in idxs])
            if np.min(dx_max) <= r + tol:
                idxs = np.where(dx_max <= r + tol)[0]
                active_walls.extend([(i, 1) for i in idxs])
            if np.min(dy_min) <= r + tol:
                idxs = np.where(dy_min <= r + tol)[0]
                active_walls.extend([(i, 2) for i in idxs])
            if np.min(dy_max) <= r + tol:
                idxs = np.where(dy_max <= r + tol)[0]
                active_walls.extend([(i, 3) for i in idxs])

        return r, active_pairs, active_walls

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
        dists_to_origin = np.linalg.norm(pts, axis=1)
        sorted_idx = np.argsort(dists_to_origin)
        pts = pts[sorted_idx[:N]]
        
        return pts

    def soft_min_logsumexp(values, beta):
        """
        Computes a smooth approximation of the minimum of values using log-sum-exp.
        min(v) ≈ -(1/beta) * log(sum(exp(-beta * v)))
        """
        # For numerical stability
        min_val = np.min(values)
        # If beta is very large, exp(-beta * (v - min_val)) might underflow for large v-min_val
        # But we only care about the gradient direction.
        # To avoid overflow in exp, we shift by min_val.
        # log(sum(exp(-beta * v))) = -beta * min_val + log(sum(exp(-beta * (v - min_val))))
        # So -1/beta * log(...) = min_val - 1/beta * log(sum(exp(-beta * (v - min_val))))
        
        # If beta is huge, the term inside log is dominated by the min.
        # Let's just use the stable form directly.
        # However, for very large beta, the gradient becomes very sharp.
        
        # We want the gradient of the soft min.
        # f(v) = -1/beta * log(sum(exp(-beta * v)))
        # df/dv_i = (exp(-beta * v_i)) / sum(exp(-beta * v_j))
        
        # To compute this stably:
        # Let m = min(v).
        # exp(-beta * v_i) = exp(-beta * (v_i - m)) * exp(-beta * m)
        # sum(exp(-beta * v_j)) = exp(-beta * m) * sum(exp(-beta * (v_j - m)))
        # Ratio = exp(-beta * (v_i - m)) / sum(exp(-beta * (v_j - m)))
        
        # We need to be careful if beta * (v_i - m) is very large positive (underflow to 0) or large negative (overflow).
        # Since v_i >= m, v_i - m >= 0. So -beta * (v_i - m) <= 0.
        # exp(negative) is safe from overflow. It might underflow to 0, which is correct for non-min elements.
        
        shifted = -beta * (values - min_val)
        # Clip to avoid underflow issues if beta is too large? 
        # If beta * (v_i - m) > 700, exp is 0. That's fine.
        
        exp_shifted = np.exp(shifted)
        sum_exp = np.sum(exp_shifted)
        
        if sum_exp == 0:
             # This shouldn't happen if min_val is included, but just in case
             return min_val, np.zeros_like(values)
             
        probs = exp_shifted / sum_exp
        grad = probs
        
        val = min_val - (1.0 / beta) * np.log(sum_exp)
        
        return val, grad

    def objective_and_grad(x_flat, beta):
        """
        Objective function and its gradient for L-BFGS-B.
        We want to maximize the minimum distance.
        Equivalent to minimizing -min_dist.
        Using soft min: minimize -soft_min(distances).
        
        The distances are:
        1. Pairwise distances / 2
        2. Wall distances
        
        We combine them into a single vector of 'effective radii' and take the soft min.
        """
        centers = x_flat.reshape(N, 2)
        
        # Wall distances
        dx_min = centers[:, 0] + 0.5
        dx_max = 0.5 - centers[:, 0]
        dy_min = centers[:, 1] + 0.5
        dy_max = 0.5 - centers[:, 1]
        
        # Pairwise distances
        # We need the gradient of the soft min of all these values.
        # It's more efficient to compute the pairwise distances and their gradients.
        
        # For N=200, pairwise is 19900 pairs.
        # dist_matrix is 200x200.
        dist_matrix = cdist(centers, centers)
        np.fill_diagonal(dist_matrix, np.inf)
        
        # We only care about the upper triangle for pairs
        i_idx, j_idx = np.triu_indices(N, k=1)
        d_pairs = dist_matrix[i_idx, j_idx] / 2.0
        
        # Wall distances
        d_walls = np.concatenate([dx_min, dx_max, dy_min, dy_max])
        
        # Combine all constraints into one vector
        # Note: The soft min is over the union of these sets.
        # f = soft_min(concat(d_pairs, d_walls))
        # We want to minimize -f.
        
        all_dists = np.concatenate([d_pairs, d_walls])
        
        soft_min_val, probs = soft_min_logsumexp(all_dists, beta)
        
        # Objective is -soft_min_val
        obj = -soft_min_val
        
        # Gradient
        # d(-soft_min)/d(centers) = - d(soft_min)/d(centers)
        # d(soft_min)/d(v_k) = probs[k]
        # So we need to backpropagate probs to centers.
        
        grad_flat = np.zeros_like(x_flat)
        
        # 1. Pairwise contributions
        # d_pairs[k] = ||c_i - c_j|| / 2
        # d(d_pairs)/d(c_i) = (c_i - c_j) / ||c_i - c_j||
        # d(d_pairs)/d(c_j) = (c_j - c_i) / ||c_i - c_j||
        
        # We have probs for each pair.
        # grad_centers[i] += sum_{j} probs_pair[i,j] * (c_i - c_j)/||c_i - c_j||
        
        # To do this efficiently, we can use the full dist_matrix and probability matrix.
        # Let P be the NxN matrix of probabilities for pairs (symmetric).
        # P[i,j] = probs[k] if (i,j) is the k-th pair.
        
        P = np.zeros((N, N))
        P[i_idx, j_idx] = probs[:len(d_pairs)]
        P[j_idx, i_idx] = probs[:len(d_pairs)]
        
        # Gradient of distance d(i,j) wrt c_i is (c_i - c_j) / d(i,j)
        # We can compute this using broadcasting.
        # diff = centers[:, None, :] - centers[None, :, :]  (N, N, 2)
        # dists = np.linalg.norm(diff, axis=2)
        # grad_dir = diff / dists[:, :, None]
        
        # This is O(N^2) memory and compute. N=200 -> 40k vectors. Acceptable.
        
        # Optimization: Only compute for non-zero probabilities? 
        # With soft min, many probabilities are non-zero but small.
        # For beta=100, it's quite sparse. But let's just do the full vectorized op for simplicity first.
        
        # Actually, computing cdist again is redundant. We have dist_matrix.
        # But we need the unit vectors.
        
        # Let's compute the gradient of the soft min with respect to the distance vector, 
        # then use the chain rule.
        
        # grad_obj = - (d soft_min / d centers)
        # d soft_min / d centers = sum_k probs[k] * (d v_k / d centers)
        
        # Pair part:
        # diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        # dists_full = np.linalg.norm(diff, axis=2)
        # np.fill_diagonal(dists_full, 1.0) # Avoid div by zero
        # unit_vecs = diff / dists_full[:, :, np.newaxis]
        # # unit_vecs[i, j] is vector from j to i.
        # # d ||c_i - c_j|| / d c_i = (c_i - c_j) / ||c_i - c_j|| = unit_vecs[i, j]
        # # d ||c_i - c_j|| / d c_j = (c_j - c_i) / ||c_i - c_j|| = -unit_vecs[i, j]
        
        # grad_pair_centers = np.sum(P[:, :, np.newaxis] * unit_vecs, axis=1)
        
        # Wait, P[i,j] is prob for pair (i,j).
        # Contribution to grad c_i from pair (i,j) is P[i,j] * unit_vecs[i,j]
        # Contribution to grad c_j from pair (i,j) is P[i,j] * (-unit_vecs[i,j])
        
        # So grad c_i = sum_j P[i,j] * unit_vecs[i,j]
        
        # Let's compute unit_vecs efficiently.
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        # dists_full is dist_matrix but with diagonal 0. We need to handle diagonal.
        # We have dist_matrix from cdist.
        # unit_vecs[i,j] = (c_i - c_j) / dist(i,j)
        
        # To avoid division by zero on diagonal, set diagonal of dist_matrix to 1 temporarily?
        # No, P diagonal is 0, so we can just set unit_vecs diagonal to 0.
        
        dists_for_div = dist_matrix.copy()
        np.fill_diagonal(dists_for_div, 1.0)
        unit_vecs = diff / dists_for_div[:, :, np.newaxis]
        np.fill_diagonal(unit_vecs, 0.0) # Not strictly needed if P diag is 0, but safe
        
        grad_pair_centers = np.sum(P[:, :, np.newaxis] * unit_vecs, axis=1)
        
        # 2. Wall part
        # d_walls vector: [dx_min (N), dx_max (N), dy_min (N), dy_max (N)]
        # probs_wall = probs[len(d_pairs):]
        p_walls = probs[len(d_pairs):]
        
        # dx_min = x + 0.5. d/dx = 1.
        # dx_max = 0.5 - x. d/dx = -1.
        # dy_min = y + 0.5. d/dy = 1.
        # dy_max = 0.5 - y. d/dy = -1.
        
        grad_wall_centers = np.zeros((N, 2))
        grad_wall_centers[:, 0] += p_walls[:N]       # dx_min
        grad_wall_centers[:, 0] -= p_walls[N:2*N]    # dx_max
        grad_wall_centers[:, 1] += p_walls[2*N:3*N]  # dy_min
        grad_wall_centers[:, 1] -= p_walls[3*N:4*N]  # dy_max
        
        total_grad_centers = grad_pair_centers + grad_wall_centers
        
        # Objective is -soft_min, so gradient is -total_grad_centers
        grad_flat = -total_grad_centers.flatten()
        
        return obj, grad_flat

    def relax_bottlenecks(centers, active_pairs, active_walls, step_size, temperature):
        """
        Performs a single relaxation step targeting only the active constraints.
        """
        new_centers = centers.copy()
        
        for i, j in active_pairs:
            vec = new_centers[i] - new_centers[j]
            dist = np.linalg.norm(vec)
            if dist < 1e-8:
                angle = np.random.uniform(0, 2 * np.pi)
                vec = np.array([math.cos(angle), math.sin(angle)])
                dist = 1.0
            else:
                vec = vec / dist
            
            force = step_size * vec
            new_centers[i] += force
            new_centers[j] -= force
            
        for i, wall_idx in active_walls:
            force = np.zeros(2)
            if wall_idx == 0:
                force[0] = step_size
            elif wall_idx == 1:
                force[0] = -step_size
            elif wall_idx == 2:
                force[1] = step_size
            elif wall_idx == 3:
                force[1] = -step_size
                
            new_centers[i] += force
            
        noise_scale = temperature * 0.01
        if noise_scale > 0:
            new_centers += np.random.normal(0, noise_scale, size=new_centers.shape)
            
        new_centers = np.clip(new_centers, -0.5 + 1e-9, 0.5 - 1e-9)
        
        return new_centers

    # --- Main Optimization Loop ---
    
    # 1. Initialization
    centers = initialize_hex_grid()
    best_centers = centers.copy()
    best_r, _, _ = compute_radius_and_bottlenecks(centers)
    
    # 2. Gradient-Based Optimization (L-BFGS-B)
    # We run this for a few seconds to refine the hex grid.
    # Beta controls the sharpness of the soft min. Start moderate, increase later?
    # Or just use a fixed high beta for "almost exact" min.
    
    current_time = time.time()
    # Allocate ~60s for gradient optimization
    grad_deadline = start_time + 60.0
    
    # We might do multiple runs with different betas or restarts?
    # Let's stick to one main run first to ensure stability.
    
    x0 = centers.flatten()
    
    # Constraints: centers must be in [-0.5, 0.5]
    # L-BFGS-B supports bounds.
    bounds = [(-0.5 + 1e-9, 0.5 - 1e-9)] * (2 * N)
    
    # Define the function for minimize
    # We want to minimize -radius.
    # Our objective returns obj, grad.
    
    # We will use a fixed beta for this phase.
    # Beta=100 is usually good for N=200.
    beta = 100.0
    
    def fun(x):
        return objective_and_grad(x, beta)
        
    # Run L-BFGS-B
    # maxiter needs to be large enough.
    # We check time inside the callback? No, minimize doesn't support time-based stop easily.
    # We can wrap it.
    
    stop_flag = [False]
    def callback(xk):
        if time.time() > grad_deadline - 2.0:
            stop_flag[0] = True
        return stop_flag[0]
        
    # Use a wrapper to handle the stop flag
    # Actually, scipy minimize doesn't stop cleanly on callback return True in all versions?
    # It does.
    
    # Let's try running it.
    # To be safe, we'll run in a way that we can interrupt.
    # Since we can't easily interrupt the C code, we rely on the callback.
    
    try:
        res = minimize(fun, x0, method='L-BFGS-B', bounds=bounds, 
                       options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-12, 'disp': False},
                       callback=callback)
        
        new_centers = res.x.reshape(N, 2)
        new_r, _, _ = compute_radius_and_bottlenecks(new_centers)
        
        if new_r > best_r:
            best_r = new_r
            best_centers = new_centers.copy()
            centers = new_centers.copy()
            
    except Exception as e:
        pass # Keep best_centers

    # 3. Active Set Polish
    # Now that we are in a good basin, use the high-precision active set method to polish.
    # This is faster and more precise for the final bits.
    
    polish_deadline = deadline - 2.0
    current_step = 0.001
    current_temp = 0.0
    
    stuck_counter = 0
    max_stuck = 50
    
    while time.time() < polish_deadline:
        current_r, active_pairs, active_walls = compute_radius_and_bottlenecks(centers)
        
        if not active_pairs and not active_walls:
            break
            
        new_centers = relax_bottlenecks(centers, active_pairs, active_walls, current_step, current_temp)
        
        new_r, _, _ = compute_radius_and_bottlenecks(new_centers)
        
        if new_r >= current_r:
            centers = new_centers
            stuck_counter = 0
            if new_r > best_r:
                best_r = new_r
                best_centers = centers.copy()
            # Reduce step size if improving
            current_step *= 0.9
        else:
            stuck_counter += 1
            # If stuck, try a small random perturbation or reduce step
            if stuck_counter > max_stuck:
                # Perturb active points slightly
                for i, j in active_pairs:
                    centers[i] += np.random.normal(0, 1e-5, 2)
                    centers[j] += np.random.normal(0, 1e-5, 2)
                for i, w in active_walls:
                    centers[i] += np.random.normal(0, 1e-5, 2)
                centers = np.clip(centers, -0.5 + 1e-9, 0.5 - 1e-9)
                stuck_counter = 0
                current_step = max(current_step * 1.5, 1e-4)
            else:
                # Reduce step size
                current_step *= 0.8
                if current_step < 1e-8:
                     break

    # Ensure validity
    if not np.all(np.isfinite(best_centers)):
        m = int(math.ceil(math.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        best_centers = grid[:N]
        
    return best_centers

# EVOLVE-BLOCK-END
