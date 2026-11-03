import numpy as np
import time
import math
from scipy.optimize import minimize

N = 200

def construct():
    start_time = time.time()
    deadline = start_time + 145.0

    def compute_radius(centers):
        if centers.shape != (N, 2):
            return 0.0
        if not np.all(np.isfinite(centers)):
            return 0.0
        if np.any(np.abs(centers) > 0.5 + 1e-9):
            return 0.0
        wall_dists = 0.5 - np.abs(centers)
        min_wall = np.min(wall_dists)
        dist_matrix = np.zeros((N, N))
        for i in range(N):
            dist_matrix[i] = np.linalg.norm(centers - centers[i], axis=1)
        np.fill_diagonal(dist_matrix, np.inf)
        min_pair = np.min(dist_matrix)
        r = min(min_wall, min_pair / 2.0)
        return max(0.0, r)

    def soft_min(values, power=8):
        """
        Smooth approximation of min(values).
        Uses the negative power sum: - (sum(-v^p))^{-1/p}
        This is differentiable and approximates min as p -> inf.
        """
        # Clip to avoid overflow/underflow issues
        v = np.clip(values, 1e-12, None)
        # Use log-sum-exp trick for stability if needed, but for N=200 and moderate p, direct is ok
        # Actually, for min, we want: min(v) ~= - (sum((-v)^p))^{-1/p}
        # Let's use: min(v) ~= - (sum(exp(-p * v)))^{-1/p} * (1/p)? No.
        # Standard soft min: m(v) = - (1/p) * log(sum(exp(-p * v)))
        # This is the LogSumExp of -v divided by -p.
        # Let's use the power mean approximation:
        # M_p(v) = ( (1/n) sum v^p )^{1/p}. As p -> -inf, M_p -> min.
        # Let's use p = -8.
        # M_{-8}(v) = ( (1/n) sum v^{-8} )^{-1/8}
        # This is stable if v > 0.
        n = len(values)
        sum_inv_p = np.sum(v ** (-power))
        return (sum_inv_p / n) ** (-1.0 / power)

    def objective_and_grad(x, target_r):
        """
        Objective: Maximize the minimum distance to boundaries and other circles.
        We minimize the negative of a soft minimum of all constraints.
        
        Constraints:
        1. Wall distances: d_i = 0.5 - |x_i| (for each coord)
        2. Pair distances: d_ij = ||x_i - x_j||
        
        We want to maximize min(all d's).
        Equivalent to minimizing -soft_min(all d's).
        
        Additionally, we want to push towards target_r.
        Actually, a simpler approach:
        Minimize: sum( max(0, target_r - d_k)^2 ) + lambda * (target_r - soft_min(d's))^2 ?
        
        Let's stick to maximizing the soft min of distances.
        To encourage larger radii, we can scale the distances or add a penalty for small distances.
        
        Better approach for inflation:
        We want to find x such that min_dist(x) >= r_target.
        This is a feasibility problem.
        
        Alternative: 
        Minimize: -soft_min(distances)
        This will push all distances up.
        
        Let's use:
        f(x) = -soft_min(all_dist_constraints, power=10)
        
        Grad:
        d/dx [-soft_min(v)] = - d/dx soft_min(v)
        soft_min(v) = (sum v^{-p} / n)^{-1/p}
        Let S = sum v^{-p} / n
        soft_min = S^{-1/p}
        d(soft_min)/dv_k = (-1/p) * S^{-1/p - 1} * dS/dv_k
        dS/dv_k = (1/n) * (-p) * v_k^{-p-1}
        d(soft_min)/dv_k = (-1/p) * S^{-1/p - 1} * (-p/n) * v_k^{-p-1}
        = (1/n) * S^{-1/p - 1} * v_k^{-p-1}
        
        So grad_f = - sum_k (d(soft_min)/dv_k) * (dv_k/dx)
        
        """
        x = x.reshape(N, 2)
        
        # Compute all distances
        # Wall distances
        wall_dists = 0.5 - np.abs(x)  # Shape (N, 2)
        
        # Pair distances
        # For N=200, 200*199/2 = 19900 pairs.
        # Vectorized distance calculation
        diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]  # (N, N, 2)
        dists = np.sqrt(np.sum(diff ** 2, axis=2))  # (N, N)
        # Zero out diagonal
        dists = np.where(np.eye(N, dtype=bool), np.inf, dists)
        
        # Flatten distances for soft_min
        # We only care about the minimum, so including inf is fine if handled, but better to exclude.
        # Let's collect finite distances.
        wall_flat = wall_dists.flatten()
        pair_idx = np.triu_indices(N, k=1)
        pair_flat = dists[pair_idx]
        
        all_dists = np.concatenate([wall_flat, pair_flat])
        
        # Soft min
        p = 10
        v = np.clip(all_dists, 1e-12, None)
        S = np.sum(v ** (-p)) / len(v)
        soft_min_val = S ** (-1.0 / p)
        
        f_val = -soft_min_val
        
        # Gradient
        grad = np.zeros_like(x)
        
        # Coefficient for each distance in soft_min gradient
        # d(soft_min)/dv_k = (1/n) * S^{-1/p - 1} * v_k^{-p-1}
        coeff_base = (1.0 / len(v)) * S ** (-1.0 / p - 1.0)
        
        # Wall gradients
        # v_k is wall_dist. d(wall_dist)/dx = -sign(x)
        # grad_f = - sum coeff_k * d(v_k)/dx
        # coeff_k = coeff_base * v_k^{-p-1}
        
        wall_v = np.clip(wall_dists, 1e-12, None)
        wall_coeff = coeff_base * wall_v ** (-p - 1)
        
        # d(f)/dx = - (d(soft_min)/dx)
        # d(soft_min)/dx = sum_k (d(soft_min)/dv_k) * (dv_k/dx)
        # dv_k/dx = -sign(x)
        # So d(soft_min)/dx = - sum_k wall_coeff_k * sign(x)
        # grad_f = - (- sum ...) = sum_k wall_coeff_k * sign(x)
        
        signs = np.sign(x)
        # Handle zero sign (gradient 0)
        signs[signs == 0] = 1.0 # Arbitrary, but contribution is 0 if dist is large? No, if x=0, sign is 0.
        # Actually if x=0, wall_dist = 0.5. It's not the min usually.
        # If it is the min, we need to push away. Sign is ambiguous. Let's use 1.
        
        grad += wall_coeff * signs
        
        # Pair gradients
        # v_k is dist_ij.
        # d(dist_ij)/dx_i = (x_i - x_j) / dist_ij
        # d(dist_ij)/dx_j = (x_j - x_i) / dist_ij
        
        pair_v = np.clip(pair_flat, 1e-12, None)
        pair_coeff = coeff_base * pair_v ** (-p - 1)
        
        # We need to accumulate gradients for each pair.
        # This is O(N^2).
        # For each pair (i, j), coeff is pair_coeff[k].
        # grad_i += coeff * (x_i - x_j) / dist
        # grad_j -= coeff * (x_i - x_j) / dist
        
        # To vectorize:
        # diff is (N, N, 2).
        # dists is (N, N).
        # We need pair_coeff mapped to (N, N).
        coeff_matrix = np.zeros((N, N))
        coeff_matrix[pair_idx] = pair_coeff
        coeff_matrix = (coeff_matrix + coeff_matrix.T) / 2.0 # Symmetric
        
        # Safe division
        dist_safe = np.where(dists < 1e-9, 1e-9, dists)
        
        # grad_pair = sum_j coeff_ij * (x_i - x_j) / dist_ij
        # This is: sum_j coeff_ij * diff[i, j] / dist_safe[i, j]
        
        # diff[i, j] is vector.
        # term = coeff_matrix / dist_safe  (N, N)
        term = coeff_matrix / dist_safe
        
        # grad_i = sum_j term[i, j] * diff[i, j]
        # np.einsum('ij,ijk->ik', term, diff)
        
        grad_pairs = np.einsum('ij,ijk->ik', term, diff)
        
        # f = -soft_min
        # grad_f = - grad_soft_min
        # grad_soft_min = wall_part + pair_part
        # So grad_f = - (wall_part + pair_part)
        
        # Wait, I derived grad_f = sum wall_coeff * sign(x) earlier.
        # Let's re-verify signs.
        # f = - soft_min
        # df/dx = - d(soft_min)/dx
        # d(soft_min)/dx = sum_k (d sm/dv_k) * (dv_k/dx)
        
        # Wall: dv/dx = -sign(x)
        # d sm/dx_wall = - sum_k c_k sign(x_k)
        # df/dx_wall = sum_k c_k sign(x_k)
        
        # Pair: dv/dx_i = (x_i - x_j)/d
        # d sm/dx_i = sum_j c_ij (x_i - x_j)/d_ij
        # df/dx_i = - sum_j c_ij (x_i - x_j)/d_ij
        
        # So:
        # grad_wall = wall_coeff * signs
        # grad_pair = - grad_pairs
        
        grad = wall_coeff * signs - grad_pairs
        
        return f_val, grad.flatten()

    def hexagonal_init(n_rows, n_cols, d_scale=1.0):
        d = d_scale / (n_cols + 0.5)
        row_spacing = d * math.sqrt(3) / 2
        total_height = (n_rows - 1) * row_spacing + d
        if total_height > 1.0 + 1e-6:
            return None
        pts = []
        for row in range(n_rows):
            y = -0.5 + d/2 + row * row_spacing
            if row % 2 == 0:
                n_in_row = n_cols
                x_start = -0.5 + d/2
            else:
                n_in_row = n_cols - 1
                x_start = -0.5 + d
            for col in range(n_in_row):
                x = x_start + col * d
                pts.append([x, y])
        pts = np.array(pts)
        if len(pts) >= N:
            pts = pts[:N]
        elif len(pts) < N:
            return None
        return pts

    def square_grid_init(m):
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        return grid[:N]

    def optimize_from_init(init, time_limit):
        x0 = init.flatten()
        
        # Bounds
        bounds = [(-0.5, 0.5)] * (2 * N)
        
        start = time.time()
        
        # Staged optimization:
        # 1. Optimize with lower power (smoother) to get out of bad local minima
        # 2. Optimize with higher power (sharper) to refine
        
        best_x = x0.copy()
        best_r = compute_radius(init)
        
        # Stage 1: Smooth
        p_smooth = 4
        def obj_smooth(x):
            f, g = objective_and_grad(x, 0)
            # We need to modify objective_and_grad to take power
            # Let's refactor objective to take power
            return f, g
            
        # Since objective_and_grad is hardcoded to p=10, let's make a wrapper or modify it.
        # For now, let's just run p=10 but with multiple restarts.
        # Or modify objective to accept power.
        
        # Let's modify objective_and_grad to accept power.
        
        # Re-defining inner function to allow power change
        def obj_with_power(x, power):
            x = x.reshape(N, 2)
            wall_dists = 0.5 - np.abs(x)
            diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
            dists = np.sqrt(np.sum(diff ** 2, axis=2))
            dists = np.where(np.eye(N, dtype=bool), np.inf, dists)
            
            wall_flat = wall_dists.flatten()
            pair_idx = np.triu_indices(N, k=1)
            pair_flat = dists[pair_idx]
            
            all_dists = np.concatenate([wall_flat, pair_flat])
            
            v = np.clip(all_dists, 1e-12, None)
            S = np.sum(v ** (-power)) / len(v)
            soft_min_val = S ** (-1.0 / power)
            
            f_val = -soft_min_val
            
            grad = np.zeros_like(x)
            
            coeff_base = (1.0 / len(v)) * S ** (-1.0 / power - 1.0)
            
            # Wall
            wall_v = np.clip(wall_dists, 1e-12, None)
            wall_coeff = coeff_base * wall_v ** (-power - 1)
            signs = np.sign(x)
            signs[signs == 0] = 1.0
            grad += wall_coeff * signs
            
            # Pair
            pair_v = np.clip(pair_flat, 1e-12, None)
            pair_coeff = coeff_base * pair_v ** (-power - 1)
            
            coeff_matrix = np.zeros((N, N))
            coeff_matrix[pair_idx] = pair_coeff
            coeff_matrix = (coeff_matrix + coeff_matrix.T) / 2.0
            
            dist_safe = np.where(dists < 1e-9, 1e-9, dists)
            term = coeff_matrix / dist_safe
            grad_pairs = np.einsum('ij,ijk->ik', term, diff)
            
            grad = grad - grad_pairs
            
            return f_val, grad.flatten()
        
        # Stage 1: p=4
        if time.time() < start + time_limit * 0.4:
            res = minimize(
                lambda x: obj_with_power(x, 4)[0],
                x0,
                jac=lambda x: obj_with_power(x, 4)[1],
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 500, 'gtol': 1e-5}
            )
            if res.success or res.x is not None:
                r_curr = compute_radius(res.x.reshape(N, 2))
                if r_curr > best_r:
                    best_r = r_curr
                    best_x = res.x.copy()
        
        # Stage 2: p=10
        if time.time() < start + time_limit * 0.8:
            res = minimize(
                lambda x: obj_with_power(x, 10)[0],
                best_x,
                jac=lambda x: obj_with_power(x, 10)[1],
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 1000, 'gtol': 1e-6}
            )
            if res.success or res.x is not None:
                r_curr = compute_radius(res.x.reshape(N, 2))
                if r_curr > best_r:
                    best_r = r_curr
                    best_x = res.x.copy()
        
        # Stage 3: p=20 (Very sharp, fine tuning)
        if time.time() < start + time_limit:
            res = minimize(
                lambda x: obj_with_power(x, 20)[0],
                best_x,
                jac=lambda x: obj_with_power(x, 20)[1],
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 2000, 'gtol': 1e-8}
            )
            if res.success or res.x is not None:
                r_curr = compute_radius(res.x.reshape(N, 2))
                if r_curr > best_r:
                    best_r = r_curr
                    best_x = res.x.copy()
                    
        return best_x.reshape(N, 2), best_r

    # Main logic
    best_centers = None
    best_radius = 0.0
    
    # Generate initializations
    inits = []
    
    # Hexagonal
    for n_rows in range(13, 17):
        if time.time() > deadline - 30:
            break
        n_cols = math.ceil(N / n_rows)
        for d_scale in [0.95, 1.0, 1.05]:
            pts = hexagonal_init(n_rows, n_cols, d_scale)
            if pts is not None:
                inits.append(pts)
                
    # Square Grid
    m = int(np.ceil(np.sqrt(N)))
    grid = square_grid_init(m)
    inits.append(grid)
    
    # Random
    np.random.seed(42) # Reproducibility for first run? Or random?
    for _ in range(3):
        if time.time() > deadline - 30:
            break
        rand_pts = np.random.uniform(-0.49, 0.49, (N, 2))
        inits.append(rand_pts)
        
    # Optimize
    for i, init in enumerate(inits):
        if time.time() > deadline - 10:
            break
            
        # Estimate time per init
        remaining = deadline - time.time()
        t_per_init = remaining / (len(inits) - i)
        t_limit = min(t_per_init, 20.0) # Cap per init
        
        if t_limit < 2.0:
            break
            
        centers, r = optimize_from_init(init, t_limit)
        
        if r > best_radius:
            best_radius = r
            best_centers = centers.copy()
            
    # Final check
    if best_centers is None:
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        best_centers = grid[:N]
        
    return best_centers

# EVOLVE-BLOCK-END
