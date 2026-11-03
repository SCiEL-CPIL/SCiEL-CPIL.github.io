import numpy as np
import time
import math

N = 200

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import cdist
from scipy.optimize import minimize

def construct():
    start_time = time.time()
    deadline = start_time + 145.0

    def compute_radius(centers):
        if centers.shape != (N, 2):
            return 0.0
        if not np.all(np.isfinite(centers)):
            return 0.0
        wall_dists = 0.5 - np.abs(centers)
        min_wall = np.min(wall_dists)
        dist_matrix = cdist(centers, centers)
        np.fill_diagonal(dist_matrix, np.inf)
        min_pair = np.min(dist_matrix)
        r = min(min_wall, min_pair / 2.0)
        return max(0.0, r)

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

    def repulsive_energy(centers, p=4.0, d0=0.05):
        """
        Smooth repulsive energy: sum_{i<j} max(0, d0 - d_ij)^p
        Plus wall penalty: sum_i max(0, -wall_dist_i)^p
        This is differentiable and smooth.
        """
        n = len(centers)
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dists, np.inf)
        
        # Pairwise repulsion: penalize distances less than d0
        # Use upper triangle to avoid double counting
        iu = np.triu_indices(n, 1)
        pair_dists = dists[iu]
        violations = np.maximum(0, d0 - pair_dists)
        pair_energy = np.sum(violations ** p)
        
        # Wall penalty
        wall_dists = 0.5 - np.abs(centers)
        flat_wall = wall_dists.flatten()
        wall_violations = np.maximum(0, -flat_wall)
        wall_energy = np.sum(wall_violations ** p)
        
        return pair_energy + wall_energy

    def repulsive_grad(centers, p=4.0, d0=0.05):
        """Gradient of repulsive_energy w.r.t. centers."""
        n = len(centers)
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dists, 1.0)  # avoid div by 0
        
        grad = np.zeros_like(centers)
        
        # Pairwise
        # d/dc_i [ (d0 - |c_i - c_j|)^p ] = p*(d0 - d)^(p-1) * (c_i - c_j)/d * (-1) ? 
        # Let v = max(0, d0 - d). If v > 0: d(v)/dc_i = -(c_i - c_j)/d
        # d(v^p)/dc_i = p * v^(p-1) * (-(c_i - c_j)/d)
        
        # Vectorized:
        # Only compute for pairs where d < d0
        mask = dists < d0
        # Exclude diagonal
        mask = np.where(np.eye(n, dtype=bool), False, mask)
        
        # v = d0 - dists (positive where mask is true)
        v = np.where(mask, d0 - dists, 0.0)
        
        # factor = p * v^(p-1) / dists
        # Handle v=0: v^(p-1) is 0 if p>1, which is fine.
        # But if p=1, v^0 = 1. Here p=4, so v^3.
        # Safe division: where mask is false, factor is 0.
        
        with np.errstate(divide='ignore', invalid='ignore'):
            factor = np.where(mask, p * (v ** (p-1)) / dists, 0.0)
        
        # grad_i = - sum_j factor[i,j] * diff[i,j]
        # diff[i,j] = c_i - c_j
        # So grad_i = - sum_j factor[i,j] * (c_i - c_j)
        
        contrib = np.sum(factor[:, :, np.newaxis] * diff, axis=1)
        grad -= contrib
        
        # Wall gradient
        # E_wall = sum (max(0, -w_k))^p
        # d/dc_k = p * (-w_k)^(p-1) * (-1) * sign(c_k)  if w_k < 0
        # = -p * (-w_k)^(p-1) * sign(c_k)
        
        wall_dists = 0.5 - np.abs(centers)
        flat_wall = wall_dists.flatten()
        wall_grad_flat = np.zeros_like(flat_wall)
        
        # Indices where wall is violated (w < 0)
        viol_mask = flat_wall < 0
        if np.any(viol_mask):
            # w = 0.5 - |c|, so -w = |c| - 0.5
            neg_w = -flat_wall[viol_mask]
            # d(-w)/dc = sign(c)
            # d((-w)^p)/dc = p*(-w)^(p-1) * sign(c)
            # d(max(0,-w)^p)/dc = same
            # But wait, E_wall = sum (max(0, -w))^p.
            # Let u = -w. If u>0, E = u^p. dE/dc = p*u^(p-1) * du/dc.
            # u = -(0.5 - |c|) = |c| - 0.5.
            # du/dc = sign(c).
            # So dE/dc = p * u^(p-1) * sign(c).
            
            # We are computing grad of E.
            # So grad_wall = p * u^(p-1) * sign(c)
            
            idx = np.where(viol_mask)[0]
            c_flat = centers.flatten()
            sign_c = np.sign(c_flat[idx])
            # Handle c=0
            sign_c[sign_c == 0] = 1.0 # Arbitrary, but gradient is 0 anyway if u=0? No, u>0 here.
            
            wall_grad_flat[idx] = p * (neg_w ** (p-1)) * sign_c
        
        grad += wall_grad_flat.reshape(n, 2)
        
        return grad

    def stochastic_gradient_descent(centers, time_limit=10.0, p=4.0, d0=0.05):
        """
        SGD with noise injection to escape local minima.
        """
        centers = centers.copy()
        start = time.time()
        
        lr = 0.01
        noise_scale = 0.005
        momentum = 0.9
        velocity = np.zeros_like(centers)
        
        # Adaptive d0: start small, maybe increase?
        # Or just fix d0.
        
        while time.time() < start + time_limit:
            # Compute gradient
            grad = repulsive_grad(centers, p=p, d0=d0)
            
            # Add noise
            noise = np.random.randn(*centers.shape) * noise_scale
            
            # Update with momentum
            velocity = momentum * velocity - lr * (grad + noise)
            centers += velocity
            
            # Clamp to bounds
            centers[:, 0] = np.clip(centers[:, 0], -0.5, 0.5)
            centers[:, 1] = np.clip(centers[:, 1], -0.5, 0.5)
            
            # Decay noise and lr
            lr *= 0.999
            noise_scale *= 0.999
            
        return centers

    def bottleneck_elimination(centers, time_limit=5.0):
        """
        Local search: repeatedly find the closest pair (or wall violation) and push them apart.
        """
        centers = centers.copy()
        start = time.time()
        
        while time.time() < start + time_limit:
            # Find bottleneck
            wall_dists = 0.5 - np.abs(centers)
            min_wall = np.min(wall_dists)
            
            dist_matrix = cdist(centers, centers)
            np.fill_diagonal(dist_matrix, np.inf)
            min_pair = np.min(dist_matrix)
            
            if min_wall < min_pair / 2.0:
                # Wall is bottleneck
                idx = np.argmin(wall_dists) // 2
                axis = np.argmin(wall_dists) % 2
                # Push inwards
                if centers[idx, axis] > 0:
                    centers[idx, axis] -= 0.001
                else:
                    centers[idx, axis] += 0.001
            else:
                # Pair is bottleneck
                i, j = np.unravel_index(np.argmin(dist_matrix), (N, N))
                if i == j:
                    continue
                diff = centers[i] - centers[j]
                dist = np.linalg.norm(diff)
                if dist < 1e-10:
                    # Random kick
                    centers[i] += np.random.randn(2) * 0.001
                    centers[j] -= np.random.randn(2) * 0.001
                else:
                    direction = diff / dist
                    # Push i away from j, j away from i
                    step = 0.001
                    centers[i] += direction * step
                    centers[j] -= direction * step
            
            # Clamp
            centers[:, 0] = np.clip(centers[:, 0], -0.5, 0.5)
            centers[:, 1] = np.clip(centers[:, 1], -0.5, 0.5)
            
        return centers

    def soft_min_polish(centers, time_limit=10.0):
        """
        Final polish using L-BFGS on soft-min objective.
        """
        def soft_min_obj(x, k=50.0):
            c = x.reshape(-1, 2)
            wall_dists = 0.5 - np.abs(c)
            flat_wall = wall_dists.flatten()
            dist_matrix = cdist(c, c)
            np.fill_diagonal(dist_matrix, np.inf)
            flat_pair = dist_matrix.flatten()
            
            vals = np.concatenate([flat_wall, flat_pair / 2.0])
            min_val = np.min(vals)
            # Stable log-sum-exp
            neg_k_vals = -k * vals
            max_neg_k_val = np.max(neg_k_vals)
            log_sum_exp = max_neg_k_val + np.log(np.sum(np.exp(neg_k_vals - max_neg_k_val)))
            soft_min_val = -log_sum_exp / k
            return -soft_min_val
        
        def soft_min_grad(x, k=50.0):
            c = x.reshape(-1, 2)
            wall_dists = 0.5 - np.abs(c)
            flat_wall = wall_dists.flatten()
            dist_matrix = cdist(c, c)
            np.fill_diagonal(dist_matrix, np.inf)
            flat_pair = dist_matrix.flatten()
            
            vals = np.concatenate([flat_wall, flat_pair / 2.0])
            neg_k_vals = -k * vals
            max_neg_k_val = np.max(neg_k_vals)
            exp_vals = np.exp(neg_k_vals - max_neg_k_val)
            sum_exp = np.sum(exp_vals)
            probs = exp_vals / sum_exp
            
            grad_vals = -probs
            grad_centers = np.zeros_like(c)
            
            # Wall part
            grad_wall_flat = grad_vals[:len(flat_wall)]
            grad_wall = grad_wall_flat.reshape(N, 2)
            sign_c = np.sign(c)
            grad_centers += grad_wall * sign_c
            
            # Pair part
            grad_pair_flat = grad_vals[len(flat_wall):]
            diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
            dists = np.linalg.norm(diff, axis=2)
            np.fill_diagonal(dists, 1.0)
            
            grad_pair_matrix = grad_pair_flat.reshape(N, N)
            factor = 0.5 * grad_pair_matrix / dists
            contrib_i = np.sum(factor[:, :, np.newaxis] * diff, axis=1)
            grad_centers += contrib_i
            
            return grad_centers.flatten()
        
        x0 = centers.flatten()
        bounds = [(-0.5, 0.5)] * (2 * N)
        
        try:
            result = minimize(
                soft_min_obj,
                x0,
                method='L-BFGS-B',
                jac=soft_min_grad,
                bounds=bounds,
                options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            return result.x.reshape(-1, 2)
        except Exception:
            return centers

    # Main logic
    best_centers = None
    best_radius = 0.0
    
    # 1. Generate initial configurations
    inits = []
    
    # Hexagonal
    for n_rows in range(13, 17):
        if time.time() > deadline - 60:
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
    for _ in range(3):
        if time.time() > deadline - 60:
            break
        rand_pts = np.random.uniform(-0.49, 0.49, (N, 2))
        inits.append(rand_pts)
    
    # 2. For each init, run the new strategy: SGD -> Bottleneck -> Polish
    for init in inits:
        if time.time() > deadline - 30:
            break
            
        # Quick eval
        r_init = compute_radius(init)
        if r_init > best_radius:
            best_radius = r_init
            best_centers = init.copy()
            
        # Phase 1: Stochastic Gradient Descent (Repulsive Field)
        if time.time() < deadline - 20:
            sgd_centers = stochastic_gradient_descent(init, time_limit=5.0, p=4.0, d0=0.05)
            r_sgd = compute_radius(sgd_centers)
            if r_sgd > best_radius:
                best_radius = r_sgd
                best_centers = sgd_centers.copy()
                
        # Phase 2: Bottleneck Elimination
        if time.time() < deadline - 10:
            # Use best so far from this init (either init or sgd)
            current = best_centers if (best_centers is init or np.allclose(best_centers, sgd_centers)) else (sgd_centers if r_sgd > r_init else init)
            # Actually, just polish the best of init and sgd
            cand = sgd_centers if r_sgd > r_init else init
            bottleneck_centers = bottleneck_elimination(cand, time_limit=3.0)
            r_bottleneck = compute_radius(bottleneck_centers)
            if r_bottleneck > best_radius:
                best_radius = r_bottleneck
                best_centers = bottleneck_centers.copy()
                
        # Phase 3: Soft-Min Polish
        if time.time() < deadline - 5:
            # Polish the best of the above
            cand = best_centers
            polished = soft_min_polish(cand, time_limit=5.0)
            r_polished = compute_radius(polished)
            if r_polished > best_radius:
                best_radius = r_polished
                best_centers = polished.copy()
                
    # Final global polish if time remains
    if time.time() < deadline - 2 and best_centers is not None:
        final_polished = soft_min_polish(best_centers, time_limit=2.0)
        r_final = compute_radius(final_polished)
        if r_final > best_radius:
            best_radius = r_final
            best_centers = final_polished.copy()
            
    if best_centers is None:
        # Fallback
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        best_centers = grid[:N]
        
    return best_centers
# EVOLVE-BLOCK-END
