import numpy as np
import time
import math
from scipy.spatial.distance import cdist
from scipy.optimize import minimize

N = 200

# EVOLVE-BLOCK-START
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

    def generate_hex_target(n_rows=14, n_cols=15):
        """
        Generate a target density grid for the hexagonal lattice.
        This serves as the 'source' for the optimal transport analogy.
        We create a coarse grid representing the ideal hex packing.
        """
        # Hexagonal packing density is ~0.9069
        # We want a target that encourages points to be in hex-like positions.
        # Instead of explicit OT, we use a 'target potential' V_target(x)
        # such that minimizing sum V_target(c_i) pushes points towards high-density regions.
        # The high-density regions of a hex lattice are the lattice points themselves.
        
        # Let's create a set of 'ideal' hex points and use them to define a potential.
        # d = spacing
        # For N=200 in unit square, approximate d ~ 0.07
        d = 0.07
        row_spacing = d * math.sqrt(3) / 2
        
        pts = []
        y = -0.5 + d/2
        row_idx = 0
        while y < 0.5:
            if row_idx % 2 == 0:
                x_start = -0.5 + d/2
                x = x_start
                while x < 0.5:
                    pts.append([x, y])
                    x += d
            else:
                x_start = -0.5 + d
                x = x_start
                while x < 0.5:
                    pts.append([x, y])
                    x += d
            y += row_spacing
            row_idx += 1
            
        return np.array(pts)

    def target_potential_and_grad(centers, target_pts, sigma=0.05):
        """
        Compute potential V(x) = sum_j exp(-|x - t_j|^2 / (2*sigma^2))
        and its gradient.
        Minimizing sum V(c_i) is NOT what we want. We want to MAXIMIZE sum V(c_i)
        to push points towards the target density peaks.
        So we minimize -sum V(c_i).
        """
        n = len(centers)
        # dist_sq[i, j] = |c_i - t_j|^2
        diff = centers[:, np.newaxis, :] - target_pts[np.newaxis, :, :]
        dist_sq = np.sum(diff**2, axis=2)
        
        # V_i = sum_j exp(-dist_sq[i,j] / (2*sigma^2))
        # grad V_i / grad c_i = sum_j exp(...) * (-(c_i - t_j) / sigma^2)
        
        coeff = 1.0 / (2.0 * sigma * sigma)
        exp_term = np.exp(-dist_sq * coeff)
        
        # Potential value (to be maximized, so objective is -sum)
        val = np.sum(exp_term, axis=1)
        
        # Gradient
        # grad_i = - sum_j exp(-dist_sq[i,j] * coeff) * (c_i - t_j) / sigma^2
        # factor_j = exp_term[i,j] / sigma^2
        factor = exp_term / (sigma * sigma)
        
        # grad_i = - sum_j factor[i,j] * diff[i,j]
        grad = -np.sum(factor[:, :, np.newaxis] * diff, axis=1)
        
        return -np.sum(val), grad

    def repulsive_potential_and_grad(centers, r_target=0.035, k=500.0):
        """
        Smooth repulsive potential to prevent overlap.
        V_rep = sum_{i<j} k * (r_target - d_ij)^2 for d_ij < r_target
        Actually, let's use a soft penalty that is always active but stronger when close.
        V_rep = sum_{i<j} exp(-d_ij / r_target) is not great for gradients at large d.
        Let's use V_rep = sum_{i<j} max(0, r_target - d_ij)^2
        This is non-smooth at d=r_target.
        Smooth version: V_rep = sum_{i<j} (r_target - d_ij + sqrt((r_target - d_ij)^2 + eps))^2 / 2? No.
        
        Let's use: V_rep = sum_{i<j} (r_target^2 - d_ij^2)^2 / d_ij^2 ? No.
        
        Standard smooth repulsion: V(d) = exp(-d^2 / r0^2) is always positive.
        Gradient: dV/dc_i = -2/r0^2 * exp(...) * (c_i - c_j)
        This pushes points apart, but the force decays quickly.
        For packing, we want a 'hard' constraint feel.
        
        Let's combine:
        Objective = -lambda_hex * sum V_hex(c_i) + lambda_rep * sum_{i<j} V_rep(c_i, c_j)
        
        V_rep(d) = (r_target - d)^2 for d < r_target, 0 otherwise? Non-smooth.
        Use: V_rep(d) = 0.5 * (r_target - d)^2 * (1 + tanh(k*(r_target - d)))?
        
        Let's stick to a simple smooth repulsion that is strong at small d.
        V_rep(d) = exp(-d / r_scale)
        grad V_rep / dc_i = -1/r_scale * exp(-d/r_scale) * (c_i - c_j) / d * (c_i - c_j) ?
        d/dx exp(-x/a) = -1/a exp(-x/a).
        d/d c_i exp(-|c_i-c_j|/a) = -1/a exp(-|c_i-c_j|/a) * (c_i-c_j)/|c_i-c_j|
        
        This is good.
        """
        n = len(centers)
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dists, np.inf)
        
        r_scale = 0.02
        exp_term = np.exp(-dists / r_scale)
        
        # Sum over j for each i, excluding i=j
        # We only want i<j to avoid double counting, but for gradient sum it's symmetric.
        # Let's sum over all j != i.
        val = np.sum(exp_term, axis=1)
        
        # Gradient
        # grad_i = -1/r_scale * sum_j exp(-d_ij/r_scale) * (c_i - c_j) / d_ij
        with np.errstate(divide='ignore', invalid='ignore'):
            factor = np.where(dists > 1e-6, exp_term / (r_scale * dists), 0.0)
        
        grad = -np.sum(factor[:, :, np.newaxis] * diff, axis=1)
        
        return np.sum(val), grad

    def total_objective_and_grad(centers, target_pts, lam_hex=10.0, lam_rep=1.0, sigma=0.05, r_target=0.035):
        """
        Minimizes: -lam_hex * sum V_hex(c_i) + lam_rep * sum V_rep(c_i, c_j)
        """
        val_hex, grad_hex = target_potential_and_grad(centers, target_pts, sigma)
        val_rep, grad_rep = repulsive_potential_and_grad(centers, r_target)
        
        val = -lam_hex * val_hex + lam_rep * val_rep
        grad = -lam_hex * grad_hex + lam_rep * grad_rep
        
        return val, grad

    def optimize_with_l_bfgs(centers0, target_pts, time_limit=10.0):
        """
        Run L-BFGS-B on the combined objective.
        """
        def obj(x):
            c = x.reshape(-1, 2)
            v, g = total_objective_and_grad(c, target_pts)
            return v, g
        
        x0 = centers0.flatten()
        bounds = [(-0.5, 0.5)] * (2 * N)
        
        try:
            result = minimize(
                obj,
                x0,
                method='L-BFGS-B',
                jac=True,
                bounds=bounds,
                options={'maxiter': 200, 'ftol': 1e-8, 'gtol': 1e-6, 'maxls': 50}
            )
            return result.x.reshape(-1, 2)
        except Exception:
            return centers0

    def local_polish(centers, time_limit=5.0):
        """
        Standard local polish to fix any remaining overlaps or wall violations.
        Uses a soft-min objective to maximize the minimum radius.
        """
        def soft_min_obj(x, k=100.0):
            c = x.reshape(-1, 2)
            wall_dists = 0.5 - np.abs(c)
            flat_wall = wall_dists.flatten()
            dist_matrix = cdist(c, c)
            np.fill_diagonal(dist_matrix, np.inf)
            flat_pair = dist_matrix.flatten()
            
            vals = np.concatenate([flat_wall, flat_pair / 2.0])
            min_val = np.min(vals)
            neg_k_vals = -k * vals
            max_neg_k_val = np.max(neg_k_vals)
            log_sum_exp = max_neg_k_val + np.log(np.sum(np.exp(neg_k_vals - max_neg_k_val)))
            soft_min_val = -log_sum_exp / k
            return -soft_min_val
        
        def soft_min_grad(x, k=100.0):
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
                options={'maxiter': 100, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            return result.x.reshape(-1, 2)
        except Exception:
            return centers

    # Main Logic
    best_centers = None
    best_radius = 0.0
    
    # 1. Generate Target
    target_pts = generate_hex_target()
    
    # 2. Generate Initial Configurations
    inits = []
    
    # Jittered Hex Grid
    d = 0.07
    row_spacing = d * math.sqrt(3) / 2
    pts = []
    y = -0.5 + d/2
    row_idx = 0
    while y < 0.5 and len(pts) < N:
        if row_idx % 2 == 0:
            x_start = -0.5 + d/2
            x = x_start
            while x < 0.5 and len(pts) < N:
                pts.append([x, y])
                x += d
        else:
            x_start = -0.5 + d
            x = x_start
            while x < 0.5 and len(pts) < N:
                pts.append([x, y])
                x += d
        y += row_spacing
        row_idx += 1
        
    if len(pts) >= N:
        hex_init = np.array(pts[:N])
        # Add small jitter to break symmetry
        hex_init_jittered = hex_init + np.random.randn(N, 2) * 0.001
        inits.append(hex_init_jittered)
        
    # Square Grid
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    if len(grid) >= N:
        inits.append(grid[:N] + np.random.randn(N, 2) * 0.001)
        
    # Random
    inits.append(np.random.uniform(-0.49, 0.49, (N, 2)))

    # 3. Optimization Loop
    for init in inits:
        if time.time() > deadline - 30:
            break
            
        r_init = compute_radius(init)
        if r_init > best_radius:
            best_radius = r_init
            best_centers = init.copy()
            
        # Phase 1: Optimal Transport Style Optimization (Gradient Descent on Potential)
        if time.time() < deadline - 10:
            ot_centers = optimize_with_l_bfgs(init, target_pts, time_limit=10.0)
            r_ot = compute_radius(ot_centers)
            if r_ot > best_radius:
                best_radius = r_ot
                best_centers = ot_centers.copy()
                
        # Phase 2: Local Polish
        if time.time() < deadline - 5:
            polished = local_polish(best_centers, time_limit=5.0)
            r_polished = compute_radius(polished)
            if r_polished > best_radius:
                best_radius = r_polished
                best_centers = polished.copy()
                
    if best_centers is None:
        # Fallback
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        best_centers = grid[:N]
        
    return best_centers
# EVOLVE-BLOCK-END
