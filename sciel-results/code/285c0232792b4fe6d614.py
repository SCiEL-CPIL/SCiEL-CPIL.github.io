"""
Combined approach: Vectorized Smooth Surrogate Optimization for Sphere Packing.
Uses soft-min via log-sum-exp with vectorized gradients for speed.
Implements beta-annealing for global-to-local convergence and a final direct polish.
"""
import numpy as np
import time
import scipy.optimize as opt
import scipy.special as sp

N = 50
CUBE_HALF = 0.5
DEADLINE_SECONDS = 145.0  # Safety margin before 150s kill

def construct():
    start_time = time.time()
    
    def elapsed():
        return time.time() - start_time
    
    def remaining():
        return DEADLINE_SECONDS - elapsed()

    def compute_radius(positions):
        """Compute the exact minimum radius for given positions."""
        positions = np.asarray(positions, dtype=np.float64)
        if positions.shape != (N, 3) or not np.all(np.isfinite(positions)):
            return 0.0
        if np.any(np.abs(positions) > 0.5 + 1e-9):
            return 0.0
        
        # Pairwise distances
        diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
        d = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(d, np.inf)
        min_pair_dist = np.min(d)
        
        # Wall distances
        dist_to_walls = np.min(0.5 - np.abs(positions), axis=1)
        min_wall_dist = np.min(dist_to_walls)
        
        return min(min_pair_dist / 2.0, min_wall_dist)

    def smooth_objective_and_grad(c_flat, beta):
        """
        Computes the smooth surrogate objective (-soft_min) and its gradient.
        c_flat: (3N,)
        beta: temperature parameter
        Returns: (value, grad_flat)
        """
        c = c_flat.reshape(N, 3)
        
        # 1. Calculate all "radius candidates" g_k
        # Pairwise: d_ij / 2
        tri_idx = np.triu_indices(N, k=1)
        diff = c[tri_idx[0]] - c[tri_idx[1]]
        d_pairs = np.linalg.norm(diff, axis=1) / 2.0
        
        # Walls: 6 distances per point
        # d_x_pos = 0.5 - c[:,0], d_x_neg = c[:,0] + 0.5, etc.
        d_x_pos = 0.5 - c[:, 0]
        d_x_neg = c[:, 0] + 0.5
        d_y_pos = 0.5 - c[:, 1]
        d_y_neg = c[:, 1] + 0.5
        d_z_pos = 0.5 - c[:, 2]
        d_z_neg = c[:, 2] + 0.5
        
        wall_dists = np.concatenate([d_x_pos, d_x_neg, d_y_pos, d_y_neg, d_z_pos, d_z_neg])
        
        # Combine all g_k
        g_all = np.concatenate([d_pairs, wall_dists])
        
        # 2. Soft Min Calculation
        # f = -1/beta * log(sum(exp(-beta * g)))
        # Using logsumexp for stability
        log_sum_exp = sp.logsumexp(-beta * g_all)
        val = -log_sum_exp / beta
        
        # 3. Gradient Calculation
        # grad_f = sum_j (weights_j * dg_j/dc)
        # weights_j = exp(-beta * g_j) / sum(exp(-beta * g))
        # log(weights_j) = -beta * g_j - log_sum_exp
        weights = np.exp(-beta * g_all - log_sum_exp)
        
        grad_c = np.zeros_like(c)
        
        # Pairwise part
        # d(d_ij/2)/dc_i = (c_i - c_j) / (2 * d_ij) * 2? No.
        # d_ij = ||c_i - c_j||. d(d_ij/2)/dc_i = 0.5 * (c_i - c_j) / ||c_i - c_j||
        # We have d_pairs = d_ij / 2. So d(d_pairs)/dc_i = (c_i - c_j) / (2 * d_ij) * 2? 
        # Let's re-verify.
        # Let f = 0.5 * ||x-y||. grad_x f = 0.5 * (x-y)/||x-y||.
        # In our code, d_pairs[k] is the value g_k.
        # diff = c[i] - c[j]. norm = 2 * d_pairs[k].
        # grad term = weights[k] * 0.5 * diff / (2 * d_pairs[k])?
        # Wait, d_pairs[k] = 0.5 * ||diff||. So ||diff|| = 2 * d_pairs[k].
        # grad_x (0.5 * ||diff||) = 0.5 * diff / ||diff|| = 0.5 * diff / (2 * d_pairs[k]) = diff / (4 * d_pairs[k]).
        # Let's stick to the standard form:
        # g_k = 0.5 * ||c_i - c_j||
        # dg_k/dc_i = 0.5 * (c_i - c_j) / ||c_i - c_j||
        
        n_pairs = len(d_pairs)
        # Vectorize pairwise gradient accumulation? 
        # Since N=50, n_pairs=1225. A loop is acceptable but vectorized is better if possible.
        # However, accumulating into grad_c requires scattering.
        # We can use np.add.at or just loop since it's only 1225 iterations.
        
        for k in range(n_pairs):
            w = weights[k]
            if w < 1e-12:
                continue
            i = tri_idx[0][k]
            j = tri_idx[1][k]
            d_val = d_pairs[k]
            if d_val < 1e-12:
                continue
            # diff = c[i] - c[j]
            # grad contribution to c[i]: w * 0.5 * diff / (2*d_val) ?
            # No, d_val = 0.5 * ||diff||.
            # grad = 0.5 * diff / ||diff|| = 0.5 * diff / (2 * d_val) = diff / (4 * d_val).
            # Let's re-calculate carefully.
            # g = 0.5 * norm(diff).
            # d g / d c_i = 0.5 * diff / norm(diff).
            # norm(diff) = 2 * d_val.
            # So d g / d c_i = 0.5 * diff / (2 * d_val) = diff / (4 * d_val).
            
            factor = w / (4.0 * d_val)
            grad_c[i] += factor * diff[k]
            grad_c[j] -= factor * diff[k]
            
        # Wall part
        # Weights for walls start at index n_pairs
        w_start = n_pairs
        
        # d_x_pos = 0.5 - c[:,0] -> grad -1
        grad_c[:, 0] -= weights[w_start : w_start+N]
        # d_x_neg = c[:,0] + 0.5 -> grad +1
        grad_c[:, 0] += weights[w_start+N : w_start+2*N]
        
        # d_y_pos
        grad_c[:, 1] -= weights[w_start+2*N : w_start+3*N]
        # d_y_neg
        grad_c[:, 1] += weights[w_start+3*N : w_start+4*N]
        
        # d_z_pos
        grad_c[:, 2] -= weights[w_start+4*N : w_start+5*N]
        # d_z_neg
        grad_c[:, 2] += weights[w_start+5*N : w_start+6*N]
        
        # We want to MINIMIZE -soft_min.
        # So objective is -val, gradient is -grad_c.
        return -val, -grad_c.flatten()

    def local_optimize_smooth(c_init, beta, maxiter, time_budget):
        """Run L-BFGS-B with smooth objective."""
        if remaining() < time_budget:
            return c_init
        
        bounds = [(-0.499999, 0.499999)] * (N * 3)
        
        def obj_flat(x):
            val, _ = smooth_objective_and_grad(x, beta)
            return val
            
        def grad_flat(x):
            _, grad = smooth_objective_and_grad(x, beta)
            return grad
            
        try:
            res = opt.minimize(
                obj_flat, c_init.flatten(), jac=grad_flat, method='L-BFGS-B',
                bounds=bounds, options={'maxiter': maxiter, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            return res.x.reshape(N, 3)
        except Exception:
            return c_init

    def local_optimize_direct(c_init, maxiter, time_budget):
        """Run L-BFGS-B on the exact non-smooth objective."""
        if remaining() < time_budget:
            return c_init
            
        bounds = [(-0.499999, 0.499999)] * (N * 3)
        
        def obj_flat(x):
            return -compute_radius(x.reshape(N, 3))
            
        try:
            res = opt.minimize(
                obj_flat, c_init.flatten(), method='L-BFGS-B',
                bounds=bounds, options={'maxiter': maxiter, 'ftol': 1e-12}
            )
            return res.x.reshape(N, 3)
        except Exception:
            return c_init

    def make_initial_grid():
        """4x4x4 grid, take first 50."""
        m = 4
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
        return grid[:N].copy()

    def make_bcc_lattice():
        """Body-centered cubic inspired."""
        positions = []
        a = 0.5
        spacing = (2*a) / 3.0
        for iz in range(3):
            for iy in range(3):
                for ix in range(3):
                    x = -a + (ix + 0.5) * spacing
                    y = -a + (iy + 0.5) * spacing
                    z = -a + (iz + 0.5) * spacing
                    positions.append([x, y, z])
                    # Add BCC center
                    if len(positions) < N:
                        positions.append([x + spacing/2, y + spacing/2, z + spacing/2])
        return np.array(positions[:N])

    def make_random(seed):
        np.random.seed(seed)
        return np.random.uniform(-0.45, 0.45, size=(N, 3))

    best_positions = None
    best_radius = -1.0
    
    initializations = [
        ("grid", make_initial_grid),
        ("bcc", make_bcc_lattice),
        ("rand0", lambda: make_random(0)),
        ("rand1", lambda: make_random(42)),
        ("rand2", lambda: make_random(123)),
        ("rand3", lambda: make_random(777)),
    ]
    
    # Phase 1: Global Optimization with Smooth Surrogate (Annealed Beta)
    for name, init_func in initializations:
        if remaining() < 20:
            break
            
        try:
            c = init_func()
            c = np.clip(c, -0.499999, 0.499999)
            
            # Anneal beta: start low (smooth) to high (sharp)
            betas = [20.0, 50.0, 100.0, 200.0]
            
            for beta in betas:
                if remaining() < 10:
                    break
                c = local_optimize_smooth(c, beta, maxiter=200, time_budget=5)
            
            # Phase 2: Direct Polish
            if remaining() > 5:
                c = local_optimize_direct(c, maxiter=100, time_budget=5)
            
            r = compute_radius(c)
            if r > best_radius:
                best_radius = r
                best_positions = c.copy()
                
        except Exception:
            continue

    # Phase 3: Perturbations of Best Solution
    if best_positions is not None and remaining() > 20:
        for perturb_seed in range(3):
            if remaining() < 10:
                break
            np.random.seed(1000 + perturb_seed)
            c_pert = best_positions + np.random.normal(0, 0.01, size=best_positions.shape)
            c_pert = np.clip(c_pert, -0.499999, 0.499999)
            
            # Refine perturbed solution
            for beta in [50.0, 100.0]:
                if remaining() < 5:
                    break
                c_pert = local_optimize_smooth(c_pert, beta, maxiter=100, time_budget=3)
                
            if remaining() > 5:
                c_pert = local_optimize_direct(c_pert, maxiter=50, time_budget=3)
                
            r = compute_radius(c_pert)
            if r > best_radius:
                best_radius = r
                best_positions = c_pert.copy()

    # Final Polish
    if best_positions is not None and remaining() > 5:
        best_positions = local_optimize_direct(best_positions, maxiter=200, time_budget=5)
        best_positions = np.clip(best_positions, -0.499999, 0.499999)

    if best_positions is None:
        best_positions = make_initial_grid()
        
    return best_positions

# EVOLVE-BLOCK-END
