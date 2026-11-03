"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
import scipy.optimize as opt
import scipy.special as sp
import random

def construct():
    start_time = time.time()
    # Leave 5s buffer for 150s limit
    time_limit = 145.0 

    def get_radius(c):
        # c is (N, 3)
        # Pairwise distances
        diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        d = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(d, np.inf)
        min_pair_dist = np.min(d)
        
        # Wall distances
        # Distance to each wall: 0.5 - x, x + 0.5, 0.5 - y, y + 0.5, 0.5 - z, z + 0.5
        # min_i min_k (0.5 - |c_ik|) is equivalent to min(0.5 - max|c_ik|)
        # Actually, min distance to boundary is min(0.5 - c, c + 0.5) for each coordinate
        dist_to_walls = np.min(np.minimum(c + 0.5, 0.5 - c), axis=1)
        min_wall_dist = np.min(dist_to_walls)
        
        return min(min_pair_dist / 2.0, min_wall_dist)

    def objective_and_grad(c_flat, beta):
        # c_flat is (3N,)
        c = c_flat.reshape(N, 3)
        
        # Calculate distances
        # Pairwise
        diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        d = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(d, np.inf)
        
        # Wall distances
        # dist_i = min(0.5 - c_ik, c_ik + 0.5) for k=0,1,2
        # This is min over 6 values per point.
        # Let's compute the 6 wall distances explicitly for gradient
        # Walls: x=0.5, x=-0.5, y=0.5, y=-0.5, z=0.5, z=-0.5
        # d_x_pos = 0.5 - c[:,0]
        # d_x_neg = c[:,0] + 0.5
        # etc.
        
        d_x_pos = 0.5 - c[:, 0]
        d_x_neg = c[:, 0] + 0.5
        d_y_pos = 0.5 - c[:, 1]
        d_y_neg = c[:, 1] + 0.5
        d_z_pos = 0.5 - c[:, 2]
        d_z_neg = c[:, 2] + 0.5
        
        # Soft min of pairwise distances (divided by 2 later? No, objective is min(dist/2, wall_dist))
        # Let's define g_k as the actual radius contributions.
        # g_pair_ij = d_ij / 2
        # g_wall_i = dist_to_wall_i
        
        # We want to maximize min(g_k).
        # Equivalent to minimizing -min(g_k).
        # Soft min: -1/beta * log(sum(exp(-beta * g_k)))
        
        # Vectorize g_k
        # Pairwise:
        # We need upper triangle to avoid double counting? 
        # The min is the same whether we include all or just half. 
        # Including all doubles the sum in logsumexp, which adds log(2)/beta to the result. 
        # This is a constant shift for fixed N, so it doesn't affect the argmin, 
        # but it might affect the gradient magnitude slightly? 
        # Actually, logsumexp(A) vs logsumexp(2A) is different.
        # Let's stick to unique pairs to be standard.
        
        tri_idx = np.triu_indices(N, k=1)
        d_pairs = d[tri_idx] / 2.0
        
        # Wall distances:
        # For each point i, the wall distance is the min of 6 values.
        # The overall min wall distance is min over all i and all 6 walls.
        # So we have N * 6 wall constraints.
        wall_dists = np.concatenate([
            d_x_pos, d_x_neg, d_y_pos, d_y_neg, d_z_pos, d_z_neg
        ])
        
        # Combine all g_k
        # Note: We want to maximize the minimum.
        # If any g_k is negative, the radius is negative (invalid/overlap).
        # The soft min will try to increase the smallest ones.
        
        g_all = np.concatenate([d_pairs, wall_dists])
        
        # Soft min
        # f = -1/beta * log(sum(exp(-beta * g)))
        # For numerical stability, use logsumexp
        # logsumexp(-beta * g)
        log_sum_exp = sp.logsumexp(-beta * g_all)
        val = -log_sum_exp / beta
        
        # Gradient
        # d/dg_j (-1/beta * log(sum_k exp(-beta g_k)))
        # = -1/beta * (1/sum exp) * sum_k (-beta exp(-beta g_k) * delta_kj)
        # = exp(-beta g_j) / sum_k exp(-beta g_k)
        # = softmax(-beta * g)_j
        
        # We need to map this back to c.
        # grad_c = sum_j (softmax_j * dg_j/dc)
        
        # Compute softmax weights
        # weights = exp(-beta * g) / sum(exp(-beta * g))
        # To avoid underflow, use the stable version:
        # max_g = np.max(g_all)
        # weights = exp(-beta * (g_all - max_g)) / sum(exp(-beta * (g_all - max_g)))
        # But logsumexp already handles this internally.
        # Let's compute the weights manually for gradient.
        
        # p_j = exp(-beta g_j) / S
        # log p_j = -beta g_j - log S
        # p_j = exp(-beta g_j - log_sum_exp)
        
        weights = np.exp(-beta * g_all - log_sum_exp)
        
        # grad_obj = -1/beta * d(log_sum_exp)/dc
        # d(log_sum_exp)/dc = sum_j (-beta * weights_j) * dg_j/dc
        # So grad_obj = sum_j weights_j * dg_j/dc
        
        grad_c = np.zeros_like(c)
        
        # Pairwise part
        # d(d_ij/2)/dc_i = (c_i - c_j) / d_ij
        # d(d_ij/2)/dc_j = (c_j - c_i) / d_ij
        
        # We need to accumulate gradients for each pair
        # Loop is slow, let's vectorize if possible, but N=50 is small.
        # N_pairs = 1225. Looping 1225 times in Python is ~1ms. Acceptable.
        
        for k in range(len(d_pairs)):
            i, j = tri_idx[0][k], tri_idx[1][k]
            w = weights[k]
            if w < 1e-12: continue
            
            diff_ij = c[i] - c[j]
            d_ij = np.linalg.norm(diff_ij)
            if d_ij > 1e-12:
                # grad wrt c_i: w * (c_i - c_j) / d_ij
                # Note: d_pair = d_ij / 2, so d(d_pair)/dc_i = (c_i - c_j) / (2 * d_ij) * 2? 
                # No. d_pair = 0.5 * ||c_i - c_j||.
                # d/dc_i (0.5 * ||c_i - c_j||) = 0.5 * (c_i - c_j) / ||c_i - c_j||
                # So factor is 0.5.
                grad_c[i] += w * 0.5 * diff_ij / d_ij
                grad_c[j] -= w * 0.5 * diff_ij / d_ij
        
        # Wall part
        # Indices in g_all:
        # 0 to N-1: d_x_pos (0.5 - c[:,0])
        # N to 2N-1: d_x_neg (c[:,0] + 0.5)
        # 2N to 3N-1: d_y_pos
        # 3N to 4N-1: d_y_neg
        # 4N to 5N-1: d_z_pos
        # 5N to 6N-1: d_z_neg
        
        # Derivatives:
        # d(d_x_pos)/dc[:,0] = -1
        # d(d_x_neg)/dc[:,0] = 1
        # d(d_y_pos)/dc[:,1] = -1
        # d(d_y_neg)/dc[:,1] = 1
        # d(d_z_pos)/dc[:,2] = -1
        # d(d_z_neg)/dc[:,2] = 1
        
        # Weights for walls start at index len(d_pairs)
        w_start = len(d_pairs)
        
        # d_x_pos weights: w_start to w_start+N-1
        grad_c[:, 0] -= weights[w_start : w_start+N]
        # d_x_neg weights: w_start+N to w_start+2N-1
        grad_c[:, 0] += weights[w_start+N : w_start+2*N]
        
        # d_y_pos weights: w_start+2N to w_start+3N-1
        grad_c[:, 1] -= weights[w_start+2*N : w_start+3*N]
        # d_y_neg weights: w_start+3N to w_start+4N-1
        grad_c[:, 1] += weights[w_start+3*N : w_start+4*N]
        
        # d_z_pos weights: w_start+4N to w_start+5N-1
        grad_c[:, 2] -= weights[w_start+4*N : w_start+5*N]
        # d_z_neg weights: w_start+5N to w_start+6N-1
        grad_c[:, 2] += weights[w_start+5*N : w_start+6*N]
        
        # The objective we want to MINIMIZE is -soft_min.
        # So we return -val and -grad_c.
        return -val, -grad_c.flatten()

    def run_l_bfgs(c_init, beta_start, beta_end, max_iter, time_budget):
        # Anneal beta
        n_restarts = 3
        betas = np.linspace(beta_start, beta_end, n_restarts)
        
        best_c = c_init.copy()
        best_r = get_radius(c_init)
        
        for beta in betas:
            # Run L-BFGS-B
            # We need to check time
            if time.time() - start_time > time_budget:
                break
                
            # Flatten
            c_flat = c_init.flatten()
            
            # Bounds
            bounds = [(-0.5, 0.5)] * (3 * N)
            
            # Callback to check time? L-BFGS-B doesn't support time-based stopping easily in all versions.
            # We'll just run it and hope it finishes. If it's too slow, we reduce max_iter.
            # For N=50, L-BFGS-B is quite fast.
            
            try:
                res = opt.minimize(
                    lambda x: objective_and_grad(x, beta)[0],
                    c_flat,
                    jac=lambda x: objective_and_grad(x, beta)[1],
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-8, 'maxfun': 10000}
                )
                
                c_new = res.x.reshape(N, 3)
                r_new = get_radius(c_new)
                
                if r_new > best_r:
                    best_r = r_new
                    best_c = c_new.copy()
                
                c_init = c_new.copy()
            except Exception as e:
                print(f"Optimization error: {e}")
                continue
                
        return best_c, best_r

    # 1. Generate Initial Candidates
    
    # Candidate 1: FCC Lattice
    # FCC unit cell has 4 atoms. 
    # We need 50 atoms. 
    # Approximate lattice constant for 50 spheres in unit cube?
    # Volume per sphere = 1/50. 
    # Sphere radius r. Volume = 4/3 pi r^3.
    # Packing fraction ~ 0.74.
    # 50 * 4/3 pi r^3 * 0.74 = 1 => r^3 = 1 / (50 * 4.18 * 0.74) = 1/154.7 => r ~ 0.188.
    # Wait, the known best is 0.1359. Why?
    # Because they are in a CUBE. Boundary effects are huge.
    # FCC lattice constant a. 4 spheres per cell.
    # Distance between centers in FCC is a/sqrt(2).
    # If we fit a 3x3x3 block of FCC cells? 3^3 * 4 = 108 spheres. Too many.
    # 2x2x2 block? 2^3 * 4 = 32 spheres.
    # 3x2x2? 48 spheres. Close to 50.
    # Let's try to generate an FCC-like grid and perturb.
    
    # Simple Grid Init (Fallback/Backup)
    m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    c_grid = grid[:N].copy()
    
    # Random Init
    np.random.seed(42)
    c_rand = np.random.uniform(-0.45, 0.45, (N, 3))
    
    # FCC-like Init
    # Create a 3x3x3 grid of FCC units? No, let's just use a dense random with repulsion.
    # Or better, take the grid and perturb it randomly.
    
    candidates = [
        c_grid,
        c_rand,
        c_grid + np.random.uniform(-0.05, 0.05, (N, 3)),
        c_rand + np.random.uniform(-0.05, 0.05, (N, 3))
    ]
    
    # Filter candidates to ensure they are inside the cube
    valid_candidates = []
    for c in candidates:
        if np.all(c >= -0.5) and np.all(c <= 0.5):
            valid_candidates.append(c)
    
    if not valid_candidates:
        valid_candidates = [c_grid]

    best_overall_c = valid_candidates[0].copy()
    best_overall_r = get_radius(valid_candidates[0])
    
    # 2. Optimization Loop
    # We have ~135s left.
    # Let's try to run L-BFGS on a few candidates.
    # Beta schedule: Start low (10), end high (1000).
    
    # Time per candidate?
    # L-BFGS with 1000 iterations on 150 vars is fast.
    # Let's allocate 30s per candidate.
    
    for i, c_init in enumerate(valid_candidates):
        if time.time() - start_time > time_limit - 10:
            break
            
        # Run optimization
        c_opt, r_opt = run_l_bfgs(c_init, beta_start=10.0, beta_end=500.0, max_iter=500, time_budget=time_limit - 10)
        
        if r_opt > best_overall_r:
            best_overall_r = r_opt
            best_overall_c = c_opt.copy()
            
        # Optional: Perturb the result and run again if time permits
        if time.time() - start_time < time_limit - 30 and i < 2:
            # Perturb
            c_pert = c_opt + np.random.normal(0, 0.01, (N, 3))
            c_pert = np.clip(c_pert, -0.5, 0.5)
            c_opt2, r_opt2 = run_l_bfgs(c_pert, beta_start=50.0, beta_end=1000.0, max_iter=300, time_budget=time_limit - 10)
            if r_opt2 > best_overall_r:
                best_overall_r = r_opt2
                best_overall_c = c_opt2.copy()

    # 3. Final Polish
    # If we have time left, do a high-beta run on the best candidate
    if time.time() - start_time < time_limit - 5:
        c_polish = best_overall_c + np.random.normal(0, 0.001, (N, 3))
        c_polish = np.clip(c_polish, -0.5, 0.5)
        # Use very high beta
        c_final, r_final = run_l_bfgs(c_polish, beta_start=1000.0, beta_end=5000.0, max_iter=200, time_budget=time_limit - 2)
        if r_final > best_overall_r:
            best_overall_r = r_final
            best_overall_c = c_final.copy()

    return best_overall_c

# EVOLVE-BLOCK-END
