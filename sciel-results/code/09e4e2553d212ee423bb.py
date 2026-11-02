import numpy as np
from scipy.optimize import minimize

N = 50

def construct():
    # Objective: Maximize min(clearances)
    # Clearances are:
    # 1. Pairwise distances / 2
    # 2. Distance to walls (0.5 - max(|x|, |y|, |z|))
    
    # We will minimize -soft_min(clearances, beta)
    # soft_min(x, beta) = -1/beta * log(sum(exp(-beta * x)))
    
    def objective_and_grad(params, beta, gamma_wall=100.0):
        centers = params.reshape(N, 3)
        
        # 1. Pairwise clearances
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        iu = np.triu_indices(N, k=1)
        pair_clears = dists[iu] / 2.0
        
        # 2. Wall clearances using soft-max for differentiability
        abs_c = np.abs(centers)
        max_abs = np.max(abs_c, axis=1, keepdims=True)
        exp_terms_wall = np.exp(gamma_wall * (abs_c - max_abs))
        sum_exp_wall = np.sum(exp_terms_wall, axis=1)
        soft_max_wall = max_abs.flatten() + (1.0 / gamma_wall) * np.log(sum_exp_wall)
        wall_clears = 0.5 - soft_max_wall
        
        # Combine clearances
        n_pairs = len(pair_clears)
        clears = np.concatenate([pair_clears, wall_clears])
        
        # Soft-min of all clearances
        m = np.max(clears)
        exp_terms_soft = np.exp(-beta * (clears - m))
        sum_exp_soft = np.sum(exp_terms_soft)
        
        # Objective: -soft_min
        soft_min_val = -m - (1.0 / beta) * np.log(sum_exp_soft)
        val = -soft_min_val
        
        # Gradient
        probs = exp_terms_soft / sum_exp_soft
        
        grad_centers = np.zeros_like(centers)
        
        # Pairwise part
        for idx in range(n_pairs):
            w = probs[idx]
            if w < 1e-12: continue
            i, j = iu[0][idx], iu[1][idx]
            diff_ij = centers[i] - centers[j]
            dist_ij = np.linalg.norm(diff_ij)
            if dist_ij < 1e-8: continue
            # d(pair_clear)/dx_i = (x_i - x_j) / (2 * dist_ij)
            grad_vec = diff_ij / (2.0 * dist_ij)
            grad_centers[i] += w * grad_vec
            grad_centers[j] -= w * grad_vec
            
        # Wall part
        # d(wall_clear)/dx_i = - d(soft_max_wall)/dx_i
        # d(soft_max_wall)/dx_i,k = sign(x_i,k) * exp(gamma_wall * (|x_i,k| - max_abs_i)) / sum_exp_wall_i
        for i in range(N):
            w = probs[n_pairs + i]
            if w < 1e-12: continue
            signs = np.sign(centers[i])
            # probs_wall_i_k = exp_terms_wall[i, k] / sum_exp_wall[i]
            grad_wall_vec = signs * (exp_terms_wall[i] / sum_exp_wall[i])
            grad_centers[i] -= w * grad_wall_vec
            
        return val, grad_centers.flatten()

    def optimize(centers_init, max_iters=1000):
        # Annealing schedule
        betas = [10, 50, 100, 200, 500]
        current_centers = centers_init.copy()
        
        for beta in betas:
            res = minimize(
                lambda p: objective_and_grad(p, beta)[0],
                current_centers.flatten(),
                jac=lambda p: objective_and_grad(p, beta)[1],
                method='L-BFGS-B',
                bounds=[(-0.5, 0.5)] * (N * 3),
                options={'maxiter': max_iters // len(betas), 'gtol': 1e-8}
            )
            current_centers = res.x.reshape(N, 3)
            
        return current_centers

    def score(centers):
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dists, np.inf)
        min_pair = np.min(dists) / 2.0
        
        max_c = np.max(np.abs(centers), axis=1)
        min_wall = 0.5 - np.min(max_c)
        
        return min(min_pair, min_wall)

    # Initializations
    best_centers = None
    best_score = -np.inf
    
    for seed in range(10):
        np.random.seed(seed)
        # Random start
        c_init = np.random.uniform(-0.45, 0.45, (N, 3))
        
        # Or FCC-like start
        if seed % 2 == 0:
             # Perturb FCC
             # Generate FCC in unit cube [0,1]
             pts = []
             for i in range(3):
                 for j in range(3):
                     for k in range(3):
                         pts.append([i/3, j/3, k/3])
                         pts.append([(i+0.5)/3, (j+0.5)/3, k/3])
                         pts.append([(i+0.5)/3, j/3, (k+0.5)/3])
                         pts.append([i/3, (j+0.5)/3, (k+0.5)/3])
             pts = np.array(pts)
             # Center at 0
             pts = pts - 0.5
             # Scale to fit?
             # Just add noise
             c_init = pts[:N] + np.random.normal(0, 0.05, (N, 3))
             c_init = np.clip(c_init, -0.45, 0.45)
        
        c_opt = optimize(c_init, max_iters=200)
        s = score(c_opt)
        
        if s > best_score:
            best_score = s
            best_centers = c_opt

    return best_centers
