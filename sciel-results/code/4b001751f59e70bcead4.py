import numpy as np
from scipy.optimize import minimize

N = 50

def construct():
    best_score = -np.inf
    best_centers = None
    
    # Define the smooth objective and gradient for soft-min of distances
    def soft_min_obj_and_grad(params, beta, gamma_wall=10.0):
        centers = params.reshape(N, 3)
        
        # Pairwise distances
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        iu = np.triu_indices(N, k=1)
        d_pairs = dists[iu]
        
        # Wall distances (smoothed)
        abs_centers = np.abs(centers)
        max_abs = np.max(abs_centers, axis=1, keepdims=True)
        exp_terms = np.exp(gamma_wall * (abs_centers - max_abs))
        sum_exp = np.sum(exp_terms, axis=1)
        soft_max_val = max_abs.flatten() + (1.0 / gamma_wall) * np.log(sum_exp.flatten())
        w = 0.5 - soft_max_val
        
        # Combine all distances
        exp_pairs = np.exp(-beta * d_pairs)
        exp_walls = np.exp(-beta * w)
        
        S = np.sum(exp_pairs) + np.sum(exp_walls)
        
        # Objective
        val = -1.0 / beta * np.log(S)
        
        # Gradient with respect to d
        grad_d_pairs = exp_pairs / S
        grad_d_walls = exp_walls / S
        
        # Gradient with respect to params
        grad_params = np.zeros_like(centers)
        
        # Pairwise part
        for idx in range(len(iu[0])):
            i, j = iu[0][idx], iu[1][idx]
            d = d_pairs[idx]
            if d < 1e-6:
                continue
            weight = grad_d_pairs[idx]
            diff_ij = centers[i] - centers[j]
            grad_params[i] += weight * (diff_ij / d)
            grad_params[j] -= weight * (diff_ij / d)
            
        # Wall part
        for i in range(N):
            weight = grad_d_walls[i]
            if weight == 0: continue
            probs = exp_terms[i] / sum_exp[i]
            signs = np.sign(centers[i])
            grad_params[i] -= weight * signs * probs
            
        return val, grad_params.flatten()

    def optimize_start(centers_init, verbose=False):
        centers = centers_init.copy()
        # Annealing schedule
        betas = [5.0, 15.0, 40.0, 100.0]
        
        for beta in betas:
            res = minimize(
                lambda p: soft_min_obj_and_grad(p, beta)[0],
                centers.flatten(),
                jac=lambda p: soft_min_obj_and_grad(p, beta)[1],
                method='BFGS',
                options={'maxiter': 50, 'gtol': 1e-5}
            )
            centers = res.x.reshape(N, 3)
            if verbose:
                print(f"Beta {beta}: Val {res.fun}")
                
        return centers

    # Start 1: Random
    np.random.seed(42)
    c1 = np.random.uniform(-0.4, 0.4, (N, 3))
    
    best_c = c1
    best_r = -1
    
    for k in range(5):
        np.random.seed(100 + k)
        c_init = np.random.uniform(-0.45, 0.45, (N, 3))
        c_opt = optimize_start(c_init)
        
        # Calculate true min radius
        diff = c_opt[:, np.newaxis, :] - c_opt[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dists, np.inf)
        min_pair = np.min(dists) / 2.0
        
        max_c = np.max(np.abs(c_opt), axis=1)
        min_wall = 0.5 - np.min(max_c)
        
        r = min(min_pair, min_wall)
        
        if r > best_r:
            best_r = r
            best_c = c_opt
            
    return best_c
