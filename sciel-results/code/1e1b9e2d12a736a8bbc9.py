import numpy as np
from scipy.optimize import minimize
import time

N = 50

# EVOLVE-BLOCK-START
def soft_min_objective(x, beta):
    """
    x: shape (N, 2)
    Returns (-soft_min, -gradient) to be minimized.
    """
    x = x.reshape(N, 2)
    
    # Boundary distances
    dists_bound = 1.0 - np.linalg.norm(x, axis=1)
    
    # Pairwise distances
    diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
    dists_pair = np.linalg.norm(diff, axis=2) / 2.0
    mask = np.triu(np.ones((N, N), dtype=bool), k=1)
    dists_pair = dists_pair[mask]
    
    all_dists = np.concatenate([dists_pair, dists_bound])
    
    # Soft min M = -1/beta * logsumexp(-beta * all_dists)
    w = -beta * all_dists
    w_max = np.max(w)
    if w_max > 700:
        # Huge penalty for invalid configs
        return 1e10, np.zeros_like(x)
        
    log_sum_exp = w_max + np.log(np.sum(np.exp(w - w_max)))
    M_val = -1.0 / beta * log_sum_exp
    
    # We want to maximize M, so minimize -M
    obj_val = -M_val
    
    # Gradient of M
    exp_w = np.exp(w - w_max)
    sum_exp_w = np.sum(exp_w)
    weights = exp_w / sum_exp_w
    
    grad_M = np.zeros_like(x)
    
    # Boundary
    w_bound = weights[len(dists_pair):]
    for i in range(N):
        norm_i = np.linalg.norm(x[i])
        if norm_i > 1e-8:
            grad_M[i] += w_bound[i] * (-x[i] / norm_i)
            
    # Pairwise
    idx_pairs = np.where(mask)
    w_pair = weights[:len(dists_pair)]
    for k in range(len(dists_pair)):
        i, j = idx_pairs[0][k], idx_pairs[1][k]
        diff_ij = x[i] - x[j]
        norm_diff = np.linalg.norm(diff_ij)
        if norm_diff > 1e-8:
            grad_ij = diff_ij / (2.0 * norm_diff)
            grad_M[i] += w_pair[k] * grad_ij
            grad_M[j] += w_pair[k] * (-grad_ij)
            
    # Return negative for minimization
    return obj_val, -grad_M

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    best_x = None
    best_score = -np.inf
    
    beta = 30.0
    
    inits = []
    
    # 1. Hexagonal lattice
    spacing = 2.0 / (np.sqrt(N) + 1.0)
    k = int(1.0 / spacing) + 2
    rows = [((i + 0.5 * (j % 2)) * spacing, j * spacing * np.sqrt(3.0) / 2.0)
            for i in range(-k, k + 1) for j in range(-k, k + 1)]
    points = np.array(rows)
    radius = np.hypot(points[:, 0], points[:, 1])
    points, radius = points[radius <= 1.0 - spacing / 2.0], radius[radius <= 1.0 - spacing / 2.0]
    if len(points) >= N:
        p0 = points[:N] + np.random.normal(0, 0.01, (N, 2))
        norms = np.linalg.norm(p0, axis=1)
        mask = norms > 0.99
        p0[mask] = p0[mask] * (0.99 / norms[mask])
        inits.append(p0)
        
    # 2. Random inits
    np.random.seed(42)
    for _ in range(20):
        angles = np.random.uniform(0, 2*np.pi, N)
        r = np.sqrt(np.random.uniform(0, 1, N))
        x = r * np.cos(angles)
        y = r * np.sin(angles)
        p_rand = np.column_stack((x, y))
        inits.append(p_rand)
        
    # 3. Ring init
    angles = np.linspace(0, 2*np.pi, N, endpoint=False)
    r = 0.8
    x = r * np.cos(angles)
    y = r * np.sin(angles)
    p_ring = np.column_stack((x, y)) + np.random.normal(0, 0.05, (N, 2))
    norms = np.linalg.norm(p_ring, axis=1)
    mask = norms > 0.99
    p_ring[mask] = p_ring[mask] * (0.99 / norms[mask])
    inits.append(p_ring)

    for i_init, x0 in enumerate(inits):
        if time.time() > deadline:
            break
            
        try:
            res = minimize(
                soft_min_objective, 
                x0, 
                args=(beta,), 
                method='L-BFGS-B', 
                jac=True,
                options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 50}
            )
            x_opt = res.x.reshape(N, 2)
            
            # Calculate actual score
            dists_bound = 1.0 - np.linalg.norm(x_opt, axis=1)
            diff = x_opt[:, np.newaxis, :] - x_opt[np.newaxis, :, :]
            dists_pair = np.linalg.norm(diff, axis=2) / 2.0
            mask = np.triu(np.ones((N, N), dtype=bool), k=1)
            min_pair = np.min(dists_pair[mask])
            min_bound = np.min(dists_bound)
            score = min(min_pair, min_bound)
            
            if min_bound < -1e-8 or min_pair < -1e-8:
                continue
            if not np.all(np.isfinite(x_opt)):
                continue
                
            if score > best_score:
                best_score = score
                best_x = x_opt
                
        except Exception:
            pass

    if best_x is None:
        return inits[0]
        
    return best_x

# EVOLVE-BLOCK-END
