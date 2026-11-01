import numpy as np
import time
from scipy.optimize import minimize

N = 300

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    time_limit = 145.0  # Safety margin
    
    # 1. Initial Hexagonal Lattice
    spacing = 0.06
    while True:
        k = int(1.0 / spacing) + 2
        xs = []
        ys = []
        for j in range(-k, k + 1):
            for i in range(-k, k + 1):
                x = (i + 0.5 * (j % 2)) * spacing
                y = j * spacing * np.sqrt(3.0) / 2.0
                if np.hypot(x, y) <= 1.0:
                    xs.append(x)
                    ys.append(y)
        pts = np.array(list(zip(xs, ys)))
        if len(pts) >= N:
            radii = np.hypot(pts[:, 0], pts[:, 1])
            idx = np.argsort(radii)
            x0 = pts[idx[:N]].astype(np.float64).flatten()
            break
        spacing *= 0.95
        
    def compute_min_dist(x):
        X = x.reshape(N, 2)
        diff = X[:, np.newaxis, :] - X[np.newaxis, :, :]
        dist_sq = np.sum(diff**2, axis=2)
        np.fill_diagonal(dist_sq, np.inf)
        dists = np.sqrt(dist_sq)
        iu = np.triu_indices(N, 1)
        pair_dists = dists[iu]
        
        radii = np.hypot(X[:, 0], X[:, 1])
        bound_dists = 1.0 - radii
        
        all_dists = np.concatenate([pair_dists, bound_dists])
        return np.min(all_dists)

    best_x = x0
    best_score = compute_min_dist(x0)
    
    def objective(x):
        X = x.reshape(N, 2)
        diff = X[:, np.newaxis, :] - X[np.newaxis, :, :]
        dist_sq = np.sum(diff**2, axis=2)
        np.fill_diagonal(dist_sq, np.inf)
        dists = np.sqrt(dist_sq)
        iu = np.triu_indices(N, 1)
        pair_dists = dists[iu]
        
        radii = np.hypot(X[:, 0], X[:, 1])
        bound_dists = 1.0 - radii
        
        all_dists = np.concatenate([pair_dists, bound_dists])
        
        beta = current_beta[0]
        m = np.min(all_dists)
        # Shift to prevent underflow
        shifted = all_dists - m
        # If beta is large, exp(-beta * shifted) for shifted > 0 is 0.
        # We sum exp(-beta * shifted).
        # The term with shifted=0 (the min) is 1.
        # So sum >= 1.
        # log(sum) >= 0.
        # val = -m + log(sum)/beta.
        
        # Optimization: if beta is very large, we can just return -m
        # because the other terms vanish.
        # But for gradient, we need the soft part.
        
        # Clip shifted to avoid overflow in exp if beta is small?
        # exp(-beta * shifted). If shifted is large, exp is 0.
        # If shifted is negative? No, m is min, so shifted >= 0.
        
        exp_terms = np.exp(-beta * shifted)
        s = np.sum(exp_terms)
        
        # If s is 0 (shouldn't happen since min term is 1), handle it
        if s < 1e-300:
             return m
             
        val = -m + np.log(s) / beta
        return val

    current_beta = [10.0]
    bounds = [(-1.0, 1.0)] * (2 * N)
    
    x = x0.copy()
    
    # List of betas to try
    betas = [10, 50, 100, 500, 1000, 2000, 5000]
    
    for beta in betas:
        if time.time() - start_time > time_limit:
            break
            
        current_beta[0] = beta
        
        # Use L-BFGS-B
        # maxiter: number of iterations.
        # Each iteration does a line search (multiple function evals).
        # Let's allow 50 iterations.
        
        try:
            res = minimize(objective, x, method='L-BFGS-B', bounds=bounds, 
                           options={'maxiter': 50, 'ftol': 1e-12, 'gtol': 1e-12})
            
            # Check if result is valid
            if np.all(np.isfinite(res.x)):
                new_score = compute_min_dist(res.x)
                if new_score > best_score:
                    best_score = new_score
                    best_x = res.x.copy()
                    x = res.x.copy()
                else:
                    # If it didn't improve, revert x to best_x for next stage?
                    # Or keep res.x?
                    # Usually, if it didn't improve the hard min, it might have improved the soft min?
                    # But we track hard min.
                    # Let's revert to best_x to avoid drifting into bad local minima.
                    x = best_x.copy()
        except Exception:
            pass
            
        # Check time
        if time.time() - start_time > time_limit:
            break
            
    # Final check
    if np.all(np.isfinite(best_x)):
        return best_x.reshape(N, 2)
    else:
        return x0.reshape(N, 2)

# EVOLVE-BLOCK-END
