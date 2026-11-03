import numpy as np
import time
from scipy.optimize import minimize

N = 50

def compute_radius(centers):
    # centers: (N, 3)
    # Check bounds
    if np.any(centers < -0.5) or np.any(centers > 0.5):
        return -1.0
    # Pairwise distances
    dists = []
    for i in range(N):
        for j in range(i+1, N):
            dists.append(np.linalg.norm(centers[i] - centers[j]))
    # Wall distances
    for i in range(N):
        for k in range(3):
            dists.append(0.5 - abs(centers[i][k]))
    
    if not dists:
        return -1.0
    min_dist = min(dists)
    return min_dist / 2.0

def softmin(d, alpha=20.0):
    # d: array of distances
    # Handle very small distances to avoid overflow/underflow?
    # exp(-alpha * d)
    # If d is large, exp is small.
    # If d is 0, exp is 1.
    # sum exp
    # log sum
    # -1/alpha * log sum
    # We want to MINIMIZE -softmin, so MAXIMIZE softmin.
    # softmin approximates min(d).
    # As alpha -> inf, softmin -> min(d).
    # We want to maximize min(d).
    # So we minimize -softmin(d).
    
    # To avoid numerical issues, shift d?
    # min(d) is what we care about.
    # Let's just compute it.
    # d might have zeros.
    # exp(-alpha * 0) = 1.
    # If d is large, e.g. 1, alpha=20, exp(-20) ~ 2e-9.
    # Sum is dominated by smallest d.
    
    # Use log-sum-exp trick for stability?
    # log(sum(exp(-alpha * d)))
    # = -alpha * min(d) + log(sum(exp(-alpha * (d - min(d)))))
    # softmin = -1/alpha * [ -alpha * min(d) + log(sum(exp(-alpha * (d - min(d))))) ]
    # = min(d) - 1/alpha * log(sum(exp(-alpha * (d - min(d)))))
    
    min_d = np.min(d)
    # d - min_d >= 0
    # exp(-alpha * (d - min_d)) <= 1
    # Sum is between 1 and len(d)
    # Log is between 0 and log(len(d))
    # softmin is between min_d - log(len(d))/alpha and min_d
    # This is a good approximation.
    
    shifted = d - min_d
    exp_terms = np.exp(-alpha * shifted)
    log_sum = np.log(np.sum(exp_terms))
    return min_d - (1.0/alpha) * log_sum

def objective(x, alpha=20.0):
    centers = x.reshape(N, 3)
    # Compute distances
    d_list = []
    for i in range(N):
        for j in range(i+1, N):
            d_list.append(np.linalg.norm(centers[i] - centers[j]))
    for i in range(N):
        for k in range(3):
            d_list.append(0.5 - abs(centers[i][k]))
    
    d = np.array(d_list)
    sm = softmin(d, alpha)
    return -sm

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    # Initial guess
    m = 4
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    x0 = grid[:N].flatten()
    
    bounds = [(-0.5, 0.5)] * (3 * N)
    
    best_x = x0
    best_r = compute_radius(x0.reshape(N, 3))
    
    # Try a few different alphas or restarts?
    # Let's try one run with alpha=20.
    
    try:
        res = minimize(objective, x0, args=(20.0,), method='L-BFGS-B', bounds=bounds, options={'maxiter': 500, 'maxfun': 5000})
        if time.time() < deadline:
            x_opt = res.x
            # Clip to bounds just in case
            x_opt = np.clip(x_opt, -0.5, 0.5)
            centers = x_opt.reshape(N, 3)
            r_opt = compute_radius(centers)
            if r_opt > best_r:
                best_r = r_opt
                best_x = x_opt
    except Exception:
        pass
        
    # If time allows, try another run with different alpha or perturbation
    if time.time() < deadline - 20.0:
        # Perturb initial guess
        x0_pert = x0 + np.random.normal(0, 0.01, size=x0.shape)
        x0_pert = np.clip(x0_pert, -0.49, 0.49)
        try:
            res2 = minimize(objective, x0_pert, args=(50.0,), method='L-BFGS-B', bounds=bounds, options={'maxiter': 500, 'maxfun': 5000})
            if time.time() < deadline:
                x_opt2 = res2.x
                x_opt2 = np.clip(x_opt2, -0.5, 0.5)
                centers2 = x_opt2.reshape(N, 3)
                r_opt2 = compute_radius(centers2)
                if r_opt2 > best_r:
                    best_r = r_opt2
                    best_x = x_opt2
        except Exception:
            pass
            
    return best_x.reshape(N, 3)
