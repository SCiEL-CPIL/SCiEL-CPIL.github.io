import numpy as np
from scipy.optimize import minimize
import time

def compute_score_and_grad(points):
    n = len(points)
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    
    mask = ~np.eye(n, dtype=bool)
    d_upper = dists[mask]
    d_upper = np.maximum(d_upper, 1e-15)
    log_sum = np.sum(np.log(d_upper))
    
    D = np.max(dists)
    if D < 1e-15:
        return -1e10, np.zeros_like(points)
    
    score = 2 * log_sum - 56 * np.log(D)
    
    # Gradient
    # grad_score = 2 * sum_{j != k} (p_k - p_j) / d_kj^2 - 56 * grad_log_D
    grad = np.zeros_like(points)
    for k in range(n):
        # Sum over j != k
        # vectorized: (points[k] - points) / dists[k]**2, with dists[k,k]=1 to avoid div0
        d_k = dists[k].copy()
        d_k[k] = 1.0
        # term = (points[k] - points) / d_k**2
        term = (points[k] - points) / (d_k**2)
        term[k] = 0.0 # self term is 0
        grad[k] += 2.0 * np.sum(term, axis=0)
        
    # Gradient of log D
    # D = max d_ij. Let (a,b) be the pair.
    # If unique max, grad log D = 0.5 * (grad d_ab / D)
    # grad d_ab / d_ab = (p_a - p_b)/d_ab^2 for p_a, and (p_b - p_a)/d_ab^2 for p_b
    # So grad log D = (p_a - p_b)/d_ab^2 for p_a, and (p_b - p_a)/d_ab^2 for p_b
    # If multiple max, take average? Or just one?
    # For smoothness, we can use the one with largest value.
    idx = np.unravel_index(np.argmax(dists), dists.shape)
    a, b = idx
    if a != b:
        d_ab = dists[a, b]
        # grad log D wrt p_a: (p_a - p_b) / d_ab^2
        # grad log D wrt p_b: (p_b - p_a) / d_ab^2
        vec = (points[a] - points[b]) / (d_ab**2)
        grad[a] -= 56.0 * vec
        grad[b] -= 56.0 * (-vec)
        
    return score, grad

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    angles = np.linspace(0, 2*np.pi, 8, endpoint=False)
    x0 = np.column_stack([np.cos(angles), np.sin(angles)]).flatten()
    
    def neg_score(x):
        pts = x.reshape(8, 2)
        s, _ = compute_score_and_grad(pts)
        return -s
        
    def neg_grad(x):
        pts = x.reshape(8, 2)
        _, g = compute_score_and_grad(pts)
        return -g.flatten()
        
    best_score = -neg_score(x0)
    best_pts = x0.reshape(8, 2)
    
    # L-BFGS-B
    try:
        res = minimize(neg_score, x0, jac=neg_grad, method='L-BFGS-B', 
                       options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8})
        if -res.fun > best_score:
            best_score = -res.fun
            best_pts = res.x.reshape(8, 2)
    except Exception:
        pass
        
    if time.time() > deadline - 30:
        # Normalize
        d = best_pts[:, np.newaxis, :] - best_pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(d**2, axis=2))
        D = np.max(dists)
        if D > 1e-10:
            best_pts = best_pts * (2.0 / D)
        return best_pts.tolist()

    # Random restarts
    for i in range(30):
        if time.time() > deadline - 20:
            break
        x_pert = x0 + np.random.normal(0, 0.1, 16)
        try:
            res = minimize(neg_score, x_pert, jac=neg_grad, method='L-BFGS-B', 
                           options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-6})
            if -res.fun > best_score:
                best_score = -res.fun
                best_pts = res.x.reshape(8, 2)
        except Exception:
            pass

    # Normalize
    d = best_pts[:, np.newaxis, :] - best_pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(d**2, axis=2))
    D = np.max(dists)
    if D > 1e-10:
        best_pts = best_pts * (2.0 / D)
    return best_pts.tolist()
