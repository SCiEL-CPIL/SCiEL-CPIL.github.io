import numpy as np
import scipy.optimize as so
import time

N = 14
D = 3

def _pairwise_sq_dists(P):
    P2 = np.sum(P**2, axis=1)
    dist_sq = P2[:, None] + P2[None, :] - 2.0 * (P @ P.T)
    np.maximum(dist_sq, 0, out=dist_sq)
    iu = np.triu_indices(N, 1)
    return dist_sq[iu]

def _objective(P, beta=10.0, mu=1000.0):
    centroid = np.mean(P, axis=0)
    Pc = P - centroid
    scale = np.sum(Pc**2)
    scale_penalty = mu * (scale - 1.0)**2
    
    D2 = _pairwise_sq_dists(Pc)
    D2_clipped = np.clip(D2, 1e-12, 1e12)
    
    max_D2 = np.max(D2_clipped)
    log_sum_exp_max = max_D2 + np.log(np.sum(np.exp(beta * (D2_clipped - max_D2))))
    soft_max = log_sum_exp_max / beta
    
    min_D2 = np.min(D2_clipped)
    log_sum_exp_min = min_D2 - np.log(np.sum(np.exp(-beta * (D2_clipped - min_D2))))
    soft_min = log_sum_exp_min / beta
    
    ratio = soft_max / soft_min
    return ratio + scale_penalty

def _optimize_from_start(start_P, max_iter=1000):
    x0 = start_P.flatten()
    def obj_func(x):
        return _objective(x.reshape(N, D))
        
    try:
        res = so.minimize(obj_func, x0, method='Powell', options={'maxiter': max_iter, 'disp': False})
        P_opt = res.x.reshape(N, D)
        D2 = _pairwise_sq_dists(P_opt)
        if np.min(D2) < 1e-10:
            return np.inf, P_opt
        score = np.max(D2) / np.min(D2)
        return score, P_opt
    except Exception:
        return np.inf, start_P

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    best_score = np.inf
    best_P = None
    
    starts = []
    
    # 1. Grid
    side = int(np.ceil(N ** (1 / D)))
    grid = list(np.meshgrid(np.arange(side), np.arange(side), np.arange(side), indexing='ij'))
    grid = np.array([p for p in zip(*grid)])
    grid_P = grid[:N].astype(float)
    grid_P -= np.mean(grid_P, axis=0)
    grid_P /= np.sqrt(np.sum(grid_P**2))
    starts.append(grid_P)
    
    # 2. Fibonacci Sphere
    def fib_sphere(n):
        pts = []
        phi = np.pi * (3 - np.sqrt(5))
        for i in range(n):
            z = (i + 0.5) / n
            r = np.sqrt(1 - z**2)
            theta = phi * i
            pts.append([np.cos(theta) * r, np.sin(theta) * r, z])
        return np.array(pts)
        
    sphere_P = fib_sphere(N)
    sphere_P /= np.sqrt(np.sum(sphere_P**2))
    starts.append(sphere_P)
    
    # 3. Random
    np.random.seed(42)
    for _ in range(50):
        rand_P = np.random.randn(N, D)
        rand_P -= np.mean(rand_P, axis=0)
        norm = np.sqrt(np.sum(rand_P**2))
        if norm > 1e-10:
            rand_P /= norm
            starts.append(rand_P)
            
    for i, start_P in enumerate(starts):
        if time.time() > deadline - 5.0:  # Stop if less than 5s left
            break
            
        score, P_opt = _optimize_from_start(start_P, max_iter=500)
        
        if score < best_score:
            best_score = score
            best_P = P_opt
            
    if best_P is None:
        best_P = starts[0]
        
    return best_P
