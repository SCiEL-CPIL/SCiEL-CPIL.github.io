"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

def _softmin_obj_grad(c, beta):
    """
    Computes the objective (negative soft-min distance) and its gradient.
    c: (N, 3) array of centers.
    beta: sharpness parameter.
    """
    # 1. Pairwise distances
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dist = np.linalg.norm(diff, axis=2)
    dist_safe = np.where(dist < 1e-8, 1e-8, dist)
    u = diff / dist_safe[:, :, np.newaxis]
    
    mask = np.triu(np.ones((N, N), dtype=bool), k=1)
    dist_upper = dist[mask]
    exp_pair = np.exp(-beta * dist_upper)
    
    # 2. Wall distances (2 * dist to wall)
    # x_wall[i, k] = 2 * dist(c_i, face_k)
    x_wall = np.empty((N, 6))
    x_wall[:, 0] = 1.0 - 2.0 * c[:, 0] # -x face
    x_wall[:, 1] = 1.0 + 2.0 * c[:, 0] # +x face
    x_wall[:, 2] = 1.0 - 2.0 * c[:, 1] # -y face
    x_wall[:, 3] = 1.0 + 2.0 * c[:, 1] # +y face
    x_wall[:, 4] = 1.0 - 2.0 * c[:, 2] # -z face
    x_wall[:, 5] = 1.0 + 2.0 * c[:, 2] # +z face
    
    exp_wall = np.exp(-beta * x_wall)
    
    S = np.sum(exp_pair) + np.sum(exp_wall)
    
    # Objective: g = (1/beta) * log(S)
    obj = (1.0 / beta) * np.log(S)
    
    # Gradient
    # p_pair = exp_pair / S
    p_pair = exp_pair / S
    P = np.zeros((N, N))
    P[mask] = p_pair
    
    # grad_softmin_pair[i] = sum_j P[i, j] * u[i, j]
    grad_softmin_pair = np.sum(P[:, :, np.newaxis] * u, axis=1)
    
    # Wall gradients
    p_wall = exp_wall / S
    coeffs = np.array([[-2, 0, 0],
                       [2, 0, 0],
                       [0, -2, 0],
                       [0, 2, 0],
                       [0, 0, -2],
                       [0, 0, 2]], dtype=float)
    
    grad_softmin_wall = p_wall @ coeffs.T
    
    grad_softmin = grad_softmin_pair + grad_softmin_wall
    
    # We minimize g = -softmin, so grad_g = -grad_softmin
    grad_g = -grad_softmin
    
    return obj, grad_g

def _score(c):
    """Compute the true maximum radius."""
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dist = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dist, np.inf)
    min_pair = np.min(dist) / 2.0
    
    w = np.min(np.stack([
        0.5 - c[:, 0], 0.5 + c[:, 0],
        0.5 - c[:, 1], 0.5 + c[:, 1],
        0.5 - c[:, 2], 0.5 + c[:, 2]
    ], axis=1), axis=1)
    min_wall = np.min(w)
    
    return min(min_pair, min_wall)

def construct():
    start_time = time.time()
    deadline = start_time + 145.0 # Safety margin
    
    best_c = None
    best_score = -1.0
    
    # Initializations
    inits = []
    
    # 1. Grid
    m = 4
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    inits.append(grid[:N])
    
    # 2. Random
    rng = np.random.default_rng(42)
    for _ in range(5):
        inits.append(rng.uniform(-0.4, 0.4, size=(N, 3)))
        
    # 3. Jittered Grid
    inits.append(grid[:N] + rng.normal(0, 0.05, size=(N, 3)))
    
    betas = [10, 20, 50, 100, 200, 500]
    bounds = [(-0.5, 0.5)] * 3 * N
    
    for init in inits:
        if time.time() > deadline - 15:
            break
            
        c = init.copy()
        c = np.clip(c, -0.49, 0.49)
        
        for beta in betas:
            if time.time() > deadline - 5:
                break
                
            def fun(x):
                return _softmin_obj_grad(x, beta)[0]
            
            def jac(x):
                return _softmin_obj_grad(x, beta)[1]
                
            res = minimize(
                fun,
                c,
                jac=jac,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 100, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            
            c = res.x
            c = np.clip(c, -0.5, 0.5)
            
            s = _score(c)
            if s > best_score:
                best_score = s
                best_c = c.copy()
                
    # Final polish
    if best_c is not None and time.time() < deadline - 2:
        for beta in [1000, 2000]:
            if time.time() > deadline - 1:
                break
            def fun(x):
                return _softmin_obj_grad(x, beta)[0]
            
            def jac(x):
                return _softmin_obj_grad(x, beta)[1]
                
            res = minimize(
                fun,
                best_c,
                jac=jac,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            c = res.x
            c = np.clip(c, -0.5, 0.5)
            s = _score(c)
            if s > best_score:
                best_score = s
                best_c = c.copy()
                
    return best_c
# EVOLVE-BLOCK-END
