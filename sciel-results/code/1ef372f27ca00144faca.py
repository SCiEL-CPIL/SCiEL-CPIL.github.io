import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
import time

N = 50

def _get_distances(c):
    d = cdist(c, c)
    iu = np.triu_indices(N, 1)
    d_pair = d[iu] / 2.0
    wx = 0.5 - np.abs(c[:, 0])
    wy = 0.5 - np.abs(c[:, 1])
    d_wall = np.minimum(wx, wy)
    return np.concatenate([d_pair, d_wall])

def _objective_and_grad(c_flat, k):
    c = c_flat.reshape(N, 2)
    d = _get_distances(c)
    exponents = -k * d
    max_exp = np.max(exponents)
    weights = np.exp(exponents - max_exp)
    sum_weights = np.sum(weights)
    val = -1.0 / k * (max_exp + np.log(sum_weights))
    
    grad_c = np.zeros((N, 2))
    
    i_idx, j_idx = np.triu_indices(N, 1)
    w_pair = weights[:len(i_idx)] / sum_weights
    
    diff = c[i_idx] - c[j_idx]
    dists = np.linalg.norm(diff, axis=1)
    dists[dists < 1e-8] = 1e-8
    unit = diff / dists[:, np.newaxis]
    
    contrib_i = 0.5 * w_pair[:, np.newaxis] * unit
    contrib_j = -0.5 * w_pair[:, np.newaxis] * unit
    
    np.add.at(grad_c, i_idx, contrib_i)
    np.add.at(grad_c, j_idx, contrib_j)
    
    w_wall = weights[len(i_idx):] / sum_weights
    wx = 0.5 - np.abs(c[:, 0])
    wy = 0.5 - np.abs(c[:, 1])
    x_limiting = wx < wy
    y_limiting = wy <= wx
    
    grad_x = np.zeros(N)
    grad_y = np.zeros(N)
    
    sx = np.sign(c[x_limiting, 0])
    grad_x[x_limiting] = -sx
    
    sy = np.sign(c[y_limiting, 1])
    grad_y[y_limiting] = -sy
    
    grad_c[:, 0] += w_wall * grad_x
    grad_c[:, 1] += w_wall * grad_y
    
    grad_flat = (1.0 / k) * grad_c.flatten()
    return -val, grad_flat

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    best_c = None
    best_score = -np.inf
    
    # Fallback valid solution
    m = 8
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    fallback = grid[:N]
    
    inits = []
    for _ in range(5):
        c = np.random.uniform(-0.4, 0.4, (N, 2))
        inits.append(c)
        
    k_values = [1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0]
    
    for c0 in inits:
        if time.time() > deadline - 20: break
        c_curr = c0.copy()
        for k in k_values:
            if time.time() > deadline - 10: break
            res = minimize(_objective_and_grad, c_curr.flatten(), args=(k,), method='L-BFGS-B', jac=True, options={'maxiter': 50, 'ftol': 1e-10, 'gtol': 1e-8})
            c_curr = res.x.reshape(N, 2)
        d = _get_distances(c_curr)
        score = np.min(d)
        if score > best_score:
            best_score = score
            best_c = c_curr.copy()
            
    if best_c is not None and time.time() < deadline - 5:
        c_curr = best_c.copy()
        for k in [200.0, 500.0]:
            res = minimize(_objective_and_grad, c_curr.flatten(), args=(k,), method='L-BFGS-B', jac=True, options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-10})
            c_curr = res.x.reshape(N, 2)
        d = _get_distances(c_curr)
        score = np.min(d)
        if score > best_score:
            best_score = score
            best_c = c_curr.copy()
            
    if best_c is None:
        return fallback
    return best_c
