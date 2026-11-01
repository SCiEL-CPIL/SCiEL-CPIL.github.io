import numpy as np
from scipy.optimize import minimize
import time

N = 14
D = 3

def compute_distances(X):
    # X is (N, D)
    # Return (N, N) distance matrix
    # d[i,j] = ||X[i] - X[j]||
    diff = X[:, np.newaxis, :] - X[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    return dists

def smooth_ratio_and_grad(X, beta):
    X = X.reshape(N, D)
    dists = compute_distances(X)
    # Mask diagonal
    mask = ~np.eye(N, dtype=bool)
    d = dists[mask]
    
    # Soft max
    max_d = np.max(d)
    # Shift for stability
    shifted_d = d - max_d
    exp_shifted = np.exp(beta * shifted_d)
    S = np.sum(exp_shifted)
    log_max = max_d + np.log(S) / beta
    # grad of log_max w.r.t d
    grad_log_max_d = exp_shifted / S  # This is the weight for each d
    
    # Soft min
    min_d = np.min(d)
    shifted_d_min = min_d - d
    exp_shifted_min = np.exp(beta * shifted_d_min)
    T = np.sum(exp_shifted_min)
    log_min = min_d - np.log(T) / beta
    # grad of log_min w.r.t d
    # d(log_min)/d(d_k) = -1/beta * d(log T)/d(d_k)
    # T = sum exp(beta(min_d - d_j))
    # dT/dd_k = exp(beta(min_d - d_k)) * (-beta)
    # d(log T)/dd_k = (dT/dd_k)/T = -beta * exp_shifted_min[k] / T
    # d(log_min)/dd_k = -1/beta * (-beta * w_k) = w_k
    grad_log_min_d = exp_shifted_min / T
    
    # Objective: (max/min)^2 = exp(2 * (log_max - log_min))
    log_ratio = log_max - log_min
    val = np.exp(2 * log_ratio)
    
    # Gradient w.r.t X
    # L = log_max - log_min
    # dL/dd_k = grad_log_max_d[k] - grad_log_min_d[k]
    dL_dd = grad_log_max_d - grad_log_min_d
    
    # We need dL/dX
    # dL = sum_k dL_dd[k] * dd_k
    # dd_k = d ||x_i - x_j|| / d x_a
    # For each pair (i,j), contribution to grad X[i] and X[j]
    
    grad_X = np.zeros_like(X)
    
    # Iterate over pairs
    # This might be slow in Python, vectorize if possible
    # d_k corresponds to pair (i,j) where mask is True
    # Let's extract indices
    i_idx, j_idx = np.where(mask)
    
    # dL_dd is 1D array of length N*(N-1)/2
    # We need to accumulate gradients
    # For each pair k, weight w_k = dL_dd[k]
    # diff_k = X[i_k] - X[j_k]
    # norm_k = d_k
    # grad_d_k = diff_k / norm_k
    # grad_X[i_k] += w_k * grad_d_k
    # grad_X[j_k] -= w_k * grad_d_k
    
    # Vectorized approach:
    # Create arrays of differences
    # This is tricky with 1D dL_dd mapping to 2D pairs.
    # Let's use a loop for clarity and speed (N=14 is small, 91 pairs)
    for k in range(len(dL_dd)):
        i, j = i_idx[k], j_idx[k]
        w = dL_dd[k]
        if w == 0: continue
        diff = X[i] - X[j]
        dist = dists[i, j]
        if dist < 1e-9:
            grad = np.zeros(D)
        else:
            grad = diff / dist
        grad_X[i] += w * grad
        grad_X[j] -= w * grad
        
    # Total gradient of val = 2 * val * dL/dX
    grad_val = 2 * val * grad_X
    
    return val, grad_val

def construct():
    start_time = time.time()
    deadline = start_time + 140.0 # 10s margin
    
    best_val = float('inf')
    best_X = None
    
    # Initializations
    inits = []
    
    # 1. Random
    for _ in range(20):
        X = np.random.randn(N, D)
        inits.append(X)
        
    # 2. Spherical Fibonacci
    golden_ratio = (1 + np.sqrt(5)) / 2
    y = np.arange(N) - N/2 + 0.5
    r = np.sqrt(1 - (y/N)**2)
    theta = np.mod(np.arange(N) * np.pi * 2 / golden_ratio, 2 * np.pi)
    X_sphere = np.column_stack([np.cos(theta)*r, y/N, np.sin(theta)*r])
    inits.append(X_sphere)
    
    # 3. Icosahedron + 2
    # Icosahedron vertices
    phi = (1 + np.sqrt(5)) / 2
    ico_verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ])
    ico_verts = ico_verts / np.linalg.norm(ico_verts, axis=1)[:, np.newaxis]
    # Add 2 points at origin? No, min dist 0.
    # Add 2 points slightly off center?
    # Or just use 12 ico points and 2 random points near center?
    # Let's just add 2 random points
    X_ico = np.vstack([ico_verts, np.random.randn(2, D) * 0.1])
    inits.append(X_ico)
    
    # 4. Grid (from seed, but centered and scaled)
    side = int(np.ceil(N ** (1 / D)))
    grid = list(__import__('itertools').product(range(side), repeat=D))[:N]
    X_grid = np.array(grid, dtype=float)
    X_grid = (X_grid - X_grid.mean(axis=0))
    inits.append(X_grid)
    
    beta = 50.0
    
    for X0 in inits:
        if time.time() > deadline:
            break
            
        def obj_and_grad(x):
            X = x.reshape(N, D)
            val, grad = smooth_ratio_and_grad(X, beta)
            return val, grad.reshape(-1)
            
        try:
            res = minimize(obj_and_grad, X0.flatten(), method='L-BFGS-B', jac=True, options={'maxiter': 200, 'ftol': 1e-12})
            val = res.fun
            if val < best_val:
                best_val = val
                best_X = res.x.reshape(N, D)
        except Exception:
            pass
            
    if best_X is None:
        return np.array(inits[0])
        
    return best_X
