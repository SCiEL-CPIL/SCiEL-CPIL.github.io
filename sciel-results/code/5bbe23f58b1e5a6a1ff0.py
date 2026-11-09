import numpy as np
from scipy.optimize import minimize
import time

N = 100

def compute_clearances(X):
    """
    Compute all pairwise distances and distances to walls.
    Returns a flat array of clearances.
    """
    # Pairwise distances
    # X shape (N, 3)
    # dist shape (N, N)
    diff = X[:, np.newaxis, :] - X[np.newaxis, :, :]
    dist = np.linalg.norm(diff, axis=2)
    # Mask diagonal
    np.fill_diagonal(dist, np.inf)
    # Take upper triangle to avoid duplicates
    iu = np.triu_indices(N, k=1)
    pair_dists = dist[iu]
    
    # Distances to walls
    # min(0.5 - x, x + 0.5) for each coord
    wall_dists = np.min(np.stack([
        0.5 - X[:, 0], X[:, 0] + 0.5,
        0.5 - X[:, 1], X[:, 1] + 0.5,
        0.5 - X[:, 2], X[:, 2] + 0.5
    ]), axis=0)
    
    return np.concatenate([pair_dists, wall_dists])

def soft_min_radius_and_grad(X, beta):
    """
    Compute soft min radius and its gradient.
    """
    D = compute_clearances(X)
    # Soft min
    # To avoid overflow, subtract max
    max_D = np.max(D)
    exp_terms = np.exp(-beta * (D - max_D))
    sum_exp = np.sum(exp_terms)
    R_soft = max_D - (1.0 / beta) * np.log(sum_exp)
    
    # Gradient
    # w_k = exp(-beta * (D_k - max_D)) / sum_exp
    w = exp_terms / sum_exp
    
    # Gradient of D wrt X
    # D is concatenation of pair_dists and wall_dists
    grad_X = np.zeros_like(X)
    
    # Pairwise part
    pair_dists = D[:len(w) - N] # Length N(N-1)/2
    wall_dists = D[len(w) - N:] # Length N
    
    # Reconstruct pairwise distances for gradient
    # This is inefficient if we recompute. Better to compute grad directly in the loop or vectorize.
    # Let's vectorize the pairwise gradient part.
    # We need d/dX_i (||X_i - X_j||) = (X_i - X_j) / ||X_i - X_j||
    
    diff = X[:, np.newaxis, :] - X[np.newaxis, :, :]
    dist = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dist, 1.0) # avoid div by 0
    
    # Unit vectors
    unit_vecs = diff / dist[:, :, np.newaxis]
    
    # Weights for pairs
    # iu indices
    iu = np.triu_indices(N, k=1)
    pair_weights = w[:len(pair_dists)]
    
    # Accumulate gradients
    # For each pair (i, j), grad_X[i] += w * unit_vec[i,j], grad_X[j] -= w * unit_vec[i,j]
    # Vectorized accumulation is tricky. 
    # Loop over pairs is O(N^2) = 5000, which is fine for Python if not too slow.
    # But N=100, 5000 iterations per gradient call. L-BFGS might call this 100-1000 times.
    # 5000 * 1000 = 5M ops. Should be okay in 120s? Maybe.
    # Let's try a faster way using matrix operations if possible, or just simple loops.
    
    # Actually, we can use the fact that:
    # sum_{j != i} w_{ij} * (X_i - X_j) / d_{ij}
    # Let's just loop for clarity and correctness, and optimize if needed.
    
    for k in range(len(pair_dists)):
        i, j = iu[0][k], iu[1][k]
        w_k = pair_weights[k]
        d_k = dist[i, j]
        if d_k < 1e-12: continue
        vec = (X[i] - X[j]) / d_k
        grad_X[i] += w_k * vec
        grad_X[j] -= w_k * vec
        
    # Wall part
    # d/dx_i (0.5 - x_i) = -1, d/dx_i (x_i + 0.5) = 1
    # The min is taken over 6 boundaries.
    # The gradient is -1 for the active boundary if it's the "0.5 - x" type, +1 if "x + 0.5" type.
    # Specifically, for each sphere i and coord c:
    # If 0.5 - X[i,c] is the min, grad is -1.
    # If X[i,c] + 0.5 is the min, grad is 1.
    
    for i in range(N):
        w_i = wall_dists[i]
        # Check which wall is closest
        # x: 0.5 - x vs x + 0.5
        if 0.5 - X[i, 0] <= X[i, 0] + 0.5:
            grad_X[i, 0] += -w_i
        else:
            grad_X[i, 0] += w_i
            
        if 0.5 - X[i, 1] <= X[i, 1] + 0.5:
            grad_X[i, 1] += -w_i
        else:
            grad_X[i, 1] += w_i
            
        if 0.5 - X[i, 2] <= X[i, 2] + 0.5:
            grad_X[i, 2] += -w_i
        else:
            grad_X[i, 2] += w_i
            
    # We want to MAXIMIZE R_soft, so minimize -R_soft
    # Gradient of -R_soft is -grad_R_soft
    return R_soft, -grad_X

def objective(X, beta):
    val, grad = soft_min_radius_and_grad(X, beta)
    return -val, -grad

def construct():
    start_time = time.time()
    deadline = start_time + 110.0 # 10s safety margin
    
    # Initial guess: Grid
    # 4x5x5 grid
    m_x, m_y, m_z = 4, 5, 5
    x = np.linspace(-0.5 + 1.0/(2*m_x), 0.5 - 1.0/(2*m_x), m_x)
    y = np.linspace(-0.5 + 1.0/(2*m_y), 0.5 - 1.0/(2*m_y), m_y)
    z = np.linspace(-0.5 + 1.0/(2*m_z), 0.5 - 1.0/(2*m_z), m_z)
    
    X_init = np.array([(xi, yi, zi) for zi in z for yi in y for xi in x])
    
    best_X = X_init
    best_score = -1e9
    
    # Try a few optimizations with increasing beta
    betas = [5.0, 10.0, 20.0, 50.0, 100.0]
    
    for beta in betas:
        if time.time() > deadline:
            break
            
        # Random perturbations
        for attempt in range(5):
            if time.time() > deadline:
                break
                
            X0 = best_X + np.random.normal(0, 0.01, best_X.shape)
            X0 = np.clip(X0, -0.49, 0.49)
            
            try:
                res = minimize(
                    lambda x: objective(x, beta),
                    X0,
                    jac=True,
                    method='L-BFGS-B',
                    options={'maxiter': 500, 'ftol': 1e-8, 'gtol': 1e-8}
                )
                if res.success:
                    X_opt = res.x
                    # Compute actual radius
                    D = compute_clearances(X_opt)
                    r = np.min(D)
                    if r > best_score:
                        best_score = r
                        best_X = X_opt
            except Exception:
                pass
                
    # Final polish with very high beta?
    # Just return the best found
    return best_X

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

N = 100

def compute_clearances(X):
    diff = X[:, np.newaxis, :] - X[np.newaxis, :, :]
    dist = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dist, np.inf)
    iu = np.triu_indices(N, k=1)
    pair_dists = dist[iu]
    
    wall_dists = np.min(np.stack([
        0.5 - X[:, 0], X[:, 0] + 0.5,
        0.5 - X[:, 1], X[:, 1] + 0.5,
        0.5 - X[:, 2], X[:, 2] + 0.5
    ]), axis=0)
    
    return np.concatenate([pair_dists, wall_dists])

def soft_min_radius_and_grad(X, beta):
    D = compute_clearances(X)
    max_D = np.max(D)
    exp_terms = np.exp(-beta * (D - max_D))
    sum_exp = np.sum(exp_terms)
    R_soft = max_D - (1.0 / beta) * np.log(sum_exp)
    
    w = exp_terms / sum_exp
    
    grad_X = np.zeros_like(X)
    
    pair_dists = D[:len(w) - N]
    wall_dists = D[len(w) - N:]
    
    diff = X[:, np.newaxis, :] - X[np.newaxis, :, :]
    dist = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dist, 1.0)
    
    unit_vecs = diff / dist[:, :, np.newaxis]
    
    iu = np.triu_indices(N, k=1)
    pair_weights = w[:len(pair_dists)]
    
    for k in range(len(pair_dists)):
        i, j = iu[0][k], iu[1][k]
        w_k = pair_weights[k]
        d_k = dist[i, j]
        if d_k < 1e-12: continue
        vec = (X[i] - X[j]) / d_k
        grad_X[i] += w_k * vec
        grad_X[j] -= w_k * vec
        
    for i in range(N):
        w_i = wall_dists[i]
        if 0.5 - X[i, 0] <= X[i, 0] + 0.5:
            grad_X[i, 0] += -w_i
        else:
            grad_X[i, 0] += w_i
            
        if 0.5 - X[i, 1] <= X[i, 1] + 0.5:
            grad_X[i, 1] += -w_i
        else:
            grad_X[i, 1] += w_i
            
        if 0.5 - X[i, 2] <= X[i, 2] + 0.5:
            grad_X[i, 2] += -w_i
        else:
            grad_X[i, 2] += w_i
            
    return R_soft, -grad_X

def objective(X, beta):
    val, grad = soft_min_radius_and_grad(X, beta)
    return -val, -grad

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    m_x, m_y, m_z = 4, 5, 5
    x = np.linspace(-0.5 + 1.0/(2*m_x), 0.5 - 1.0/(2*m_x), m_x)
    y = np.linspace(-0.5 + 1.0/(2*m_y), 0.5 - 1.0/(2*m_y), m_y)
    z = np.linspace(-0.5 + 1.0/(2*m_z), 0.5 - 1.0/(2*m_z), m_z)
    
    X_init = np.array([(xi, yi, zi) for zi in z for yi in y for xi in x])
    
    best_X = X_init
    best_score = -1e9
    
    betas = [5.0, 10.0, 20.0, 50.0, 100.0]
    
    for beta in betas:
        if time.time() > deadline:
            break
            
        for attempt in range(5):
            if time.time() > deadline:
                break
                
            X0 = best_X + np.random.normal(0, 0.01, best_X.shape)
            X0 = np.clip(X0, -0.49, 0.49)
            
            try:
                res = minimize(
                    lambda x: objective(x, beta),
                    X0,
                    jac=True,
                    method='L-BFGS-B',
                    options={'maxiter': 500, 'ftol': 1e-8, 'gtol': 1e-8}
                )
                if res.success:
                    X_opt = res.x
                    D = compute_clearances(X_opt)
                    r = np.min(D)
                    if r > best_score:
                        best_score = r
                        best_X = X_opt
            except Exception:
                pass
                
    return best_X
# EVOLVE-BLOCK-END
