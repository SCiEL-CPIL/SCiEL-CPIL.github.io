"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

N = 14
D = 3

# EVOLVE-BLOCK-START
def _pairwise_sq_dists(X):
    """Compute all pairwise squared distances for n points in d dimensions."""
    # Use broadcasting: (n,1,d) - (1,n,d) -> (n,n,d)
    diff = X[:, np.newaxis, :] - X[np.newaxis, :, :]
    sq_dists = np.sum(diff ** 2, axis=2)
    return sq_dists


def _score(X):
    """Compute (max/min)^2 of pairwise distances."""
    sq_dists = _pairwise_sq_dists(X)
    # Zero out diagonal
    n = X.shape[0]
    np.fill_diagonal(sq_dists, np.inf)
    min_sq = np.min(sq_dists)
    max_sq = np.max(sq_dists)
    if min_sq <= 0:
        return np.inf
    return (max_sq / min_sq) ** 2


def _objective(X, softness=1000.0):
    """
    Smooth surrogate: minimize soft-max of log(sq_dists) - soft-min of log(sq_dists)
    This is smooth and differentiable.
    """
    sq_dists = _pairwise_sq_dists(X)
    n = X.shape[0]
    # Mask diagonal with large value so it doesn't affect min/max
    mask = ~np.eye(n, dtype=bool)
    sq_dists_masked = np.where(mask, sq_dists, np.inf)
    
    # Work with log distances for numerical stability
    # Add small epsilon to avoid log(0)
    eps = 1e-12
    log_dists = np.log(sq_dists_masked + eps)
    
    # Soft max and soft min using log-sum-exp
    # soft_max(x) = (1/t) * log(sum(exp(t*x)))
    # soft_min(x) = -(1/t) * log(sum(exp(-t*x)))
    t = softness
    max_log = np.max(log_dists)
    min_log = np.min(log_dists)
    
    exp_t = np.exp(t * (log_dists - max_log))
    soft_max_val = max_log + np.log(np.sum(exp_t)) / t
    
    exp_neg_t = np.exp(-t * (log_dists - min_log))
    soft_min_val = min_log - np.log(np.sum(exp_neg_t)) / t
    
    return soft_max_val - soft_min_val


def _objective_and_grad(X, softness=1000.0):
    """Compute objective and gradient."""
    sq_dists = _pairwise_sq_dists(X)
    n = X.shape[0]
    mask = ~np.eye(n, dtype=bool)
    sq_dists_masked = np.where(mask, sq_dists, np.inf)
    
    eps = 1e-12
    log_dists = np.log(sq_dists_masked + eps)
    
    t = softness
    max_log = np.max(log_dists)
    min_log = np.min(log_dists)
    
    exp_t = np.exp(t * (log_dists - max_log))
    sum_exp_t = np.sum(exp_t)
    soft_max_val = max_log + np.log(sum_exp_t) / t
    
    exp_neg_t = np.exp(-t * (log_dists - min_log))
    sum_exp_neg_t = np.sum(exp_neg_t)
    soft_min_val = min_log - np.log(sum_exp_neg_t) / t
    
    obj = soft_max_val - soft_min_val
    
    # Gradient computation
    # d(soft_max)/d(log_dists) = exp_t / sum_exp_t
    # d(soft_min)/d(log_dists) = -exp_neg_t / sum_exp_neg_t
    # d(log_dists)/d(sq_dists) = 1/(sq_dists + eps)
    # d(sq_dists)/dX[i,j] = 2*(X[i] - X[j]) for i != j
    
    grad_log_dists_max = exp_t / sum_exp_t
    grad_log_dists_min = -exp_neg_t / sum_exp_neg_t
    
    # Total gradient wrt log_dists
    grad_log_dists = grad_log_dists_max - grad_log_dists_min
    
    # Gradient wrt sq_dists
    grad_sq_dists = grad_log_dists / (sq_dists_masked + eps)
    
    # Gradient wrt X
    grad_X = np.zeros_like(X)
    for i in range(n):
        for j in range(n):
            if i != j:
                # d(sq_dist_ij)/dX[i] = 2*(X[i] - X[j])
                # d(sq_dist_ij)/dX[j] = 2*(X[j] - X[i]) = -2*(X[i] - X[j])
                coeff = grad_sq_dists[i, j]
                grad_X[i] += 2 * coeff * (X[i] - X[j])
                grad_X[j] -= 2 * coeff * (X[i] - X[j])
    
    return obj, grad_X


def _lbfgs_minimize(objective, x0, max_iter=1000, tol=1e-10):
    """
    Simple L-BFGS implementation.
    """
    n = x0.size
    m = 10  # memory size
    
    x = x0.copy()
    f, g = objective(x)
    
    # L-BFGS memory
    s_list = []
    y_list = []
    rho_list = []
    
    r = g.copy()
    
    for it in range(max_iter):
        if np.linalg.norm(r) < tol:
            break
        
        # Two-loop recursion for p = -H*r
        alpha = np.zeros(len(s_list))
        q = r.copy()
        for i in range(len(s_list) - 1, -1, -1):
            alpha[i] = rho_list[i] * np.dot(s_list[i], q)
            q -= alpha[i] * y_list[i]
        
        gamma = 1.0
        if len(s_list) > 0:
            gamma = np.dot(s_list[-1], y_list[-1]) / np.dot(y_list[-1], y_list[-1])
        
        p = gamma * q
        for i in range(len(s_list)):
            beta = rho_list[i] * np.dot(y_list[i], p)
            p += (alpha[i] - beta) * s_list[i]
        
        p = -p
        
        # Line search (backtracking)
        step = 1.0
        f_new, g_new = objective(x + step * p)
        c1 = 1e-4
        while f_new > f + c1 * step * np.dot(r, p) and step > 1e-8:
            step *= 0.5
            f_new, g_new = objective(x + step * p)
        
        # Update
        s = step * p
        y = g_new - r
        sy = np.dot(s, y)
        
        if sy > 1e-12:
            rho = 1.0 / sy
            s_list.append(s)
            y_list.append(y)
            rho_list.append(rho)
            if len(s_list) > m:
                s_list.pop(0)
                y_list.pop(0)
                rho_list.pop(0)
        
        x = x + s
        f = f_new
        r = g_new
    
    return x, f


def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    best_X = None
    best_score = np.inf
    
    # Multiple random restarts
    n_restarts = 50
    
    for restart in range(n_restarts):
        if time.time() > deadline:
            break
        
        # Random initialization: points on a sphere with some randomness
        # Start with points roughly evenly distributed
        np.random.seed(restart * 1000 + 42)
        
        # Initialize with random points in a ball
        X = np.random.randn(N, D)
        # Normalize to unit scale
        X /= np.linalg.norm(X, axis=1, keepdims=True)
        X *= 1.0
        
        # Optimize using L-BFGS
        def obj_wrapper(X_flat):
            X_reshaped = X_flat.reshape(N, D)
            return _objective(X_reshaped, softness=500.0)
        
        X_flat = X.flatten()
        X_opt, f_opt = _lbfgs_minimize(obj_wrapper, X_flat, max_iter=500, tol=1e-8)
        X_opt = X_opt.reshape(N, D)
        
        # Compute actual score
        score = _score(X_opt)
        
        if score < best_score:
            best_score = score
            best_X = X_opt.copy()
        
        # Polish with tighter softness
        if time.time() < deadline - 5.0:
            def obj_wrapper_tight(X_flat):
                X_reshaped = X_flat.reshape(N, D)
                return _objective(X_reshaped, softness=5000.0)
            
            X_flat = X_opt.flatten()
            X_opt2, f_opt2 = _lbfgs_minimize(obj_wrapper_tight, X_flat, max_iter=200, tol=1e-10)
            X_opt2 = X_opt2.reshape(N, D)
            
            score2 = _score(X_opt2)
            if score2 < best_score:
                best_score = score2
                best_X = X_opt2.copy()
    
    if best_X is None:
        # Fallback: simple grid
        side = int(np.ceil(N ** (1 / D)))
        import itertools
        grid = list(itertools.product(range(side), repeat=D))[:N]
        best_X = np.array(grid, dtype=float)
    
    return best_X
# EVOLVE-BLOCK-END
