"""Improved program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm

# Precompute constants for speed
_SQRT5 = np.sqrt(5.0)

# --- Shekel 4D Implementation ---
def shekel_4d(x):
    """
    Standard Shekel 4D function.
    x: (4,) array or (N, 4)
    """
    A = np.array([
        [4, 1, 8, 3],
        [16, 16, 16, 16],
        [28, 28, 28, 28],
        [50, 50, 50, 50],
        [2, 2, 2, 2],
        [25, 25, 25, 25],
        [50, 50, 50, 50],
        [11, 11, 11, 11],
        [7, 7, 7, 7],
        [37, 37, 37, 37]
    ])
    c = np.array([
        0.10, 0.15, 0.10, 0.10, 0.15,
        0.10, 0.10, 0.10, 0.15, 0.10
    ])
    x = np.atleast_2d(x)
    diff = x[:, np.newaxis, :] - A[np.newaxis, :, :]
    sq_dist = np.sum(diff ** 2, axis=2)
    denom = sq_dist + c[np.newaxis, :]
    vals = np.sum(1.0 / denom, axis=1)
    return -vals / 1000.0

def create_instance(seed):
    """
    Create a shifted Shekel instance.
    """
    rng = np.random.RandomState(seed)
    x_std_min = np.array([4.0, 1.0, 8.0, 3.0])
    x_new_min = rng.uniform(1.0, 9.0, size=4)
    shift = x_new_min - x_std_min
    
    def f_instance(x):
        x_shifted = x - shift
        return shekel_4d(x_shifted)
        
    f_min_val = shekel_4d(x_std_min)[0]
    return f_instance, f_min_val

# --- GP Implementation (Optimized) ---
def matern52_kernel(X1, X2, length_scale=0.2):
    """
    Compute Matern 5/2 kernel between X1 (N,D) and X2 (M,D).
    Returns (N, M) kernel matrix.
    """
    X1_sq = np.sum(X1 ** 2, axis=1)[:, np.newaxis]
    X2_sq = np.sum(X2 ** 2, axis=1)[np.newaxis, :]
    dist_sq = X1_sq + X2_sq - 2.0 * (X1 @ X2.T)
    dist_sq = np.maximum(dist_sq, 0.0)
    dist = np.sqrt(dist_sq)
    
    r = dist / length_scale
    kernel = (1.0 + _SQRT5 * r + 5.0 * r ** 2) * np.exp(-_SQRT5 * r)
    return kernel

def predict_gp(X_train, y_train, X_test, length_scale=0.2, noise=1e-6):
    """
    X_train: (N, D)
    y_train: (N,)
    X_test: (M, D)
    Returns: mu (M,), sigma (M,)
    """
    N = X_train.shape[0]
    
    y_mean = np.mean(y_train)
    y_std = np.std(y_train) + 1e-8
    y_std_y = (y_train - y_mean) / y_std
    
    K = matern52_kernel(X_train, X_train, length_scale)
    K[np.diag_indices_from(K)] += noise + 1e-9
    
    try:
        L = np.linalg.cholesky(K)
    except np.linalg.LinAlgError:
        K += 1e-7 * np.eye(N)
        try:
            L = np.linalg.cholesky(K)
        except np.linalg.LinAlgError:
            K_inv = np.linalg.pinv(K)
            alpha = K_inv @ y_std_y
            K_test = matern52_kernel(X_test, X_train, length_scale)
            mu_scaled = K_test @ alpha
            v_test = K_inv @ K_test.T
            var_scaled = 1.0 - np.diagonal(v_test)
            var_scaled = np.maximum(var_scaled, 1e-8)
            mu = mu_scaled * y_std + y_mean
            sigma = np.sqrt(var_scaled) * y_std
            return mu, sigma
    
    v = np.linalg.solve(L, y_std_y)
    alpha = np.linalg.solve(L.T, v)
    
    K_test = matern52_kernel(X_test, X_train, length_scale)
    
    mu_scaled = K_test @ alpha
    v_test = np.linalg.solve(L, K_test.T)
    var_scaled = 1.0 - np.sum(v_test ** 2, axis=0)
    var_scaled = np.maximum(var_scaled, 1e-8)
    
    mu = mu_scaled * y_std + y_mean
    sigma = np.sqrt(var_scaled) * y_std
    
    return mu, sigma

# --- Acquisition Functions ---
def acq_ei(mu, sigma, f_best, cost):
    """Expected Improvement, cost-aware."""
    sigma_safe = np.maximum(sigma, 1e-9)
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    return ei / cost

def acq_ucb(mu, sigma, f_best, cost, beta):
    """UCB, cost-aware."""
    improve = f_best - (mu - beta * sigma)
    return improve / cost

def acq_log_ei(mu, sigma, f_best, cost):
    """Log EI, cost-aware."""
    sigma_safe = np.maximum(sigma, 1e-9)
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    return np.log(ei + 1e-9) / cost

# --- Simulation Engine ---
def run_simulation(seed, acq_func, beta=None, max_cost=30.0):
    """
    Run the BO loop for one instance.
    Returns simple regret.
    """
    f, f_true_min = create_instance(seed)
    
    rng = np.random.RandomState(seed + 100)
    X = rng.uniform(0, 10, size=(5, 4))
    Y = np.array([f(x) for x in X])
    total_cost = np.sum(1 + np.mean(X / 10.0, axis=1))
    
    f_best = np.min(Y)
    
    length_scale = 0.2
    noise = 1e-6
    
    while total_cost < max_cost:
        X_cand = rng.uniform(0, 10, size=(2000, 4))
        
        mu, sigma = predict_gp(X, Y, X_cand, length_scale, noise)
        
        cost = 1 + np.mean(X_cand / 10.0, axis=1)
        
        if beta is not None:
            scores = acq_func(mu, sigma, f_best, cost, beta)
        else:
            scores = acq_func(mu, sigma, f_best, cost)
            
        idx = np.argmax(scores)
        x_new = X_cand[idx]
        
        y_new = f(x_new)
        
        X = np.vstack([X, x_new])
        Y = np.append(Y, y_new)
        
        c_new = 1 + np.mean(x_new / 10.0)
        total_cost += c_new
        
        if y_new < f_best:
            f_best = y_new
            
    regret = f_best - f_true_min
    return regret

# --- Default Acquisition: UCB with beta=2.0 ---
ACQ_FUNC = acq_ucb
BETA = 2.0

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function.
    mu, sigma, cost: arrays of shape (2000,)
    f_best: scalar
    Returns: array of 2000 finite scores (higher = better)
    """
    scores = ACQ_FUNC(mu, sigma, f_best, cost, BETA)
    
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    return scores
# EVOLVE-BLOCK-END
