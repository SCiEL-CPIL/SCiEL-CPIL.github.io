# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

# Precompute pooling matrices for slopes -1, 0, 1, 2
# For each cell (y, x), compute the projected value and map to a bin index
def _build_pooling_matrices(n=10):
    """Build pooling matrices for slopes -1, 0, 1, 2."""
    y_idx, x_idx = np.indices((n, n))
    
    # For each slope, compute the projected value and find unique values
    slopes_info = {}
    for slope in [-1, 0, 1, 2]:
        vals = x_idx + slope * y_idx  # shape (n, n)
        vals_flat = vals.flatten()
        unique_vals = np.unique(vals_flat)
        # Map each cell to its bin index
        val_to_bin = {v: i for i, v in enumerate(unique_vals)}
        bin_indices = np.array([val_to_bin[v] for v in vals_flat])
        # Build pooling matrix: (k, 100) where k = len(unique_vals)
        k = len(unique_vals)
        M = np.zeros((k, n * n))
        for i, b in enumerate(bin_indices):
            M[b, i] = 1.0
        slopes_info[slope] = M
    
    return slopes_info

# Precompute at module level
_POOLING = _build_pooling_matrices(10)
_M_D = _POOLING[-1]   # (k_D, 100)
_M_0 = _POOLING[0]    # (k_0, 100)
_M_1 = _POOLING[1]    # (k_1, 100)
_M_2 = _POOLING[2]    # (k_2, 100)

def _entropy_from_pooled(M, w):
    """Compute Shannon entropy in bits from pooled distribution M @ w."""
    pooled = M @ w  # shape (k,)
    total = pooled.sum()
    if total <= 1e-15:
        return 0.0
    probs = pooled / total
    # Only count positive probabilities
    mask = probs > 1e-15
    if not np.any(mask):
        return 0.0
    pm = probs[mask]
    return float(-np.sum(pm * np.log2(pm)))

def _compute_ratio_fast(P):
    """Compute D/M ratio for a 10x10 distribution P[y,x] using precomputed matrices."""
    P = np.clip(P, 0, None)
    s = P.sum()
    if s <= 1e-15:
        return 0.0
    P = P / s
    w = P.flatten()
    
    D = _entropy_from_pooled(_M_D, w)
    h0 = _entropy_from_pooled(_M_0, w)
    h1 = _entropy_from_pooled(_M_1, w)
    h2 = _entropy_from_pooled(_M_2, w)
    M = max(h0, h1, h2)
    
    if M <= 1e-9:
        return 0.0
    
    return D / M

def _neg_ratio_params(params):
    """Negative ratio for minimization, params are log-weights."""
    w = np.exp(params)
    w = w / w.sum()
    P = w.reshape(10, 10)
    return -_compute_ratio_fast(P)

def _optimize_with_lbfgs(start_logP, deadline):
    """Optimize using L-BFGS-B on log-probability parameters."""
    x0 = start_logP.flatten().copy()
    
    options = {
        'maxiter': 500,
        'ftol': 1e-12,
        'gtol': 1e-8,
        'maxls': 50
    }
    
    result = minimize(_neg_ratio_params, x0, method='L-BFGS-B', options=options)
    
    if time.time() > deadline:
        return None
    
    best_params = result.x
    w = np.exp(best_params)
    w = w / w.sum()
    best_P = w.reshape(10, 10)
    best_P = np.clip(best_P, 0, None)
    total = best_P.sum()
    if total > 1e-15:
        best_P = best_P / total
    
    return best_P

def _local_search(P_init, deadline, n_steps=1000, step_size=0.1):
    """Simple coordinate descent / random perturbation local search."""
    P = P_init.copy()
    best_ratio = _compute_ratio_fast(P)
    
    for step in range(n_steps):
        if time.time() > deadline:
            break
        
        # Random perturbation in log-space
        logP = np.log(P + 1e-15)
        logP_perturbed = logP + np.random.normal(0, step_size, (10, 10))
        P_perturbed = np.exp(logP_perturbed)
        P_perturbed = P_perturbed / P_perturbed.sum()
        
        ratio = _compute_ratio_fast(P_perturbed)
        if ratio > best_ratio:
            best_ratio = ratio
            P = P_perturbed.copy()
            # Decrease step size slightly
            step_size *= 0.99
        else:
            step_size *= 0.95
    
    return P, best_ratio

def arithmetic_kakeya_distribution() -> np.ndarray:
    """
    Arithmetic Kakeya conjecture (entropy formulation). Construct a joint probability
    distribution P[y, x] of two discrete random variables X, Y supported on an n x n grid
    (here n = 10) so as to maximize the entropy projection ratio

        H(pi_{r_inf}) / max_i H(pi_{r_i}),

    where pi_r(x, y) = x + r * y projects the grid along slope r, H is the Shannon entropy
    (in bits) of the resulting projected distribution, the finite slopes are
    r_1, ..., r_k = 0.0, 1.0, 2.0 and the distinguished slope is r_inf = -1.0.

    Returns:
        P: np.ndarray of shape (10, 10), non-negative and summing to 1, where P[y, x] is the
            joint probability that X = x and Y = y.
    """
    n = 10
    start_time = time.time()
    deadline = start_time + 570  # 570 second deadline (30s safety margin)
    
    # Start with uniform distribution
    P_uniform = np.ones((n, n), dtype=np.float64) / (n * n)
    best_P = P_uniform.copy()
    best_ratio = _compute_ratio_fast(P_uniform)
    
    # Generate several random initial points
    np.random.seed(42)
    num_restarts = 40
    
    for restart in range(num_restarts):
        # Check deadline
        if time.time() > deadline - 60:  # Leave 60 seconds safety margin
            break
        
        # Generate random initial distribution
        if restart == 0:
            # Start near uniform
            logP = np.log(P_uniform) + np.random.normal(0, 0.1, (n, n))
        elif restart % 4 == 1:
            # Emphasize diagonal (x = y)
            P_init = np.zeros((n, n))
            for i in range(n):
                P_init[i, i] = 1.0
            P_init /= P_init.sum()
            logP = np.log(P_init + 1e-10) + np.random.normal(0, 0.5, (n, n))
        elif restart % 4 == 2:
            # Emphasize anti-diagonal (x + y = constant)
            P_init = np.zeros((n, n))
            for i in range(n):
                for j in range(n):
                    if i + j == n - 1:
                        P_init[i, j] = 1.0
            P_init /= P_init.sum()
            logP = np.log(P_init + 1e-10) + np.random.normal(0, 0.5, (n, n))
        elif restart % 4 == 3:
            # Emphasize x=0 or y=0
            P_init = np.zeros((n, n))
            P_init[:, 0] = 1.0
            P_init[0, :] = 1.0
            P_init /= P_init.sum()
            logP = np.log(P_init + 1e-10) + np.random.normal(0, 0.3, (n, n))
        else:
            # Random distribution
            P_init = np.random.dirichlet(np.ones(n * n))
            P_init = P_init.reshape(n, n)
            logP = np.log(P_init + 1e-10)
        
        # Optimize with L-BFGS-B
        result_P = _optimize_with_lbfgs(logP, deadline)
        
        if result_P is not None:
            ratio = _compute_ratio_fast(result_P)
            if ratio > best_ratio:
                best_ratio = ratio
                best_P = result_P.copy()
        
        # Follow up with local search if we have time
        if time.time() < deadline - 30:
            P_ls, ratio_ls = _local_search(best_P, deadline - 30, n_steps=500, step_size=0.15)
            if ratio_ls > best_ratio:
                best_ratio = ratio_ls
                best_P = P_ls.copy()
        
        # Check deadline
        if time.time() > deadline - 60:
            break
    
    # Final validation
    best_P = np.clip(best_P, 0, None)
    total = best_P.sum()
    if total <= 1e-9:
        return P_uniform
    best_P = best_P / total
    
    # Ensure no NaN or inf
    if not np.all(np.isfinite(best_P)):
        return P_uniform
    
    return best_P

# EVOLVE-BLOCK-END
