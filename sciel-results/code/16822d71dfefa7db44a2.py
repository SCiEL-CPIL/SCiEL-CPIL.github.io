# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

def _shannon_entropy(p):
    """Compute Shannon entropy in bits, ignoring zero probabilities."""
    p = np.asarray(p, dtype=np.float64)
    p = p[p > 1e-15]
    if len(p) == 0:
        return 0.0
    return float(-np.sum(p * np.log2(p)))

def _compute_D_M(P):
    """Compute D = H(x-y) and M = max(H(x), H(x+y), H(x+2y))."""
    n = 10
    x = np.arange(n)
    y = np.arange(n)
    X, Y = np.meshgrid(x, y, indexing='ij')  # X[i,j]=i, Y[i,j]=j where i=y, j=x
    
    # Wait, the problem says P[y, x]. So index 0 is y, index 1 is x.
    # Let's be precise.
    # P has shape (10, 10). P[i, j] where i is y, j is x.
    # x in 0..9, y in 0..9.
    
    # D: distribution of x - y
    # x - y ranges from -9 to 9.
    # Let's create arrays for x and y values.
    # x_vals[y, x] = x
    # y_vals[y, x] = y
    
    x_vals = np.tile(np.arange(10), (10, 1))
    y_vals = np.tile(np.arange(10)[:, np.newaxis], (1, 10))
    
    # D: x - y
    diff = x_vals - y_vals
    diff_vals = diff.ravel()
    # Unique values in diff are -9..9
    diff_probs = np.zeros(19)
    for val in diff_vals:
        idx = val + 9
        diff_probs[idx] += P.ravel()[np.where(diff_vals == val)[0]]
    # More efficient way:
    diff_probs = np.zeros(19)
    for i in range(10):
        for j in range(10):
            idx = (j - i) + 9
            diff_probs[idx] += P[i, j]
    D = _shannon_entropy(diff_probs)
    
    # M: max(H(x), H(x+y), H(x+2y))
    # H(x): distribution of x
    x_probs = np.zeros(10)
    for i in range(10):
        for j in range(10):
            x_probs[j] += P[i, j]
    H0 = _shannon_entropy(x_probs)
    
    # H(x+y): x+y ranges 0..18
    sum_vals = x_vals + y_vals
    sum_probs = np.zeros(19)
    for i in range(10):
        for j in range(10):
            idx = (j + i)
            sum_probs[idx] += P[i, j]
    H1 = _shannon_entropy(sum_probs)
    
    # H(x+2y): x+2y ranges 0..27
    two_sum_vals = x_vals + 2 * y_vals
    two_sum_probs = np.zeros(28)
    for i in range(10):
        for j in range(10):
            idx = (j + 2 * i)
            two_sum_probs[idx] += P[i, j]
    H2 = _shannon_entropy(two_sum_probs)
    
    M = max(H0, H1, H2)
    return D, M

def _objective(theta):
    """
    Objective function to minimize: -D / M + regularization.
    theta is a 100-dim vector.
    P = softmax(theta) / sum(softmax(theta))
    """
    P_flat = np.exp(theta - np.max(theta))
    P_flat = P_flat / np.sum(P_flat)
    P = P_flat.reshape(10, 10)
    
    D, M = _compute_D_M(P)
    
    if M < 1e-9:
        return 1e6
    
    # We want to maximize D/M. So minimize -D/M.
    # Add a small regularization to prevent extreme values?
    # Actually, softmax handles it.
    return -D / M

def _compute_value(P):
    D, M = _compute_D_M(P)
    if M < 1e-9:
        return 0.0
    return D / M

def arithmetic_kakeya_distribution() -> np.ndarray:
    n = 10
    start_time = time.time()
    deadline = start_time + 600
    
    # Initial uniform distribution
    P_uniform = np.ones((n, n), dtype=np.float64) / (n * n)
    best_P = P_uniform.copy()
    best_value = _compute_value(best_P)
    
    # Multiple initializations
    inits = [
        np.zeros(100), # uniform
        np.random.normal(0, 1, 100),
        np.random.normal(0, 2, 100),
        # Anti-diagonal emphasis
        np.zeros(100),
    ]
    
    # Create specific inits
    init1 = np.zeros(100)
    init2 = np.random.normal(0, 1, 100)
    init3 = np.random.normal(0, 2, 100)
    
    # Emphasis on x-y = 0 (diagonal)
    init4 = np.zeros(100)
    for i in range(10):
        for j in range(10):
            if i == j:
                init4[i*10+j] = 5.0
            else:
                init4[i*10+j] = -5.0
    
    # Emphasis on x-y = 1
    init5 = np.zeros(100)
    for i in range(10):
        for j in range(10):
            if j == i + 1:
                init5[i*10+j] = 5.0
            elif j == i - 1:
                init5[i*10+j] = 5.0
            else:
                init5[i*10+j] = -5.0
                
    inits = [init1, init2, init3, init4, init5]
    
    for init in inits:
        if time.time() > deadline - 60:
            break
            
        try:
            res = minimize(_objective, init, method='L-BFGS-B', 
                          options={'maxiter': 1000, 'ftol': 1e-10, 'gtol': 1e-8})
            if res.success:
                P_flat = np.exp(res.x - np.max(res.x))
                P_flat = P_flat / np.sum(P_flat)
                P = P_flat.reshape(10, 10)
                val = _compute_value(P)
                if val > best_value:
                    best_value = val
                    best_P = P.copy()
        except Exception:
            pass
    
    # Final check and normalization
    best_P = np.clip(best_P, 0, None)
    s = np.sum(best_P)
    if s > 0:
        best_P /= s
    else:
        best_P = P_uniform
    
    # Ensure finite
    best_P = np.nan_to_num(best_P, nan=0.0, posinf=0.0, neginf=0.0)
    
    return best_P

# EVOLVE-BLOCK-END
