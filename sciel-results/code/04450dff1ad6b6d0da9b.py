import numpy as np
import time
from scipy.signal import fftconvolve
from scipy.optimize import minimize

N = 50

def compute_ratio(h):
    h = np.asarray(h, dtype=np.float64)
    h = np.maximum(h, 0.0)
    l1 = np.sum(h)
    if l1 < 1e-15:
        return 0.0
    h_norm = h / l1
    h2_sq = np.sum(h_norm**2)
    conv = fftconvolve(h_norm, h_norm, mode='full')
    conv_inf = np.max(conv)
    if conv_inf < 1e-15:
        return 0.0
    return h2_sq / conv_inf

def is_sidon_set(indices):
    """Check if a set of indices forms a B2 (Sidon) set."""
    indices = sorted(indices)
    sums = set()
    for i in range(len(indices)):
        for j in range(i, len(indices)):
            s = indices[i] + indices[j]
            if s in sums:
                return False
            sums.add(s)
    return True

def find_max_sidon_set(n):
    """
    Greedy algorithm to find a maximal Sidon set in {0, ..., n-1}.
    This typically finds the Mian-Chowla sequence, which is optimal or near-optimal for small n.
    """
    s = set()
    for i in range(n):
        # Check if adding i maintains the Sidon property
        # We need to check sums of i with existing elements, and i+i
        # against existing sums
        new_sums = []
        for j in s:
            new_sums.append(i + j)
        new_sums.append(2 * i)
        
        # Check for collisions with existing sums
        # We can maintain a set of all current sums
        # But for simplicity and speed (n=50), we can just re-check or maintain a sum set.
        
        # Let's maintain a sum set for efficiency
        if not hasattr(find_max_sidon_set, 'sums'):
            find_max_sidon_set.sums = set()
        
        # Actually, resetting is safer if called multiple times, but here we call once.
        # Let's just do a clean check for correctness.
        
        # Re-check logic:
        # If we add i, new sums are {i+j for j in s} U {2i}.
        # These must be distinct from each other and from existing sums.
        
        # Existing sums are all pairs from s.
        # Let's just use the is_sidon_set function on s U {i} for simplicity, it's fast enough for N=50.
        
        candidate = list(s) + [i]
        if is_sidon_set(candidate):
            s.add(i)
            
    return sorted(list(s))

def optimize_weights(support):
    """
    Optimize weights for a given support to maximize the ratio.
    We use log-variables to handle positivity, or just bounded optimization.
    Since uniform is likely optimal for Sidon sets, we start there.
    """
    k = len(support)
    if k == 0:
        return np.zeros(N), 0.0
    
    # Initial guess: uniform
    x0 = np.ones(k) / k
    
    def objective(x):
        # x are the weights
        h = np.zeros(N)
        h[support] = x
        s = np.sum(h)
        if s < 1e-15:
            return 0.0
        h = h / s
        return -compute_ratio(h) # Minimize negative ratio
        
    def jac(x):
        # Numerical gradient or analytical?
        # For k <= 10, numerical is fine and robust.
        eps = 1e-7
        grad = np.zeros(k)
        base = objective(x)
        for i in range(k):
            x_plus = x.copy()
            x_plus[i] += eps
            # Normalize? The objective function normalizes h.
            # So we can just evaluate objective(x_plus).
            grad[i] = (objective(x_plus) - base) / eps
        return grad

    bounds = [(0.0, 1.0) for _ in range(k)]
    
    # Use L-BFGS-B
    try:
        res = minimize(objective, x0, method='L-BFGS-B', jac=jac, bounds=bounds, options={'maxiter': 1000, 'ftol': 1e-12})
        x_opt = res.x
        x_opt = np.maximum(x_opt, 0.0)
        x_opt = x_opt / np.sum(x_opt)
        
        h_opt = np.zeros(N)
        h_opt[support] = x_opt
        sc = compute_ratio(h_opt)
        return h_opt, sc
    except Exception:
        h_opt = np.zeros(N)
        h_opt[support] = 1.0 / k
        sc = compute_ratio(h_opt)
        return h_opt, sc

def construct():
    t0 = time.time()
    deadline = t0 + 140.0 # Safety margin
    
    best_h = np.zeros(N, dtype=np.float64)
    best_h[0] = 1.0
    best_score = compute_ratio(best_h)
    
    # 1. Find the largest Sidon set
    sidon_support = find_max_sidon_set(N)
    
    # 2. Optimize weights for this support
    if time.time() < deadline - 10.0:
        h_opt, sc = optimize_weights(sidon_support)
        if sc > best_score:
            best_score = sc
            best_h = h_opt.copy()
            
    # 3. Try a few other random supports of similar size?
    # Or just try to improve the Sidon set?
    # The greedy Sidon set is likely optimal.
    
    # 4. Final check
    return list(best_h)

# Run once at import to verify? No, construct is the entry point.
# But we can pre-compute if we want to be safe, though construct should be fast.
