import numpy as np
import time
from scipy.optimize import minimize

N = 50
START_TIME = time.time()
TIME_LIMIT = 145.0  # Leave margin before 150s kill

def compute_ratio_normalized(h):
    """
    Compute the ratio ||h||_2^2 / (||h||_1 * ||h*h||_inf) for step function on [-1/4, 1/4].
    """
    h = np.asarray(h, dtype=float)
    if np.any(h < 0):
        return 0.0
    dx = 1.0 / N  # interval width
    l1 = dx * np.sum(h)
    if l1 <= 0:
        return 0.0
    l2_sq = dx * np.sum(h**2)
    conv = np.convolve(h, h, mode='full')
    conv_scaled = conv * dx
    conv_inf = np.max(conv_scaled)
    if conv_inf <= 0:
        return 0.0
    return l2_sq / (l1 * conv_inf)

def neg_ratio(h):
    """Negative of the ratio for minimization."""
    val = compute_ratio_normalized(h)
    if val <= 0:
        return 1e10
    return -val

def construct():
    np.random.seed(42)
    
    # The ratio R = ||h||_2^2 / (||h||_1 * ||h*h||_inf) is homogeneous of degree -1.
    # For a fixed shape (direction in the cone of nonnegative vectors), R is determined by shape.
    # We can fix ||h||_1 = 1 and maximize ||h||_2^2 / ||h*h||_inf.
    # 
    # Key insight: a single spike maximizes concentration. Let's verify by trying
    # several structured families and using scipy optimization on a reduced parameterization.
    
    # Strategy: parameterize h as a nonnegative vector with sum=1, then optimize.
    # Use scipy minimize with bounds [0, 1] and a constraint sum(h)=1.
    # To handle the non-smoothness of ||h*h||_inf, we use Nelder-Mead on the logit-transformed
    # variables or just direct search.
    
    # First, evaluate simple candidates:
    candidates = []
    
    # Single spike at various positions
    for pos in [0, N//4, N//2, 3*N//4, N-1]:
        h = np.zeros(N)
        h[pos] = 1.0
        candidates.append(h)
    
    # Two adjacent spikes
    for pos in [0, N//2 - 1, N//2]:
        h = np.zeros(N)
        h[pos] = 0.5
        h[pos+1] = 0.5
        candidates.append(h)
    
    # Block of k consecutive ones
    for k in [1, 2, 3, 5, 10]:
        for start in [0, N//2 - k//2]:
            h = np.zeros(N)
            h[start:start+k] = 1.0/k
            candidates.append(h)
    
    # Gaussian-like shapes with varying width
    for sigma in [0.5, 1.0, 2.0, 3.0, 5.0]:
        x = np.linspace(-5, 5, N)
        h = np.exp(-x**2 / (2*sigma**2))
        h = h / np.sum(h)
        candidates.append(h)
    
    # Evaluate all candidates
    best_val = 0.0
    best_shape = candidates[0].copy()
    for c in candidates:
        val = compute_ratio_normalized(c)
        if val > best_val:
            best_val = val
            best_shape = c.copy()
    
    # Now try to improve with scipy Nelder-Mead on the shape
    # Parameterize: h_i = softmax(z_i) * scale, but we fix sum=1
    # Use the fact that for sum=1, we maximize ||h||_2^2 / ||h*h||_inf
    # which equals (sum h_i^2) / (max_k sum_j h_j h_{k-j})
    
    def objective(z):
        """z is a vector of length N, we map to h = exp(z) / sum(exp(z))"""
        z_clipped = np.clip(z, -50, 50)  # avoid overflow
        h = np.exp(z_clipped)
        s = np.sum(h)
        if s <= 0:
            return 1e10
        h = h / s
        return neg_ratio(h)
    
    # Start from best candidate
    h_init = best_shape.copy()
    # Convert to log-space for optimization
    h_init_safe = np.maximum(h_init, 1e-10)
    z_init = np.log(h_init_safe)
    
    if time.time() - START_TIME < TIME_LIMIT - 30:
        try:
            res = minimize(objective, z_init, method='Nelder-Mead',
                          options={'maxiter': 5000, 'xatol': 1e-8, 'fatol': 1e-10,
                                   'adaptive': True})
            z_opt = res.x
            z_opt_clipped = np.clip(z_opt, -50, 50)
            h_opt = np.exp(z_opt_clipped)
            s = np.sum(h_opt)
            if s > 0:
                h_opt = h_opt / s
                val_opt = compute_ratio_normalized(h_opt)
                if val_opt > best_val:
                    best_val = val_opt
                    best_shape = h_opt.copy()
        except Exception:
            pass
    
    # Try a few more random restarts with Nelder-Mead if time permits
    if time.time() - START_TIME < TIME_LIMIT - 20:
        for restart in range(5):
            if time.time() - START_TIME > TIME_LIMIT - 10:
                break
            # Random initial shape
            h_rand = np.random.dirichlet(np.ones(N) * 0.5)  # sparse-ish
            h_rand_safe = np.maximum(h_rand, 1e-10)
            z_rand = np.log(h_rand_safe)
            
            try:
                res = minimize(objective, z_rand, method='Nelder-Mead',
                              options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-10,
                                       'adaptive': True})
                z_opt = res.x
                z_opt_clipped = np.clip(z_opt, -50, 50)
                h_opt = np.exp(z_opt_clipped)
                s = np.sum(h_opt)
                if s > 0:
                    h_opt = h_opt / s
                    val_opt = compute_ratio_normalized(h_opt)
                    if val_opt > best_val:
                        best_val = val_opt
                        best_shape = h_opt.copy()
            except Exception:
                pass
    
    # Return unit L1 norm shape
    result = np.maximum(0, best_shape)
    s = np.sum(result)
    if s > 0:
        result = result / s
    return [float(x) for x in result]
