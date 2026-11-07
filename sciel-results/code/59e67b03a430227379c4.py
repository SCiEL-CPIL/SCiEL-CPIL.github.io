"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from scipy.optimize import minimize

N = 50

# EVOLVE-BLOCK-START
def compute_value(h):
    """Compute the value metric for given heights."""
    h = np.asarray(h, dtype=float)
    if h.size != N:
        h = h[:N]
        if h.size < N:
            h = np.pad(h, (0, N - h.size), 'constant')
    
    if np.any(h < 0) or not np.all(np.isfinite(h)):
        return 0.0

    g = np.convolve(h, h)
    m = len(g)
    y = np.zeros(m + 2)
    y[1:-1] = g
    w = 1.0 / (m + 1)
    
    y_sq_terms = y[:m+1]**2 + y[:m+1]*y[1:m+2] + y[1:m+2]**2
    l2 = (w / 3.0) * np.sum(y_sq_terms)
    
    norm_1 = np.sum(np.abs(g)) / (m + 1)
    norm_inf = np.max(np.abs(g))
    
    if norm_1 == 0 or norm_inf == 0:
        return 0.0
    
    value = l2 / (norm_1 * norm_inf)
    return value

def neg_value_log(z):
    """Objective function for log-parameterized optimization."""
    h = np.exp(z)
    # Prevent overflow/underflow issues by clipping z if necessary, 
    # though Nelder-Mead usually keeps z in reasonable bounds if initialized well.
    # h must be finite.
    if not np.all(np.isfinite(h)):
        return 1e10
    
    val = compute_value(h)
    if val <= 0:
        return 1e10
    return -val

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    best_h = np.ones(N)
    best_val = compute_value(best_h)
    
    # Generate diverse starting points in log-space
    # We want shapes that maximize l2 / (norm_1 * norm_inf).
    # This ratio is scale-invariant for l2/norm_1, but norm_inf scales with h^2.
    # Wait, l2 scales with h^4 (integral of g^2 where g~h^2).
    # norm_1 scales with h^2 (sum g ~ h^2).
    # norm_inf scales with h^2 (max g ~ h^2).
    # So value scales with h^4 / (h^2 * h^2) = constant.
    # The value is scale-invariant.
    # So we can fix the scale, e.g., max(h)=1 or sum(h)=1, to reduce dimensionality.
    # However, log-space optimization handles scale naturally.
    
    x = np.linspace(-1, 1, N)
    starts_log = []
    
    # 1. Gaussian bumps
    for c in np.linspace(-0.8, 0.8, 5):
        for s in np.linspace(0.2, 1.0, 5):
            h = np.exp(-((x - c)**2) / (2 * s**2))
            h = h / np.max(h) # Normalize
            starts_log.append(np.log(h + 1e-9))
            
    # 2. Cosine bumps
    for c in np.linspace(-0.8, 0.8, 5):
        for s in np.linspace(0.3, 1.0, 5):
            h = np.where(np.abs(x - c) < s, 0.5 * (1 + np.cos(np.pi * (x - c) / s)), 0.0)
            h = np.maximum(0, h)
            h = h / np.max(h)
            starts_log.append(np.log(h + 1e-9))
            
    # 3. Triangular bumps
    for c in np.linspace(-0.8, 0.8, 5):
        for s in np.linspace(0.3, 1.0, 5):
            h = np.maximum(0, 1 - np.abs(x - c) / s)
            h = h / np.max(h)
            starts_log.append(np.log(h + 1e-9))
            
    # 4. Box bumps
    for c in np.linspace(-0.8, 0.8, 5):
        for s in np.linspace(0.3, 1.0, 5):
            h = np.where(np.abs(x - c) < s, 1.0, 0.0)
            h = h / np.max(h)
            starts_log.append(np.log(h + 1e-9))
            
    # 5. Uniform (seed)
    h_uni = np.ones(N)
    starts_log.append(np.log(h_uni))

    # Evaluate starts to find top candidates
    start_scores = []
    for z0 in starts_log:
        if time.time() > deadline - 10:
            break
        h0 = np.exp(z0)
        val = compute_value(h0)
        start_scores.append((val, z0))
        
    start_scores.sort(key=lambda x: x[0], reverse=True)
    
    # Run Nelder-Mead on top diverse starts
    # Nelder-Mead is derivative-free and good for non-smooth functions (like max in norm_inf)
    top_k = 10
    for i in range(min(top_k, len(start_scores))):
        if time.time() > deadline - 10:
            break
            
        val_init, z0 = start_scores[i]
        
        # Check if this start is already better
        h_init = np.exp(z0)
        if val_init > best_val:
            best_val = val_init
            best_h = h_init
            
        try:
            # Nelder-Mead options
            res = minimize(
                neg_value_log, 
                z0, 
                method='Nelder-Mead', 
                options={
                    'xatol': 1e-6, 
                    'fatol': 1e-6, 
                    'maxiter': 500, 
                    'adaptive': True
                }
            )
            
            if res.fun < -best_val:
                h_opt = np.exp(res.x)
                val_opt = compute_value(h_opt)
                if val_opt > best_val:
                    best_val = val_opt
                    best_h = h_opt
        except Exception:
            pass

    # Final polish with L-BFGS-B on the best solution found
    # L-BFGS-B uses gradients (approximated) and might converge faster locally
    if time.time() < deadline - 5:
        try:
            # Convert best_h to log space for optimization
            z_best = np.log(np.maximum(best_h, 1e-9))
            
            res = minimize(
                neg_value_log, 
                z_best, 
                method='Nelder-Mead', 
                options={
                    'xatol': 1e-8, 
                    'fatol': 1e-8, 
                    'maxiter': 1000, 
                    'adaptive': True
                }
            )
            
            if res.fun < -best_val:
                h_opt = np.exp(res.x)
                val_opt = compute_value(h_opt)
                if val_opt > best_val:
                    best_val = val_opt
                    best_h = h_opt
        except Exception:
            pass

    # Ensure non-negativity and finiteness
    best_h = np.maximum(0, best_h)
    best_h = np.nan_to_num(best_h, nan=0.0, posinf=1e6, neginf=0.0)
    
    return best_h.tolist()
# EVOLVE-BLOCK-END
