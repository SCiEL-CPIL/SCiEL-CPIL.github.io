import numpy as np
import time
from scipy.optimize import minimize

N = 50
DEADLINE = 115.0  # Safety margin from 120s limit

def compute_value_exact(h):
    """
    Compute the exact value as defined in the task contract.
    """
    h = np.array(h, dtype=np.float64)
    if np.any(h < -1e-12) or not np.all(np.isfinite(h)):
        return -np.inf
    h = np.clip(h, 0, None)
    
    g = np.convolve(h, h)
    m = len(g)
    
    # y = [0, g[0], ..., g[m-1], 0]
    y = np.zeros(m + 2)
    y[1:m+1] = g
    
    w = 1.0 / (m + 1)
    
    # l2 = sum over i in 0..m of (w / 3) * (y[i]^2 + y[i]*y[i+1] + y[i+1]^2)
    # Vectorized computation of the sum
    # terms_i = y[i]^2 + y[i]*y[i+1] + y[i+1]^2 for i=0..m
    # Note: y has length m+2. We sum i from 0 to m.
    # y[0] to y[m] are the first terms, y[1] to y[m+1] are the second.
    y_i = y[:m+1]
    y_ip1 = y[1:m+2]
    terms = y_i**2 + y_i * y_ip1 + y_ip1**2
    l2 = (w / 3.0) * np.sum(terms)
    
    norm_1 = np.sum(np.abs(g)) / (m + 1)
    norm_inf = np.max(np.abs(g))
    
    if norm_1 * norm_inf == 0:
        return -np.inf
        
    return l2 / (norm_1 * norm_inf)

def compute_value_smooth(h, beta=50.0):
    """
    Compute a smooth approximation of the value for gradient-based optimization.
    Uses log-sum-exp to approximate max(g).
    """
    h = np.array(h, dtype=np.float64)
    if not np.all(np.isfinite(h)):
        return -np.inf
    # Clip to ensure non-negativity for the exact eval later, 
    # but for the smooth objective, we can allow small negatives 
    # if the penalty is high, or just clip. 
    # Clipping makes the gradient zero for negative parts, which is fine for L-BFGS-B 
    # if we bound x >= 0.
    h = np.clip(h, 0, None)
    
    g = np.convolve(h, h)
    m = len(g)
    
    y = np.zeros(m + 2)
    y[1:m+1] = g
    
    w = 1.0 / (m + 1)
    
    y_i = y[:m+1]
    y_ip1 = y[1:m+2]
    terms = y_i**2 + y_i * y_ip1 + y_ip1**2
    l2 = (w / 3.0) * np.sum(terms)
    
    norm_1 = np.sum(g) / (m + 1)  # g is non-negative since h is non-negative
    
    # Smooth max approximation: (1/beta) * log(sum(exp(beta * g)))
    # To avoid overflow, subtract max
    g_max = np.max(g)
    if g_max == 0:
        return -np.inf
    smooth_max = (1.0 / beta) * np.log(np.sum(np.exp(beta * (g - g_max)))) + g_max
    
    if norm_1 * smooth_max == 0:
        return -np.inf
        
    return l2 / (norm_1 * smooth_max)

def neg_value_smooth(h, beta=50.0):
    val = compute_value_smooth(h, beta)
    if val == -np.inf:
        return 1e10
    return -val

def construct():
    start_time = time.time()
    
    # We will use L-BFGS-B on the raw 50 variables.
    # Bounds: [0, infinity) for each height.
    # Since the objective is homogeneous of degree 0, the scale of h doesn't matter.
    # We can fix the scale to prevent numerical issues, e.g., sum(h) = 1 or max(h)=1.
    # However, L-BFGS-B handles unbounded domains well. 
    # To improve conditioning, we might normalize h internally or just let it float.
    # Let's try without explicit normalization first, but start with reasonable scales.
    
    best_val = -np.inf
    best_h = np.ones(N) / N  # Seed: uniform heights
    
    # Initializations
    # 1. Uniform (Seed)
    # 2. Gaussian centered
    # 3. Triangular
    # 4. Random
    # 5. "Hump" (concentrated)
    
    def get_initializations():
        inits = []
        # 1. Uniform
        inits.append(np.ones(N) / N)
        
        # 2. Gaussian
        x = np.linspace(-1, 1, N)
        g = np.exp(-x**2 / (2 * 0.2**2))
        inits.append(g / np.sum(g))
        
        # 3. Triangular
        t = np.abs(2 * np.linspace(0, 1, N) - 1)
        tri = 1 - t
        inits.append(tri / np.sum(tri))
        
        # 4. Hump (very concentrated)
        hump = np.zeros(N)
        hump[N//2 - 2 : N//2 + 3] = 1.0
        inits.append(hump / np.sum(hump))
        
        # 5-10. Random
        rng = np.random.default_rng(50)
        for _ in range(6):
            r = rng.random(N)
            # Add some structure? No, pure random for diversity
            inits.append(r / np.sum(r))
            
        # 11. Low freq sine
        s = np.sin(np.pi * np.linspace(0, 1, N))
        inits.append(s / np.sum(s))
        
        return inits

    inits = get_initializations()
    
    bounds = [(0, None)] * N
    
    # Optimization loop
    for i, h0 in enumerate(inits):
        if time.time() > start_time + DEADLINE:
            break
            
        try:
            # Use L-BFGS-B
            # maxiter needs to be sufficient but not too high to avoid wasting time
            # maxfun similar
            res = minimize(
                neg_value_smooth,
                h0,
                method='L-BFGS-B',
                bounds=bounds,
                options={
                    'maxiter': 1000,
                    'maxfun': 10000,
                    'ftol': 1e-12,
                    'gtol': 1e-8,
                    'iprint': 0
                }
            )
            
            h_opt = res.x
            # Evaluate exact value
            val = compute_value_exact(h_opt)
            
            if val > best_val:
                best_val = val
                best_h = h_opt
                
        except Exception as e:
            pass
            
    # If we have time, do a second pass with higher beta for smoother max approximation
    # or just refine the best solution found with a direct local search or higher precision L-BFGS-B
    
    # Refine best solution found
    if time.time() < start_time + DEADLINE:
        try:
            # Use a higher beta for the smooth approximation to get closer to the true max
            # This makes the objective sharper, requiring more careful optimization
            res2 = minimize(
                lambda h: neg_value_smooth(h, beta=100.0),
                best_h,
                method='L-BFGS-B',
                bounds=bounds,
                options={
                    'maxiter': 1000,
                    'maxfun': 10000,
                    'ftol': 1e-14,
                    'gtol': 1e-10,
                    'iprint': 0
                }
            )
            h_refined = res2.x
            val_refined = compute_value_exact(h_refined)
            
            if val_refined > best_val:
                best_val = val_refined
                best_h = h_refined
        except Exception:
            pass
            
    # One more pass with even higher beta if time permits
    if time.time() < start_time + DEADLINE - 10:
        try:
            res3 = minimize(
                lambda h: neg_value_smooth(h, beta=200.0),
                best_h,
                method='L-BFGS-B',
                bounds=bounds,
                options={
                    'maxiter': 500,
                    'maxfun': 5000,
                    'ftol': 1e-15,
                    'gtol': 1e-12,
                    'iprint': 0
                }
            )
            h_refined2 = res3.x
            val_refined2 = compute_value_exact(h_refined2)
            
            if val_refined2 > best_val:
                best_val = val_refined2
                best_h = h_refined2
        except Exception:
            pass

    # Final check: ensure non-negative and finite
    best_h = np.clip(best_h, 0, None)
    if not np.all(np.isfinite(best_h)):
        best_h = np.ones(N) / N
        
    return list(best_h)

# EVOLVE-BLOCK-START
# The construct function is defined above
# EVOLVE-BLOCK-END
