import numpy as np
import time
from scipy.optimize import minimize

N = 50
DEADLINE = 145.0  # Safety margin from 150s limit

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
    
    y_i = y[:m+1]
    y_ip1 = y[1:m+2]
    terms = y_i**2 + y_i * y_ip1 + y_ip1**2
    l2 = (w / 3.0) * np.sum(terms)
    
    norm_1 = np.sum(np.abs(g)) / (m + 1)
    norm_inf = np.max(np.abs(g))
    
    if norm_1 * norm_inf == 0:
        return -np.inf
        
    return l2 / (norm_1 * norm_inf)

def params_to_h(params):
    """
    Convert spectral parameters to step heights h.
    params structure: [C, A1, B1, f1, A2, B2, f2, ...]
    We use K harmonics.
    Total params: 1 (C) + 3*K
    """
    K = 5
    C = params[0]
    h = np.full(N, C**2)
    
    for j in range(K):
        base = 1 + 3*j
        Aj = params[base]
        Bj = params[base+1]
        fj = params[base+2]
        
        # Frequencies are in cycles per N samples. 
        # To ensure smoothness and avoid aliasing issues in the discrete sum,
        # we can constrain fj or just let it be. 
        # However, since we are summing cos/sin, the result is periodic.
        # The "heights" are on a grid of N points.
        # h_k = | C + sum (Aj cos(2 pi fj k/N) + Bj sin(2 pi fj k/N)) |^2
        
        k = np.arange(N)
        angle = 2 * np.pi * fj * k / N
        signal = C + Aj * np.cos(angle) + Bj * np.sin(angle)
        h += signal**2 - C**2 # Add the contribution of this harmonic
        
    # Actually, the definition above is slightly off if we want to sum squares of a superposition.
    # Let S_k = C + sum_{j=1}^K (Aj cos(...) + Bj sin(...))
    # h_k = S_k^2
    # So:
    S = np.full(N, C)
    for j in range(K):
        base = 1 + 3*j
        Aj = params[base]
        Bj = params[base+1]
        fj = params[base+2]
        k = np.arange(N)
        angle = 2 * np.pi * fj * k / N
        S += Aj * np.cos(angle) + Bj * np.sin(angle)
        
    h = S**2
    return h

def objective(params):
    """
    Objective function for optimization.
    """
    h = params_to_h(params)
    val = compute_value_exact(h)
    if not np.isfinite(val):
        return 1e10
    return -val

def construct():
    start_time = time.time()
    
    # Number of harmonics to use in the spectral representation
    K = 5
    n_params = 1 + 3 * K
    
    best_val = -np.inf
    best_params = None
    
    # Define bounds for parameters
    # C can be large
    # A, B can be large
    # f can be any real, but effectively periodic. Let's bound f to [0, 10] to cover low freqs.
    # Actually, since it's discrete, f and f+N are equivalent.
    # Let's use bounds to help L-BFGS-B.
    bounds = [(0, 10)] * n_params # Start with positive bounds, we can adjust if needed.
    # Wait, C, A, B can be negative? 
    # If C is negative, S can still be anything. 
    # But h = S^2 is invariant to sign of S.
    # Let's allow C, A, B to be in [-5, 5] and f in [0, 5].
    # Actually, scaling of params doesn't matter for the shape, but it affects the magnitude of h.
    # The objective is scale-invariant (ratio of norms). 
    # So we can fix the scale of h, e.g., sum(h)=1 or max(h)=1.
    # But params_to_h doesn't enforce that.
    # We can just let the optimizer find the scale, or normalize h inside objective.
    # Normalizing h inside objective makes it scale-invariant and potentially better conditioned.
    
    def objective_normalized(params):
        h = params_to_h(params)
        # Normalize h to have unit L2 norm or similar? 
        # The value is scale invariant. 
        # If we don't normalize, the optimizer might drift in scale.
        # Let's normalize h to have max value 1.
        if np.max(h) > 0:
            h = h / np.max(h)
        val = compute_value_exact(h)
        if not np.isfinite(val):
            return 1e10
        return -val

    # We will use objective_normalized
    
    # Initializations
    # 1. All zeros (uniform h)
    # 2. Single low frequency
    # 3. Random
    
    rng = np.random.default_rng(50)
    
    inits = []
    
    # Init 1: Uniform (C=1, others 0)
    p0 = np.zeros(n_params)
    p0[0] = 1.0
    inits.append(p0)
    
    # Init 2: Low freq sine (C=0, A1=1, f1=1)
    p1 = np.zeros(n_params)
    p1[1] = 1.0 # A1
    p1[3] = 1.0 # f1
    inits.append(p1)
    
    # Init 3: Low freq cosine (C=0, B1=1, f1=1)
    p2 = np.zeros(n_params)
    p2[2] = 1.0 # B1
    p2[3] = 1.0 # f1
    inits.append(p2)
    
    # Init 4: Two freqs
    p3 = np.zeros(n_params)
    p3[1] = 1.0 # A1
    p3[3] = 1.0 # f1
    p3[4] = 0.5 # A2
    p3[6] = 2.0 # f2
    inits.append(p3)
    
    # Random inits
    for _ in range(10):
        p = rng.uniform(-2, 2, n_params)
        inits.append(p)
    
    # Optimize each init
    for i, p0 in enumerate(inits):
        if time.time() > start_time + DEADLINE:
            break
            
        try:
            res = minimize(
                objective_normalized,
                p0,
                method='L-BFGS-B',
                bounds=bounds,
                options={
                    'maxiter': 500,
                    'maxfun': 5000,
                    'ftol': 1e-10,
                    'gtol': 1e-7,
                    'iprint': 0
                }
            )
            
            p_opt = res.x
            h_opt = params_to_h(p_opt)
            if np.max(h_opt) > 0:
                h_opt = h_opt / np.max(h_opt)
            
            val = compute_value_exact(h_opt)
            
            if val > best_val:
                best_val = val
                best_params = p_opt.copy()
                
        except Exception:
            pass
            
    # If we have time, refine the best solution
    if best_params is not None and time.time() < start_time + DEADLINE - 10:
        try:
            res2 = minimize(
                objective_normalized,
                best_params,
                method='L-BFGS-B',
                bounds=bounds,
                options={
                    'maxiter': 1000,
                    'maxfun': 10000,
                    'ftol': 1e-12,
                    'gtol': 1e-9,
                    'iprint': 0
                }
            )
            p_refined = res2.x
            h_refined = params_to_h(p_refined)
            if np.max(h_refined) > 0:
                h_refined = h_refined / np.max(h_refined)
            
            val_refined = compute_value_exact(h_refined)
            
            if val_refined > best_val:
                best_val = val_refined
                best_params = p_refined.copy()
        except Exception:
            pass
            
    # Construct final h
    if best_params is not None:
        final_h = params_to_h(best_params)
    else:
        final_h = np.ones(N) / N
        
    # Ensure non-negative and finite
    final_h = np.clip(final_h, 0, None)
    if not np.all(np.isfinite(final_h)) or np.max(final_h) == 0:
        final_h = np.ones(N) / N
        
    # Normalize to a reasonable scale (doesn't affect value, but good practice)
    # The evaluator uses the raw values. Scale invariance means any scale works.
    # Let's keep max=1 for stability.
    if np.max(final_h) > 0:
        final_h = final_h / np.max(final_h)
        
    return list(final_h)

# EVOLVE-BLOCK-START
# The construct function is defined above
# EVOLVE-BLOCK-END
