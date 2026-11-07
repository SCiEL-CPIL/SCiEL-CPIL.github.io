import numpy as np
import time
from scipy.optimize import minimize

N = 50

def _compute_value(h):
    """
    Computes the objective value for a given height vector h.
    """
    h = np.asarray(h, dtype=float)
    if np.any(h < 0):
        h = np.clip(h, 0, None)
    h_sum = np.sum(h)
    if h_sum == 0:
        return 0.0
    
    g = np.convolve(h, h, mode='full')
    m = len(g)
    
    y = np.zeros(m + 2)
    y[1:-1] = g
    
    w = 1.0 / (m + 1)
    
    y_curr = y[:m+1]
    y_next = y[1:m+2]
    terms = y_curr**2 + y_curr * y_next + y_next**2
    S = np.sum(terms)
    l2 = (w / 3.0) * S
    
    norm_1 = np.sum(np.abs(g)) / (m + 1)
    norm_inf = np.max(np.abs(g))
    
    if norm_1 * norm_inf == 0:
        return 0.0
        
    value = l2 / (norm_1 * norm_inf)
    return value

def _knots_to_h(params):
    """
    Convert knot parameters to a height vector of length N.
    
    params structure (length 14):
    - 7 knot positions (x_0..x_6) in [0, N-1], sorted
    - 7 amplitudes (a_0..a_6) > 0
    
    The function is piecewise linear interpolating these points.
    To ensure non-negativity, we use softplus or exp for amplitudes?
    Actually, simpler: enforce knots sorted and amplitudes positive via exp.
    
    Let's use:
    x_i = sigmoid(raw_x_i) * (N-1)
    a_i = exp(raw_a_i)
    
    But we need to sort x_i. Sorting is non-differentiable but okay for NM.
    However, NM doesn't need gradients.
    
    To make it smoother for NM, we can just let the optimizer find sorted knots.
    But if knots cross, the piecewise linear interpolation might behave weirdly.
    Better approach: 
    1. Take 7 raw positions, sort them.
    2. Take 7 raw amplitudes, exp them.
    3. Interpolate.
    
    Wait, if we sort, the identity of which amplitude goes with which knot changes.
    This creates a non-smooth objective landscape for the optimizer if it swaps knots.
    
    Alternative: Fix the order of knots? No, we want to find optimal positions.
    
    Let's stick to the previous "tent" idea but with variable widths?
    The prompt suggests "piecewise linear 'tent' basis with only 5-7 variable knots".
    
    Let's try a different parametrization that avoids sorting issues:
    Use 7 "peaks" (tents) but allow them to be asymmetric?
    Or just use 7 tents (14 params: 7 centers, 7 widths/amps).
    
    Actually, the current program uses 4 tents (12 params).
    The directive says "5-7 variable knots". A piecewise linear function with 7 knots is defined by 7 (x,y) pairs.
    If we sort x, we must sort y accordingly to maintain the function definition.
    This permutation symmetry is hard for NM.
    
    Let's use 7 tents again, but with more flexible initial guesses and maybe more iterations.
    Or, better: Use 7 "Gaussian-like" tents (using exp(-((x-c)/w)^2)) which are smooth and positive.
    This avoids the sharp corners of linear tents which might be suboptimal for the L2/L1/Inf ratio?
    Actually, the objective is discrete. Sharp features might help maximize norm_inf while keeping norm_1 low?
    Or smooth features maximize the L2 term?
    
    Let's try 7 Gaussian tents.
    Params: 7 centers, 7 log_widths, 7 log_amps. Total 21 params.
    This is a larger search space than 12, but more expressive.
    """
    K = 7
    centers = params[0:K]
    log_widths = params[K:2*K]
    log_amps = params[2*K:3*K]
    
    widths = np.exp(log_widths)
    amps = np.exp(log_amps)
    
    x = np.arange(N)
    h = np.zeros(N)
    
    for i in range(K):
        c = centers[i]
        w = widths[i]
        a = amps[i]
        
        # Gaussian tent
        h += a * np.exp(-((x - c) / w)**2)
        
    return h

def _objective_gauss(params):
    """
    Objective function for Nelder-Mead minimization (returns negative value).
    """
    try:
        h = _knots_to_h(params)
        val = _compute_value(h)
        return -val
    except Exception:
        return 1e10

def construct():
    start_time = time.time()
    deadline = start_time + 140.0 # Safety margin
    
    K = 7
    num_params = 3 * K
    
    # Initial candidates
    candidates = []
    
    # 1. Symmetric multi-lobe: 7 tents evenly spaced
    c0 = np.linspace(5, 45, K)
    w0 = np.full(K, 3.0)
    a0 = np.full(K, 1.0)
    p0 = np.concatenate([c0, np.log(w0), np.log(a0)])
    candidates.append(p0)
    
    # 2. Central peak + side lobes
    c1 = np.array([25, 15, 35, 5, 45, 10, 40])
    w1 = np.array([5.0, 3.0, 3.0, 2.0, 2.0, 2.0, 2.0])
    a1 = np.array([1.0, 0.5, 0.5, 0.2, 0.2, 0.2, 0.2])
    p1 = np.concatenate([c1, np.log(w1), np.log(a1)])
    candidates.append(p1)
    
    # 3. Random starts
    np.random.seed(50)
    for _ in range(30):
        c = np.random.uniform(0, N, K)
        w = np.random.uniform(1.0, 10.0, K)
        a = np.random.uniform(0.1, 5.0, K)
        p = np.concatenate([c, np.log(w), np.log(a)])
        candidates.append(p)
        
    # Evaluate all candidates
    scored = []
    for p in candidates:
        val = _compute_value(_knots_to_h(p))
        scored.append((val, p))
    
    scored.sort(key=lambda x: x[0], reverse=True)
    
    best_val = scored[0][0]
    best_params = scored[0][1].copy()
    
    # Polish top candidates
    # 21 dimensions is high for NM, so we limit iterations and try top few.
    top_candidates = [p for _, p in scored[:5]]
    
    for p_init in top_candidates:
        if time.time() > deadline - 15:
            break
            
        try:
            res = minimize(
                _objective_gauss,
                p_init,
                method='Nelder-Mead',
                options={
                    'maxiter': 3000,
                    'xatol': 1e-8,
                    'fatol': 1e-10,
                    'adaptive': True
                }
            )
            
            val_opt = -res.fun
            if val_opt > best_val:
                best_val = val_opt
                best_params = res.x.copy()
        except Exception:
            pass
            
    # Final check
    if best_params is None:
        return list(np.ones(N) / N)
        
    h_best = _knots_to_h(best_params)
    
    # Ensure non-negative and finite
    h_best = np.clip(h_best, 0, None)
    if np.sum(h_best) == 0:
        h_best = np.ones(N) / N
        
    return list(h_best)

# EVOLVE-BLOCK-START
# The construct function is defined above.
# EVOLVE-BLOCK-END
