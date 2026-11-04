import numpy as np
from scipy.optimize import differential_evolution

N = 50
dx = 0.01
x = -0.25 + (np.arange(N) + 0.5) * dx

def _compute_score(params):
    """
    params: [a, b, c1, c2, s1, s2]
    h = |a * w1 + b * w2|^2
    w1, w2 are Gaussian pulses centered at c1, c2 with widths s1, s2.
    Designed for disjoint supports to maximize ||h||_2^2 / (||h||_1 * ||h*h||_inf)
    """
    a, b, c1, c2, s1, s2 = params
    
    # Ensure widths are positive and reasonable
    s1 = max(abs(s1), 0.002)
    s2 = max(abs(s2), 0.002)
    
    # Clip centers to domain
    c1 = np.clip(c1, -0.25, 0.25)
    c2 = np.clip(c2, -0.25, 0.25)
    
    # Compute Gaussian windows
    w1 = np.exp(-0.5 * ((x - c1) / s1) ** 2)
    w2 = np.exp(-0.5 * ((x - c2) / s2) ** 2)
    
    # Sum of weighted windows
    f = a * w1 + b * w2
    
    # Squared modulus
    h = f ** 2
    h = np.maximum(h, 0.0)
    
    l1 = np.sum(h) * dx
    if l1 <= 0:
        return -1.0
        
    l2_sq = np.sum(h**2) * dx
    
    # Autoconvolution
    conv = np.convolve(h, h, mode='full') * dx
    sup_norm = np.max(conv)
    
    if sup_norm <= 0:
        return -1.0
        
    return l2_sq / (l1 * sup_norm)

def _neg_score(params):
    s = _compute_score(params)
    if s < 0:
        return 1e10
    return -s

def construct():
    """
    Construct 50 nonnegative heights as |a*w1 + b*w2|^2 where w1, w2 are
    disjointly supported Gaussian pulses. Optimize separation, widths, amplitudes.
    """
    # Parameters: a, b, c1, c2, s1, s2
    # Strategy: enforce separation by parameterizing c1 and c2 with a gap
    
    bounds = [
        (0.1, 1.0),      # a (amplitude 1)
        (0.1, 1.0),      # b (amplitude 2)
        (-0.25, -0.05),  # c1 (center 1, left half)
        (0.05, 0.25),    # c2 (center 2, right half) - ensures separation
        (0.005, 0.08),   # s1 (width 1)
        (0.005, 0.08)    # s2 (width 2)
    ]
    
    best_score = -np.inf
    best_params = None
    
    try:
        # Use differential evolution with tight bounds for disjoint supports
        result = differential_evolution(
            _neg_score, 
            bounds, 
            maxiter=500, 
            popsize=30, 
            tol=1e-10, 
            seed=42,
            init='sobol'
        )
        best_params = result.x
        best_score = -result.fun
    except Exception as e:
        # Fallback: Grid + random search
        np.random.seed(42)
        
        # Try a grid of separations and widths
        for sep in [0.1, 0.15, 0.2, 0.3, 0.4]:
            for s in [0.01, 0.02, 0.03, 0.04, 0.05]:
                for a in [0.5, 1.0]:
                    for b in [0.5, 1.0]:
                        c1 = -0.25 + 0.05
                        c2 = 0.25 - 0.05
                        params = [a, b, c1, c2, s, s]
                        sc = _compute_score(params)
                        if sc > best_score:
                            best_score = sc
                            best_params = params
        
        # Random refinement
        for _ in range(5000):
            a = np.random.uniform(0.1, 1.0)
            b = np.random.uniform(0.1, 1.0)
            c1 = np.random.uniform(-0.25, -0.05)
            c2 = np.random.uniform(0.05, 0.25)
            s1 = np.random.uniform(0.005, 0.08)
            s2 = np.random.uniform(0.005, 0.08)
            params = [a, b, c1, c2, s1, s2]
            sc = _compute_score(params)
            if sc > best_score:
                best_score = sc
                best_params = params
        
        if best_params is None:
            return [1.0] * N

    a, b, c1, c2, s1, s2 = best_params
    s1 = max(abs(s1), 0.002)
    s2 = max(abs(s2), 0.002)
    c1 = np.clip(c1, -0.25, 0.25)
    c2 = np.clip(c2, -0.25, 0.25)
    
    w1 = np.exp(-0.5 * ((x - c1) / s1) ** 2)
    w2 = np.exp(-0.5 * ((x - c2) / s2) ** 2)
    
    f = a * w1 + b * w2
    h = f ** 2
    h = np.maximum(h, 0.0)
    
    return [float(v) for v in h]
