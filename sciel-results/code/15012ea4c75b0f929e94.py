import numpy as np
from scipy.optimize import differential_evolution

N = 50
dx = 0.01
x = -0.25 + (np.arange(N) + 0.5) * dx

def _compute_score(params):
    """
    params: [a, b, c1, c2, s1, s2]
    h = (a * w1 + b * w2)^2
    w1, w2 are Gaussian pulses centered at c1, c2 with widths s1, s2.
    Returns the ratio ||h||_2^2 / (||h||_1 * ||h*h||_inf)
    """
    a, b, c1, c2, s1, s2 = params
    
    # Ensure widths are positive and reasonable
    s1 = max(abs(s1), 0.001)
    s2 = max(abs(s2), 0.001)
    
    # Clip centers to domain to keep pulses relevant
    c1 = np.clip(c1, -0.25, 0.25)
    c2 = np.clip(c2, -0.25, 0.25)
    
    # Compute Gaussian windows
    # Using exp(-0.5 * ((x-c)/s)^2)
    w1 = np.exp(-0.5 * ((x - c1) / s1) ** 2)
    w2 = np.exp(-0.5 * ((x - c2) / s2) ** 2)
    
    # Sum of weighted windows
    f = a * w1 + b * w2
    
    # Squared modulus (which is just square since real)
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
    # Parameters: a, b, c1, c2, s1, s2
    # a, b: amplitudes. Scale invariance means only ratio matters, but we bound them.
    # c1, c2: centers.
    # s1, s2: widths.
    
    # To encourage disjoint supports, we can initialize or bound such that |c1-c2| is large.
    # However, DE explores the whole space. We can add a soft penalty for overlap if needed,
    # or just let the optimizer find the best configuration.
    # The theoretical optimum for such ratios often involves sharp peaks or specific separations.
    
    bounds = [
        (0.0, 1.0),   # a
        (0.0, 1.0),   # b
        (-0.25, 0.25), # c1
        (-0.25, 0.25), # c2
        (0.001, 0.1),  # s1 (width)
        (0.001, 0.1)   # s2 (width)
    ]
    
    try:
        result = differential_evolution(
            _neg_score, 
            bounds, 
            maxiter=300, 
            popsize=25, 
            tol=1e-8, 
            seed=42,
            init='sobol'
        )
        best_params = result.x
        best_score = -result.fun
    except Exception as e:
        # Fallback: Random search
        np.random.seed(42)
        best_score = -np.inf
        best_params = None
        
        for _ in range(2000):
            a = np.random.uniform(0.1, 1.0)
            b = np.random.uniform(0.1, 1.0)
            c1 = np.random.uniform(-0.2, 0.2)
            c2 = np.random.uniform(-0.2, 0.2)
            # Ensure some separation
            if abs(c1 - c2) < 0.05:
                c2 = c1 + np.sign(c2-c1 if c2!=c1 else 1) * 0.1
            s1 = np.random.uniform(0.005, 0.05)
            s2 = np.random.uniform(0.005, 0.05)
            params = [a, b, c1, c2, s1, s2]
            s = _compute_score(params)
            if s > best_score:
                best_score = s
                best_params = params
        
        if best_params is None:
            return [1.0] * N

    a, b, c1, c2, s1, s2 = best_params
    s1 = max(abs(s1), 0.001)
    s2 = max(abs(s2), 0.001)
    c1 = np.clip(c1, -0.25, 0.25)
    c2 = np.clip(c2, -0.25, 0.25)
    
    w1 = np.exp(-0.5 * ((x - c1) / s1) ** 2)
    w2 = np.exp(-0.5 * ((x - c2) / s2) ** 2)
    
    f = a * w1 + b * w2
    h = f ** 2
    h = np.maximum(h, 0.0)
    
    return [float(v) for v in h]
