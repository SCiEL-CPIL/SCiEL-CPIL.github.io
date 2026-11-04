import numpy as np
from scipy.optimize import differential_evolution

N = 50
dx = 0.01
x = -0.25 + (np.arange(N) + 0.5) * dx

def _compute_score(params):
    """
    params: [c1, c2, s1, s2, b]
    a is fixed to 1.0.
    Returns the ratio ||h||_2^2 / (||h||_1 * ||h*h||_inf)
    """
    c1, c2, s1, s2, b = params
    
    s1 = max(abs(s1), 0.001)
    s2 = max(abs(s2), 0.001)
    b = abs(b)
    
    c1 = np.clip(c1, -0.25, 0.25)
    c2 = np.clip(c2, -0.25, 0.25)
    
    w1 = np.exp(-0.5 * ((x - c1) / s1) ** 2)
    w2 = np.exp(-0.5 * ((x - c2) / s2) ** 2)
    
    h = (w1 + b * w2) ** 2
    h = np.maximum(h, 0.0)
    
    l1 = np.sum(h) * dx
    if l1 <= 0:
        return -1.0
        
    l2_sq = np.sum(h**2) * dx
    
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
    # We will search for the optimal parameters using differential evolution
    # which is robust to local minima and can handle the non-smooth boundaries
    # (like the max function for sigma) better than gradient-based methods if we
    # don't clip too aggressively inside the objective, but here we clip for safety.
    
    # Bounds:
    # c1, c2: [-0.25, 0.25]
    # s1, s2: [0.001, 0.5]  (very narrow to wide)
    # b: [0.0, 20.0]
    
    bounds = [(-0.25, 0.25), (-0.25, 0.25), (0.001, 0.5), (0.001, 0.5), (0.0, 20.0)]
    
    # Use differential evolution for global optimization
    # maxiter and popsize control the search depth and breadth
    # tol controls convergence
    
    try:
        result = differential_evolution(
            _neg_score, 
            bounds, 
            maxiter=200, 
            popsize=20, 
            tol=1e-10, 
            seed=42,
            init='sobol'
        )
        best_params = result.x
        best_score = -result.fun
    except Exception as e:
        # Fallback to a simpler random search if DE fails
        np.random.seed(42)
        best_score = -np.inf
        best_params = None
        
        for _ in range(1000):
            c1 = np.random.uniform(-0.24, 0.24)
            c2 = np.random.uniform(-0.24, 0.24)
            s1 = np.random.uniform(0.001, 0.1)
            s2 = np.random.uniform(0.001, 0.1)
            b = np.random.uniform(0.0, 10.0)
            params = [c1, c2, s1, s2, b]
            s = _compute_score(params)
            if s > best_score:
                best_score = s
                best_params = params
        
        if best_params is None:
            return [1.0] * N

    c1, c2, s1, s2, b = best_params
    s1 = max(abs(s1), 0.001)
    s2 = max(abs(s2), 0.001)
    b = abs(b)
    c1 = np.clip(c1, -0.25, 0.25)
    c2 = np.clip(c2, -0.25, 0.25)
    
    w1 = np.exp(-0.5 * ((x - c1) / s1) ** 2)
    w2 = np.exp(-0.5 * ((x - c2) / s2) ** 2)
    
    h = (w1 + b * w2) ** 2
    h = np.maximum(h, 0.0)
    
    return [float(v) for v in h]
