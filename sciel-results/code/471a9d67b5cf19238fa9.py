import numpy as np
from scipy.optimize import minimize
import time

def _score(pts):
    """
    Calculate the log-score: sum(log(d_ij)) - n * log(D).
    Higher is better.
    """
    n = len(pts)
    if n != 8:
        return -np.inf
    
    # Calculate pairwise distances
    dists = []
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(pts[i] - pts[j])
            if d < 1e-12:
                return -np.inf
            dists.append(d)
    
    D = max(dists)
    if D < 1e-12:
        return -np.inf
        
    log_prod = sum(np.log(d) for d in dists)
    log_score = log_prod - n * np.log(D)
    return log_score

def _objective(x):
    """
    Objective function for minimization (negated log-score).
    """
    pts = x.reshape(8, 2)
    s = _score(pts)
    if np.isinf(s) or np.isnan(s):
        return 1e10
    return -s

def construct():
    start_time = time.time()
    deadline = 145.0  # Leave 5s for overhead/return
    
    # Regular Octagon
    angles = np.linspace(0, 2 * np.pi, 8, endpoint=False)
    init_pts = np.column_stack([np.cos(angles), np.sin(angles)])
    init_x = init_pts.flatten()
    
    best_score = _score(init_pts)
    best_x = init_x
    
    # Optimization settings
    # Use Nelder-Mead for robustness without gradients
    # and Powell as an alternative
    
    rng = np.random.default_rng(12345)
    
    # List of initial guesses
    guesses = [init_x]
    
    # Add perturbed versions
    for scale in [0.05, 0.1, 0.2, 0.5, 1.0]:
        for _ in range(5):
            noise = rng.normal(0, scale, size=init_x.shape)
            guesses.append(init_x + noise)
            
    # Add a "double square" guess? 
    # Square 1: (1,0), (-1,0), (0,1), (0,-1)
    # Square 2: Rotated 45 deg? That's octagon.
    # Maybe rotate by 22.5? Still octagon.
    # Maybe non-regular?
    # Let's just rely on the random perturbations of the octagon.
    
    for i, x0 in enumerate(guesses):
        if time.time() > deadline:
            break
            
        try:
            # Method 1: Nelder-Mead
            res_nm = minimize(_objective, x0, method='Nelder-Mead', 
                              options={'maxiter': 1000, 'xatol': 1e-5, 'fatol': 1e-5})
            
            if res_nm.fun < -best_score:
                best_score = -res_nm.fun
                best_x = res_nm.x
                
            # Method 2: Powell (sometimes better for this)
            # Only if time permits
            if time.time() < deadline - 10:
                res_pw = minimize(_objective, x0, method='Powell',
                                  options={'maxiter': 200, 'ftol': 1e-8})
                if res_pw.fun < -best_score:
                    best_score = -res_pw.fun
                    best_x = res_pw.x
                    
        except Exception as e:
            pass

    # Final scaling to diameter 2
    pts = best_x.reshape(8, 2)
    dists = [np.linalg.norm(pts[i] - pts[j]) for i in range(8) for j in range(i + 1, 8)]
    D = max(dists)
    if D > 1e-12:
        pts = pts * (2.0 / D)
        
    # Ensure distinct points (they should be, but just in case)
    # Convert to list of lists
    return [[float(p[0]), float(p[1])] for p in pts]
