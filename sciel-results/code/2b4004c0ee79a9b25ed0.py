import numpy as np
from scipy.optimize import minimize
import time

N = 100
SHIFTS = [1, 2, 3, 4, 5]

def compute_overlaps(h):
    overlaps = []
    for k in SHIFTS:
        shifted_h = np.roll(h, k)
        ov = np.sum(h * (1 - shifted_h))
        overlaps.append(ov)
    return np.array(overlaps)

def make_objective(beta):
    def objective(h):
        overlaps = compute_overlaps(h)
        m = np.max(overlaps)
        # Soft max
        return m + np.log(np.sum(np.exp(beta * (overlaps - m)))) / beta
    return objective

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    best_h = None
    best_score = float('inf')
    
    np.random.seed(42)
    
    def project_sum(x, target_sum):
        current_sum = np.sum(x)
        diff = target_sum - current_sum
        x = x + diff / N
        return np.clip(x, 0, 1)
        
    # Generate starting points
    starts = []
    starts.append(np.full(N, 0.5))
    alt = np.array([0.0 if i % 2 == 0 else 1.0 for i in range(N)])
    starts.append(alt)
    
    for _ in range(30):
        r = np.random.rand(N)
        r = project_sum(r, 50.0)
        starts.append(r)
        
    constraints = [
        {'type': 'eq', 'fun': lambda x: np.sum(x) - 50.0}
    ]
    bounds = [(0.0, 1.0)] * N

    for h0 in starts:
        if time.time() > deadline:
            break
            
        # Anneal beta
        betas = [1.0, 5.0, 10.0, 20.0]
        current_h = h0.copy()
        
        for beta in betas:
            if time.time() > deadline:
                break
                
            try:
                res = minimize(
                    make_objective(beta), 
                    current_h, 
                    method='L-BFGS-B', 
                    bounds=bounds,
                    constraints=constraints,
                    options={'maxiter': 500, 'ftol': 1e-9, 'gtol': 1e-9}
                )
                current_h = res.x
            except Exception:
                pass
        
        # Final evaluation with hard max
        current_h = project_sum(current_h, 50.0)
        overlaps = compute_overlaps(current_h)
        score = np.max(overlaps)
        
        if score < best_score:
            best_score = score
            best_h = current_h.copy()

    if best_h is None:
        return np.full(N, 0.5)
        
    return best_h
