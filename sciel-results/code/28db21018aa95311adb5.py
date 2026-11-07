import numpy as np
import time
from scipy.optimize import minimize

N = 100

def _value(h):
    c = np.correlate(h, 1 - h, mode="full")
    return np.max(c) / N * 2

def _objective(x):
    # x is 99-dimensional, h[0] determined by sum constraint
    h = np.zeros(N)
    h[:N-1] = x
    h[N-1] = N / 2 - np.sum(x)
    # Clip to [0,1] - but this makes it non-smooth
    # Better: use sigmoid-like or just optimize and project
    return _value(h)

def construct():
    start = time.time()
    deadline = start + 140  # safety margin
    
    # Start from seed
    h = np.full(N, 0.5)
    best_h = h.copy()
    best_val = _value(h)
    
    # Use scipy minimize with SLSQP
    # Parameterize: h[i] for i=0..N-2, h[N-1] = 50 - sum(h[0..N-2])
    # But this makes it hard to enforce bounds.
    
    # Alternative: use all N variables with sum constraint
    from scipy.optimize import LinearConstraint
    
    def obj(x):
        h = x
        # Project to ensure sum = 50 and bounds [0,1]
        h = np.clip(h, 0, 1)
        # Adjust for sum
        diff = 50 - np.sum(h)
        if diff > 0:
            # Need to add
            mask = h < 1
            if np.sum(mask) == 0:
                return _value(h)
            h[mask] += diff / np.sum(mask)
            h = np.clip(h, 0, 1)
        elif diff < 0:
            mask = h > 0
            if np.sum(mask) == 0:
                return _value(h)
            h[mask] += diff / np.sum(mask)
            h = np.clip(h, 0, 1)
        return _value(h)
    
    # Try multiple random restarts
    rng = np.random.default_rng(42)
    
    for trial in range(50):
        if time.time() > deadline:
            break
        
        if trial == 0:
            x0 = np.full(N, 0.5)
        else:
            # Random binary-ish start
            x0 = rng.random(N)
            x0 = np.clip(x0 * 2 - 0.5, 0, 1)  # bias toward 0/1
            # Or just random
            x0 = rng.random(N)
        
        # Scale to sum = 50
        s = np.sum(x0)
        if s > 0:
            x0 = x0 * 50 / s
        x0 = np.clip(x0, 0, 1)
        
        res = minimize(obj, x0, method='L-BFGS-B', 
                       options={'maxiter': 1000, 'ftol': 1e-12})
        
        if res.success and res.fun < best_val:
            best_val = res.fun
            best_h = np.clip(res.x, 0, 1)
            # Ensure sum
            diff = 50 - np.sum(best_h)
            if abs(diff) > 1e-9:
                mask = (best_h < 1) if diff > 0 else (best_h > 0)
                if np.sum(mask) > 0:
                    best_h[mask] += diff / np.sum(mask)
                    best_h = np.clip(best_h, 0, 1)
    
    return best_h
