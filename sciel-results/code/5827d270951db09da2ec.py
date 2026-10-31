import numpy as np
from scipy.optimize import minimize
import time

N = 400

def autoconv_max_abs(a):
    """Compute max absolute value of autoconvolution of a."""
    n = len(a)
    c = np.correlate(a, a, mode='full')
    return np.max(np.abs(c))

def objective(params):
    a = params
    s = np.sum(a)
    if abs(s) < 1e-12:
        return 1e10
    a_norm = a / s
    return autoconv_max_abs(a_norm)

def local_sign_flip_optimize(a, max_iters=500, tol=1e-10):
    """Local optimization: flip signs of individual entries to reduce max |autoconv|."""
    a = a.copy()
    s = np.sum(a)
    if abs(s) < 1e-12:
        return a
    
    a_norm = a / s
    current_score = autoconv_max_abs(a_norm)
    
    for _ in range(max_iters):
        improved = False
        # Try flipping each non-zero entry
        nonzero_idx = np.nonzero(a)[0]
        np.random.shuffle(nonzero_idx)
        
        for idx in nonzero_idx:
            old_val = a[idx]
            a[idx] = -a[idx]
            s = np.sum(a)
            if abs(s) < 1e-12:
                a[idx] = old_val
                continue
            a_norm = a / s
            new_score = autoconv_max_abs(a_norm)
            
            if new_score < current_score - tol:
                current_score = new_score
                improved = True
            else:
                a[idx] = old_val
        
        if not improved:
            break
    
    s = np.sum(a)
    if abs(s) < 1e-12:
        return a
    return a / s

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    # Start with dense baseline
    best_a = np.ones(N)
    best_score = autoconv_max_abs(best_a) / (np.sum(best_a)**2)
    
    np.random.seed(42)
    
    # Try sparse random signed sequences
    for density in [0.1, 0.2, 0.3, 0.4, 0.5]:
        if time.time() > deadline - 20:
            break
        
        n_trials = 5
        for trial in range(n_trials):
            if time.time() > deadline - 20:
                break
            
            # Create sparse sequence with random signs
            mask = np.random.rand(N) < density
            a = np.zeros(N)
            a[mask] = np.random.choice([-1.0, 1.0], size=np.sum(mask))
            
            # Ensure non-zero sum
            if abs(np.sum(a)) < 1e-12:
                a[np.random.randint(N)] += 1.0
            
            # Local sign-flip optimization
            a_opt = local_sign_flip_optimize(a, max_iters=300)
            
            s = np.sum(a_opt)
            if abs(s) < 1e-12:
                continue
            a_norm = a_opt / s
            score = autoconv_max_abs(a_norm)
            
            if score < best_score:
                best_score = score
                best_a = a_norm.copy()
    
    return best_a
