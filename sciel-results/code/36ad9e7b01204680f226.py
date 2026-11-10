import numpy as np
import time

def compute_P_and_grad(support, weights):
    # ... implementation ...
    return P, grad

def beat_the_average_measure():
    start_time = time.time()
    deadline = start_time + 600
    
    # Seed
    n_seed = 32
    support_seed = np.arange(n_seed, dtype=np.float64)
    weights_seed = np.ones(n_seed, dtype=np.float64) / n_seed
    P_seed, _ = compute_P_and_grad(support_seed, weights_seed)
    
    best_P = P_seed
    best_support = support_seed
    best_weights = weights_seed
    
    # Optimization on larger grid
    n_opt = 64
    support_opt = np.arange(n_opt, dtype=np.float64)
    
    # Try a few restarts
    for restart in range(5):
        if time.time() > deadline - 60: break
        
        if restart == 0:
            w = np.ones(n_opt) / n_opt
        else:
            w = np.random.dirichlet(np.ones(n_opt))
        
        for step in range(100):
            if time.time() > deadline - 10: break
            P, grad = compute_P_and_grad(support_opt, w)
            if P > best_P:
                best_P = P
                best_support = support_opt.copy()
                best_weights = w.copy()
            
            # Gradient step
            # Normalize gradient?
            alpha = 0.1 # learning rate
            w_new = w + alpha * grad
            # Project to simplex
            w_new = np.maximum(w_new, 0)
            if np.sum(w_new) > 1e-9:
                w_new /= np.sum(w_new)
            else:
                w_new = w
            w = w_new
            
    return best_support, best_weights
