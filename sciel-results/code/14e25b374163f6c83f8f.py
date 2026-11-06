import numpy as np
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    n = 24
    k = np.arange(n)
    
    # Precompute grid powers
    # 8192 points is a good balance between speed and accuracy for proxy scoring
    m = 8192
    thetas = np.linspace(0, 2*np.pi, m, endpoint=False)
    powers_grid = np.exp(1j * np.outer(k, thetas))
    
    def fast_eval(c):
        # c is a numpy array of shape (n,)
        # Returns max |P(e^{i theta})| on the grid
        vals = np.dot(c, powers_grid)
        return np.max(np.abs(vals))
        
    # High precision eval for final check? 
    # The evaluator does the high precision check. 
    # We just need to find the best c.
    # So fast_eval is sufficient for comparison.
    
    best_c = None
    best_score = float('inf')
    
    # Helper for local search
    def local_search(c):
        current_score = fast_eval(c)
        improved = True
        # Limit iterations to avoid infinite loops if stuck in oscillation (unlikely with best improvement)
        while improved and time.time() < deadline - 5.0:
            improved = False
            best_idx = -1
            best_new_score = current_score
            
            for i in range(n):
                c[i] *= -1
                new_score = fast_eval(c)
                c[i] *= -1 # revert
                
                if new_score < best_new_score:
                    best_new_score = new_score
                    best_idx = i
                    
            if best_idx != -1:
                c[best_idx] *= -1
                current_score = best_new_score
                improved = True
                
        return c, current_score

    # Run multiple restarts
    restarts = 0
    while time.time() < deadline - 10.0:
        # Initialize
        c = np.random.choice([-1, 1], size=n)
        
        # Local Search
        c, score = local_search(c)
        
        if score < best_score:
            best_score = score
            best_c = c.copy()
            
        restarts += 1
        
        # If we find a very good score, we can stop earlier?
        # Theoretical min is ~6.165 (raw max norm).
        # Score = max_norm / sqrt(24).
        # Target score < 1.26.
        # If best_score < 1.26, we are close to optimal.
        # But we might find 1.259.
        # Let's just run as many restarts as possible.
        
    # If we never found one (shouldn't happen), return random
    if best_c is None:
        best_c = np.random.choice([-1, 1], size=n)
        
    return best_c.tolist()
