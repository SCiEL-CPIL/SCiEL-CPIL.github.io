import numpy as np
import time

N = 400

def construct():
    start_time = time.time()
    deadline = start_time + 100  # 100s safety margin
    
    # Initial guess: all ones
    x = np.ones(N)
    
    def calc_score(x):
        # Autoconvolution
        c = np.correlate(x, x, mode='full')
        max_c = np.max(np.abs(c))
        s = np.sum(x)
        s2 = s * s
        if s2 == 0:
            return float('inf')
        return max_c / s2

    best_x = x.copy()
    best_score = calc_score(best_x)
    
    # Local search: try flipping random signs
    # Since we suspect all-ones is optimal, we just verify a few neighbors.
    rng = np.random.default_rng(42)
    
    iterations = 0
    while time.time() < deadline:
        iterations += 1
        # Flip a random index
        idx = rng.integers(0, N)
        x_new = best_x.copy()
        x_new[idx] *= -1
        
        score_new = calc_score(x_new)
        
        if score_new < best_score:
            best_x = x_new
            best_score = score_new
        # If we found a better one, continue. Otherwise, maybe stop early if many failures?
        # But keep it simple.
        
        if iterations > 1000:
            break

    return best_x
