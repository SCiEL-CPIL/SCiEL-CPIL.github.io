import numpy as np
import time

def construct():
    start_time = time.time()
    timeout = 110.0  # Leave 10s margin for the 120s limit
    
    n = 24
    
    # Generate Golay-Rudin-Shapiro sequence as initial guess
    def gsr(n):
        p, q = [1], [1]
        while len(p) < n:
            p, q = p + q, p + [-c for c in q]
        return p[:n]
    
    best_coeffs = gsr(n)
    
    # Setup fast proxy evaluation
    # 8192 points is a good balance between accuracy and speed for N=24
    M = 8192
    angles = np.linspace(0, 2 * np.pi, M, endpoint=False)
    z = np.exp(1j * angles)
    
    def score(coeffs):
        # coeffs is a list of integers +1 or -1
        # np.polynomial.polynomial.polyval expects [c0, c1, ..., cn]
        vals = np.polynomial.polynomial.polyval(z, coeffs)
        return np.max(np.abs(vals))
        
    best_score = score(best_coeffs)
    
    # Local Search with Random Restarts
    # Strategy: Alternate between improving the current best and random restarts
    # Use First-Improvement Hill Climbing
    
    while time.time() - start_time < timeout:
        # Decide starting point
        if np.random.rand() < 0.7:
            current = best_coeffs[:]
        else:
            current = [1 if np.random.rand() < 0.5 else -1 for _ in range(n)]
            
        current_score = score(current)
        
        # Hill Climb
        # Try flipping bits in random order until no single flip improves
        while time.time() - start_time < timeout:
            improved = False
            indices = np.random.permutation(n)
            for i in indices:
                # Flip bit i
                old_val = current[i]
                current[i] = -old_val
                new_score = score(current)
                if new_score < current_score:
                    current_score = new_score
                    improved = True
                    # Optional: break and restart permutation? 
                    # Usually, continuing with the current permutation is fine.
                else:
                    # Revert
                    current[i] = old_val
                    
            if not improved:
                break
                
        if current_score < best_score:
            best_score = current_score
            best_coeffs = current[:]
            
    return best_coeffs
