N = 100

# EVOLVE-BLOCK-START
import numpy as np
import scipy.optimize
import time

def construct():
    start_time = time.time()
    best_h = None
    best_score = float('inf')
    
    # Run several random restarts to escape local minima
    for i in range(30):
        if time.time() - start_time > 100:
            break
            
        h0 = np.random.rand(N)
        h0 = h0 / np.sum(h0) * (N/2)
        
        def objective(h_raw):
            # Bounds penalty
            pen_bounds = 1000 * (np.sum(np.maximum(0, h_raw - 1)) + np.sum(np.maximum(0, -h_raw)))
            
            h = np.clip(h_raw, 0, 1)
            g = 1 - h
            c = np.correlate(h, g, mode='full')
            
            # Exclude shift 0 (index N-1)
            # c indices: 0 to 2N-2
            # Shift 0 is at N-1
            max_c = max(np.max(c[:N-1]), np.max(c[N:]))
            
            # Sum penalty
            pen_sum = 1000 * abs(np.sum(h) - N/2)
            
            return max_c + pen_bounds + pen_sum

        # Use Nelder-Mead for derivative-free optimization
        res = scipy.optimize.minimize(objective, h0, method='Nelder-Mead', 
                                      options={'maxiter': 2000, 'xatol': 1e-5, 'fatol': 1e-5})
        
        h_opt = np.clip(res.x, 0, 1)
        
        # Ensure sum is exactly N/2
        current_sum = np.sum(h_opt)
        diff = N/2 - current_sum
        if abs(diff) > 1e-9:
            # Distribute diff to all elements
            h_opt += diff / N
            h_opt = np.clip(h_opt, 0, 1)
            
            # If clipping changed the sum, adjust again (simple iteration)
            for _ in range(10):
                current_sum = np.sum(h_opt)
                diff = N/2 - current_sum
                if abs(diff) < 1e-12:
                    break
                h_opt += diff / N
                h_opt = np.clip(h_opt, 0, 1)
                
            # Final precise fix: adjust one element if still off
            current_sum = np.sum(h_opt)
            diff = N/2 - current_sum
            if abs(diff) > 1e-12:
                # Find index with most room
                idx = np.argmax(1 - h_opt) if diff > 0 else np.argmax(h_opt)
                h_opt[idx] += diff
                h_opt[idx] = np.clip(h_opt[idx], 0, 1)

        # Calculate true score
        g = 1 - h_opt
        c = np.correlate(h_opt, g, mode='full')
        max_c = max(np.max(c[:N-1]), np.max(c[N:]))
        
        if max_c < best_score:
            best_score = max_c
            best_h = h_opt

    return best_h
# EVOLVE-BLOCK-END
