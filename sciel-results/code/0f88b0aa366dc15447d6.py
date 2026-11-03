import numpy as np
import time

N = 400

def construct():
    # Deadline guard
    start_time = time.time()
    timeout = 110 # 10s margin
    
    # Initialize
    # Start with a random sequence
    x = np.random.choice([1, -1], N)
    
    # Calculate initial cost
    c = np.convolve(x, x, 'full')
    current_max = np.max(np.abs(c))
    best_max = current_max
    best_x = x.copy()
    
    # Simulated Annealing / Hill Climbing
    # Temperature
    T = 1.0
    T_min = 0.01
    cooling = 0.999
    
    while time.time() - start_time < timeout:
        # Pick random index
        i = np.random.randint(N)
        
        # Flip
        x[i] *= -1
        
        # Recalculate
        c_new = np.convolve(x, x, 'full')
        new_max = np.max(np.abs(c_new))
        
        # Acceptance
        if new_max < current_max:
            current_max = new_max
            if new_max < best_max:
                best_max = new_max
                best_x = x.copy()
        else:
            # Revert
            x[i] *= -1
            # If SA, we might accept worse, but for max-min, hill climbing is often good.
            # Let's try simple hill climbing first, it's robust.
            # If stuck, we might need restarts.
            
        # Cool down
        T *= cooling
        
        # If we hit a local minimum (no improvements in last X steps), restart?
        # For now, keep it simple.
        
    return best_x
