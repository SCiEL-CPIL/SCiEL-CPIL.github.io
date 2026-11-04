import numpy as np
import time

N = 400

def construct():
    start_time = time.time()
    deadline = 145.0  # Safety margin
    
    # Initialize with random +/- 1
    rng = np.random.RandomState(42)
    h = rng.choice([-1, 1], size=N).astype(np.float64)
    
    def get_psl(h):
        # Compute aperiodic autocorrelation
        c = np.correlate(h, h, mode='full')
        # c[N-1] is the main lobe (N)
        # Sidelobes are the rest
        # We want max absolute value of sidelobes
        # To avoid boundary issues in indexing, we take all except the center
        # c has length 2N-1. Center is N-1.
        # Sidelobes are c[0:N-1] and c[N:2N-1]
        left = c[:N-1]
        right = c[N:]
        if len(left) == 0 and len(right) == 0:
            return 0.0
        return np.max(np.abs(np.concatenate([left, right])))
        
    current_cost = get_psl(h)
    best_cost = current_cost
    best_h = h.copy()
    
    # Simulated Annealing Parameters
    T = 100.0
    T_min = 0.01
    cooling_rate = 0.995
    
    no_improve_count = 0
    max_no_improve = 2000
    
    while time.time() < deadline:
        # Pick a random index to flip
        i = rng.randint(N)
        old_val = h[i]
        h[i] = -old_val
        
        new_cost = get_psl(h)
        
        delta = new_cost - current_cost
        
        # Accept if better or with probability exp(-delta/T)
        if delta < 0 or rng.rand() < np.exp(-delta / T):
            current_cost = new_cost
            no_improve_count = 0
            if new_cost < best_cost:
                best_cost = new_cost
                best_h = h.copy()
                no_improve_count = 0
            else:
                no_improve_count += 1
        else:
            h[i] = old_val
            no_improve_count += 1
            
        # Cool down
        T *= cooling_rate
        if T < T_min:
            T = T_min
            
        # If stuck for too long, restart with a new random sequence
        # This helps escape deep local minima
        if no_improve_count > max_no_improve:
            h = rng.choice([-1, 1], size=N).astype(np.float64)
            current_cost = get_psl(h)
            if current_cost < best_cost:
                best_cost = current_cost
                best_h = h.copy()
            no_improve_count = 0
            # Reheat
            T = 50.0

    return best_h
