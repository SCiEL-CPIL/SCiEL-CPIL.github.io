import numpy as np
import time

N = 400

# EVOLVE-BLOCK-START
def construct():
    # Simulated Annealing to minimize sum |C_k|^3 for k=1..N-1
    
    rng = np.random.default_rng(42)
    x = rng.integers(0, 2, N) * 2 - 1
    
    def calc_cost(x):
        # Use np.correlate for exact integer arithmetic
        c = np.correlate(x, x, mode='full')
        # c[N-1] is lag 0.
        # We sum |C_k|^3 for k=1..N-1.
        # Due to symmetry, sum_{k=1}^{N-1} |C_k|^3 is half the sum of all non-zero lags.
        # We just take the positive lags part c[N:]
        pos_lags = c[N:]
        return np.sum(np.abs(pos_lags)**3)

    current_cost = calc_cost(x)
    best_x = x.copy()
    best_cost = current_cost
    
    # SA Parameters
    # Costs are in range of ~10^6 for random.
    # Delta can be ~10^5.
    start_temp = 50000.0
    end_temp = 1.0
    steps = 200000
    
    t_start = time.time()
    time_limit = 110.0
    
    for i in range(steps):
        if time.time() - t_start > time_limit:
            break
            
        # Geometric cooling
        alpha = end_temp / start_temp
        temp = start_temp * (alpha ** (i / steps))
        
        # Random move: flip one random bit
        idx = rng.integers(0, N)
        old_val = x[idx]
        x[idx] = -old_val
        
        new_cost = calc_cost(x)
        delta_cost = new_cost - current_cost
        
        if delta_cost < 0:
            # Accept
            current_cost = new_cost
            if new_cost < best_cost:
                best_cost = new_cost
                best_x = x.copy()
        else:
            # Accept with probability exp(-delta/temp)
            # If temp is very small, exp(-large) is 0.
            if temp > 1e-6:
                prob = np.exp(-delta_cost / temp)
            else:
                prob = 0.0
            
            if rng.random() < prob:
                current_cost = new_cost
            else:
                # Reject
                x[idx] = old_val
                
    return best_x
# EVOLVE-BLOCK-END
