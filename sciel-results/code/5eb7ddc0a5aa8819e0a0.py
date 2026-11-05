import random
import time
import math

def calc_max_discrepancy(signs):
    n = len(signs)
    max_abs = 0
    for d in range(1, n):
        for r in range(d):
            curr_sum = 0
            for idx in range(r, n, d):
                curr_sum += signs[idx]
                abs_sum = curr_sum if curr_sum > 0 else -curr_sum
                if abs_sum > max_abs:
                    max_abs = abs_sum
    return max_abs

def construct():
    n = 256
    # Initialize with Thue-Morse
    signs = [1 if bin(i).count('1') % 2 == 0 else -1 for i in range(n)]
    
    start_time = time.time()
    limit_time = start_time + 55.0 # 5s buffer
    
    current_max = calc_max_discrepancy(signs)
    best_signs = signs[:]
    best_max = current_max
    
    # Simulated Annealing parameters
    # We need to estimate a good initial temperature.
    # The energy landscape is discrete. Let's start high.
    T = 10.0
    T_min = 0.01
    # We will run until time limit.
    # We can use a cooling schedule based on iterations or time.
    # Let's do a simple loop:
    # While time < limit:
    #   Pick random index
    #   Flip
    #   Calc new max
    #   If new_max < current_max: accept, update best
    #   Else: accept with prob exp(-(new_max - current_max) / T)
    #   Cool T down slightly
    
    # To make it more effective, we can use a multi-start strategy with SA.
    # But let's stick to one long SA run first.
    
    # Optimization: Pre-compute nothing, just fast loop.
    
    while time.time() < limit_time:
        # Choose a random index to flip
        i = random.randint(0, n - 1)
        old_sign = signs[i]
        signs[i] = -old_sign
        new_max = calc_max_discrepancy(signs)
        
        delta = new_max - current_max
        
        if delta < 0:
            # Better, accept
            current_max = new_max
            if new_max < best_max:
                best_max = new_max
                best_signs = signs[:]
        else:
            # Worse, accept with probability exp(-delta / T)
            if T > 0 and random.random() < math.exp(-delta / T):
                current_max = new_max
            else:
                # Reject, revert
                signs[i] = old_sign
        
        # Cool down
        # Linear cooling might be too slow or fast.
        # Let's try a geometric cooling or a time-based cooling.
        # Since we don't know the total iterations, let's cool based on time remaining.
        # Or just a small factor each step.
        T *= 0.9999
        
        if T < T_min:
            T = T_min
            # If T is too low, we are effectively doing hill climbing.
            # To avoid getting stuck, we can restart with a higher T if no improvement for a while.
            # For simplicity, let's just keep going.
            
    return best_signs
