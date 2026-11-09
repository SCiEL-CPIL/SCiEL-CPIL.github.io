import time
import math
import random

def max_ap_discrepancy(signs):
    n = len(signs)
    max_abs = 0
    for d in range(1, n + 1):
        for start in range(d):
            max_ending_here = 0
            max_so_far = 0
            min_ending_here = 0
            min_so_far = 0
            idx = start
            while idx < n:
                val = signs[idx]
                # Max subarray sum
                max_ending_here = max(0, max_ending_here + val)
                if max_ending_here > max_so_far:
                    max_so_far = max_ending_here
                # Min subarray sum
                min_ending_here = min(0, min_ending_here + val)
                if min_ending_here < min_so_far:
                    min_so_far = min_ending_here
                idx += d
            if max_so_far > max_abs:
                max_abs = max_so_far
            if -min_so_far > max_abs:
                max_abs = -min_so_far
    return max_abs

def construct():
    n = 256
    # Start with Thue-Morse as a good initial guess
    signs = []
    for i in range(1, n + 1):
        s2 = bin(i).count('1')
        signs.append(1 if s2 % 2 == 0 else -1)
    
    current_max = max_ap_discrepancy(signs)
    best_signs = signs[:]
    best_max = current_max
    
    start_time = time.time()
    timeout = 55 # 5s buffer
    
    # Simulated Annealing
    T = 10.0
    cooling = 0.999
    
    while (time.time() - start_time) < timeout:
        # Pick a random index to flip
        i = random.randint(0, n - 1)
        signs[i] *= -1
        new_max = max_ap_discrepancy(signs)
        
        delta = new_max - current_max
        if delta < 0:
            current_max = new_max
            if current_max < best_max:
                best_max = current_max
                best_signs = signs[:]
        else:
            # Accept with probability exp(-delta / T)
            if random.random() < math.exp(-delta / T):
                current_max = new_max
            else:
                signs[i] *= -1 # Revert
        
        T *= cooling
        if T < 0.01:
            T = 10.0 # Restart? Or just keep going? 
            # Actually, for discrepancy, we want to get out of local minima.
            # Simple restart if stuck?
            # Let's just let T cool down to 0.01 and stop or restart.
            # Given the time limit, restarting SA might be good.
            # But let's just continue with low T for now, or restart if no improvement for a while.
            # For simplicity, let's just run until time runs out.
            
    return best_signs
