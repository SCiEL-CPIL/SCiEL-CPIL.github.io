import time
import random
from math import gcd

def construct():
    start_time = time.time()
    deadline = 85.0
    n = 30
    
    # Calculate LCM of 1..30
    L = 1
    for i in range(1, n + 1):
        L = L * i // gcd(L, i)
        
    # Integer weights: w_i = L / i
    weights = [L // i for i in range(1, n + 1)]
    
    # Initialize with random coefficients in {-1, 0, 1}
    # Ensure non-zero sum
    def random_solution():
        while True:
            coeffs = [random.choice([-1, 0, 1]) for _ in range(n)]
            total = sum(c * w for c, w in zip(coeffs, weights))
            if total != 0:
                return coeffs, total

    best_coeffs, best_total = random_solution()
    best_abs = abs(best_total)
    
    # Simulated Annealing
    # Temperature schedule
    T_start = 1000.0
    T_end = 0.001
    alpha = 0.995
    
    T = T_start
    no_improve_count = 0
    
    while time.time() < deadline - 5.0:
        # Generate a neighbor
        # Pick a random index and change its coefficient
        idx = random.randint(0, n - 1)
        old_val = best_coeffs[idx]
        
        # Possible new values
        if old_val == 0:
            new_val = random.choice([-1, 1])
        else:
            # Flip sign or set to 0
            new_val = random.choice([-1, 0, 1])
            if new_val == old_val:
                new_val = -old_val if old_val != 0 else 1 # Ensure change if possible, but 0->1/-1 is handled above. 
                # Actually simpler: 
                # If 1, try -1 or 0. If -1, try 1 or 0.
                # If 0, try 1 or -1.
                pass 
        
        # Let's just pick a random different value
        options = [v for v in [-1, 0, 1] if v != old_val]
        if not options:
            continue
        new_val = random.choice(options)
        
        # Calculate delta
        delta = (new_val - old_val) * weights[idx]
        new_total = best_total + delta
        
        if new_total == 0:
            # Invalid, reject
            continue
            
        new_abs = abs(new_total)
        
        # Acceptance criterion
        if new_abs < best_abs:
            # Always accept improvement
            best_total = new_total
            best_abs = new_abs
            best_coeffs[idx] = new_val
            no_improve_count = 0
        else:
            # Accept with probability exp(-delta_abs / T)
            # delta_abs is positive here
            prob = 1.0
            if T > 0:
                prob = pow(2.718281828, -new_abs / T) # Using e
            if random.random() < prob:
                best_total = new_total
                best_abs = new_abs
                best_coeffs[idx] = new_val
            else:
                no_improve_count += 1
            
        # Cool down
        T *= alpha
        
        # If T is very small, restart with a new random seed if stuck
        if no_improve_count > 1000 or T < T_end:
             # Restart
             new_c, new_t = random_solution()
             if abs(new_t) < best_abs:
                 best_coeffs = new_c
                 best_total = new_t
                 best_abs = abs(new_t)
             T = T_start
             no_improve_count = 0
             
    # Ensure we return native Python ints
    return [int(c) for c in best_coeffs]
