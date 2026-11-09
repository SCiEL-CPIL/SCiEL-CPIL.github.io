"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import math
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin
    
    n = 30
    # Calculate LCM of 1..30
    lcm_val = 1
    for i in range(1, n + 1):
        lcm_val = lcm_val * i // gcd(lcm_val, i)
        
    # Weights for integer arithmetic: S = sum(c_i / i) = (1/LCM) * sum(c_i * LCM/i)
    weights = [lcm_val // i for i in range(1, n + 1)]
    
    best_abs_num = None
    best_coeffs = None
    
    # Initial candidate from seed
    seed_coeffs = [0] * (n - 2) + [1, -1]
    seed_num = sum(c * w for c, w in zip(seed_coeffs, weights))
    if seed_num != 0:
        best_abs_num = abs(seed_num)
        best_coeffs = seed_coeffs[:]
    
    def evaluate(coeffs):
        s = 0
        for c, w in zip(coeffs, weights):
            if c != 0:
                s += c * w
        return s
    
    def sparse_state(num_nonzero=2):
        """Generate a sparse random state with exactly num_nonzero non-zero coefficients."""
        coeffs = [0] * n
        indices = random.sample(range(n), num_nonzero)
        for i in indices:
            coeffs[i] = random.choice([-1, 1])
        return coeffs
    
    def simulated_annealing(init_coeffs, time_limit):
        nonlocal best_abs_num, best_coeffs
        
        current_coeffs = init_coeffs[:]
        current_num = evaluate(current_coeffs)
        current_abs = abs(current_num) if current_num != 0 else float('inf')
        
        # Simulated annealing parameters
        T = 1e6  # Initial temperature (scale with LCM)
        T_min = 1.0
        cooling_rate = 0.9995
        
        while time.time() < time_limit and T > T_min:
            T *= cooling_rate
            # Pick a random index to change
            idx = random.randint(0, n - 1)
            old_val = current_coeffs[idx]
            possible_vals = [v for v in [-1, 0, 1] if v != old_val]
            new_val = random.choice(possible_vals)
            
            current_coeffs[idx] = new_val
            delta = (new_val - old_val) * weights[idx]
            new_num = current_num + delta
            new_abs = abs(new_num) if new_num != 0 else float('inf')
            
            # Acceptance criterion with simulated annealing
            if new_abs < current_abs:
                # Always accept improvement
                current_num = new_num
                current_abs = new_abs
                
                # Update global best
                if best_abs_num is None or current_abs < best_abs_num:
                    best_abs_num = current_abs
                    best_coeffs = current_coeffs[:]
            else:
                # Accept worse solution with probability exp(-delta_E / T)
                if new_abs == float('inf'):
                    # New state is zero - accept with small probability to escape
                    if random.random() < 0.01:
                        current_num = new_num
                        current_abs = new_abs
                    else:
                        current_coeffs[idx] = old_val
                else:
                    delta_E = new_abs - current_abs
                    if random.random() < math.exp(-delta_E / T):
                        current_num = new_num
                        current_abs = new_abs
                    else:
                        current_coeffs[idx] = old_val
            
            # Occasionally do a larger jump to escape local minima
            if random.random() < 0.02:
                for _ in range(random.randint(2, 4)):
                    j = random.randint(0, n - 1)
                    current_coeffs[j] = random.choice([-1, 0, 1])
                current_num = evaluate(current_coeffs)
                current_abs = abs(current_num) if current_num != 0 else float('inf')
        
        return current_coeffs
    
    # Run simulated annealing from seed
    simulated_annealing(seed_coeffs, deadline)
    
    # Then run from various sparse initial states
    restarts = 0
    while time.time() < deadline:
        # Use sparse initial states (2-4 non-zero coefficients)
        num_nz = random.choice([2, 3, 3, 4])
        init = sparse_state(num_nz)
        simulated_annealing(init, deadline)
        restarts += 1
        
        # Also try some denser states occasionally
        if random.random() < 0.1:
            init = [random.choice([-1, 0, 1]) for _ in range(n)]
            simulated_annealing(init, deadline)
            restarts += 1
        
        # Break if we found the minimum possible
        if best_abs_num == 1:
            break
    
    # Return the best coefficients found
    if best_coeffs is None:
        # Fallback to seed
        return seed_coeffs
    return best_coeffs

# EVOLVE-BLOCK-END
