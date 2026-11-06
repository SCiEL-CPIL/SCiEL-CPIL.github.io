import numpy as np
import time
import random

def count_nonzero(arr):
    return np.count_nonzero(arr)

def square_nonzero_count(coeffs):
    # coeffs is a list of 12 integers
    # We assume the support is 0..11
    # If we allow gaps, we need to handle indices.
    # For now, stick to 0..11 support.
    sq = np.convolve(coeffs, coeffs)
    return count_nonzero(sq)

def construct():
    start_time = time.time()
    deadline = start_time + 50.0 # 50s margin
    
    best_count = 23
    best_coeffs = [1]*12
    
    # Try alternating signs
    alt = [1 if i%2==0 else -1 for i in range(12)]
    c = square_nonzero_count(alt)
    if c < best_count:
        best_count = c
        best_coeffs = alt.copy()
        
    # Simulated annealing
    # We'll use a temperature schedule and accept worse solutions with some probability
    
    for restart in range(200): # Max restarts, but limited by time
        if time.time() > deadline:
            break
            
        # Initialize coeffs
        init_strategy = restart % 4
        if init_strategy == 0:
            coeffs = [1]*12
        elif init_strategy == 1:
            coeffs = [1 if i%2==0 else -1 for i in range(12)]
        elif init_strategy == 2:
            coeffs = [random.randint(-2, 2) for _ in range(12)]
            # Ensure 12 non-zeros
            for i in range(12):
                if coeffs[i] == 0:
                    coeffs[i] = 1
        else:
            coeffs = [random.randint(-64, 64) for _ in range(12)]
            for i in range(12):
                if coeffs[i] == 0:
                    coeffs[i] = 1
                    
        current_count = square_nonzero_count(coeffs)
        if current_count < best_count:
            best_count = current_count
            best_coeffs = coeffs.copy()
            
        # Simulated annealing
        T = 1.0
        T_min = 0.01
        T_factor = 0.999
        
        for step in range(10000):
            if time.time() > deadline:
                break
                
            # Pick a random index to modify
            idx = random.randint(0, 11)
            old_val = coeffs[idx]
            
            # Try a new value
            if random.random() < 0.5:
                new_val = random.randint(-64, 64)
            else:
                new_val = old_val + random.randint(-3, 3)
            
            if new_val == 0 or new_val == old_val or new_val < -64 or new_val > 64:
                continue
                
            coeffs[idx] = new_val
            new_count = square_nonzero_count(coeffs)
            
            # Accept if better, or with probability exp(-(new_count - current_count)/T) if worse
            if new_count < current_count:
                current_count = new_count
                if new_count < best_count:
                    best_count = new_count
                    best_coeffs = coeffs.copy()
            else:
                # Metropolis criterion
                delta = new_count - current_count
                if delta > 0 and random.random() < np.exp(-delta / T):
                    current_count = new_count
                else:
                    coeffs[idx] = old_val
            
            T *= T_factor
            if T < T_min:
                T = T_min
                
        if best_count == 12: # Theoretical min?
            break

    # Return the best coeffs found
    # Ensure it meets the criteria
    # Length 12, all non-zero, in [-64, 64]
    # The function returns the coefficient list.
    # The evaluator expects a list of length <= 41.
    # If our support is 0..11, the list is length 12.
    # This is valid.
    
    # One check: "nonzero final coefficient".
    # Our list ends at index 11. If coeffs[11] != 0, it's fine.
    # We ensured all are non-zero.
    
    return best_coeffs
