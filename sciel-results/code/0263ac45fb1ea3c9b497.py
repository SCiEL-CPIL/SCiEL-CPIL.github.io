import numpy as np
import random
import time
import math

def construct():
    start_time = time.time()
    best_c = None
    best_score = 100  # Max possible is 23, but start high
    
    # We want to minimize the number of non-zero terms in the square
    # Exponents are fixed to 0..11 to minimize the support size of the sumset (23)
    # We optimize coefficients c[0..11]
    
    def get_score(c):
        # c is a list or array of 12 non-zero integers
        # Compute square
        # Using numpy convolution for speed
        conv = np.convolve(c, c)
        return np.count_nonzero(conv)
        
    def simulate_annealing(initial_c, max_time=50):
        c = initial_c.copy()
        current_score = get_score(c)
        T = 5.0  # Initial temperature
        cooling_rate = 0.995
        min_T = 0.01
        
        while time.time() - start_time < max_time and T > min_T:
            # Generate a neighbor
            idx = random.randint(0, 11)
            old_val = c[idx]
            
            # Choose a new value
            # Bias towards small integers as they are more likely to cancel
            if random.random() < 0.7:
                new_val = random.choice([-4, -3, -2, -1, 1, 2, 3, 4])
            else:
                new_val = random.randint(-10, 10)
            
            if new_val == 0:
                continue
            if new_val == old_val:
                continue
            if abs(new_val) > 64:
                continue
                
            c[idx] = new_val
            new_score = get_score(c)
            
            delta_E = new_score - current_score
            
            # Accept if better, or with probability exp(-delta_E/T)
            if delta_E < 0 or random.random() < math.exp(-delta_E / T):
                current_score = new_score
            else:
                c[idx] = old_val
                
            T *= cooling_rate
            
        return c, current_score

    # Run multiple restarts
    num_restarts = 20
    for i in range(num_restarts):
        if time.time() - start_time > 55:
            break
            
        # Initialize
        # Mix of random and structured
        if i % 3 == 0:
            # Alternating signs
            c_init = [1 if j % 2 == 0 else -1 for j in range(12)]
        elif i % 3 == 1:
            # Random small
            c_init = [random.choice([-2, -1, 1, 2]) for _ in range(12)]
        else:
            # Random from wider range
            c_init = [random.randint(-5, 5) for _ in range(12)]
            c_init = [1 if x == 0 else x for x in c_init]
            
        c_res, score_res = simulate_annealing(np.array(c_init, dtype=int))
        
        if score_res < best_score:
            best_score = score_res
            best_c = c_res.copy()
            
        # If we found a very low score, we might stop early?
        # Theoretical min for n=12 is likely around 10-15?
        # 23 is the upper bound for AP exponents.
        if best_score <= 10:
            break

    # Fallback if something went wrong
    if best_c is None:
        best_c = np.array([1]*12, dtype=int)
        
    return [int(x) for x in best_c]
