# EVOLVE-BLOCK-START
from fractions import Fraction
import random
import math
import time

def construct():
    random.seed(42)
    
    N = 30
    recips_frac = [Fraction(1, i + 1) for i in range(N)]
    recips_float = [1.0 / (i + 1) for i in range(N)]
    
    def calc_sum_frac(coeff):
        s = Fraction(0, 1)
        for i, c in enumerate(coeff):
            if c != 0:
                s += c * recips_frac[i]
        return s

    def calc_sum_float(coeff):
        s = 0.0
        for i, c in enumerate(coeff):
            if c != 0:
                s += c * recips_float[i]
        return s

    best_exact_sum = None
    best_coeff = None
    best_float_val = float('inf')
    
    def update_best(coeff):
        nonlocal best_exact_sum, best_coeff, best_float_val
        exact_sum = calc_sum_frac(coeff)
        if exact_sum == 0:
            return
        abs_val = abs(exact_sum)
        if best_exact_sum is None or abs_val < best_exact_sum:
            best_exact_sum = abs_val
            best_coeff = coeff[:]
            best_float_val = abs(calc_sum_float(coeff)) # Update float estimate too

    def simulated_annealing(start_coeff, start_time, duration=5.0):
        nonlocal best_exact_sum, best_coeff, best_float_val
        
        current_coeff = start_coeff[:]
        current_sum_float = calc_sum_float(current_coeff)
        
        # If we start with 0, perturb
        if current_sum_float == 0.0:
            idx = random.randint(0, N-1)
            current_coeff[idx] = random.choice([-1, 1])
            current_sum_float = calc_sum_float(current_coeff)
        
        # Initialize best from this run if it's better
        if current_sum_float != 0.0:
            update_best(current_coeff)
            
        T = 0.1
        alpha = 0.995
        iters = 0
        max_iters = 200000 # Adjust based on speed
        
        while iters < max_iters and (time.time() - start_time) < duration:
            # Generate move
            i = random.randint(0, N - 1)
            old_val = current_coeff[i]
            new_val = random.choice([-1, 0, 1])
            if new_val == old_val:
                continue
                
            # Calculate delta
            delta = (new_val - old_val) * recips_float[i]
            new_sum_float = current_sum_float + delta
            
            # Check for zero (float approximation)
            # If new sum is very close to 0, we need to be careful.
            # But we want non-zero.
            
            if new_sum_float == 0.0:
                # Reject moves to exactly 0.0 in float? 
                # It's better to reject to avoid getting stuck at 0.
                continue
                
            # Acceptance
            if abs(new_sum_float) < abs(current_sum_float):
                # Improvement
                current_coeff[i] = new_val
                current_sum_float = new_sum_float
                # Periodically check exact value
                if iters % 1000 == 0:
                    update_best(current_coeff)
            else:
                # Worse move
                # Calculate delta_cost = |new| - |curr|
                delta_cost = abs(new_sum_float) - abs(current_sum_float)
                if delta_cost < 0:
                     # This shouldn't happen if we checked improvement, but safe
                     current_coeff[i] = new_val
                     current_sum_float = new_sum_float
                else:
                    # Boltzmann factor
                    # Avoid overflow
                    if T > 0:
                        prob = math.exp(-delta_cost / T)
                        if random.random() < prob:
                            current_coeff[i] = new_val
                            current_sum_float = new_sum_float
                            
            T *= alpha
            iters += 1
            
        # Final check
        update_best(current_coeff)

    # Run multiple SA instances with different seeds/starts
    # Total time budget 55s
    
    start_total = time.time()
    
    # Initial greedy to get a good starting point
    c_greedy = [0] * N
    s_greedy = 0.0
    for i in range(N - 1, -1, -1):
        best_c = 0
        best_s = s_greedy
        for c in [-1, 0, 1]:
            s_new = s_greedy + c * recips_float[i]
            if abs(s_new) < abs(best_s):
                best_s = s_new
                best_c = c
        c_greedy[i] = best_c
        s_greedy = best_s
    update_best(c_greedy)
    
    # Run SA
    num_runs = 50
    time_per_run = 1.0
    
    for run in range(num_runs):
        if time.time() - start_total > 55.0:
            break
            
        # Generate start
        if run % 2 == 0:
            # Random sparse
            coeff = [0] * N
            num_nonzero = random.randint(5, 20)
            indices = random.sample(range(N), num_nonzero)
            for idx in indices:
                coeff[idx] = random.choice([-1, 1])
        else:
            # Perturb current best
            if best_coeff is not None:
                coeff = best_coeff[:]
                for _ in range(random.randint(1, 5)):
                    idx = random.randint(0, N-1)
                    coeff[idx] = random.choice([-1, 0, 1])
            else:
                coeff = [random.choice([-1, 0, 1]) for _ in range(N)]
                
        simulated_annealing(coeff, time.time(), time_per_run)

    # Fallback
    if best_coeff is None:
        best_coeff = [0] * N
        best_coeff[0] = 1
        
    return best_coeff
# EVOLVE-BLOCK-END
