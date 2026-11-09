import time
import random
import math

def construct():
    start_time = time.time()
    deadline = 85.0  # 5s margin for 90s limit
    
    n = 30
    recip = [1.0 / i for i in range(1, n + 1)]
    
    best_sum = float('inf')
    best_coeffs = [0] * n
    
    # Theoretical minimum non-zero value is 1/LCM(1..30)
    # LCM(1..30) = 2329089562800
    # 1/LCM ~ 4.2935e-13
    theoretical_min = 4.2935e-13
    
    # Simulated Annealing
    T_start = 0.1
    T_end = 1e-7
    cooling = 0.9999
    
    max_restarts = 500
    steps_per_restart = 20000
    
    for _ in range(max_restarts):
        if time.time() > deadline:
            break
        if best_sum < theoretical_min * 1.1:
            break
            
        # Initialize random coefficients
        coeffs = [random.choice([-1, 0, 1]) for _ in range(n)]
        
        # Calculate initial sum
        current_sum = sum(c * r for c, r in zip(coeffs, recip))
        current_abs = abs(current_sum)
        
        if current_abs < best_sum:
            best_sum = current_abs
            best_coeffs = coeffs[:]
            
        T = T_start
        for _ in range(steps_per_restart):
            if time.time() > deadline:
                return best_coeffs
            if best_sum < theoretical_min * 1.1:
                return best_coeffs
                
            # Pick a random index
            idx = random.randint(0, n - 1)
            current_val = coeffs[idx]
            
            # Pick a new value different from current
            # Efficient way: pick from {-1, 0, 1}
            # If it matches, pick again or force
            new_val = random.choice([-1, 0, 1])
            if new_val == current_val:
                # Force a change
                # If current is -1, pick 0 or 1
                # If current is 0, pick -1 or 1
                # If current is 1, pick -1 or 0
                # Simplest: just loop until different (max 2 tries usually)
                new_val = random.choice([-1, 0, 1])
                if new_val == current_val:
                    new_val = -1 if current_val != -1 else 1
            
            delta = (new_val - current_val) * recip[idx]
            new_sum = current_sum + delta
            new_abs = abs(new_sum)
            
            # Update global best
            if new_abs < best_sum:
                best_sum = new_abs
                best_coeffs = coeffs[:]
                
            # Acceptance
            if new_abs < current_abs:
                # Accept improvement
                coeffs[idx] = new_val
                current_sum = new_sum
                current_abs = new_abs
            else:
                # Accept worse with probability
                delta_E = new_abs - current_abs
                if delta_E == 0:
                    accept = True
                else:
                    # exp(-delta_E / T)
                    # If delta_E is large relative to T, prob is 0
                    if delta_E > 50 * T:
                        accept = False
                    else:
                        accept = random.random() < math.exp(-delta_E / T)
                
                if accept:
                    coeffs[idx] = new_val
                    current_sum = new_sum
                    current_abs = new_abs
            
            T *= cooling
            if T < T_end:
                T = T_end

    return best_coeffs
