import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Leave 5s margin for evaluation overhead
    
    n = 30
    
    # Calculate LCM of 1..30 to convert to integer arithmetic
    # LCM(1..30) = 2329089562800
    lcm_val = 1
    for i in range(1, n + 1):
        lcm_val = lcm_val * i // math.gcd(lcm_val, i)
        
    # Precompute integer weights: w[i] = lcm / (i+1)
    # Sum S = sum(c[i] / (i+1)) = (sum(c[i] * w[i])) / lcm
    # Minimizing |S| is equivalent to minimizing |sum(c[i] * w[i])|
    weights = [lcm_val // (i + 1) for i in range(n)]
    
    # Initial state: all zeros is sum 0, which is invalid.
    # Start with a random valid state or a simple non-zero state.
    # c[0]=1, rest 0 -> sum = 1. Integer sum = lcm.
    best_coeffs = [0] * n
    best_coeffs[0] = 1
    best_abs_int = lcm_val  # Corresponds to abs(S) = 1
    
    def get_abs_sum(coeffs):
        s = 0
        for i, c in enumerate(coeffs):
            if c != 0:
                s += c * weights[i]
        return abs(s)
        
    # Simulated Annealing
    # State: coeffs list, current_abs_int
    # Move: pick random index, change value to one of {-1, 0, 1} \ {current}
    
    current_coeffs = [random.choice([-1, 0, 1]) for _ in range(n)]
    # Ensure current state is not zero sum
    current_sum_int = 0
    for i, c in enumerate(current_coeffs):
        current_sum_int += c * weights[i]
    while current_sum_int == 0:
        idx = random.randint(0, n - 1)
        old = current_coeffs[idx]
        new = random.choice([v for v in [-1, 0, 1] if v != old])
        current_coeffs[idx] = new
        current_sum_int += (new - old) * weights[idx]
        
    current_abs_int = abs(current_sum_int)
    
    if current_abs_int < best_abs_int:
        best_abs_int = current_abs_int
        best_coeffs = current_coeffs[:]
        
    # SA Parameters
    T = 10000.0  # Initial temperature
    cooling = 0.999
    min_T = 1.0
    
    while time.time() < deadline:
        if T < min_T:
            # Restart with higher temperature if we've cooled down too much
            # Or just break if we've spent enough time
            # Let's try a few restarts if time permits
            if random.random() < 0.1:
                T = 10000.0
            else:
                break
                
        # Pick a random index
        i = random.randint(0, n - 1)
        old_val = current_coeffs[i]
        
        # Pick a new value different from old_val
        candidates = [v for v in [-1, 0, 1] if v != old_val]
        new_val = random.choice(candidates)
        
        # Calculate delta in integer sum
        # new_sum = current_sum + (new_val - old_val) * weights[i]
        # We track current_sum_int (signed) to allow accurate delta calculation
        # But we only stored abs. Let's keep track of signed sum for efficiency.
        # Refactor: Keep current_sum_int in scope.
        
        delta = (new_val - old_val) * weights[i]
        new_sum_int = current_sum_int + delta
        
        if new_sum_int == 0:
            # Invalid move, revert (don't accept)
            continue
            
        new_abs_int = abs(new_sum_int)
        delta_abs = new_abs_int - current_abs_int
        
        # Accept if improvement (delta_abs < 0)
        # Or accept with probability exp(-delta_abs / T) if worsening
        if delta_abs < 0:
            # Accept improvement
            current_coeffs[i] = new_val
            current_sum_int = new_sum_int
            current_abs_int = new_abs_int
            
            if current_abs_int < best_abs_int:
                best_abs_int = current_abs_int
                best_coeffs = current_coeffs[:]
        else:
            # Deterioration
            # Probability of acceptance
            # To avoid overflow in exp, if delta_abs/T is large, prob is ~0
            # Use log for stability if needed, but T is large initially.
            # If delta_abs is huge, exp(-large) is 0.
            if random.random() < math.exp(-delta_abs / T):
                current_coeffs[i] = new_val
                current_sum_int = new_sum_int
                current_abs_int = new_abs_int
                
        T *= cooling

    return best_coeffs
