"""Rewritten program: Simulated Annealing with Greedy Init and Pair-Swaps to minimize |sum c_i/i|."""
import time
import random
from fractions import Fraction

# Precompute denominators 1..30
N = 30
DENOMS = list(range(1, N + 1))

# LCM of 1..30 for fast integer evaluation
def _lcm(a, b):
    from math import gcd
    return a * b // gcd(a, b)

LCM = 1
for d in DENOMS:
    LCM = _lcm(LCM, d)

# Precompute weights: w_i = LCM // i, so sum c_i * w_i = LCM * sum c_i/i
WEIGHTS = [LCM // d for d in DENOMS]

def _eval_int(coeffs):
    """Return integer S = sum c_i * WEIGHTS[i-1]. This equals LCM * (sum c_i/i)."""
    s = 0
    for i in range(N):
        c = coeffs[i]
        if c != 0:
            s += c * WEIGHTS[i]
    return s

def _abs_val(coeffs):
    """Return |sum c_i/i| as a Fraction."""
    s = _eval_int(coeffs)
    if s == 0:
        return Fraction(0)
    return Fraction(abs(s), LCM)

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # safety margin before 90s kill

    best_coeffs = None
    best_abs = None
    best_int_sum = None

    def update_best(coeffs, int_sum):
        nonlocal best_coeffs, best_abs, best_int_sum
        abs_int_sum = abs(int_sum)
        if best_int_sum is None or abs_int_sum < best_int_sum:
            best_int_sum = abs_int_sum
            best_coeffs = coeffs[:]
            best_abs = Fraction(abs_int_sum, LCM)

    # 1. Greedy Initialization
    # Sort indices by weight (largest to smallest) to cancel large terms first
    indices = list(range(N))
    indices.sort(key=lambda i: WEIGHTS[i], reverse=True)
    
    greedy_coeffs = [0] * N
    current_sum = 0
    for i in indices:
        w = WEIGHTS[i]
        # Choose sign to minimize |current_sum +/- w|
        if abs(current_sum + w) < abs(current_sum - w):
            greedy_coeffs[i] = 1
            current_sum += w
        else:
            greedy_coeffs[i] = -1
            current_sum -= w
            
    # Check if zero (invalid), if so, force a change
    if current_sum == 0:
        # Flip the smallest weight term to make it nonzero
        min_w_idx = min(indices, key=lambda i: WEIGHTS[i])
        greedy_coeffs[min_w_idx] = 1 if greedy_coeffs[min_w_idx] != 1 else -1
        current_sum += 2 * WEIGHTS[min_w_idx] if greedy_coeffs[min_w_idx] == 1 else -2 * WEIGHTS[min_w_idx]
        # Recalculate sum properly to be safe
        current_sum = _eval_int(greedy_coeffs)

    update_best(greedy_coeffs, current_sum)

    # 2. Simulated Annealing
    random.seed(42)
    
    # Parameters
    T = 10000.0  # Initial temperature (in integer sum units)
    T_min = 1.0
    alpha = 0.9995  # Cooling rate
    
    cur_coeffs = greedy_coeffs[:]
    cur_sum = current_sum
    cur_abs = abs(cur_sum)
    
    # Precompute neighbors for single flips? No, N=30 is small.
    
    steps = 0
    while time.time() < deadline:
        # Generate a move
        if random.random() < 0.5:
            # Single flip
            i = random.randint(0, N - 1)
            old_val = cur_coeffs[i]
            # Pick new value from {-1, 0, 1} \ {old_val}
            options = [v for v in (-1, 0, 1) if v != old_val]
            new_val = random.choice(options)
            
            delta = (new_val - old_val) * WEIGHTS[i]
            new_sum = cur_sum + delta
            new_abs = abs(new_sum)
            
            # Acceptance
            delta_E = new_abs - cur_abs
            if delta_E < 0 or random.random() < pow(2.718281828, -delta_E / T):
                cur_coeffs[i] = new_val
                cur_sum = new_sum
                cur_abs = new_abs
                update_best(cur_coeffs, cur_sum)
        else:
            # Pair swap
            i = random.randint(0, N - 1)
            j = random.randint(0, N - 1)
            if i == j:
                continue
            
            old_i = cur_coeffs[i]
            old_j = cur_coeffs[j]
            
            # Pick new values
            opts_i = [v for v in (-1, 0, 1) if v != old_i]
            opts_j = [v for v in (-1, 0, 1) if v != old_j]
            new_i = random.choice(opts_i)
            new_j = random.choice(opts_j)
            
            delta = (new_i - old_i) * WEIGHTS[i] + (new_j - old_j) * WEIGHTS[j]
            new_sum = cur_sum + delta
            new_abs = abs(new_sum)
            
            # Acceptance
            delta_E = new_abs - cur_abs
            if delta_E < 0 or random.random() < pow(2.718281828, -delta_E / T):
                cur_coeffs[i] = new_i
                cur_coeffs[j] = new_j
                cur_sum = new_sum
                cur_abs = new_abs
                update_best(cur_coeffs, cur_sum)
                
        T *= alpha
        if T < T_min:
            # Restart with randomization or just stop?
            # Let's try a random restart from the best found so far with perturbation
            if best_coeffs:
                cur_coeffs = best_coeffs[:]
                # Perturb randomly
                for _ in range(5):
                    k = random.randint(0, N-1)
                    cur_coeffs[k] = random.choice([-1, 0, 1])
                cur_sum = _eval_int(cur_coeffs)
                cur_abs = abs(cur_sum)
                if cur_abs == 0:
                    # Force nonzero
                    cur_coeffs[random.randint(0, N-1)] = 1
                    cur_sum = _eval_int(cur_coeffs)
                    cur_abs = abs(cur_sum)
            T = 10000.0
            # Break if we've been running for a long time and T reset doesn't help?
            # We have a deadline, so we will keep going until deadline.
            
        steps += 1

    # Final check
    if best_coeffs is None:
        # Fallback
        best_coeffs = [0] * 28 + [1, -1]
    
    # Ensure nonzero
    final_sum = _eval_int(best_coeffs)
    if final_sum == 0:
        # Find a way to make it nonzero with minimal increase
        # Just flip the smallest weight term
        min_w_idx = min(range(N), key=lambda i: WEIGHTS[i])
        old_v = best_coeffs[min_w_idx]
        new_v = 1 if old_v != 1 else -1
        best_coeffs[min_w_idx] = new_v
        final_sum = _eval_int(best_coeffs)
        
    return best_coeffs
