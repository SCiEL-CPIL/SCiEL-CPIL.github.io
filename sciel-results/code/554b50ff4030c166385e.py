"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from math import gcd
from functools import reduce
from fractions import Fraction

def construct():
    deadline = time.time() + 85  # 90s limit, keep 5s margin
    
    # Precompute L = lcm(1, ..., 30) and weights
    def lcm(a, b):
        return a * b // gcd(a, b)
    
    L = reduce(lcm, range(1, 31))
    # weights[i] corresponds to coefficient of 1/(i+1), so weight = L/(i+1)
    weights = [L // (i + 1) for i in range(30)]
    
    # The sum S = sum(c_i / (i+1)) = sum(c_i * weights[i]) / L
    # So minimizing |S| is equivalent to minimizing |sum(c_i * weights[i])|
    # where c_i in {-1, 0, 1}
    
    # Use simulated annealing on the integer weights
    # State: coeffs (list of 30 values in {-1, 0, 1})
    # Objective: minimize |sum(c_i * weights[i])|, with constraint that it's nonzero
    
    def compute_integer_sum(coeffs):
        """Compute the integer sum sum(c_i * weights[i])."""
        s = 0
        for i, c in enumerate(coeffs):
            if c != 0:
                s += c * weights[i]
        return s
    
    def abs_val(v):
        return abs(v)
    
    # Simulated annealing
    best_sum = None
    best_coeffs = None
    
    def update_best(coeffs, int_sum):
        nonlocal best_sum, best_coeffs
        if int_sum == 0:
            return
        if best_sum is None or abs(int_sum) < abs(best_sum):
            best_sum = int_sum
            best_coeffs = list(coeffs)
    
    random.seed(12345)
    
    # Multiple restarts with simulated annealing
    num_restarts = 50
    for restart in range(num_restarts):
        if time.time() > deadline:
            break
        
        # Random initial state
        coeffs = [random.choice([-1, 0, 1]) for _ in range(30)]
        current_sum = compute_integer_sum(coeffs)
        if current_sum == 0:
            # Perturb
            idx = random.randint(0, 29)
            coeffs[idx] = random.choice([-1, 1])
            current_sum = compute_integer_sum(coeffs)
        
        update_best(coeffs, current_sum)
        
        # Simulated annealing parameters
        T_start = max(abs(current_sum), 1000)
        T_end = 1
        num_iterations = 5000
        cooling_rate = (T_end / T_start) ** (1.0 / num_iterations) if T_start > 0 else 1.0
        
        T = T_start
        for it in range(num_iterations):
            if time.time() > deadline:
                break
            
            # Pick a random index to change
            i = random.randint(0, 29)
            old_val = coeffs[i]
            
            # Try a new value different from old
            new_val = random.choice([-1, 0, 1])
            if new_val == old_val:
                continue
            
            coeffs[i] = new_val
            new_sum = compute_integer_sum(coeffs)
            
            # Compute energy difference
            old_abs = abs(current_sum) if current_sum != 0 else 0
            new_abs = abs(new_sum) if new_sum != 0 else 10**18  # Penalize zero
            
            if new_sum == 0:
                # Reject zero sum
                coeffs[i] = old_val
                T *= cooling_rate
                continue
            
            delta = new_abs - old_abs
            
            if delta < 0:
                # Accept improvement
                current_sum = new_sum
                update_best(coeffs, current_sum)
            else:
                # Accept with probability exp(-delta/T)
                if T > 0:
                    import math
                    prob = math.exp(-delta / T)
                    if random.random() < prob:
                        current_sum = new_sum
                else:
                    coeffs[i] = old_val
            
            T *= cooling_rate
        
        # After SA, do local search to polish
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for i in range(30):
                if time.time() > deadline:
                    break
                old_val = coeffs[i]
                for new_val in (-1, 0, 1):
                    if new_val == old_val:
                        continue
                    coeffs[i] = new_val
                    new_sum = compute_integer_sum(coeffs)
                    if new_sum == 0:
                        coeffs[i] = old_val
                        continue
                    if abs(new_sum) < abs(current_sum):
                        current_sum = new_sum
                        improved = True
                        break
                    else:
                        coeffs[i] = old_val
                if improved:
                    update_best(coeffs, current_sum)
                    break
        
        update_best(coeffs, current_sum)
    
    # Additional targeted local searches from structured starts
    structured_starts = []
    # All ones
    structured_starts.append([1] * 30)
    # All -1
    structured_starts.append([-1] * 30)
    # Alternating
    structured_starts.append([1 if i % 2 == 0 else -1 for i in range(30)])
    # First 15 ones, last 15 -1s
    structured_starts.append([1] * 15 + [-1] * 15)
    # Sparse: only a few nonzeros
    for _ in range(10):
        c = [0] * 30
        for _ in range(random.randint(3, 8)):
            idx = random.randint(0, 29)
            c[idx] = random.choice([-1, 1])
        structured_starts.append(c)
    
    for start in structured_starts:
        if time.time() > deadline:
            break
        coeffs = list(start)
        current_sum = compute_integer_sum(coeffs)
        if current_sum == 0:
            idx = random.randint(0, 29)
            coeffs[idx] = random.choice([-1, 1])
            current_sum = compute_integer_sum(coeffs)
        
        update_best(coeffs, current_sum)
        
        # Local search
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for i in range(30):
                if time.time() > deadline:
                    break
                old_val = coeffs[i]
                for new_val in (-1, 0, 1):
                    if new_val == old_val:
                        continue
                    coeffs[i] = new_val
                    new_sum = compute_integer_sum(coeffs)
                    if new_sum == 0:
                        coeffs[i] = old_val
                        continue
                    if abs(new_sum) < abs(current_sum):
                        current_sum = new_sum
                        improved = True
                        break
                    else:
                        coeffs[i] = old_val
                if improved:
                    update_best(coeffs, current_sum)
                    break
        
        update_best(coeffs, current_sum)
    
    if best_coeffs is None:
        # Fallback: return a valid nonzero sum
        return [0] * 28 + [1, -1]
    
    return best_coeffs
# EVOLVE-BLOCK-END
