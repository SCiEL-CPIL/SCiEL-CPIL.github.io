import math
import time
from itertools import product

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 5 second safety margin before 90s limit
    
    N = math.factorial(12)
    # 12! = 2^10 * 3^5 * 5^2 * 7^1 * 11^1
    target_exp = [10, 5, 2, 1, 1]
    
    # Generate all divisors of N greater than 12
    divisors = []
    for e2 in range(11):
        for e3 in range(6):
            for e5 in range(3):
                for e7 in range(2):
                    for e11 in range(2):
                        d = (2 ** e2) * (3 ** e3) * (5 ** e5) * (7 ** e7) * (11 ** e11)
                        if d > 12:
                            divisors.append((d, [e2, e3, e5, e7, e11]))
    
    # Sort divisors by value
    divisors.sort(key=lambda x: x[0])
    n_divs = len(divisors)
    
    # Precompute divisor values and exponents for faster access
    div_vals = [d[0] for d in divisors]
    div_exps = [d[1] for d in divisors]
    
    # Use a more efficient backtracking with better pruning
    # Key insight: try larger factors first to reach target product faster
    # Also, we can prune if remaining product can't be achieved
    
    def can_complete(idx, current_exp, remaining_slots):
        """Check if it's possible to complete the factorization from idx onwards"""
        # If we already have the target, we're good
        if current_exp == target_exp:
            return True
        
        # If we've used all slots or no more divisors, check if we can reach target
        if idx < 0 or remaining_slots <= 0:
            return current_exp == target_exp
        
        # Calculate the maximum product we can still achieve
        # by taking the largest possible divisors from idx down to 0
        max_remaining = 1
        slots = remaining_slots
        for i in range(idx, -1, -1):
            if slots <= 0:
                break
            # Check if we can take this divisor
            feasible = True
            for j in range(5):
                if current_exp[j] + div_exps[i][j] > target_exp[j]:
                    feasible = False
                    break
            if feasible:
                max_remaining *= div_vals[i]
                slots -= 1
        
        # If max_remaining is less than what we need, prune
        # What we need is N / product of current_exp
        current_product = 1
        for j in range(5):
            current_product *= (2 ** current_exp[j]) if j == 0 else \
                               (3 ** current_exp[j]) if j == 1 else \
                               (5 ** current_exp[j]) if j == 2 else \
                               (7 ** current_exp[j]) if j == 3 else \
                               (11 ** current_exp[j])
        
        if current_product * max_remaining < N:
            return False
        
        return True
    
    def find_subset(idx, current_exp, count, path):
        """Backtracking to find a valid subset"""
        # Check time deadline
        if time.time() > deadline:
            return False
        
        # Pruning: if count > 12, return False
        if count > 12:
            return False
        
        # Check if current_exp equals target_exp
        if current_exp == target_exp:
            return True
        
        # If idx < 0, we can't add more elements
        if idx < 0:
            return False
        
        # Pruning: check if completion is possible
        if not can_complete(idx, current_exp, 12 - count):
            return False
        
        # Option 1: Pick divisors[idx] first (larger factors first for faster convergence)
        d_val = div_vals[idx]
        d_exp = div_exps[idx]
        new_exp = [current_exp[i] + d_exp[i] for i in range(5)]
        if all(new_exp[i] <= target_exp[i] for i in range(5)):
            path.append(d_val)
            if find_subset(idx - 1, new_exp, count + 1, path):
                return True
            path.pop()
        
        # Option 2: Do not pick divisors[idx]
        if find_subset(idx - 1, current_exp, count, path):
            return True
            
        return False

    # Iterate M from smallest to largest
    # We check if a solution exists using only divisors up to index i
    best_solution = None
    
    for i in range(n_divs):
        # Check time deadline
        if time.time() > deadline:
            break
        
        M = div_vals[i]
        path = []
        
        # Try to find a solution using divisors[0..i]
        if find_subset(i, [0, 0, 0, 0, 0], 0, path):
            # Found a valid solution with maximum factor M
            return path
    
    # If no solution found (should not happen, [N] is a solution)
    return [N]
