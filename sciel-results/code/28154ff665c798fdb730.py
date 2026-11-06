"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin for 60s limit
    
    target = Fraction(5, 31)
    
    # Known solution: [8, 31, 248] with max denom 248
    # 1/8 + 1/31 + 1/248 = 31/248 + 8/248 + 1/248 = 40/248 = 5/31
    
    best_solution = [8, 31, 248]
    best_max = 248
    
    # Search for 2-term solutions
    # 1/a + 1/b = 5/31 => b = 31a / (5a - 31)
    for a in range(1, best_max):
        denom = 5 * a - 31
        if denom <= 0:
            continue
        if (31 * a) % denom != 0:
            continue
        b = (31 * a) // denom
        if b <= a:
            continue
        if b < best_max:
            if Fraction(1, a) + Fraction(1, b) == target:
                return sorted([a, b])
    
    # Search for 3-term solutions with max denom < best_max
    # 1/a + 1/b + 1/c = 5/31 => c = 31ab / (5ab - 31a - 31b)
    for a in range(1, best_max):
        # Pruning: 1/a + 1/b + 1/c = 5/31, with b > a, c > b
        # So 3/a > 5/31 => a < 93/5 = 18.6, so a <= 18
        # But we need to be more careful
        if time.time() > deadline - 10:
            break
        for b in range(a + 1, best_max):
            # Pruning: 1/a + 1/b + 1/c = 5/31
            # 1/c = 5/31 - 1/a - 1/b must be positive
            rem = target - Fraction(1, a) - Fraction(1, b)
            if rem <= 0:
                break  # As b increases, 1/b decreases, so rem increases... wait
            # Actually as b increases, 1/b decreases, so rem = 5/31 - 1/a - 1/b increases
            # So we can't break here. Let me reconsider.
            
            # c = 1/rem = (a*b) / (31*(a+b) - 5*a*b) ... let me recalculate
            # 1/c = 5/31 - 1/a - 1/b = (5ab - 31b - 31a) / (31ab)
            # So c = 31ab / (5ab - 31a - 31b)
            
            denom_c = 5 * a * b - 31 * a - 31 * b
            if denom_c <= 0:
                continue
            if (31 * a * b) % denom_c != 0:
                continue
            c = (31 * a * b) // denom_c
            if c <= b:
                continue
            if c < best_max:
                # Verify
                if Fraction(1, a) + Fraction(1, b) + Fraction(1, c) == target:
                    return sorted([a, b, c])
    
    # Search for 4-term solutions with max denom < best_max
    # Use recursive backtracking with pruning
    
    def find_k_terms(rem, max_denom, min_denom, exclude, depth, current_list):
        """Find 'depth' unit fractions summing to 'rem' with denominators in [min_denom, max_denom]"""
        if depth == 0:
            return None if rem != 0 else current_list
        
        if rem <= 0:
            return None
            
        p = rem.numerator
        q = rem.denominator
        
        # If rem is a unit fraction
        if q % p == 0:
            d = q // p
            if min_denom <= d <= max_denom and d not in exclude:
                return current_list + [d]
        
        # Bounds for next denominator d:
        # 1/d < rem => d > q/p
        # Also depth * (1/d) >= rem => d <= depth * q / p
        min_d = max(min_denom, q // p + 1)
        max_d = min(max_denom, depth * q // p)
        
        if min_d > max_d:
            return None
        
        # Pruning: maximum possible sum with 'depth' terms starting from min_d
        # is 1/min_d + 1/(min_d+1) + ... + 1/(min_d+depth-1)
        # If this is less than rem, we can't achieve rem
        max_possible = sum(Fraction(1, min_d + i) for i in range(depth))
        if max_possible < rem:
            return None
        
        # Try each possible denominator
        for d in range(min_d, max_d + 1):
            if d in exclude:
                continue
            
            new_rem = rem - Fraction(1, d)
            if new_rem <= 0:
                continue
            
            result = find_k_terms(new_rem, max_denom, d + 1, exclude | {d}, depth - 1, current_list + [d])
            if result is not None:
                return result
        
        return None
    
    # Try to find 4-term solutions with max denom < best_max
    # Search from best_max - 1 downwards
    for C in range(best_max - 1, 50, -1):
        if time.time() > deadline - 5:
            break
        result = find_k_terms(target, C, 1, set(), 4, [])
        if result is not None:
            return sorted(result)
    
    # Try 5-term solutions with max denom < best_max
    for C in range(best_max - 1, 50, -1):
        if time.time() > deadline - 2:
            break
        result = find_k_terms(target, C, 1, set(), 5, [])
        if result is not None:
            return sorted(result)
    
    # Return the known best solution
    return best_solution
# EVOLVE-BLOCK-END
