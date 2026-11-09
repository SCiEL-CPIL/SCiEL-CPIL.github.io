"""Find Egyptian fraction of 5/31 minimizing the largest denominator."""
# EVOLVE-BLOCK-START
import time
from fractions import Fraction
from itertools import combinations

TARGET = Fraction(5, 31)
MAX_DENOM = 10_000_000
MAX_TERMS = 256
DEADLINE = 55.0

def construct():
    start_time = time.time()
    
    def time_left():
        return DEADLINE - (time.time() - start_time)
    
    # First, get a baseline greedy solution
    r = TARGET
    greedy_vals = []
    while r > 0:
        d = (r.denominator + r.numerator - 1) // r.numerator
        greedy_vals.append(d)
        r -= Fraction(1, d)
    greedy_max = max(greedy_vals)
    
    best_solution = greedy_vals
    best_max = greedy_max
    
    # Try to find a solution with smaller max denominator
    # Use a recursive search with strong pruning
    
    def search(remaining, min_d, max_d, used, current_vals):
        nonlocal best_solution, best_max
        if time_left() < 0.5:
            return False
        if remaining == 0:
            if current_vals:
                m = max(current_vals)
                if m < best_max:
                    best_max = m
                    best_solution = list(current_vals)
            return True
        if not current_vals and remaining > 0:
            pass
        
        # Pruning: minimum number of terms needed
        # Each term <= 1/min_d, so need at least remaining * min_d terms
        min_terms_needed = (remaining.numerator * min_d + remaining.denominator - 1) // remaining.denominator
        # Max terms we can add
        max_terms_available = MAX_TERMS - len(current_vals)
        if min_terms_needed > max_terms_available:
            return False
        
        # Next denominator bounds
        # Lower: 1/d <= remaining => d >= ceil(1/remaining)
        min_possible = (remaining.denominator + remaining.numerator - 1) // remaining.numerator
        if min_possible < min_d:
            min_possible = min_d
        
        # Upper: we need to be able to complete with remaining terms
        # Sum of next (max_terms_available) terms, each <= 1/(min_possible + i) for i=0..
        # Simpler: 1/d >= remaining / max_terms_available => d <= max_terms_available / remaining
        max_possible = (max_terms_available * remaining.numerator + remaining.denominator - 1) // remaining.denominator
        if max_possible > max_d:
            max_possible = max_d
        
        if min_possible > max_possible:
            return False
        
        # Try denominators in order (prefer smaller to find compact solutions)
        for d in range(min_possible, max_possible + 1):
            if d in used:
                continue
            if d > max_d:
                break
            new_rem = remaining - Fraction(1, d)
            if new_rem < 0:
                break
            used.add(d)
            current_vals.append(d)
            found = search(new_rem, d + 1, max_d, used, current_vals)
            used.remove(d)
            current_vals.pop()
            if found:
                return True
        
        return False
    
    # Try increasing max_d bounds
    # Start with small bounds
    bounds_to_try = []
    # Add some strategic bounds
    for b in [10, 15, 20, 25, 30, 40, 50, 75, 100, 150, 200, 300, 500, 750, 1000]:
        if b < best_max:
            bounds_to_try.append(b)
    
    # Also try bounds near the current best if it's not too large
    if best_max <= 5000:
        for b in range(min(best_max - 1, 2000), max(best_max - 200, 100), -50):
            if b < best_max and b not in bounds_to_try:
                bounds_to_try.append(b)
        bounds_to_try.sort()
    
    for bound in bounds_to_try:
        if time_left() < 2.0:
            break
        if bound >= best_max:
            continue
        used = set()
        search(TARGET, 1, bound, used, [])
    
    # Verify
    s = sum(Fraction(1, d) for d in best_solution)
    if s != TARGET:
        return greedy_vals
    
    return best_solution
# EVOLVE-BLOCK-END
