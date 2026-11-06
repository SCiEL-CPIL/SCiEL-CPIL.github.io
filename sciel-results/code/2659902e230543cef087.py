"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from fractions import Fraction

def _greedy_complete(r, start_d, max_denom, used_set):
    """Complete r using greedy, all denominators >= start_d, <= max_denom, distinct."""
    values = []
    r = Fraction(r)
    d_min = start_d
    while r > 0:
        # Greedy: pick smallest d >= d_min such that 1/d <= r
        # d >= ceil(1/r)
        if r.numerator == 0:
            break
        d = (r.denominator + r.numerator - 1) // r.numerator
        if d < d_min:
            d = d_min
        if d > max_denom:
            return None  # Can't complete within max_denom
        if d in used_set:
            # Try next
            d += 1
            while d <= max_denom and d in used_set:
                d += 1
            if d > max_denom:
                return None
        values.append(d)
        used_set.add(d)
        r -= Fraction(1, d)
        d_min = d + 1
    return values

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s
    
    target = Fraction(5, 31)
    best_solution = None
    best_max_denom = float('inf')
    
    # Minimum possible denominator: 1/d <= 5/31 => d >= 31/5 = 6.2, so d >= 7
    min_d = 7
    max_d_upper = 10000000
    
    # Try starting denominators from min_d upward
    # For each, do greedy completion and track max denominator
    # The first denominator is the largest term, so we want it as small as possible
    # but the greedy completion may introduce larger denominators
    
    # Strategy: try many first denominators, use greedy, record max
    # Also try some bounded search for better completions
    
    # Phase 1: Greedy with various starting denominators
    for first_d in range(min_d, min_d + 5000):
        if time.time() > deadline:
            break
        
        remainder = target - Fraction(1, first_d)
        if remainder < 0:
            break  # first_d too small, 1/first_d > target
        
        used = {first_d}
        completion = _greedy_complete(remainder, first_d + 1, max_d_upper, used)
        
        if completion is not None:
            all_denoms = [first_d] + completion
            max_denom = max(all_denoms)
            if max_denom < best_max_denom:
                best_max_denom = max_denom
                best_solution = all_denoms
    
    # Phase 2: If we have a good solution, try to improve by searching
    # around the best first_d with bounded DFS
    if best_solution is not None:
        best_first = best_solution[0]
        # Try first denominators near the best
        search_range = range(max(min_d, best_first - 20), best_first + 20)
        for first_d in search_range:
            if time.time() > deadline:
                break
            
            remainder = target - Fraction(1, first_d)
            if remainder < 0:
                continue
            
            used = {first_d}
            # Try greedy but with a cap on max denominator to see if we can do better
            max_allowed = best_max_denom - 1  # Try to beat current best
            if max_allowed < first_d + 1:
                continue
                
            completion = _greedy_complete(remainder, first_d + 1, max_allowed, used)
            if completion is not None:
                all_denoms = [first_d] + completion
                max_denom = max(all_denoms)
                if max_denom < best_max_denom:
                    best_max_denom = max_denom
                    best_solution = all_denoms
    
    # Phase 3: Try a few more random starting points if time remains
    if time.time() < deadline - 5.0:
        import random
        random.seed(42)
        for _ in range(10000):
            if time.time() > deadline:
                break
            first_d = random.randint(min_d, min_d + 50000)
            remainder = target - Fraction(1, first_d)
            if remainder < 0:
                continue
            used = {first_d}
            completion = _greedy_complete(remainder, first_d + 1, best_max_denom - 1, used)
            if completion is not None:
                all_denoms = [first_d] + completion
                max_denom = max(all_denoms)
                if max_denom < best_max_denom:
                    best_max_denom = max_denom
                    best_solution = all_denoms
    
    if best_solution is None:
        # Fallback: pure greedy from 7
        r = target
        values = []
        while r > 0:
            d = (r.denominator + r.numerator - 1) // r.numerator
            values.append(d)
            r -= Fraction(1, d)
        best_solution = values
    
    return best_solution
# EVOLVE-BLOCK-END
