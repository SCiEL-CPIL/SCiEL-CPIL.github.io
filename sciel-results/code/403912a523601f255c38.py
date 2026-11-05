"""Rewritten program: integer local search over {-1,0,1}^30 to minimize |sum c_i/i|."""
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

    # Initial solution: the seed [0]*28 + [1, -1]
    init = [0] * 28 + [1, -1]
    best_coeffs = init[:]
    best_abs = _abs_val(best_coeffs)

    # Multi-start local search
    random.seed(42)
    
    def local_search(coeffs, time_limit):
        """Hill-climbing / first-improvement local search. Returns (coeffs, abs_val)."""
        nonlocal best_coeffs, best_abs
        cur = coeffs[:]
        cur_int = _eval_int(cur)
        cur_abs = abs(cur_int)
        
        # Track best within this local search
        ls_best = cur[:]
        ls_best_abs = cur_abs
        
        improved = True
        while improved:
            improved = False
            # Try flipping each position
            for i in range(N):
                if time.time() > time_limit:
                    break
                # Try setting to each value in {-1, 0, 1} if different
                old = cur[i]
                for new_val in (-1, 0, 1):
                    if new_val == old:
                        continue
                    delta = (new_val - old) * WEIGHTS[i]
                    new_int = cur_int + delta
                    new_abs = abs(new_int)
                    if new_abs < cur_abs:
                        # Accept improvement
                        cur[i] = new_val
                        cur_int = new_int
                        cur_abs = new_abs
                        improved = True
                        if cur_abs < ls_best_abs:
                            ls_best = cur[:]
                            ls_best_abs = cur_abs
                        break  # restart scan from 0 for first-improvement
    
        # Update global best
        if ls_best_abs < best_abs.numerator * LCM // best_abs.denominator if best_abs != 0 else float('inf'):
            # Simpler comparison: compare as Fraction
            pass
        # Actually just compare Fractions properly
        ls_abs_frac = Fraction(ls_best_abs, LCM)
        if best_abs is None or ls_abs_frac < best_abs:
            best_abs = ls_abs_frac
            best_coeffs = ls_best[:]
        
        return cur, Fraction(cur_abs, LCM)

    # Multiple random starts
    num_starts = 0
    while time.time() < deadline:
        # Generate a random starting point
        if num_starts == 0:
            start = init[:]
        else:
            start = [random.choice([-1, 0, 1]) for _ in range(N)]
        
        # Ensure it's not zero sum (though we'll check anyway)
        # Run local search with a time budget
        time_budget = deadline - (time.time() - start_time)
        if time_budget < 1.0:
            break
        
        # Give each local search a fraction of remaining time
        ls_time_limit = start_time + (time.time() - start_time) + min(5.0, time_budget * 0.8)
        
        result, result_abs = local_search(start, ls_time_limit)
        num_starts += 1
        
        # If we found |N| = 1 (i.e., numerator of reduced fraction is 1), that's likely optimal
        # The minimal nonzero |sum c_i/i| for 30 terms... let's check if we can do better
        if result_abs.numerator == 1:
            # This is likely the minimum possible nonzero value
            break
    
    # Final check: ensure the result is valid (nonzero sum, all coeffs in {-1,0,1})
    final = best_coeffs
    final_int = _eval_int(final)
    if final_int == 0:
        # Fallback to seed
        final = [0] * 28 + [1, -1]
    
    # Verify all coefficients are in {-1, 0, 1}
    for i in range(N):
        if final[i] not in (-1, 0, 1):
            final[i] = 0
    
    return final
