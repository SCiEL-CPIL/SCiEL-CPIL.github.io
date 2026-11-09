"""Bootstrap portfolio: MODEL-AND-OPTIMIZE for Egyptian fraction of 5/31."""
# EVOLVE-BLOCK-START
import time
from fractions import Fraction

TARGET = Fraction(5, 31)
MAX_DENOM = 10_000_000
MAX_TERMS = 256
DEADLINE = 60.0  # safety margin

def _greedy_complete(remaining: Fraction, current_max: int, used: set):
    """Complete the Egyptian fraction greedily, ensuring distinct denominators.
    Returns list of denominators or None if fails."""
    result = []
    r = remaining
    last_d = current_max
    while r > 0:
        # Greedy step: ceil(1/r)
        if r.numerator == 0:
            break
        d = (r.denominator + r.numerator - 1) // r.numerator
        # Ensure distinct and > last_d (to keep increasing order)
        while d in used or d <= last_d:
            d += 1
        if d > MAX_DENOM:
            return None
        if len(result) + len(used) >= MAX_TERMS:
            return None
        r -= Fraction(1, d)
        result.append(d)
        last_d = d
    if r != 0:
        return None
    return result

def _solve_single(first_d, used):
    """Given first denominator, greedily complete. Return full list or None."""
    rem = TARGET - Fraction(1, first_d)
    if rem < 0:
        return None
    used2 = set(used) | {first_d}
    comp = _greedy_complete(rem, first_d, used2)
    if comp is None:
        return None
    return [first_d] + comp

def _try_set(denoms):
    """Verify a set of denominators sums to TARGET and meets constraints."""
    if not denoms:
        return False
    if len(denoms) > MAX_TERMS:
        return False
    s = sum(Fraction(1, d) for d in denoms)
    if s != TARGET:
        return False
    if len(set(denoms)) != len(denoms):
        return False
    if any(d < 1 or d > MAX_DENOM for d in denoms):
        return False
    return True

def construct():
    # Known optimal or near-optimal solution from literature/search
    # 5/31 = 1/8 + 1/31 + 1/248
    # Check: 1/8 + 1/31 + 1/248 = 31/248 + 8/248 + 1/248 = 40/248 = 5/31
    candidate = [8, 31, 248]
    
    if _try_set(candidate):
        return candidate
    
    # Fallback if the hardcoded solution is somehow invalid (should not happen)
    # Use a fast greedy with a small search space
    start_time = time.time()
    best_result = None
    best_max_denom = float('inf')
    
    # Try first few denominators
    for first_d in range(6, 20):
        if time.time() - start_time > DEADLINE - 2:
            break
        rem = TARGET - Fraction(1, first_d)
        if rem <= 0:
            continue
        
        # Try second denominators near the greedy choice
        greedy_second = (rem.denominator + rem.numerator - 1) // rem.numerator
        for second_d in range(max(first_d + 1, greedy_second - 10), greedy_second + 10):
            if second_d > MAX_DENOM:
                break
            rem2 = rem - Fraction(1, second_d)
            if rem2 < 0:
                continue
            if rem2 == 0:
                result = [first_d, second_d]
                if _try_set(result):
                    if max(result) < best_max_denom:
                        best_max_denom = max(result)
                        best_result = result[:]
                continue
            
            # Try third denominators near the greedy choice
            greedy_third = (rem2.denominator + rem2.numerator - 1) // rem2.numerator
            for third_d in range(max(second_d + 1, greedy_third - 10), greedy_third + 10):
                if third_d > MAX_DENOM:
                    break
                rem3 = rem2 - Fraction(1, third_d)
                if rem3 < 0:
                    continue
                if rem3 == 0:
                    result = [first_d, second_d, third_d]
                    if _try_set(result):
                        if max(result) < best_max_denom:
                            best_max_denom = max(result)
                            best_result = result[:]
                    continue
                
                # If rem3 is not zero, we could try a 4th term, but [8,31,248] is 3 terms.
                # If we haven't found it, a greedy completion might work but is risky.
                # Given the hardcoded solution, this path is unlikely to be needed.
                pass

    if best_result:
        return best_result
        
    # Final fallback: pure greedy
    r = TARGET
    result = []
    last_d = 0
    while r > 0:
        d = (r.denominator + r.numerator - 1) // r.numerator
        if d <= last_d:
            d = last_d + 1
        result.append(d)
        r -= Fraction(1, d)
        last_d = d
    return result
# EVOLVE-BLOCK-END
