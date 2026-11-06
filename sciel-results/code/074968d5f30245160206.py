import time
from fractions import Fraction

# Global deadline
_DEADLINE = None

def construct():
    global _DEADLINE
    start_time = time.time()
    _DEADLINE = start_time + 50  # 50s safety margin

    # We established that <8 terms are impossible.
    # Search for minimal k starting from 8.
    for k in range(8, 257):
        if time.time() > _DEADLINE:
            break
        result = _search(Fraction(1, 1), 5, k)
        if result:
            return result
            
    # If we exhaust k or time out, we must return *something* valid.
    # But the task implies a solution exists.
    # If we time out, we might return a partial result? No, must be valid.
    # Let's hope 8 works.
    raise RuntimeError("No solution found within deadline")

def _search(target, min_denom, terms_left):
    """
    Finds 'terms_left' distinct denominators >= min_denom 
    that sum to 'target'.
    Returns a list of denominators or None.
    """
    if time.time() > _DEADLINE:
        return None

    if terms_left == 0:
        if target == 0:
            return []
        else:
            return None
            
    if terms_left == 1:
        if target.numerator == 0:
            return None
        # Check if 1/d = target has an integer solution d
        if target.denominator % target.numerator != 0:
            return None
        d = target.denominator // target.numerator
        if d >= min_denom:
            return [d]
        return None

    # Pruning: Max possible sum with 'terms_left' terms starting from min_denom
    max_sum = Fraction(0, 1)
    for i in range(terms_left):
        max_sum += Fraction(1, min_denom + i)
        
    if max_sum < target:
        return None

    # Determine lower bound for d
    # 1/d <= target => d >= 1/target
    d_min = min_denom
    if target.numerator != 0:
        # d >= 1/target
        # 1/d <= target => d >= 1/target
        # Calculate ceil(1/target)
        # 1/target = target.denominator / target.numerator
        num = target.denominator
        den = target.numerator
        if num % den == 0:
            d_from_target = num // den
        else:
            d_from_target = num // den + 1
        d_min = max(d_min, d_from_target)
        
    # Iterate possible next denominators
    # We break when it's no longer possible to reach target
    for d in range(d_min, 10000000):
        if time.time() > _DEADLINE:
            return None

        term = Fraction(1, d)
        if term > target:
            break
            
        new_target = target - term
        
        # Check if new_target can be formed by terms_left-1 terms > d
        # Max sum of terms_left-1 terms starting from d+1
        max_rest = Fraction(0, 1)
        for i in range(terms_left - 1):
            max_rest += Fraction(1, d + 1 + i)
            
        if new_target > max_rest:
            # If d increases, new_target increases (1/d decreases)
            # and max_rest decreases. So it will only get worse.
            break
            
        res = _search(new_target, d + 1, terms_left - 1)
        if res is not None:
            return [d] + res
            
    return None
