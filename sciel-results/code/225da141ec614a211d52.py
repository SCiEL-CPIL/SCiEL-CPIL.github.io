import time
import random
from fractions import Fraction

_START_TIME = time.time()
_DEADLINE = 75.0

def construct():
    global _START_TIME
    _START_TIME = time.time()
    
    candidates = list(range(1, 23))
    best_set = []
    best_size = 0
    
    # Fast collision check using floats
    def check_collision_float(current_sums, new_recip):
        # current_sums is a set of floats
        # We check if any (s + new_recip) is in current_sums or in the new set
        # To be safe, we check against current_sums and a local set of new sums
        
        new_sums = set()
        for s in current_sums:
            ns = s + new_recip
            # Check if ns is close to any existing sum
            # Efficient check: just add to a set and check size?
            # No, floats are not exact. We need to check distance.
            # But for greedy, we can assume if it's not in the set, it's distinct.
            # However, floating point errors might make s1 + r == s2 even if they are distinct rationals.
            # Given the gap is ~1e-11 and float error is ~1e-16, this is safe.
            
            # We just add to a set. If len(set) != len(old) + len(new), collision.
            # But wait, if s1 + r == s2, then the set size will be smaller.
            # This works for exact arithmetic. For floats, it works if the error doesn't bridge the gap.
            # Gap is 1e-11, error is 1e-16. Safe.
            
            new_sums.add(ns)
            
        all_sums = current_sums.union(new_sums)
        
        if len(all_sums) != len(current_sums) + len(new_sums):
            return False, None
            
        return True, all_sums

    def greedy_solve(order):
        current_sums = {0.0}
        current_set = []
        
        for idx in order:
            recip = 1.0 / idx
            valid, new_sums = check_collision_float(current_sums, recip)
            if valid:
                current_sums = new_sums
                current_set.append(idx)
        
        return current_set

    # Try structured orders
    orders = [
        list(range(1, 23)),
        list(range(22, 0, -1)),
    ]
    
    # Add some random permutations
    random.seed(1)
    for _ in range(10):
        o = list(range(1, 23))
        random.shuffle(o)
        orders.append(o)

    for order in orders:
        if time.time() - _START_TIME > _DEADLINE:
            break
        result = greedy_solve(order)
        if len(result) > best_size:
            best_size = len(result)
            best_set = result

    # Random restarts
    for i in range(5000):
        if time.time() - _START_TIME > _DEADLINE:
            break
        
        order = list(range(1, 23))
        random.shuffle(order)
        result = greedy_solve(order)
        if len(result) > best_size:
            best_size = len(result)
            best_set = result

    if not best_set:
        return [1, 2, 3, 5, 7, 11, 13, 17, 19]

    # Verify with exact arithmetic
    def is_dissociated_exact(indices):
        sums = {Fraction(0)}
        for idx in indices:
            recip = Fraction(1, idx)
            new_sums = set()
            for s in sums:
                ns = s + recip
                if ns in sums or ns in new_sums:
                    return False
                new_sums.add(ns)
            sums = sums.union(new_sums)
        return True

    if not is_dissociated_exact(best_set):
        # If float check failed, try to repair or fall back
        # Simple repair: remove the last element and re-check?
        # Or just use a hardcoded good set if this fails.
        # Given the precision, this is unlikely.
        # Let's try a fallback: return the seed.
        return [1, 2, 3, 5, 7, 11, 13, 17, 19]

    return best_set
