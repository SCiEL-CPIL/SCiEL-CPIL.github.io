"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from fractions import Fraction

def _has_3ap(s):
    """Check if set s contains any non-constant 3-term AP."""
    sl = sorted(s)
    n = len(sl)
    for i in range(n):
        for j in range(i+1, n):
            a = sl[i]
            b = sl[j]
            c = 2*b - a
            if c > b and c in s:
                return True
    return False

def _recip_sum(s):
    """Compute exact reciprocal sum as Fraction."""
    total = Fraction(0, 1)
    for x in s:
        total += Fraction(1, x)
    return total

def _can_add(s, x):
    """Check if x can be added to s without creating a 3-AP."""
    for a in s:
        # a, x, c: need 2x - a in s and 2x - a > x
        c = 2*x - a
        if c > x and c in s:
            return False
    for b in s:
        # a, b, x: need 2b - a = x, i.e., a = 2b - x, need a < b and a in s
        a = 2*b - x
        if a < b and a in s:
            return False
    return True

def _greedy_solution():
    """Greedy: add elements in order, prioritize by reciprocal value (smallest first)."""
    selected = set()
    # Try adding in order 1, 2, 3, ... (maximizes reciprocal sum greedily)
    for x in range(1, 257):
        if _can_add(selected, x):
            selected.add(x)
    return selected

def _local_search(initial_set, time_limit):
    """Local search with removal and re-optimization, plus swaps."""
    deadline = time.time() + time_limit
    s = set(initial_set)
    best_score = _recip_sum(s)
    best_s = set(s)
    
    while time.time() < deadline - 2.0:
        improved = False
        
        # Phase 1: Try adding any element from [1,256] not in s
        for x in range(1, 257):
            if x not in s and _can_add(s, x):
                s.add(x)
                new_score = _recip_sum(s)
                if new_score > best_score:
                    best_score = new_score
                    best_s = set(s)
                    improved = True
                else:
                    s.discard(x)
        
        # Phase 2: Try removing one element and re-optimizing
        for x in list(s):
            s.discard(x)
            # Try to add back any element that was blocked
            for y in range(1, 257):
                if y not in s and _can_add(s, y):
                    s.add(y)
            new_score = _recip_sum(s)
            if new_score > best_score:
                best_score = new_score
                best_s = set(s)
                improved = True
            else:
                # Restore
                s = set(best_s)
        
        # Phase 3: Try swaps - remove one, add another
        for x in list(s):
            s.discard(x)
            for y in range(1, 257):
                if y not in s and _can_add(s, y):
                    s.add(y)
                    new_score = _recip_sum(s)
                    if new_score > best_score:
                        best_score = new_score
                        best_s = set(s)
                        improved = True
                    s.discard(y)
            s.add(x)  # restore
        
        if not improved:
            break
    
    return best_s

def construct():
    start_time = time.time()
    time_limit = 55.0  # Safety margin before 60s kill
    
    # Get greedy solution
    s = _greedy_solution()
    
    # Local search with time budget
    s = _local_search(s, time_limit)
    
    # Final check
    if not s:
        return [1]
    
    return sorted(s)
# EVOLVE-BLOCK-END
