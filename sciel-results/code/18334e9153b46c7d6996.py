"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
from collections import defaultdict

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 85  # safety margin before 90s limit
    
    # Start with [91, 180]
    base = set(range(91, 181))
    result = set(base)
    
    # For each a in [1, 90], check if it can be added
    # Forbidden triple: 1/a = 1/b + 1/c with distinct a,b,c
    # This means a < b and a < c, and (b-a)(c-a) = a^2
    
    def can_add(a, current_set):
        """Check if adding a to current_set creates a forbidden triple."""
        a_sq = a * a
        # For each b in current_set where b > a, check if c = a + a_sq/(b-a) exists
        # (b-a)(c-a) = a^2 => c-a = a^2/(b-a) => c = a + a^2/(b-a)
        for b in current_set:
            if b <= a:
                continue
            diff = b - a
            if a_sq % diff != 0:
                continue
            c = a + a_sq // diff
            if c == a or c == b:
                continue
            if c in current_set:
                return False
        # Also need to check: could a be b or c in a triple where some other element is 'a'?
        # No - a is the smallest, so a can only play the role of 'a' in the equation
        # But wait: we also need to check if a could be b or c in a triple
        # 1/x = 1/a + 1/c where x < a: then (a-x)(c-x) = x^2
        # 1/x = 1/b + 1/a where x < a: then (b-x)(a-x) = x^2
        # These are cases where a is NOT the minimum, so some x < a is the minimum
        # We need to check: for each x in current_set where x < a, does there exist y in current_set
        # such that 1/x = 1/a + 1/y or 1/x = 1/y + 1/a?
        # 1/x = 1/a + 1/y => y = ax/(a-x) (requires a > x)
        # 1/x = 1/y + 1/a => same thing, y = ax/(a-x)
        for x in current_set:
            if x >= a:
                continue
            diff = a - x
            if (a * x) % diff != 0:
                continue
            y = (a * x) // diff
            if y == x or y == a:
                continue
            if y in current_set:
                return False
        return True
    
    # Try adding numbers from 1 to 90 in increasing order
    for a in range(1, 91):
        if time.time() > deadline:
            break
        if can_add(a, result):
            result.add(a)
    
    # Try a second pass in different order to see if we can add more
    # Sometimes order matters
    for a in range(90, 0, -1):
        if time.time() > deadline:
            break
        if a not in result and can_add(a, result):
            result.add(a)
    
    # Local improvement: try swapping out one element for another
    # Try to remove an element and add two others
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for remove in sorted(result):
            if time.time() > deadline:
                break
            trial = set(result) - {remove}
            # Try to add back some small numbers
            for add in range(1, 91):
                if add in trial or add == remove:
                    continue
                if can_add(add, trial):
                    trial.add(add)
                    # Try to add one more
                    for add2 in range(1, 91):
                        if add2 in trial or add2 == remove or add2 == add:
                            continue
                        if can_add(add2, trial):
                            trial.add(add2)
                            if len(trial) > len(result):
                                result = trial
                                improved = True
                                break
                    if improved:
                        break
            if improved:
                break
    
    return sorted(result)
# EVOLVE-BLOCK-END
