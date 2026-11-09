import math
from functools import reduce

def construct():
    # Calculate LCM of 1 to 24
    def gcd(a, b):
        while b:
            a, b = b, a % b
        return a

    def lcm(a, b):
        return a * b // gcd(a, b)

    L = reduce(lcm, range(1, 25))
    
    # Create list of (value, x) sorted by value ascending
    # value = L / x
    items = []
    for x in range(1, 25):
        items.append((L // x, x))
    
    # Sort by value ascending
    items.sort(key=lambda p: p[0])
    
    selected = []
    # Set of all possible subset sums of the selected values
    # We use a set of integers
    sums = {0}
    
    for v, x in items:
        # Check if v is already formable by a subset of selected elements
        if v in sums:
            continue
        
        # If not, we can safely add x
        # Because v is larger than all previously selected values (or equal? no, distinct x -> distinct v)
        # Wait, are v distinct?
        # L/x are distinct for x in 1..24 because L is divisible by all, and L/x1 = L/x2 implies x1=x2.
        # So v is strictly larger than all previous v's.
        # Thus, adding v cannot create a sum that equals a previous v_old, because any sum involving v is >= v > v_old.
        # So the invariant holds.
        
        selected.append(x)
        
        # Update sums
        # new_sums = sums U {s + v for s in sums}
        # We must be careful not to modify while iterating
        new_sums = set(sums)
        for s in sums:
            new_sums.add(s + v)
        sums = new_sums
        
    return selected
