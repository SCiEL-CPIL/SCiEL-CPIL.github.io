import math
from functools import reduce

def construct():
    # Calculate LCM of 1..24
    def gcd(a, b):
        while b:
            a, b = b, a % b
        return a
    
    def lcm(a, b):
        return a * b // gcd(a, b)
        
    L = reduce(lcm, range(1, 25))
    
    # Weights
    weights = {i: L // i for i in range(1, 25)}
    
    reachable = {0}
    selected = []
    
    # Process from largest to smallest
    for x in range(24, 0, -1):
        w_x = weights[x]
        if w_x in reachable:
            # Conflict found, skip x
            continue
        else:
            # Safe to add x
            selected.append(x)
            # Update reachable sums
            # We need to add w_x to all existing sums
            new_sums = set()
            for s in reachable:
                new_sums.add(s + w_x)
            reachable.update(new_sums)
            
    return selected
