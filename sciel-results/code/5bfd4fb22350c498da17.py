import time
from math import factorial

def construct():
    # The optimal solution has a maximum factor of 33.
    # The set is {24, 25, 27, 28, 32, 33}
    # Product: 24 * 25 * 27 * 28 * 32 * 33 = 479001600 = 12!
    # All factors are distinct and > 12.
    sol = [24, 25, 27, 28, 32, 33]
    
    # Verify product to be safe
    prod = 1
    for x in sol:
        prod *= x
    if prod != factorial(12):
        # Fallback to trivial solution if verification fails (should not happen)
        return [factorial(12)]
        
    return sol
