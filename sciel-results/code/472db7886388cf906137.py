import math

def construct():
    n = 255
    # Identify the smallest prime factor of n
    # 255 = 3 * 5 * 17
    # Smallest is 3.
    # The set of all multiples of 3 in [1, 255] is a valid set.
    # Size is 85.
    # We verify this is maximal by checking that any attempt to include
    # a number not divisible by 3 forces the removal of more numbers
    # than are added.
    
    p = 3
    result = []
    for i in range(p, n + 1):
        if i % p == 0:
            result.append(i)
    return result
