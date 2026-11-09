import time

def construct():
    """
    Returns the set of all multiples of 3 in [1, 255].
    This set contains 255.
    For any distinct a, b in the set, gcd(a, b) is a multiple of 3, hence > 1.
    The size of this set is floor(255/3) = 85.
    """
    return list(range(3, 256, 3))
