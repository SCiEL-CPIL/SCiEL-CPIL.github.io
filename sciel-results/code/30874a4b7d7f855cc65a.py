def construct():
    """
    Return the smallest integer n in [1, 1000000] such that phi(n) is divisible by 120.
    
    The minimum such n is 143, since phi(143) = phi(11*13) = 10*12 = 120, which is divisible by 120.
    No smaller n has phi(n) divisible by 120 (verified by exhaustive check up to 142).
    """
    return 143
