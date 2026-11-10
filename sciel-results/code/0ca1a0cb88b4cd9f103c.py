def construct():
    """
    Returns the smallest integer n in [1, 1000000] such that phi(n) % 120 == 0.
    
    The answer is 143: phi(143) = phi(11 * 13) = 10 * 12 = 120, which is divisible by 120.
    No smaller n has phi(n) divisible by 120.
    """
    return 143
