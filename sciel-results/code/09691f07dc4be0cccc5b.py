def construct():
    """
    Return the smallest integer n in [1, 1000000] such that phi(n) is divisible by 120.
    
    Mathematical analysis:
    We seek the minimum n such that 120 | phi(n).
    Since phi(n) < n for n > 1, if phi(n) is a multiple of 120, the smallest possible value for phi(n) is 120.
    If phi(n) >= 240, then n > 240.
    The smallest n with phi(n) = 120 is 143.
    Any n < 143 must have phi(n) < 143, so phi(n) cannot be 240 or higher.
    Thus, we only need to check if any n < 143 has phi(n) = 120.
    It is a known result that 143 is the smallest solution to phi(n) = 120.
    Therefore, 143 is the global minimum for the condition 120 | phi(n).
    """
    return 143
