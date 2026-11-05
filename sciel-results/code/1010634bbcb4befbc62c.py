def construct():
    """
    Returns the smallest integer n in [1, 1000000] such that phi(n) is divisible by 120.
    
    The minimal such integer is 143, as phi(143) = phi(11 * 13) = 10 * 12 = 120.
    This is verified to be the minimum by checking all smaller candidates or via
    analytical derivation of the minimal n for phi(n) = 120.
    """
    return 143
