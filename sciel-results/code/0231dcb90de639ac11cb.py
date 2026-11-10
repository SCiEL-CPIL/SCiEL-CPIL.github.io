def construct():
    """
    Returns the smallest integer n in [1, 1000000] such that phi(n) is divisible by 120.
    
    Mathematical analysis shows that 143 (11 * 13) is the minimal such integer:
    - For n < 121, phi(n) < 120, so phi(n) cannot be a positive multiple of 120.
    - For 121 <= n < 143, direct computation of phi(n) shows none are divisible by 120.
    - phi(143) = 143 * (1 - 1/11) * (1 - 1/13) = 143 * (10/11) * (12/13) = 120.
    
    Therefore, 143 is the answer.
    """
    return 143
