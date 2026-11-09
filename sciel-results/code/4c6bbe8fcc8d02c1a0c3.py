"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START

def construct():
    """
    Return the smallest integer n in [1, 1000000] such that phi(n) is divisible by 120.
    
    Analysis:
    120 = 2^3 * 3 * 5.
    We seek min n such that 120 | phi(n).
    Since phi(n) < n for n > 1, if n < 120, phi(n) < 120, so 120 cannot divide phi(n) (as phi(n) >= 1).
    Thus, n must be >= 120.
    We check n from 120 upwards.
    If 120 | phi(n), then phi(n) must be a multiple of 120.
    For n < 240, the only possible value for phi(n) is 120.
    So we look for the smallest n such that phi(n) = 120.
    
    Manual check for n in [120, 142]:
    phi(120) = 48
    phi(121) = 110
    phi(122) = 60
    phi(123) = 80
    phi(124) = 60
    phi(125) = 100
    phi(126) = 36
    phi(127) = 126
    phi(128) = 64
    phi(129) = 84
    phi(130) = 48
    phi(131) = 130
    phi(132) = 40
    phi(133) = 108
    phi(134) = 66
    phi(135) = 72
    phi(136) = 64
    phi(137) = 136
    phi(138) = 44
    phi(139) = 138
    phi(140) = 48
    phi(141) = 92
    phi(142) = 70
    
    None of these are 120.
    Check n = 143:
    143 = 11 * 13
    phi(143) = 10 * 12 = 120.
    120 is divisible by 120.
    
    Therefore, 143 is the minimum.
    """
    # The minimal n such that phi(n) is divisible by 120 is 143.
    # phi(143) = phi(11*13) = 10*12 = 120.
    return 143

# EVOLVE-BLOCK-END
