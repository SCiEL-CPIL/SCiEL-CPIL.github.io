"""Find smallest n in [1,1000000] with phi(n) divisible by 120."""

def construct():
    """Return smallest n in [1,1000000] such that phi(n) % 120 == 0."""
    # The smallest such n is 143.
    # 143 = 11 * 13
    # phi(143) = (11-1)*(13-1) = 10*12 = 120
    # 120 is divisible by 120.
    # No n < 143 satisfies the condition.
    return 143
