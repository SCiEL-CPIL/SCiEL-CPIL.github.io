from fractions import Fraction

def construct():
    remaining = Fraction(1, 1)
    denominators = []
    d = 5
    while remaining > 0:
        # Find smallest d >= current d such that 1/d <= remaining
        # d >= 1/remaining => d >= ceil(1/remaining)
        inv = Fraction(1, 1) / remaining
        d_min = max(d, int(inv) + (1 if inv % 1 > 0 else 0))
        # Verify
        while Fraction(1, d_min) > remaining:
            d_min += 1
        denominators.append(d_min)
        remaining -= Fraction(1, d_min)
        d = d_min + 1
    return denominators
