def construct():
    # Hardcoded 12-term Egyptian fraction representation of 1
    # with distinct denominators all >= 5 and <= 10000000.
    # 1 = 1/5 + 1/6 + 1/7 + 1/10 + 1/11 + 1/12 + 1/15 + 1/18 + 1/30 + 1/36 + 1/42 + 1/110
    #
    # Verification:
    # 1/5 + 1/10 + 1/15 + 1/30 = 6/30 + 3/30 + 2/30 + 1/30 = 12/30 = 2/5
    # 2/5 + 1/11 + 1/110 = 22/55 + 5/55 + 1/110 = 27/55 + 1/110 = 54/110 + 1/110 = 55/110 = 1/2
    # 1/6 + 1/12 + 1/18 + 1/36 = 6/36 + 3/36 + 2/36 + 1/36 = 12/36 = 1/3
    # 1/3 + 1/7 + 1/42 = 14/42 + 6/42 + 1/42 = 21/42 = 1/2
    # 1/2 + 1/2 = 1
    #
    # All denominators: 5, 6, 7, 10, 11, 12, 15, 18, 30, 36, 42, 110
    # All distinct, all in [5, 10000000], count = 12.

    denominators = [5, 6, 7, 10, 11, 12, 15, 18, 30, 36, 42, 110]

    # Verify correctness (optional, but safe)
    from fractions import Fraction
    s = Fraction(0, 1)
    for d in denominators:
        s += Fraction(1, d)
    assert s == 1, f"Sum is {s}, not 1"
    assert len(set(denominators)) == len(denominators), "Denominators not distinct"
    assert all(5 <= d <= 10000000 for d in denominators), "Denominator out of range"
    assert 1 <= len(denominators) <= 256, "Count out of range"

    return denominators
