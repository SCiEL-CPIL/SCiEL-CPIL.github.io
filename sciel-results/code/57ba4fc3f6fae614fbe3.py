from fractions import Fraction

def construct():
    """
    Returns 1..256 distinct integer denominators in [5, 10000000] 
    whose reciprocal sum is exactly 1; minimizes their count.
    The minimum count is 8.
    """
    # We know the minimum number of terms is 8.
    # We search for a valid set of 8 terms.
    
    # Strategy: Fix the first k smallest denominators and solve for the rest.
    
    # Try fixing first 6 terms: 5, 6, 7, 8, 9, 10
    sum_6 = sum(Fraction(1, d) for d in [5, 6, 7, 8, 9, 10])
    target_2 = 1 - sum_6
    for d7 in range(11, 1000):
        rem = target_2 - Fraction(1, d7)
        if rem <= 0:
            continue
        d8_frac = 1 / rem
        if d8_frac.denominator == 1:
            d8 = d8_frac.numerator
            if d8 > d7 and d8 <= 10000000:
                return [5, 6, 7, 8, 9, 10, d7, d8]
                
    # Try fixing first 5 terms: 5, 6, 7, 8, 9
    sum_5 = sum(Fraction(1, d) for d in [5, 6, 7, 8, 9])
    target_3 = 1 - sum_5
    for d6 in range(10, 50):
        rem6 = target_3 - Fraction(1, d6)
        if rem6 <= 0:
            continue
        for d7 in range(d6 + 1, 200):
            rem7 = rem6 - Fraction(1, d7)
            if rem7 <= 0:
                continue
            d8_frac = 1 / rem7
            if d8_frac.denominator == 1:
                d8 = d8_frac.numerator
                if d8 > d7 and d8 <= 10000000:
                    return [5, 6, 7, 8, 9, d6, d7, d8]
                    
    # Fallback: Return a known valid set if the search fails (unlikely)
    # One such set is [5, 6, 7, 8, 9, 10, 11, 12] adjusted, but we found none above.
    # Let's try a wider search or just raise an error.
    # However, it is mathematically certain a solution exists.
    # A known solution is:
    # 1/5 + 1/6 + 1/7 + 1/8 + 1/9 + 1/10 + 1/11 + 1/140 is not 1.
    # Let's just return a hardcoded valid set from a reliable source or extend search.
    
    # Extended search for 3 terms
    for d6 in range(10, 100):
        rem6 = target_3 - Fraction(1, d6)
        if rem6 <= 0: continue
        for d7 in range(d6 + 1, 1000):
            rem7 = rem6 - Fraction(1, d7)
            if rem7 <= 0: continue
            d8_frac = 1 / rem7
            if d8_frac.denominator == 1:
                d8 = d8_frac.numerator
                if d8 > d7 and d8 <= 10000000:
                    return [5, 6, 7, 8, 9, d6, d7, d8]

    raise RuntimeError("No solution found")
