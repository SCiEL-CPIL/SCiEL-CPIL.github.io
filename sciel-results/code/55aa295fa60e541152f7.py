"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    from fractions import Fraction

    denominators = list(range(2, 49))
    n = len(denominators)
    vals = [Fraction(1, d) for d in denominators]
    
    # Helper to check if a set is valid and get its sum
    # We maintain state explicitly for speed
    
    # Greedy initial solution: include as many as possible while sum < 1
    # Sort by value descending (which is index ascending for 1/d)
    # Actually, 1/2 > 1/3 > ... so indices 0, 1, 2... are largest to smallest.
    selected = set()
    total = Fraction(0)
    for i in range(n):
        if total + vals[i] < 1:
            selected.add(i)
            total += vals[i]
    
    return list(selected)
# EVOLVE-BLOCK-END
