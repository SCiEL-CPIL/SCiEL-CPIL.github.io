"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START

def construct():
    # The set of powers of 2 up to 512 satisfies the condition.
    # No power of 2 is the arithmetic mean of any subset of other distinct powers of 2.
    # This is because the sum of distinct powers of 2 has a unique binary representation,
    # and the equation sum(T) = k * x (where x is a power of 2) has no solutions for k >= 2
    # with distinct elements in T and x not in T (or even if x in T, the coefficients don't match).
    
    max_val = 512
    selected = []
    x = 1
    while x <= max_val:
        selected.append(x)
        x *= 2
    
    return selected

# EVOLVE-BLOCK-END
