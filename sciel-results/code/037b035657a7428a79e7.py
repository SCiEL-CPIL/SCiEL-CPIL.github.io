"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    """
    Constructs a set of n=12 distinct positive integers with distinct subset sums,
    minimizing the largest element.
    
    The minimal possible largest element for such a set of size n is 2^(n-1).
    This is achieved by the superincreasing sequence where each element is
    1 plus the sum of all previous elements.
    """
    n = 12
    elements = []
    running_sum = 0
    
    for _ in range(n):
        # To ensure all subset sums are distinct, the next element must be
        # strictly greater than the sum of all previous elements.
        # To minimize the largest element, we pick the smallest such integer.
        next_element = running_sum + 1
        elements.append(next_element)
        running_sum += next_element
        
    return elements
# EVOLVE-BLOCK-END
