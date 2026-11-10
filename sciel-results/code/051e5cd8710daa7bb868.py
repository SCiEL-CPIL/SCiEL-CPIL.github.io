"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    n = 12
    # Greedy construction of a dissociated set (distinct subset sums)
    # We maintain the set of all possible differences of subset sums.
    # Initially, with an empty set, the only difference is 0.
    forbidden = {0}
    result = []
    
    # We want to find n distinct positive integers.
    # To minimize the largest element, we greedily pick the smallest
    # positive integer not in the 'forbidden' set.
    # The 'forbidden' set contains all values x such that if we added x,
    # it would create a duplicate subset sum.
    # Specifically, x is forbidden if x = s2 - s1 for some subset sums s1, s2.
    # This is equivalent to x being a signed sum of existing elements.
    
    for _ in range(n):
        # Find the smallest positive integer not in forbidden
        x = 1
        while x in forbidden:
            x += 1
        
        # Add x to the result
        result.append(x)
        
        # Update the forbidden set
        # New differences are formed by adding x to old differences,
        # subtracting x from old differences, and keeping old differences.
        # Note: We must use a copy of forbidden to iterate while updating.
        new_forbidden = set(forbidden)
        for d in forbidden:
            new_forbidden.add(d + x)
            new_forbidden.add(d - x)
        
        forbidden = new_forbidden
        
    return result
# EVOLVE-BLOCK-END
