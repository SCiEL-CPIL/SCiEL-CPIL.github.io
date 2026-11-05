"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    """
    Returns a nonempty subset of [1, 255] containing 255, with gcd(a,b)>1 
    for all distinct pairs, maximizing cardinality.
    
    The optimal strategy is to select all multiples of the prime factor p of 255
    that maximizes floor(255/p).
    255 = 3 * 5 * 17.
    p=3 -> 85 elements.
    p=5 -> 51 elements.
    p=17 -> 15 elements.
    
    It can be shown that any maximal clique must be contained within the multiples
    of a single prime factor of 255. Thus, multiples of 3 is optimal.
    """
    # 255 factors: 3, 5, 17
    # Best is 3
    return [i for i in range(1, 256) if i % 3 == 0]
# EVOLVE-BLOCK-END
