N = 100

# EVOLVE-BLOCK-START
def construct():
    """
    OneMax: maximize the number of 1s in a binary string of length N.
    
    The optimal solution is trivially all 1s.
    Direct construction: return a string of N '1's.
    """
    return "1" * N
# EVOLVE-BLOCK-END
