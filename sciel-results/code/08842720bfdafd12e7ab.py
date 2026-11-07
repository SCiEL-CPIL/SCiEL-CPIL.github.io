"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    """
    Returns a list containing a single 10x10 Latin square.
    Since no pair of orthogonal Latin squares exists for order 10,
    the maximum number of mutually orthogonal squares is 1.
    """
    n = 10
    # Construct a standard cyclic Latin square: L[i][j] = (i + j) % n
    # This is a valid Latin square.
    square = [[(i + j) % n for j in range(n)] for i in range(n)]
    
    # Ensure all values are Python int and structure is list of lists
    # The list comprehension above already produces Python ints and lists.
    
    return [square]
# EVOLVE-BLOCK-END
