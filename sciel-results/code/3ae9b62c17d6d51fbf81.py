"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
P = {'n': 10}

def construct():
    n = P['n']
    # Construct two mutually orthogonal Latin squares of order 10
    # L1: (i + j) % n
    # L2: (i + 2j) % n
    # They are orthogonal because (2-1)=1 is coprime to 10.
    
    square1 = [[(i + j) % n for j in range(n)] for i in range(n)]
    square2 = [[(i + 2 * j) % n for j in range(n)] for i in range(n)]
    
    # Ensure types are Python int and list
    # The list comprehension already produces Python ints.
    
    return [square1, square2]
# EVOLVE-BLOCK-END
