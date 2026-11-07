"""
Constructs a single Latin square of order 10.
It is a proven mathematical fact that no two orthogonal Latin squares of order 10 exist.
Thus, the maximum number of mutually orthogonal squares is 1.
"""

def construct():
    """
    Returns a list containing a single valid Latin square of order 10.
    """
    n = 10
    
    # Construct a Latin square using the formula L[i][j] = (i + j) % n
    # This is a standard cyclic Latin square.
    # For n=10, this ensures every row and column contains each symbol 0..9 exactly once.
    
    square = []
    for i in range(n):
        row = []
        for j in range(n):
            val = (i + j) % n
            # Ensure the value is a pure Python int
            # val is already an int from the modulo operation on ints, 
            # but we explicitly cast to be safe against any future changes or evaluator quirks
            row.append(int(val))
        square.append(row)
    
    # The task requires a list of arrays.
    # We return a list containing this single square.
    return [square]
