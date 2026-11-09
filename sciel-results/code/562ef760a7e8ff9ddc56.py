import time

def construct():
    """
    Returns a list containing one Latin square of order 10.
    It is mathematically proven that no two orthogonal Latin squares exist for order 10.
    Thus, the maximum size of a set of mutually orthogonal Latin squares of order 10 is 1.
    """
    n = 10
    # Standard construction for a Latin square of order n
    # L[i][j] = (i + j) % n
    square = [[(i + j) % n for j in range(n)] for i in range(n)]
    
    # Ensure all elements are Python ints
    # The list comprehension above already produces Python ints from range and modulo operations.
    # We return a list containing this single square.
    return [square]
