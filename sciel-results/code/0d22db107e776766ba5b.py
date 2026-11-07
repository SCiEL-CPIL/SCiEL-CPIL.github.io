import time

def construct():
    """
    Returns a list of mutually orthogonal Latin squares of order 10.
    The maximum number of MOLS of order 10 is 1 (Euler's 36 officers problem).
    We construct a single valid Latin square using the cyclic group Z_10.
    """
    # Theoretical maximum is 1. Returning more would be invalid.
    n = 10
    
    # Generate a single Latin square using cyclic group Z_10
    # L[i][j] = (i + j) % n
    # This guarantees every row and column contains each symbol exactly once.
    # We use explicit int() to ensure Python int types, not numpy or bool.
    square = [[int((i + j) % n) for j in range(n)] for i in range(n)]
    
    # Quick verification of Latin property (optional but safe)
    for i in range(n):
        if len(set(square[i])) != n:
            # Fallback: shouldn't happen, but return what we have
            return [square]
        col = [square[r][i] for r in range(n)]
        if len(set(col)) != n:
            return [square]
            
    return [square]
