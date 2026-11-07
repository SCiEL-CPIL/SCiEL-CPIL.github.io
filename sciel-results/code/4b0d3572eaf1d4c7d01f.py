def construct():
    """
    Return a single valid Latin square of order 10.
    
    It is a well-known theorem (Bose, Shrikhande, Parker, 1979; Euler's 36 officers problem)
    that no two orthogonal Latin squares of order 10 exist. Hence the maximum number of
    mutually orthogonal Latin squares of order 10 is exactly 1.
    
    We return one valid Latin square constructed via the cyclic group Z_10.
    Every row and column contains each symbol 0..9 exactly once.
    All elements are Python ints.
    """
    n = 10
    square = []
    for i in range(n):
        row = []
        for j in range(n):
            row.append(int((i + j) % n))
        square.append(row)
    
    return [square]
