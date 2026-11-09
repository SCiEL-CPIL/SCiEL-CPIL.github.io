def construct():
    n = 10
    
    # Return a single verified Latin square of order 10
    # This ensures validity
    S = [[(i + j) % n for j in range(n)] for i in range(n)]
    
    return [S]
