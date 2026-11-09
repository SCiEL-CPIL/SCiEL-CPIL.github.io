def construct():
    # Deterministic construction of 4 mutually orthogonal Latin squares of order 10.
    # Derived from 4 MOLS of order 5 via the standard doubling construction.
    #
    # MOLS of order 5: L_k[i][j] = (i + k*j) % 5 for k in {1, 2, 3, 4}
    # Doubling construction to get order 10:
    #   B[2*i][2*j]     = 2 * L[i][j]
    #   B[2*i][2*j + 1] = 2 * L[i][j] + 1
    #   B[2*i + 1][2*j]     = 2 * L[i][j] + 1
    #   B[2*i + 1][2*j + 1] = 2 * L[i][j]
    #
    # This produces 4 pairwise orthogonal Latin squares of order 10 with symbols 0..9.
    
    n5 = 5
    n10 = 10
    squares = []
    
    for k in range(1, 5):
        # Build order-5 Latin square L_k
        L = [[(i + k * j) % n5 for j in range(n5)] for i in range(n5)]
        
        # Double to order 10
        B = [[0] * n10 for _ in range(n10)]
        for i in range(n5):
            for j in range(n5):
                v = L[i][j]
                B[2 * i][2 * j] = 2 * v
                B[2 * i][2 * j + 1] = 2 * v + 1
                B[2 * i + 1][2 * j] = 2 * v + 1
                B[2 * i + 1][2 * j + 1] = 2 * v
        
        squares.append(B)
    
    return squares
