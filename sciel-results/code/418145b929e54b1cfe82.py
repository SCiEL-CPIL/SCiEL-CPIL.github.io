"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START

def construct():
    n = 10
    # Construct 2 mutually orthogonal Latin squares of order 10
    # using linear functions over Z_10.
    # L_k[i][j] = (a_k * i + b_k * j) % n
    # Two squares are orthogonal iff the map (i,j) -> (L1[i][j], L2[i][j]) is bijective,
    # which holds when the coefficient matrix has determinant coprime to n.
    
    # Coefficient pairs (a, b) for each square
    # Square 1: (1, 1)
    # Square 2: (1, 2) -> det = 1*2 - 1*1 = 1, gcd(1,10)=1 => orthogonal
    
    coeffs = [(1, 1), (1, 2)]
    
    squares = []
    for a, b in coeffs:
        sq = [[(a * i + b * j) % n for j in range(n)] for i in range(n)]
        squares.append(sq)
    
    # Verify each square is a valid Latin square
    for sq in squares:
        for i in range(n):
            if sorted(sq[i]) != list(range(n)):
                raise ValueError("Row not a permutation")
        for j in range(n):
            col = [sq[i][j] for i in range(n)]
            if sorted(col) != list(range(n)):
                raise ValueError("Column not a permutation")
    
    # Verify orthogonality between all pairs
    for k in range(len(squares)):
        for m in range(k + 1, len(squares)):
            pairs = set()
            for i in range(n):
                for j in range(n):
                    p = (squares[k][i][j], squares[m][i][j])
                    if p in pairs:
                        raise ValueError("Squares not orthogonal")
                    pairs.add(p)
    
    return squares

# EVOLVE-BLOCK-END
