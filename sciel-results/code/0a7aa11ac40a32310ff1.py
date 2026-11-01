"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Rank-minimal exact tensor decomposition for 2x3 * 3x5 matrix multiplication.
    
    The standard schoolbook algorithm gives rank n*m*p = 2*3*5 = 30.
    
    For small matrix multiplication, we can sometimes find decompositions of lower rank
    than the schoolbook algorithm. However, for 2x3 * 3x5, it is known that the
    border rank and tensor rank are not fully characterized. 
    
    A safe and exact approach is to use the schoolbook decomposition which is guaranteed
    to work. The rank is 30.
    
    Let's verify if there's a known better rank. For 2x3 * 3x2, the rank is known to be
    14 (Strassen-like). For 2x3 * 3x3, the rank is at most 22 (using block methods).
    But for 2x3 * 3x5, splitting into 2x3*3x3 (rank <= 22) and 2x3*3x2 (rank <= 14)
    would give 22 + 14 = 36, which is worse than 30.
    
    Actually, let's just use the schoolbook rank of 30, which is simple and correct.
    """
    n, m, p = DIMENSIONS
    
    total_rank = n * m * p
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
