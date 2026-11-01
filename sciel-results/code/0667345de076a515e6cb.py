"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
def _build_mm(n, m, p):
    """Build U, V, W for (n x m) x (m x p) matrix multiplication tensor.
    Returns U (n*m x rank), V (m*p x rank), W (p*n x rank) as lists of lists.
    """
    import numpy as np
    
    if m == 1:
        # Schoolbook: n*p terms
        rank = n * p
        U = [[0] * rank for _ in range(n)]
        V = [[0] * rank for _ in range(p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for k in range(p):
                U[i][col] = 1
                V[k][col] = 1
                W[k * n + i][col] = 1
                col += 1
        return U, V, W
    
    if n == 2 and m == 2 and p == 2:
        # Strassen decomposition for 2x2x2
        rank = 7
        U = [[0] * rank for _ in range(4)]
        V = [[0] * rank for _ in range(4)]
        W = [[0] * rank for _ in range(4)]
        
        # Strassen products:
        # P1 = (A00+A11)*(B00+B11)
        # P2 = (A10+A11)*B00
        # P3 = A00*(B01-B11)
        # P4 = A11*(B10-A10)
        # P5 = (A00+A01)*B11
        # P6 = (A10-A00)*(B00+B01)
        # P7 = (A01-A11)*(B10+B11)
        
        # U indices: A00=0, A01=1, A10=2, A11=3
        # V indices: B00=0, B01=1, B10=2, B11=3
        # W indices: C00=0, C01=1, C10=2, C11=3
        
        # P1: (A00+A11)*(B00+B11)
        U[0][0] = 1; U[3][0] = 1
        V[0][0] = 1; V[3][0] = 1
        W[0][0] = 1; W[3][0] = 1
        
        # P2: (A10+A11)*B00
        U[2][1] = 1; U[3][1] = 1
        V[0][1] = 1
        W[2][1] = 1
        
        # P3: A00*(B01-B11)
        U[0][2] = 1
        V[1][2] = 1; V[3][2] = -1
        W[1][2] = 1
        
        # P4: A11*(B10-A10)
        U[3][3] = 1
        V[2][3] = 1; V[0][3] = 0  # B10 only, no A10 in V
        # Wait, P4 = A11 * (B10 - A10) doesn't make sense dimensionally.
        # Let me reconsider. P4 should be a product of a linear form in A and a linear form in B.
        # P4 = A11 * B10 - A11 * A10? No, that's not bilinear in the right way.
        
        # I think I need to be more careful. Let me just use schoolbook for 2x2x2 as well.
        # This is getting too error-prone without verification.
        pass
    
    # Fallback: split middle dimension
    m1 = m // 2
    m2 = m - m1
    
    U1, V1, W1 = _build_mm(n, m1, p)
    U2, V2, W2 = _build_mm(n, m2, p)
    
    rank1 = len(U1[0]) if U1 else 0
    rank2 = len(U2[0]) if U2 else 0
    rank = rank1 + rank2
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Embed first subproblem (columns 0..m1-1)
    for i in range(n):
        for j in range(m1):
            for r in range(rank1):
                U[i * m + j][r] = U1[i * m1 + j][r]
    for j in range(m1):
        for k in range(p):
            for r in range(rank1):
                V[j * p + k][r] = V1[j * p + k][r]
    for i in range(n):
        for k in range(p):
            for r in range(rank1):
                W[k * n + i][r] = W1[k * n + i][r]
    
    # Embed second subproblem (columns m1..m-1)
    for i in range(n):
        for j in range(m2):
            for r in range(rank2):
                U[i * m + (m1 + j)][rank1 + r] = U2[i * m2 + j][r]
    for j in range(m2):
        for k in range(p):
            for r in range(rank2):
                V[(m1 + j) * p + k][rank1 + r] = V2[j * p + k][r]
    for i in range(n):
        for k in range(p):
            for r in range(rank2):
                W[k * n + i][rank1 + r] = W2[k * n + i][r]
    
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    U, V, W = _build_mm(n, m, p)
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
