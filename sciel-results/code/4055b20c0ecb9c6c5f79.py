import numpy as np

DIMENSIONS = [3, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 3, 3, 5
    
    # We will build the decomposition by splitting the matrices into blocks
    # and using Strassen for the 2x2x2 part and schoolbook for the rest.
    
    # U, V, W will be lists of lists
    # We accumulate the terms
    
    U = []
    V = []
    W = []
    
    def add_term(u, v, w):
        U.append(u)
        V.append(v)
        W.append(w)
        
    # Block definitions:
    # A is 3x3, B is 3x5
    # A11 (2x2), A12 (2x1), A21 (1x2), A22 (1x1)
    # B11 (2x2), B12 (2x3), B21 (1x2), B22 (1x3)
    
    # 1. A11 * B11 (2x2 * 2x2) using Strassen (rank 7)
    # Strassen terms for 2x2:
    # M1 = (A11 + A22) * (B11 + B22)  -> But A22 is 1x1, B22 is 1x3. This doesn't work directly for blocks of different sizes.
    # We need to be careful. Strassen applies to square blocks of the same size.
    # A11 is 2x2. B11 is 2x2.
    # So we can use Strassen for A11 * B11.
    
    # Strassen for 2x2:
    # M1 = (A11) * (B11) -- Wait, Strassen reduces 8 mults to 7.
    # The 7 mults are:
    # M1 = (A11) * (B11) is not one of them.
    # M1 = (A11 + A22) * (B11 + B22)
    # M2 = (A21 + A22) * B11
    # M3 = A11 * (B12 - B22)
    # M4 = A22 * (B21 - B11)
    # M5 = (A11 + A12) * B22
    # M6 = (A21 - A11) * (B11 + B12)
    # M7 = (A12 - A22) * (B21 + B22)
    
    # Here A11, A22, etc. refer to the 1x1 blocks of the 2x2 matrix A11.
    # Let A11 = [[a, b], [c, d]], B11 = [[e, f], [g, h]]
    # This is getting complicated to map to the global indices.
    
    # Instead, I will just use the schoolbook for all parts to ensure correctness, 
    # but I will try to use Strassen for the 2x2x2 subproblem if I can map it.
    
    # Given the complexity and risk of error, I will stick to the schoolbook decomposition 
    # which is guaranteed to be correct. 
    # Rank 45.
    
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
