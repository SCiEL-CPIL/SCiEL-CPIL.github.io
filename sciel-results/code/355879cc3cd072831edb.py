"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    # Strassen-style decomposition for 2x3 * 3x6.
    # Split A (2x3) into A11 (1x2), A12 (1x1), A21 (1x2), A22 (1x1).
    # Split B (3x6) into B11 (2x3), B12 (2x3), B21 (1x3), B22 (1x3).
    # Output C (2x6) with C11 (1x3), C12 (1x3), C21 (1x3), C22 (1x3).
    # 
    # Strassen-like formulas for 2x2 blocks:
    # M1 = (A11+A22) * (B11+B22)
    # M2 = (A21+A22) * B11
    # M3 = A11 * (B12-B22)
    # M4 = A22 * (B21-B11)
    # M5 = (A11+A12) * B22
    # M6 = (A21-A11) * (B11+B12)
    # M7 = (A12-A22) * (B21+B22)
    #
    # C11 = M1 + M4 - M5 - M7
    # C12 = M3 + M5
    # C21 = M2 + M7
    # C22 = M1 - M2 + M3 + M4
    #
    # This gives rank 7 if we can decompose each Mi into rank-1 terms.
    # But we need to express this as a tensor decomposition with integer/half-integer entries.
    #
    # Let's build the decomposition explicitly.
    # U has shape (n*m, rank) = (6, rank)
    # V has shape (m*p, rank) = (18, rank)
    # W has shape (p*n, rank) = (12, rank)
    #
    # The tensor is T[i,j,k] = 1 if the (i,j) entry of A times (j,k) entry of B contributes to C.
    # In the trace convention: W[k*n+i, r] is the coefficient for output position (i,k) in factor r.
    # U[i*m+j, r] is the coefficient for input A position (i,j) in factor r.
    # V[j*p+k, r] is the coefficient for input B position (j,k) in factor r.
    
    n, m, p = 2, 3, 6
    
    # We'll use rank 7 based on Strassen's method adapted for this case.
    # Each "Strassen product" Mi corresponds to one rank-1 term in the decomposition.
    # But we need to expand each Mi into actual rank-1 terms.
    
    # Actually, let's think differently. The Strassen algorithm reduces the number of scalar multiplications.
    # For our tensor decomposition, we need to express the multiplication tensor as a sum of rank-1 tensors.
    
    # Let me try a direct construction. The known minimal rank for 2x3*3x6 might be higher than 7.
    # Let's start with a known good decomposition and try to optimize.
    
    # For now, return the schoolbook to ensure correctness, then we can try to improve.
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
