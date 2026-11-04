"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Use rank-27 decomposition for 3x3x3, mapped to 3x3x6
    # This corresponds to using the schoolbook method for 3x3x3
    # and duplicating it for the two halves of the 6 columns.
    # Total rank = 27.
    
    rank = 27
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # The 3x3x3 schoolbook decomposition has rank 27.
    # We map the 3x6 problem to two 3x3 problems.
    # U is 9x27. V is 18x27. W is 18x27.
    # For the first 3 columns of B (k=0,1,2), we use the first 27 terms?
    # No, the rank-27 decomposition of 3x3x3 computes C = A B_1.
    # We need to compute C_1 = A B_1 and C_2 = A B_2.
    # If we use the same 27 terms for both, we need to separate V and W.
    # V has 18 rows. Rows 0-5 for k=0-2, rows 6-11 for k=3-5.
    # W has 18 rows. Rows 0-5 for k=0-2, rows 6-11 for k=3-5.
    
    # Let's build the 3x3x3 schoolbook decomposition first.
    # U3[i*m+j, c], V3[j*p+k, c], W3[k*n+i, c] for 3x3x3.
    # p=3 for the 3x3x3 case.
    U3 = [[0] * 27 for _ in range(9)]
    V3 = [[0] * 27 for _ in range(9)]
    W3 = [[0] * 27 for _ in range(9)]
    col = 0
    for i in range(3):
        for j in range(3):
            for k in range(3):
                U3[i * 3 + j][col] = 1
                V3[j * 3 + k][col] = 1
                W3[k * 3 + i][col] = 1
                col += 1
                
    # Now map to 3x3x6.
    # U is the same as U3.
    for i in range(9):
        for c in range(27):
            U[i][c] = U3[i][c]
            
    # V has 18 rows.
    # For k=0,1,2 (cols 0-2 of B), V[j*6+k, c] = V3[j*3+k, c].
    # For k=3,4,5 (cols 3-5 of B), V[j*6+k, c] = V3[j*3+(k-3), c].
    for j in range(3):
        for k in range(3):
            for c in range(27):
                V[j * 6 + k][c] = V3[j * 3 + k][c]
                V[j * 6 + (k + 3)][c] = V3[j * 3 + k][c]
                
    # W has 18 rows.
    # W[k*3+i, c] for k=0,1,2.
    # W[k*3+i, c] for k=3,4,5.
    for k in range(3):
        for i in range(3):
            for c in range(27):
                W[k * 3 + i][c] = W3[k * 3 + i][c]
                W[(k + 3) * 3 + i][c] = W3[k * 3 + i][c]
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
