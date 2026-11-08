"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank-56 exact tensor decomposition for multiplying a 2x4 matrix by a 4x8 matrix.
    
    Uses Strassen's algorithm applied to the 2x2 block structure of the 4x4 subproblem,
    extended to handle the 2x4 * 4x8 case.
    
    The matrix multiplication tensor T_{i,j,k}^{a,b,c} = 1 if (i,j,k) corresponds to valid indices
    in A[i,j] * B[j,k] = C[i,k].
    
    We decompose this into rank-56 factors using 7 Strassen multiplications per 2x2 block pair,
    applied to the 4 intermediate columns (split into 2 groups of 2).
    """
    n, m, p = DIMENSIONS  # n=2, m=4, p=8
    
    # We'll build U (n*m x rank), V (m*p x rank), W (p*n x rank)
    # rank = 56
    
    rank = 56
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen's 7 multiplications for 2x2 * 2x2:
    # M1 = (A11 + A22) * (B11 + B22)
    # M2 = (A21 + A22) * B11
    # M3 = A11 * (B12 - B22)
    # M4 = A22 * (B21 - B11)
    # M5 = (A11 + A12) * B22
    # M6 = (A21 - A11) * (B11 + B12)
    # M7 = (A12 - A22) * (B21 + B22)
    
    # For our case: A is 2x4, B is 4x8
    # Split A into 2x2 blocks: A11, A12 (each 1x2)
    # Split B into 2x4 blocks: B11, B12 (each 2x2), B21, B22 (each 2x2)
    # But we need to think in terms of the full tensor.
    
    # Let's think of this as: for each of the 2 output rows i, and each of the 8 output cols k,
    # C[i,k] = sum_j A[i,j] * B[j,k]
    
    # Group the 4 intermediate indices j into two pairs: {0,1} and {2,3}
    # Group the 8 output columns k into four pairs: {0,1}, {2,3}, {4,5}, {6,7}
    
    # For each pair of intermediate indices (j1, j2) in {(0,1), (2,3)} and each pair of output columns (k1,k2) in {(0,1),(2,3),(4,5),(6,7)},
    # we have a 2x2 * 2x2 subproblem:
    # A_sub[i, j1 or j2] * B_sub[j1 or j2, k1 or k2]
    
    # There are 2 (j-pairs) * 4 (k-pairs) = 8 such 2x2*2x2 subproblems.
    # Each has rank 7 (Strassen), so total rank = 8 * 7 = 56.
    
    # Let's build this systematically.
    
    # j_pairs: list of (j1, j2) pairs
    j_pairs = [(0, 1), (2, 3)]
    # k_pairs: list of (k1, k2) pairs  
    k_pairs = [(0, 1), (2, 3), (4, 5), (6, 7)]
    
    # Strassen coefficients for 2x2*2x2:
    # For multiplication M = X * Y where X, Y are 2x2:
    # M1 = (X[0,0]+X[1,1]) * (Y[0,0]+Y[1,1])
    # M2 = (X[1,0]+X[1,1]) * Y[0,0]
    # M3 = X[0,0] * (Y[0,1]-Y[1,1])
    # M4 = X[1,1] * (Y[1,0]-Y[0,0])
    # M5 = (X[0,0]+X[0,1]) * Y[1,1]
    # M6 = (X[1,0]-X[0,0]) * (Y[0,0]+Y[0,1])
    # M7 = (X[0,1]-X[1,1]) * (Y[1,0]+Y[1,1])
    
    # Result:
    # C[0,0] = M1 + M4 - M5 - M7
    # C[0,1] = M3 + M5
    # C[1,0] = M2 + M4
    # C[1,1] = M1 - M2 + M3 + M6
    
    # For our subproblem with intermediate indices (j1, j2) and output columns (k1, k2):
    # X[i, a] = A[i, j_a] for i in {0,1}, a in {0,1} (where j_a is j1 or j2)
    # Y[a, b] = B[j_a, k_b] for a in {0,1}, b in {0,1} (where k_b is k1 or k2)
    
    # The result C_sub[i, b] = sum_a X[i,a] * Y[a,b] = sum_a A[i, j_a] * B[j_a, k_b]
    # This is exactly the contribution to C[i, k_b] from intermediate indices j1, j2.
    
    col_offset = 0
    for j_pair in j_pairs:
        j1, j2 = j_pair
        for k_pair in k_pairs:
            k1, k2 = k_pair
            
            # For each of the 7 Strassen multiplications
            for s in range(7):
                col = col_offset
                col_offset += 1
                
                # Determine the Strassen formula for multiplication s
                # s=0: M1, s=1: M2, ..., s=6: M7
                
                # X coefficients (for A[i, j]):
                # We need to extract which A elements are involved and their coefficients
                # For each i in {0,1} and each j in {j1, j2}
                
                # Let's define the X and Y factors for each Strassen step
                # X is 2x2, Y is 2x2
                
                if s == 0:  # M1 = (X00+X11)*(Y00+Y11)
                    # X part: X[0,0] + X[1,1] = A[0,j1] + A[1,j2]
                    # Y part: Y[0,0] + Y[1,1] = B[j1,k1] + B[j2,k2]
                    # U: A[0,j1] coeff 1, A[1,j2] coeff 1
                    U[0 * m + j1][col] = 1
                    U[1 * m + j2][col] = 1
                    # V: B[j1,k1] coeff 1, B[j2,k2] coeff 1
                    V[j1 * p + k1][col] = 1
                    V[j2 * p + k2][col] = 1
                    # W: C[0,k1] += 1, C[0,k2] += 1, C[1,k1] += 1, C[1,k2] += 1
                    # Wait, M1 contributes to all four outputs with specific coefficients
                    # C_sub[0,0] = M1 + ..., C_sub[0,1] = M1 + ..., C_sub[1,0] = M1 + ..., C_sub[1,1] = M1 + ...
                    # From the result formulas:
                    # C[0,0] has M1 coeff 1
                    # C[0,1] has M1 coeff 0? No: C[0,1] = M3 + M5, so M1 coeff 0
                    # C[1,0] = M2 + M4, so M1 coeff 0
                    # C[1,1] = M1 - M2 + M3 + M6, so M1 coeff 1
                    # So M1 contributes to C[0,k1] with coeff 1 and C[1,k2] with coeff 1
                    W[k1 * n + 0][col] = 1   # C[0, k1]
                    W[k2 * n + 1][col] = 1   # C[1, k2]
                
                elif s == 1:  # M2 = (X10+X11)*Y00 = (A[1,j1]+A[1,j2])*B[j1,k1]
                    U[1 * m + j1][col] = 1
                    U[1 * m + j2][col] = 1
                    V[j1 * p + k1][col] = 1
                    # C[1,k1] += M2, C[1,k2] -= M2
                    # C[1,0] = M2 + M4, so M2 coeff 1 for C[1,k1]
                    # C[1,1] = M1 - M2 + M3 + M6, so M2 coeff -1 for C[1,k2]
                    W[k1 * n + 1][col] = 1
                    W[k2 * n + 1][col] = -1
                
                elif s == 2:  # M3 = X00*(Y01-Y11) = A[0,j1]*(B[j1,k2]-B[j2,k2])
                    U[0 * m + j1][col] = 1
                    V[j1 * p + k2][col] = 1
                    V[j2 * p + k2][col] = -1
                    # C[0,k2] += M3, C[1,k2] += M3
                    # C[0,1] = M3 + M5, so M3 coeff 1 for C[0,k2]
                    # C[1,1] = M1 - M2 + M3 + M6, so M3 coeff 1 for C[1,k2]
                    W[k2 * n + 0][col] = 1
                    W[k2 * n + 1][col] = 1
                
                elif s == 3:  # M4 = X11*(Y10-Y00) = A[1,j2]*(B[j2,k1]-B[j1,k1])
                    U[1 * m + j2][col] = 1
                    V[j2 * p + k1][col] = 1
                    V[j1 * p + k1][col] = -1
                    # C[0,k1] += M4, C[1,k1] += M4
                    # C[0,0] = M1 + M4 - M5 - M7, so M4 coeff 1 for C[0,k1]
                    # C[1,0] = M2 + M4, so M4 coeff 1 for C[1,k1]
                    W[k1 * n + 0][col] = 1
                    W[k1 * n + 1][col] = 1
                
                elif s == 4:  # M5 = (X00+X01)*Y11 = (A[0,j1]+A[0,j2])*B[j2,k2]
                    U[0 * m + j1][col] = 1
                    U[0 * m + j2][col] = 1
                    V[j2 * p + k2][col] = 1
                    # C[0,k1] -= M5, C[0,k2] += M5
                    # C[0,0] = M1 + M4 - M5 - M7, so M5 coeff -1 for C[0,k1]
                    # C[0,1] = M3 + M5, so M5 coeff 1 for C[0,k2]
                    W[k1 * n + 0][col] = -1
                    W[k2 * n + 0][col] = 1
                
                elif s == 5:  # M6 = (X10-X00)*(Y00+Y01) = (A[1,j1]-A[0,j1])*(B[j1,k1]+B[j1,k2])
                    U[1 * m + j1][col] = 1
                    U[0 * m + j1][col] = -1
                    V[j1 * p + k1][col] = 1
                    V[j1 * p + k2][col] = 1
                    # C[1,k2] += M6
                    # C[1,1] = M1 - M2 + M3 + M6, so M6 coeff 1 for C[1,k2]
                    W[k2 * n + 1][col] = 1
                
                elif s == 6:  # M7 = (X01-X11)*(Y10+Y11) = (A[0,j2]-A[1,j2])*(B[j2,k1]+B[j2,k2])
                    U[0 * m + j2][col] = 1
                    U[1 * m + j2][col] = -1
                    V[j2 * p + k1][col] = 1
                    V[j2 * p + k2][col] = 1
                    # C[0,k1] -= M7
                    # C[0,0] = M1 + M4 - M5 - M7, so M7 coeff -1 for C[0,k1]
                    W[k1 * n + 0][col] = -1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
