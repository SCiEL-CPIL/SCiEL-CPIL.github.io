"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    """
    Rank-49 decomposition of 4x4 matrix multiplication using Strassen's algorithm.
    
    Strassen's algorithm for 2x2 matrices uses 7 products.
    For 4x4 matrices, we partition into 2x2 blocks of 2x2 submatrices.
    The block-level Strassen uses 7 products of 2x2 matrices,
    each of which uses 7 scalar products. Total: 7*7 = 49.
    
    Convention: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k
    """
    n, m, p = DIMENSIONS  # all 4
    rank = 49
    
    U = [[0] * rank for _ in range(n * m)]  # 16 x 49
    V = [[0] * rank for _ in range(m * p)]  # 16 x 49
    W = [[0] * rank for _ in range(p * n)]  # 16 x 49
    
    # Strassen's 7 products for 2x2 matrix multiplication:
    # M1 = (A00 + A11)(B00 + B11)
    # M2 = (A10 + A11)B00
    # M3 = A00(B10 - B11)
    # M4 = A11(B01 - B10)
    # M5 = (A00 + A01)B11
    # M6 = (A10 - A00)(B00 + B01)
    # M7 = (A01 - A11)(B10 + B11)
    #
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    #
    # For 4x4, we index blocks as:
    # A = [[A00, A01], [A10, A11]] where each is 2x2
    # Block (bi, bj) of A has rows 2*bi..2*bi+1, cols 2*bj..2*bj+1
    # Similarly for B and C.
    #
    # The 7 block-level products each involve 2x2 matrices.
    # Each 2x2 product uses 7 scalar-level Strassen products.
    #
    # Total rank = 7 (block level) * 7 (scalar level) = 49
    
    # Strassen scalar-level coefficients for 2x2:
    # For computing X*Y where X,Y are 2x2:
    # The 7 products are:
    # P1 = (X00+X11)*(Y00+Y11)
    # P2 = (X10+X11)*Y00
    # P3 = X00*(Y10-Y11)
    # P4 = X11*(Y01-Y10)
    # P5 = (X00+X01)*Y11
    # P6 = (X10-X00)*(Y00+Y01)
    # P7 = (X01-X11)*(Y10+Y11)
    #
    # Result:
    # R00 = P1 + P4 - P5 + P7
    # R01 = P3 + P5
    # R10 = P2 + P4
    # R11 = P1 - P2 + P3 + P6
    
    # Define the 7 Strassen products for a 2x2 matrix multiplication.
    # Each product is defined by:
    # - For each element (a_row, a_col) of the left matrix: coefficient
    # - For each element (b_row, b_col) of the right matrix: coefficient
    # - For each element (c_row, c_col) of the result: coefficient (how this product contributes)
    
    # Strassen products for 2x2:
    # s is in 0..6
    # left_coeffs[s][a_row][a_col] = coefficient for left matrix element
    # right_coeffs[s][b_row][b_col] = coefficient for right matrix element
    # result_coeffs[s][c_row][c_col] = coefficient for result element
    
    left_coeffs = []
    right_coeffs = []
    result_coeffs = []
    
    # P1 = (X00+X11)*(Y00+Y11)
    left_coeffs.append([[1, 0], [0, 1]])  # X00 + X11
    right_coeffs.append([[1, 0], [0, 1]])  # Y00 + Y11
    # R00 += P1, R11 += P1
    result_coeffs.append([[1, 0], [0, 1]])
    
    # P2 = (X10+X11)*Y00
    left_coeffs.append([[0, 0], [1, 1]])  # X10 + X11
    right_coeffs.append([[1, 0], [0, 0]])  # Y00
    # R10 += P2, R11 -= P2
    result_coeffs.append([[0, 0], [1, -1]])
    
    # P3 = X00*(Y10-Y11)
    left_coeffs.append([[1, 0], [0, 0]])  # X00
    right_coeffs.append([[0, 0], [1, -1]])  # Y10 - Y11
    # R01 += P3, R11 += P3
    result_coeffs.append([[0, 1], [0, 1]])
    
    # P4 = X11*(Y01-Y10)
    left_coeffs.append([[0, 0], [0, 1]])  # X11
    right_coeffs.append([[0, 1], [-1, 0]])  # Y01 - Y10
    # R00 += P4, R10 += P4
    result_coeffs.append([[1, 0], [1, 0]])
    
    # P5 = (X00+X01)*Y11
    left_coeffs.append([[1, 1], [0, 0]])  # X00 + X01
    right_coeffs.append([[0, 0], [0, 1]])  # Y11
    # R00 -= P5, R01 += P5
    result_coeffs.append([[-1, 1], [0, 0]])
    
    # P6 = (X10-X00)*(Y00+Y01)
    left_coeffs.append([[-1, 0], [1, 0]])  # X10 - X00
    right_coeffs.append([[1, 1], [0, 0]])  # Y00 + Y01
    # R11 += P6
    result_coeffs.append([[0, 0], [0, 1]])
    
    # P7 = (X01-X11)*(Y10+Y11)
    left_coeffs.append([[0, 1], [0, -1]])  # X01 - X11
    right_coeffs.append([[0, 0], [1, 1]])  # Y10 + Y11
    # R00 += P7
    result_coeffs.append([[1, 0], [0, 0]])
    
    # Now for 4x4, we use block-level Strassen.
    # Block indices: bi, bj in {0,1}
    # Block (bi, bj) of a 4x4 matrix has rows 2*bi to 2*bi+1, cols 2*bj to 2*bj+1
    
    # Block-level Strassen for 4x4:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B10)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    #
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # Block-level coefficients:
    # For each of the 7 block products, we need:
    # - Which blocks of A are involved and with what coefficients
    # - Which blocks of B are involved and with what coefficients
    # - Which blocks of C receive the result and with what coefficients
    
    block_left = []  # list of 7, each is a dict or 2x2 of coefficients for A blocks
    block_right = []  # list of 7, each is a dict or 2x2 of coefficients for B blocks
    block_result = []  # list of 7, each is a dict or 2x2 of coefficients for C blocks
    
    # M1 = (A00+A11)(B00+B11)
    block_left.append([[1, 0], [0, 1]])   # A00 + A11
    block_right.append([[1, 0], [0, 1]])  # B00 + B11
    # C00 += M1, C11 += M1
    block_result.append([[1, 0], [0, 1]])
    
    # M2 = (A10+A11)B00
    block_left.append([[0, 0], [1, 1]])   # A10 + A11
    block_right.append([[1, 0], [0, 0]])  # B00
    # C10 += M2, C11 -= M2
    block_result.append([[0, 0], [1, -1]])
    
    # M3 = A00(B10-B11)
    block_left.append([[1, 0], [0, 0]])   # A00
    block_right.append([[0, 0], [1, -1]]) # B10 - B11
    # C01 += M3, C11 += M3
    block_result.append([[0, 1], [0, 1]])
    
    # M4 = A11(B01-B10)
    block_left.append([[0, 0], [0, 1]])   # A11
    block_right.append([[0, 1], [-1, 0]]) # B01 - B10
    # C00 += M4, C10 += M4
    block_result.append([[1, 0], [1, 0]])
    
    # M5 = (A00+A01)B11
    block_left.append([[1, 1], [0, 0]])   # A00 + A01
    block_right.append([[0, 0], [0, 1]])  # B11
    # C00 -= M5, C01 += M5
    block_result.append([[-1, 1], [0, 0]])
    
    # M6 = (A10-A00)(B00+B01)
    block_left.append([[-1, 0], [1, 0]])  # A10 - A00
    block_right.append([[1, 1], [0, 0]])  # B00 + B01
    # C11 += M6
    block_result.append([[0, 0], [0, 1]])
    
    # M7 = (A01-A11)(B10+B11)
    block_left.append([[0, 1], [0, -1]])  # A01 - A11
    block_right.append([[0, 0], [1, 1]])  # B10 + B11
    # C00 += M7
    block_result.append([[1, 0], [0, 0]])
    
    # Now build the full decomposition.
    # For each block product s (0..6) and each scalar product t (0..6):
    # The rank index is r = s*7 + t
    #
    # U[i*m+j, r]: i is row of A, j is col of A
    #   A[i,j] belongs to block (i//2, j//2) within the block at position (i%2, j%2)
    #   U[i*m+j, r] = block_left[s][i//2][j//2] * left_coeffs[t][i%2][j%2]
    #
    # V[j*p+k, r]: j is row of B, k is col of B
    #   B[j,k] belongs to block (j//2, k//2) within the block at position (j%2, k%2)
    #   V[j*p+k, r] = block_right[s][j//2][k//2] * right_coeffs[t][j%2][k%2]
    #
    # W[k*n+i, r]: k is row of C, i is col of C (since C is n x p = 4x4, and W[k*n+i] means row k, col i)
    #   Wait, the convention is W[k*n+i, r] where the output is C[i,k]... let me re-check.
    #   The tensor is T[i,j,k] = sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
    #   And T[i,j,k] should be 1 for the matrix multiplication tensor.
    #   C[i,k] = sum_j A[i,j] B[j,k], so the tensor has T[i,j,k] = 1 for all i,j,k.
    #   So W[k*n+i, r] contributes to C[i,k] with coefficient... 
    #   C[i,k] = sum_j sum_r A[i,j] V[j*p+k,r] W[k*n+i,r]
    #   = sum_r W[k*n+i,r] sum_j A[i,j] V[j*p+k,r]
    #   For this to equal sum_j A[i,j] B[j,k], we need:
    #   sum_r U[i*m+j,r] V[j*p+k,r] W[k*n+i,r] = 1 for all i,j,k
    #   
    #   W[k*n+i, r]: k is the row index of C (which is the column index of B), i is the column index of C (which is the row index of A).
    #   So C[i,k] where i is row, k is col. W is indexed by [k*n + i] meaning first index is k (col of C / row of B output), second is i (row of C).
    #   C[i,k] belongs to block (i//2, k//2) within the block at position (i%2, k%2)
    #   W[k*n+i, r] = block_result[s][i//2][k//2] * result_coeffs[t][i%2][k%2]
    
    for s in range(7):
        for t in range(7):
            r = s * 7 + t
            for i in range(n):  # row of A, col of C
                for j in range(m):  # col of A, row of B
                    U[i * m + j][r] = block_left[s][i // 2][j // 2] * left_coeffs[t][i % 2][j % 2]
            for j in range(m):  # row of B
                for k in range(p):  # col of B, row of C
                    V[j * p + k][r] = block_right[s][j // 2][k // 2] * right_coeffs[t][j % 2][k % 2]
            for k in range(p):  # row of C
                for i in range(n):  # col of C
                    W[k * n + i][r] = block_result[s][i // 2][k // 2] * result_coeffs[t][i % 2][k % 2]
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
