"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 9]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank-50 exact tensor decomposition for multiplying a 2x3 matrix by a 3x9 matrix.
    
    The multiplication is decomposed into:
    - 4 blocks of 2x2 * 2x2 multiplication (using Strassen's rank-7 algorithm)
    - 1 block of 2x2 * 2x1 multiplication (rank-4, schoolbook)
    - 1 block of 2x1 * 1x9 multiplication (rank-18, schoolbook)
    
    Total rank: 4*7 + 4 + 18 = 50
    
    Indexing convention:
    - U has shape (n*m, rank) = (6, 50), indexed by [i*m+j, r]
    - V has shape (m*p, rank) = (27, 50), indexed by [j*p+k, r]
    - W has shape (p*n, rank) = (18, 50), indexed by [k*n+i, r]
    """
    n, m, p = 2, 3, 9  # dimensions
    rank = 50
    
    U = [[0] * rank for _ in range(n * m)]  # (6, 50)
    V = [[0] * rank for _ in range(m * p)]  # (27, 50)
    W = [[0] * rank for _ in range(p * n)]  # (18, 50)
    
    col = 0
    
    # Strassen's 7 multiplications for 2x2 * 2x2
    # For A[i,j] with i in {0,1}, j in {0,1} and B[j,k] with j in {0,1}, k in {0,1}:
    # M1 = (a00 + a11)(b00 + b11)
    # M2 = (a10 + a11)b00
    # M3 = a00(b01 - b11)
    # M4 = a11(b10 - b00)
    # M5 = (a00 + a01)b11
    # M6 = (a10 - a00)(b00 + b01)
    # M7 = (a01 - a11)(b10 + b11)
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    def strassen_2x2(i_base, j_base, k_base, offset):
        """
        Add Strassen decomposition for 2x2 * 2x2 block.
        i_base: row offset in A (0 or 1)
        j_base: col offset in A / row offset in B (0, 1, or 2)
        k_base: col offset in B
        offset: starting column index
        Returns: next column index
        """
        nonlocal col
        cols = []
        
        # M1 = (a00 + a11)(b00 + b11)
        # U: a00, a11 contribute; V: b00, b11 contribute; W: C00, C11
        c1 = offset
        U[i_base * m + j_base][c1] += 1        # a00
        U[(i_base + 1) * m + j_base][c1] += 1  # a11
        V[j_base * p + k_base][c1] += 1        # b00
        V[(j_base + 1) * p + (k_base + 1)][c1] += 1  # b11
        W[k_base * n + i_base][c1] += 1        # C00
        W[(k_base + 1) * n + (i_base + 1)][c1] += 1  # C11
        cols.append(c1)
        col += 1
        
        # M2 = (a10 + a11)b00
        c2 = offset + 1
        U[i_base * m + (j_base + 1)][c2] += 1  # a10
        U[(i_base + 1) * m + (j_base + 1)][c2] += 1  # a11
        V[j_base * p + k_base][c2] += 1        # b00
        W[k_base * n + (i_base + 1)][c2] += 1  # C10
        W[(k_base + 1) * n + (i_base + 1)][c2] += 1  # C11
        cols.append(c2)
        col += 1
        
        # M3 = a00(b01 - b11)
        c3 = offset + 2
        U[i_base * m + j_base][c3] += 1        # a00
        V[j_base * p + (k_base + 1)][c3] += 1  # b01
        V[(j_base + 1) * p + (k_base + 1)][c3] -= 1  # -b11
        W[(k_base + 1) * n + i_base][c3] += 1  # C01
        W[(k_base + 1) * n + (i_base + 1)][c3] += 1  # C11
        cols.append(c3)
        col += 1
        
        # M4 = a11(b10 - b00)
        c4 = offset + 3
        U[(i_base + 1) * m + (j_base + 1)][c4] += 1  # a11
        V[(j_base + 1) * p + k_base][c4] += 1        # b10
        V[j_base * p + k_base][c4] -= 1              # -b00
        W[k_base * n + i_base][c4] += 1              # C00
        W[k_base * n + (i_base + 1)][c4] += 1        # C10
        cols.append(c4)
        col += 1
        
        # M5 = (a00 + a01)b11
        c5 = offset + 4
        U[i_base * m + j_base][c5] += 1        # a00
        U[i_base * m + (j_base + 1)][c5] += 1  # a01
        V[(j_base + 1) * p + (k_base + 1)][c5] += 1  # b11
        W[k_base * n + i_base][c5] -= 1        # -C00
        W[(k_base + 1) * n + i_base][c5] += 1  # C01
        cols.append(c5)
        col += 1
        
        # M6 = (a10 - a00)(b00 + b01)
        c6 = offset + 5
        U[i_base * m + (j_base + 1)][c6] += 1  # a10
        U[i_base * m + j_base][c6] -= 1        # -a00
        V[j_base * p + k_base][c6] += 1        # b00
        V[j_base * p + (k_base + 1)][c6] += 1  # b01
        W[(k_base + 1) * n + (i_base + 1)][c6] += 1  # C11
        cols.append(c6)
        col += 1
        
        # M7 = (a01 - a11)(b10 + b11)
        c7 = offset + 6
        U[i_base * m + (j_base + 1)][c7] += 1  # a01
        U[(i_base + 1) * m + (j_base + 1)][c7] -= 1  # -a11
        V[(j_base + 1) * p + k_base][c7] += 1  # b10
        V[(j_base + 1) * p + (k_base + 1)][c7] += 1  # b11
        W[k_base * n + i_base][c7] += 1        # C00
        cols.append(c7)
        col += 1
        
        return offset + 7
    
    # Block 1: A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2]
    col = strassen_2x2(0, 0, 0, col)
    
    # Block 2: A[0:2, 0:2] * B[0:2, 2:4] -> C[0:2, 2:4]
    col = strassen_2x2(0, 0, 2, col)
    
    # Block 3: A[0:2, 0:2] * B[0:2, 4:6] -> C[0:2, 4:6]
    col = strassen_2x2(0, 0, 4, col)
    
    # Block 4: A[0:2, 0:2] * B[0:2, 6:8] -> C[0:2, 6:8]
    col = strassen_2x2(0, 0, 6, col)
    
    # Block 5: A[0:2, 0:2] * B[0:2, 8:9] -> C[0:2, 8:9]
    # 2x2 * 2x1: rank-4 schoolbook
    # C08 = a00*b08 + a01*b18
    # C18 = a10*b08 + a11*b18
    c = col
    U[0 * m + 0][c] = 1    # a00
    V[0 * p + 8][c] = 1    # b08
    W[8 * n + 0][c] = 1    # C08
    col += 1
    
    c = col
    U[0 * m + 1][c] = 1    # a01
    V[1 * p + 8][c] = 1    # b18
    W[8 * n + 0][c] = 1    # C08
    col += 1
    
    c = col
    U[1 * m + 0][c] = 1    # a10
    V[0 * p + 8][c] = 1    # b08
    W[8 * n + 1][c] = 1    # C18
    col += 1
    
    c = col
    U[1 * m + 1][c] = 1    # a11
    V[1 * p + 8][c] = 1    # b18
    W[8 * n + 1][c] = 1    # C18
    col += 1
    
    # Block 6: A[0:2, 2:3] * B[2:3, 0:9] -> C[0:2, 0:9]
    # 2x1 * 1x9: rank-18 schoolbook
    # C_i,k = a_i2 * b_2k for i in {0,1}, k in {0,...,8}
    for i in range(2):
        for k in range(9):
            c = col
            U[i * m + 2][c] = 1      # a_i2
            V[2 * p + k][c] = 1      # b_2k
            W[k * n + i][c] = 1      # C_ik
            col += 1
    
    assert col == rank, f"Expected rank {rank}, got {col}"
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
