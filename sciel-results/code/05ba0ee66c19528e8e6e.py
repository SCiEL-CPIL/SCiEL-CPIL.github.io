"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 4, 4, 6

    # Build a decomposition using Strassen for 2x2x2 blocks.
    # A is 4x4, B is 4x6, C = A*B is 4x6.
    # We partition A into 2x2 blocks: A[bi][bj] for bi,bj in {0,1}
    # We partition B into 2x2 blocks: B[bj][bk] for bj in {0,1}, bk in {0,1,2}
    # C[bi][bk] = sum_j A[bi][bj] * B[bj][bk]
    
    # Strassen 2x2x2 decomposition:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)(B00)
    # M3 = (A00)(B10+B11)
    # M4 = (A11)(B01-B11)
    # M5 = (A00+A01)(B11)
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    #
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 - M4 + M6
    # C11 = M1 - M2 + M3 + M6
    
    # For each block multiplication A[bi][bj] * B[bj][bk] -> C[bi][bk]
    # We use 7 Strassen terms per block pair (bi, bj, bk).
    # Total blocks: 2 * 2 * 3 = 12 block multiplications
    # Each uses 7 terms, so total rank = 12 * 7 = 84
    
    # But we can share some Strassen terms across different bk for the same (bi, bj).
    # Actually, let's just use independent Strassen for each (bi, bj, bk) triple.
    
    # U has shape (n*m, rank) = (16, rank)
    # V has shape (m*p, rank) = (24, rank)  
    # W has shape (p*n, rank) = (24, rank)
    
    # Indexing:
    # U[i*m + j] corresponds to A[i,j]
    # V[j*p + k] corresponds to B[j,k]
    # W[k*n + i] corresponds to C[i,k]
    
    # Strassen terms for a 2x2x2 block:
    # Each term t has coefficients for A elements (a00, a01, a10, a11),
    # B elements (b00, b01, b10, b11), and C elements (c00, c01, c10, c11)
    
    strassen_terms = [
        # (A coeffs [a00,a01,a10,a11], B coeffs [b00,b01,b10,b11], C coeffs [c00,c01,c10,c11])
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),   # M1
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, 0]),   # M2
        ([1, 0, 0, 0], [0, 0, 1, 1], [0, 1, 0, 0]),   # M3
        ([0, 0, 0, 1], [0, 1, 0, -1], [1, 0, 1, 0]),  # M4
        ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]),  # M5
        ([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]),  # M6
        ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 1]),  # M7
    ]
    
    rank = 12 * 7  # 12 block multiplications * 7 Strassen terms each
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for bi in range(2):
        for bj in range(2):
            for bk in range(3):
                for t in range(7):
                    a_coeffs = strassen_terms[t][0]
                    b_coeffs = strassen_terms[t][1]
                    c_coeffs = strassen_terms[t][2]
                    
                    # A block: rows 2*bi to 2*bi+1, cols 2*bj to 2*bj+1
                    # a00 -> A[2*bi, 2*bj]
                    # a01 -> A[2*bi, 2*bj+1]
                    # a10 -> A[2*bi+1, 2*bj]
                    # a11 -> A[2*bi+1, 2*bj+1]
                    a_indices = [
                        (2*bi) * m + (2*bj),       # a00
                        (2*bi) * m + (2*bj+1),     # a01
                        (2*bi+1) * m + (2*bj),     # a10
                        (2*bi+1) * m + (2*bj+1),   # a11
                    ]
                    for idx, coeff in zip(a_indices, a_coeffs):
                        U[idx][col] += coeff
                    
                    # B block: rows 2*bj to 2*bj+1, cols 2*bk to 2*bk+1
                    # b00 -> B[2*bj, 2*bk]
                    # b01 -> B[2*bj, 2*bk+1]
                    # b10 -> B[2*bj+1, 2*bk]
                    # b11 -> B[2*bj+1, 2*bk+1]
                    b_indices = [
                        (2*bj) * p + (2*bk),       # b00
                        (2*bj) * p + (2*bk+1),     # b01
                        (2*bj+1) * p + (2*bk),     # b10
                        (2*bj+1) * p + (2*bk+1),   # b11
                    ]
                    for idx, coeff in zip(b_indices, b_coeffs):
                        V[idx][col] += coeff
                    
                    # C block: rows 2*bi to 2*bi+1, cols 2*bk to 2*bk+1
                    # c00 -> C[2*bi, 2*bk]
                    # c01 -> C[2*bi, 2*bk+1]
                    # c10 -> C[2*bi+1, 2*bk]
                    # c11 -> C[2*bi+1, 2*bk+1]
                    c_indices = [
                        (2*bk) * n + (2*bi),       # c00
                        (2*bk) * n + (2*bi+1),     # c01
                        (2*bk+1) * n + (2*bi),     # c10
                        (2*bk+1) * n + (2*bi+1),   # c11
                    ]
                    for idx, coeff in zip(c_indices, c_coeffs):
                        W[idx][col] += coeff
                    
                    col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
