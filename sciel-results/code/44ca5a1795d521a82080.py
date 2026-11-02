"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # n=2, m=3, p=4
    # Split B (3x4) into B_left (3x3) and B_right (3x1)
    # Part 1: A(2x3) * B_left(3x3) -> C_left(2x3), rank 13
    # Part 2: A(2x3) * B_right(3x1) -> C_right(2x1), rank 6
    # Total rank = 19
    
    rank = 19
    U = [[0] * rank for _ in range(n * m)]   # 6 x 19
    V = [[0] * rank for _ in range(m * p)]   # 12 x 19
    W = [[0] * rank for _ in range(p * n)]   # 8 x 19
    
    # Part 1: 2x3x3 decomposition (rank 13)
    # Using a structured decomposition. We'll use 7 terms from Strassen-like
    # for the 2x2 top-left block, plus 6 additional terms for the remaining parts.
    
    # Let's define the 13 terms for 2x3x3:
    # A is 2x3: A[i][j] for i in {0,1}, j in {0,1,2}
    # B_left is 3x3: B[j][k] for j in {0,1,2}, k in {0,1,2}
    # C_left is 2x3: C[i][k] for i in {0,1}, k in {0,1,2}
    
    # Term definitions (U_idx, V_idx, W_idx, u_val, v_val, w_val)
    # U index: i*3 + j
    # V index: j*3 + k
    # W index: k*2 + i
    
    terms_233 = []
    
    # 7 Strassen-like terms for the 2x2 block (i in {0,1}, j in {0,1}, k in {0,1})
    # S1 = A00 * B00
    terms_233.append((0*3+0, 0*3+0, 0*2+0, 1, 1, 1))  # C[0][0] += A[0][0]*B[0][0]
    # S2 = A01 * B10
    terms_233.append((0*3+1, 1*3+0, 0*2+0, 1, 1, 1))  # C[0][0] += A[0][1]*B[1][0]
    # S3 = A10 * B01
    terms_233.append((1*3+0, 0*3+1, 1*2+0, 1, 1, 1))  # C[1][1] += A[1][0]*B[0][1]
    # S4 = A11 * B11
    terms_233.append((1*3+1, 1*3+1, 1*2+1, 1, 1, 1))  # C[1][1] += A[1][1]*B[1][1]
    # S5 = (A00+A01) * B00 -> contributes to C[0][0]
    terms_233.append((0*3+0, 0*3+0, 0*2+0, 1, 1, 1))  # Already have A00*B00
    terms_233.append((0*3+1, 0*3+0, 0*2+0, 1, 1, 1))  # A01*B00
    
    # This approach is getting messy. Let me use a cleaner method.
    
    # Actually, let me just use schoolbook for 2x3x3 (18 terms) and see if we can trim.
    # But that gives 18+6=24, which is worse than seed.
    
    # Let me try a different approach: use known rank-13 decomposition.
    # I'll construct it using the following 13 terms:
    
    terms_233 = []
    
    # I'll use a systematic approach. For 2x3x3, the minimum rank is known to be 13.
    # Let me define 13 rank-1 terms.
    
    # Term 1: A[0][0] * B[0][0] -> C[0][0]
    terms_233.append((0, 0, 0, 1, 1, 1))
    # Term 2: A[0][1] * B[1][0] -> C[0][0]
    terms_233.append((1, 3, 0, 1, 1, 1))
    # Term 3: A[0][2] * B[2][0] -> C[0][0]
    terms_233.append((2, 6, 0, 1, 1, 1))
    # Term 4: A[0][0] * B[0][1] -> C[0][1]
    terms_233.append((0, 1, 2, 1, 1, 1))
    # Term 5: A[0][1] * B[1][1] -> C[0][1]
    terms_233.append((1, 4, 2, 1, 1, 1))
    # Term 6: A[0][2] * B[2][1] -> C[0][1]
    terms_233.append((2, 7, 2, 1, 1, 1))
    # Term 7: A[0][0] * B[0][2] -> C[0][2]
    terms_233.append((0, 2, 4, 1, 1, 1))
    # Term 8: A[0][1] * B[1][2] -> C[0][2]
    terms_233.append((1, 5, 4, 1, 1, 1))
    # Term 9: A[0][2] * B[2][2] -> C[0][2]
    terms_233.append((2, 8, 4, 1, 1, 1))
    # Term 10: A[1][0] * B[0][0] -> C[1][0]
    terms_233.append((3, 0, 1, 1, 1, 1))
    # Term 11: A[1][1] * B[1][0] -> C[1][0]
    terms_233.append((4, 3, 1, 1, 1, 1))
    # Term 12: A[1][2] * B[2][0] -> C[1][0]
    terms_233.append((5, 6, 1, 1, 1, 1))
    # Term 13: A[1][0] * B[0][1] -> C[1][1]
    terms_233.append((3, 1, 3, 1, 1, 1))
    
    # This is only 13 terms but they don't cover all C entries. I need to be more careful.
    
    # Let me abandon this and use a valid decomposition even if higher rank.
    # For safety, I'll return the schoolbook decomposition.
    
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
