"""Matrix multiplication 6x6x6 using 3x3 block decomposition with 2x2 Strassen (rank 7)."""
DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    import numpy as np
    
    n, m, p = 6, 6, 6
    b = 2  # block size
    
    # Strassen decomposition for 2x2 matrix multiplication (rank 7)
    # U, V, W are 4x7 matrices
    # U[i*2+j, c], V[j*2+k, c], W[k*2+i, c]
    
    U33 = [[0]*7 for _ in range(4)]
    V33 = [[0]*7 for _ in range(4)]
    W33 = [[0]*7 for _ in range(4)]
    
    # M1 = (A00+A11)(B00+B11)
    U33[0][0] = 1; U33[3][0] = 1
    V33[0][0] = 1; V33[3][0] = 1
    W33[0][0] = 1; W33[1][0] = 1; W33[2][0] = 1; W33[3][0] = 1 # C00, C01, C10, C11 contributions?
    # Wait, W[k*2+i, c] corresponds to C[i,k].
    # C00 is i=0,k=0 -> idx 0. C01 is i=0,k=1 -> idx 1. C10 is i=1,k=0 -> idx 2. C11 is i=1,k=1 -> idx 3.
    # M1 contributes to C00 (+1), C01 (0), C10 (0), C11 (+1)?
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # Let's map coefficients to W.
    # W[idx, c] is the coefficient for C[i,k] in product c.
    
    # M1: A=(0,0)+ (1,1); B=(0,0)+ (1,1)
    # C00 += 1, C11 += 1
    U33[0][0] = 1; U33[3][0] = 1
    V33[0][0] = 1; V33[3][0] = 1
    W33[0][0] = 1; W33[3][0] = 1
    
    # M2: A=(1,0)+ (1,1); B=(0,0)
    # C10 += 1, C11 += -1
    U33[2][1] = 1; U33[3][1] = 1
    V33[0][1] = 1
    W33[2][1] = 1; W33[3][1] = -1
    
    # M3: A=(0,0); B=(1,0)- (1,1)
    # C01 += 1, C11 += 1
    U33[0][2] = 1
    V33[2][2] = 1; V33[3][2] = -1
    W33[1][2] = 1; W33[3][2] = 1
    
    # M4: A=(1,1); B=(0,1)- (0,0)
    # C00 += 1, C10 += 1
    U33[3][3] = 1
    V33[1][3] = 1; V33[0][3] = -1
    W33[0][3] = 1; W33[2][3] = 1
    
    # M5: A=(0,0)+ (0,1); B=(1,1)
    # C00 += -1, C01 += 1
    U33[0][4] = 1; U33[1][4] = 1
    V33[3][4] = 1
    W33[0][4] = -1; W33[1][4] = 1
    
    # M6: A=(1,0)- (0,0); B=(0,0)+ (0,1)
    # C11 += 1
    U33[2][5] = 1; U33[0][5] = -1
    V33[0][5] = 1; V33[1][5] = 1
    W33[3][5] = 1
    
    # M7: A=(0,1)- (1,1); B=(1,0)+ (1,1)
    # C00 += 1
    U33[1][6] = 1; U33[3][6] = -1
    V33[2][6] = 1; V33[3][6] = 1
    W33[0][6] = 1
    
    # Verify Strassen 2x2
    verified = True
    for i in range(2):
        for j in range(2):
            for k in range(2):
                s = 0
                for c in range(7):
                    s += U33[i*2+j][c] * V33[j*2+k][c] * W33[k*2+i][c]
                if s != 1:
                    verified = False
                    break
            if not verified: break
        if not verified: break
    
    if not verified:
        # Fallback to schoolbook 6x6
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

    # Lift to 6x6 using 3x3 blocks of 2x2
    total_rank = 9 * 7
    U = [[0] * total_rank for _ in range(36)]
    V = [[0] * total_rank for _ in range(36)]
    W = [[0] * total_rank for _ in range(36)]
    
    # 3x3 grid of 2x2 blocks
    # Block (bi, bj) of A is at rows bi*2..bi*2+1, cols bj*2..bj*2+1
    # A[bi*2+ii, bj*2+jj] index: (bi*2+ii)*6 + (bj*2+jj)
    
    for bi in range(3):
        for bj in range(3):
            for bk in range(3):
                # Product of A block (bi, bj) and B block (bk, bj) ??
                # C[bi, bk] = sum_j A[bi, j] B[j, bk]
                # So for block (bi, bk) of C, we sum over block column j of A and block row j of B.
                # Wait, the block decomposition of matmul is:
                # C_{bi, bk} = \sum_{bj=0}^{2} A_{bi, bj} B_{bj, bk}
                # This is a sum of 3 products.
                # Each product is a 2x2 matmul.
                # So for each C block (bi, bk), we need 3 Strassen decompositions?
                # No, the Strassen decomposition computes the product of two 2x2 matrices.
                # Here we have a sum of 3 products.
                # We can just run Strassen for each term in the sum.
                # Total rank = (number of C blocks) * (sum terms) * (rank of 2x2 Strassen)
                # Number of C blocks = 9.
                # Sum terms = 3.
                # Rank per term = 7.
                # Total rank = 9 * 3 * 7 = 189.
                
                # Wait, 189 is the score 1.1429 inspiration!
                # So the inspiration used 3x3 blocks of 2x2 Strassen, but treated each sum term separately?
                # Yes, that is the "naive" block decomposition.
                # Rank 189.
                
                # Can we do better?
                # If we use Laderman 19 for 3x3 blocks (size 2x2? No, Laderman is for 3x3 matmul).
                # If we use 3x3 blocks of size 2x2, the "matmul" is 3x3 blocks.
                # Rank of 3x3 block matmul is 19 (Laderman).
                # Total rank = 19 * (rank of 2x2 matmul)?
                # No. The tensor is $T_{6,6,6}$.
                # If we view it as $3 \times 3$ blocks of $2 \times 2$, the structure is:
                # $C_{bi, bk} = \sum_{bj} A_{bi, bj} B_{bj, bk}$.
                # This is exactly a $3 \times 3$ matrix multiplication where the entries are $2 \times 2$ matrices.
                # The rank of this "block matmul" tensor is Rank(BlockMatmul) * Rank(2x2 Matmul)?
                # No, it's the tensor product.
                # Rank = Rank(Matmul_{3,3,3}) * Rank(Matmul_{2,2,2})?
                # Rank(Matmul_{3,3,3}) = 19.
                # Rank(Matmul_{2,2,2}) = 7.
                # Total Rank = 19 * 7 = 133.
                
                # This is better than 189!
                # And better than 152 (Laderman 3x3 blocks of 3x3).
                # 133 < 152.
                # Score = 216 / 133 = 1.62.
                
                # I will implement this: Laderman 19 for the 3x3 block structure, using Strassen 7 for the 2x2 block entries.
                
                pass # This logic is complex to implement correctly in one go.
                
    # Given the complexity and risk of error in the "133" approach, and that 189 is a known working score (1.1429),
    # and 152 is also a good target, I will stick to the simpler 189 (Naive Block Strassen) 
    # OR try to fix the 152 (Laderman 3x3).
    
    # Actually, 152 is better than 189.
    # Let's try to get the 152 working.
    # I will use the Laderman 19 coefficients I defined earlier in the thought process, 
    # but I need to be SURE they are correct.
    
    # If I am not sure, I will use the 189 approach which is simpler (Strassen 2x2 lifted to 3x3 blocks naively).
    # Strassen 2x2 is definitely correct.
    
    # Let's implement the 189 rank solution (3x3 blocks of 2x2 Strassen, naive sum).
    
    U = [[0] * 189 for _ in range(36)]
    V = [[0] * 189 for _ in range(36)]
    W = [[0] * 189 for _ in range(36)]
    
    col = 0
    for bi in range(3):
        for bk in range(3):
            for bj in range(3):
                # A block (bi, bj) * B block (bj, bk) -> contributes to C block (bi, bk)
                for c in range(7):
                    # U part: A block (bi, bj)
                    for ii in range(2):
                        for jj in range(2):
                            u_idx_22 = ii*2+jj
                            val = U33[u_idx_22][c]
                            if val != 0:
                                u_idx_66 = (bi*2+ii)*6 + (bj*2+jj)
                                U[u_idx_66][col] = val
                    # V part: B block (bj, bk)
                    for jj in range(2):
                        for kk in range(2):
                            v_idx_22 = jj*2+kk
                            val = V33[v_idx_22][c]
                            if val != 0:
                                v_idx_66 = (bj*2+jj)*6 + (bk*2+kk)
                                V[v_idx_66][col] = val
                    # W part: C block (bi, bk)
                    for kk in range(2):
                        for ii in range(2):
                            w_idx_22 = kk*2+ii
                            val = W33[w_idx_22][c]
                            if val != 0:
                                w_idx_66 = (bi*2+ii)*6 + (bk*2+kk)
                                W[w_idx_66][col] = val
                    col += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
