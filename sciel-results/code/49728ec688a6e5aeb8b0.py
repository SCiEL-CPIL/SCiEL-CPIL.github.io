"""Matrix multiplication 4x4 x 4x7 tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [4, 4, 7]

def _strassen_4x4x4():
    """
    Return Strassen decomposition for 4x4 x 4x4 matrix multiplication.
    Rank = 49.
    U: (16, 49), V: (16, 49), W: (16, 49)
    Indexing: U[i*4+j, c], V[j*4+k, c], W[k*4+i, c]
    """
    n = 4
    rank = 49
    U = np.zeros((n * n, rank))
    V = np.zeros((n * n, rank))
    W = np.zeros((n * n, rank))
    
    # Strassen for 4x4: split into 2x2 blocks
    # A = [[A11, A12], [A21, A22]] each 2x2
    # B = [[B11, B12], [B21, B22]] each 2x2
    # C = [[C11, C12], [C21, C22]] each 2x2
    #
    # 7 products for 2x2 Strassen:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    #
    # C11 = M1 - M4 + M6 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M3 + M2 + M5
    #
    # For 4x4, we need to apply Strassen at the 2x2 block level.
    # Each "element" in the 2x2 block matrix is a 2x2 matrix.
    # So we need Strassen for 2x2 matrix multiplication, which has rank 7.
    # But we're doing 4x4 = 2x2 blocks of 2x2, so we need 7 multiplications
    # of 2x2 matrices. Each 2x2 multiplication has rank 4 (schoolbook).
    # Total rank = 7 * 4 = 28? No, that's not right either.
    #
    # Actually for 4x4x4 Strassen: We decompose into 2x2 blocks.
    # The 7 Strassen products each involve 2x2 matrix multiplications.
    # Each 2x2x2 multiplication has rank 4 (schoolbook, since 2x2x2 has rank 7? No, rank 4 for schoolbook).
    # Wait, 2x2x2 tensor rank is 7 (Strassen) or 4 (schoolbook = 2*2*2=8? No, 2*2*2=8).
    # Hmm, 2x2x2 schoolbook rank is 8. Strassen gives 7.
    #
    # So for 4x4x4: 7 Strassen products, each requiring a 2x2x2 multiplication.
    # If we use schoolbook for each 2x2x2: 7 * 8 = 56.
    # If we use Strassen for each 2x2x2: 7 * 7 = 49.
    # That gives rank 49!
    
    # Let me build this properly.
    # 4x4 matrix indices: (i,j) where i,j in 0..3
    # Block structure: block row br = i//2, block col bc = j//2, local (lr, lc) = (i%2, j%2)
    
    # Strassen at block level (2x2 blocks):
    # M1 = (A_BR0,BC0 + A_BR1,BC1)(B_BR0,BC0 + B_BR1,BC1)
    # M2 = (A_BR1,BC0 + A_BR1,BC1)B_BR0,BC0
    # M3 = A_BR0,BC0(B_BR0,BC1 - B_BR1,BC1)
    # M4 = A_BR1,BC1(B_BR1,BC0 - B_BR0,BC0)
    # M5 = (A_BR0,BC0 + A_BR0,BC1)B_BR1,BC1
    # M6 = (A_BR1,BC0 - A_BR0,BC0)(B_BR0,BC0 + B_BR0,BC1)
    # M7 = (A_BR0,BC1 - A_BR1,BC1)(B_BR1,BC0 + B_BR1,BC1)
    
    # Each Mi is a 2x2 matrix product, done with schoolbook (rank 4 each)
    # Total: 7 * 4 = 28... but that seems too low. Let me reconsider.
    
    # Actually, I think the standard 4x4 Strassen has rank 49.
    # Let me just use a known construction.
    
    # Alternative: use the fact that matmul(4,4,4) can be done with rank 49
    # by using 7 Strassen products where each is a matmul(2,2,2) with rank 7.
    # 7 * 7 = 49.
    
    # Let me build it using nested Strassen.
    
    # For each of the 7 Strassen products at the 4x4 level,
    # we need to compute a 2x2 x 2x2 = 2x2 product.
    # Each such product uses 7 rank-1 terms (Strassen for 2x2x2).
    
    # 2x2x2 Strassen:
    # For matrices P(2x2), Q(2x2), R=P*Q:
    # N1 = (P00+P11)(Q00+Q11)
    # N2 = (P10+P11)Q00
    # N3 = P00(Q01-Q11)
    # N4 = P11(Q10-Q00)
    # N5 = (P00+P01)Q11
    # N6 = (P10-P00)(Q00+Q01)
    # N7 = (P01-P11)(Q10+Q11)
    # R00 = N1 - N4 + N6 + N7
    # R01 = N3 + N5
    # R10 = N2 + N4
    # R11 = N1 - N3 + N2 + N5
    
    # So for the 4x4 case, we have 7 block-level Strassen products.
    # Each block product is a 2x2x2 multiplication, done with 7 rank-1 terms.
    # Total: 7 * 7 = 49 rank-1 terms.
    
    # Let me index: for block-level product m (0..6), and within that, 
    # sub-product s (0..6), we have a rank-1 term.
    
    col = 0
    for m in range(7):
        for s in range(7):
            # Determine which 2x2 sub-matrices are involved in block product m
            # and which Strassen term s within that.
            
            # First, determine the A and B block combinations for Strassen product m
            # m=0: (A00+A11)(B00+B11)
            # m=1: (A10+A11)B00
            # m=2: A00(B01-B11)
            # m=3: A11(B10-B00)
            # m=4: (A00+A01)B11
            # m=5: (A10-A00)(B00+B01)
            # m=6: (A01-A11)(B10+B11)
            
            # For each, we have "left" and "right" 2x2 block combinations
            # left: which A blocks with what signs
            # right: which B blocks with what signs
            
            if m == 0:
                a_blocks = [(0, 0, 1), (1, 1, 1)]  # (br, bc, sign)
                b_blocks = [(0, 0, 1), (1, 1, 1)]
            elif m == 1:
                a_blocks = [(1, 0, 1), (1, 1, 1)]
                b_blocks = [(0, 0, 1)]
            elif m == 2:
                a_blocks = [(0, 0, 1)]
                b_blocks = [(0, 1, 1), (1, 1, -1)]
            elif m == 3:
                a_blocks = [(1, 1, 1)]
                b_blocks = [(1, 0, 1), (0, 0, -1)]
            elif m == 4:
                a_blocks = [(0, 0, 1), (0, 1, 1)]
                b_blocks = [(1, 1, 1)]
            elif m == 5:
                a_blocks = [(1, 0, 1), (0, 0, -1)]
                b_blocks = [(0, 0, 1), (0, 1, 1)]
            elif m == 6:
                a_blocks = [(0, 1, 1), (1, 1, -1)]
                b_blocks = [(1, 0, 1), (1, 1, 1)]
            
            # Now for sub-product s, determine the 2x2 Strassen term
            # s=0: (P00+P11)(Q00+Q11)
            # s=1: (P10+P11)Q00
            # s=2: P00(Q01-Q11)
            # s=3: P11(Q10-Q00)
            # s=4: (P00+P01)Q11
            # s=5: (P10-P00)(Q00+Q01)
            # s=6: (P01-P11)(Q10+Q11)
            
            if s == 0:
                p_blocks = [(0, 0, 1), (1, 1, 1)]
                q_blocks = [(0, 0, 1), (1, 1, 1)]
            elif s == 1:
                p_blocks = [(1, 0, 1), (1, 1, 1)]
                q_blocks = [(0, 0, 1)]
            elif s == 2:
                p_blocks = [(0, 0, 1)]
                q_blocks = [(0, 1, 1), (1, 1, -1)]
            elif s == 3:
                p_blocks = [(1, 1, 1)]
                q_blocks = [(1, 0, 1), (0, 0, -1)]
            elif s == 4:
                p_blocks = [(0, 0, 1), (0, 1, 1)]
                q_blocks = [(1, 1, 1)]
            elif s == 5:
                p_blocks = [(1, 0, 1), (0, 0, -1)]
                q_blocks = [(0, 0, 1), (0, 1, 1)]
            elif s == 6:
                p_blocks = [(0, 1, 1), (1, 1, -1)]
                q_blocks = [(1, 0, 1), (1, 1, 1)]
            
            # Now, for each (i,j) in the 4x4 A matrix, determine the coefficient
            # A[i,j] is in block (br, bc) = (i//2, j//2), local (lr, lc) = (i%2, j%2)
            # For the combined "P" matrix in sub-product s, P[lr,lc] gets contributions
            # from p_blocks where (br,lc) matches... wait, I need to be more careful.
            
            # The "P" matrix in the 2x2x2 Strassen is formed from the A block combination.
            # P[lr, lc] = sum over (br, bc, sign) in p_blocks of sign * A_block[br,bc][lr,lc]
            # But A_block[br,bc] is the 2x2 block at block position (br,bc).
            # A_block[br,bc][lr,lc] = A[br*2+lr, bc*2+lc]
            
            # Similarly for Q from B blocks.
            
            # For U[i*4+j, col]: coefficient of A[i,j] in this rank-1 term
            # i = br*2+lr, j = bc*2+lc
            # U[i*4+j, col] = sum over (br2, bc2, sign_p) in p_blocks of sign_p * [br2==br and bc2==bc]
            # Wait, that's not right. P[lr,lc] = sum over p_blocks of sign * A_block[br,bc][lr,lc]
            # A_block[br,bc][lr,lc] corresponds to A[br*2+lr, bc*2+lc]
            # So for A[i,j] where i=br*2+lr, j=bc*2+lc:
            # It appears in P[lr,lc] if (br,bc) is in p_blocks, with the corresponding sign.
            
            # Actually, P is a 2x2 matrix where P[lr,lc] = sum_{(br,bc,sign) in p_blocks} sign * A[br*2+lr, bc*2+lc]
            # So A[i,j] with i=br*2+lr, j=bc*2+lc contributes to P[lr,lc] with coefficient = sum of signs for (br,bc) in p_blocks.
            
            # For the rank-1 term, U[i*4+j, col] = coefficient of A[i,j] in P
            # = sum over (br2, bc2, sign) in p_blocks: sign if br2*2+lr == i and bc2*2+lc == j
            # = sum over (br2, bc2, sign) in p_blocks: sign if br2 == i//2 and bc2 == j//2
            
            for i in range(4):
                for j in range(4):
                    br, bc = i // 2, j // 2
                    coeff = 0.0
                    for br2, bc2, sign in p_blocks:
                        if br2 == br and bc2 == bc:
                            coeff += sign
                    U[i * 4 + j, col] = coeff
            
            # For V[j*4+k, col]: coefficient of B[j,k] in Q
            # Q[lr,lc] = sum_{(br,bc,sign) in q_blocks} sign * B_block[br,bc][lr,lc]
            # B_block[br,bc][lr,lc] = B[br*2+lr, bc*2+lc]
            # B[j,k] where j=br*2+lr, k=bc*2+lc
            for j in range(4):
                for k in range(4):
                    br, bc = j // 2, k // 2
                    coeff = 0.0
                    for br2, bc2, sign in q_blocks:
                        if br2 == br and bc2 == bc:
                            coeff += sign
                    V[j * 4 + k, col] = coeff
            
            # For W[k*4+i, col]: coefficient in the output
            # The output R = P*Q is a 2x2 matrix, and R[lr,lc] contributes to C_block[br_out, bc_out]
            # where br_out, bc_out are determined by which Strassen product m we're in.
            
            # For Strassen product m, the result block is:
            # m=0: M1 contributes to C00, C11 (with signs)
            # m=1: M2 contributes to C10, C11
            # m=2: M3 contributes to C01, C11
            # m=3: M4 contributes to C00, C10
            # m=4: M5 contributes to C01, C11
            # m=5: M6 contributes to C00
            # m=6: M7 contributes to C00, C11
            
            # More precisely:
            # C00 = M1 - M4 + M6 + M7
            # C01 = M3 + M5
            # C10 = M2 + M4
            # C11 = M1 - M3 + M2 + M5
            
            # So for product m, which C blocks does it contribute to, and with what sign?
            if m == 0:  # M1
                c_contribs = [(0, 0, 1), (1, 1, 1)]  # (cbr, cbc, sign)
            elif m == 1:  # M2
                c_contribs = [(1, 0, 1), (1, 1, 1)]
            elif m == 2:  # M3
                c_contribs = [(0, 1, 1), (1, 1, -1)]
            elif m == 3:  # M4
                c_contribs = [(0, 0, -1), (1, 0, 1)]
            elif m == 4:  # M5
                c_contribs = [(0, 1, 1), (1, 1, 1)]
            elif m == 5:  # M6
                c_contribs = [(0, 0, 1)]
            elif m == 6:  # M7
                c_contribs = [(0, 0, 1), (1, 1, 1)]
            
            # Now for sub-product s, R = P*Q is a 2x2 matrix.
            # R[lr,lc] = sum over Strassen terms... but we're already in a specific s.
            # Actually, R is the result of the 2x2x2 multiplication, and we're decomposing it.
            # The rank-1 term contributes to R[lr,lc] where lr,lc are determined by s.
            
            # For 2x2x2 Strassen, the 7 terms contribute to R as:
            # R00 = N1 - N4 + N6 + N7
            # R01 = N3 + N5
            # R10 = N2 + N4
            # R11 = N1 - N3 + N2 + N5
            
            # So for term s, which R elements does it contribute to?
            if s == 0:  # N1
                r_contribs = [(0, 0, 1), (1, 1, 1)]
            elif s == 1:  # N2
                r_contribs = [(1, 0, 1), (1, 1, 1)]
            elif s == 2:  # N3
                r_contribs = [(0, 1, 1), (1, 1, -1)]
            elif s == 3:  # N4
                r_contribs = [(0, 0, -1), (1, 0, 1)]
            elif s == 4:  # N5
                r_contribs = [(0, 1, 1), (1, 1, 1)]
            elif s == 5:  # N6
                r_contribs = [(0, 0, 1)]
            elif s == 6:  # N7
                r_contribs = [(0, 0, 1), (1, 1, 1)]
            
            # Now, R[lr,lc] is the result of the 2x2 block product.
            # This R is the block at position (cbr, cbc) in the output C.
            # C[cbr*2+lr, cbc*2+lc] gets contribution from R[lr,lc] with the sign from c_contribs.
            
            # So W[k*4+i, col] where k = cbr*2+lr, i = cbc*2+lc
            # Wait, the output indexing is W[k*n+i, c] where the output C[i,k]... 
            # Let me re-check the indexing convention.
            
            # The problem states: W[k*n+i, r] where n=4.
            # The tensor decomposition: T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
            # Here i is row of A (0..n-1), j is col of A / row of B (0..m-1), k is col of B (0..p-1)
            # Output C[i,k] = sum_j A[i,j]*B[j,k]
            # So W is indexed by (k, i) -> k*n + i
            
            # For our 4x4x4 case: n=m=p=4
            # W[k*4+i, col] where i is output row, k is output col.
            
            # R[lr,lc] contributes to C_block[cbr,cbc][lr,lc] = C[cbr*2+lr, cbc*2+lc]
            # So output row i = cbr*2+lr, output col k = cbc*2+lc
            # W index = k*4+i = (cbc*2+lc)*4 + (cbr*2+lr)
            
            for cbr, cbc, sign_c in c_contribs:
                for lr, lc, sign_r in r_contribs:
                    i_out = cbr * 2 + lr
                    k_out = cbc * 2 + lc
                    W[k_out * 4 + i_out, col] += sign_c * sign_r
            
            col += 1
    
    assert col == 49, f"Expected 49 columns, got {col}"
    return U, V, W


def _matmul_4x4x3():
    """
    Schoolbook decomposition for 4x4 x 4x3 matrix multiplication.
    Rank = 48.
    U: (16, 48), V: (12, 48), W: (12, 48)
    Indexing: U[i*4+j, c], V[j*3+k, c], W[k*4+i, c]
    """
    n, m, p = 4, 4, 3
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    
    assert col == 48
    return U, V, W


def construct():
    n, m, p = DIMENSIONS  # 4, 4, 7
    start_time = time.time()
    deadline = start_time + 140
    
    # Split B (4x7) into B1 (4x4) and B2 (4x3)
    # C = A * B = [A*B1 | A*B2]
    # Part 1: A(4x4) * B1(4x4) -> C1(4x4), rank 49 (Strassen)
    # Part 2: A(4x4) * B2(4x3) -> C2(4x3), rank 48 (schoolbook)
    # Total rank: 97
    
    U1, V1, W1 = _strassen_4x4x4()  # shapes (16,49), (16,49), (16,49)
    U2, V2, W2 = _matmul_4x4x3()     # shapes (16,48), (12,48), (12,48)
    
    total_rank = 49 + 48
    
    U = np.zeros((n * m, total_rank))  # (16, 97)
    V = np.zeros((m * p, total_rank))  # (28, 97)
    W = np.zeros((p * n, total_rank))  # (28, 97)
    
    # Part 1: columns 0..48
    # U1: (16, 49) -> U[:, 0:49]
    U[:, 0:49] = U1
    
    # V1: (16, 49) corresponds to B1 which is 4x4, so V1[j*4+k, c] for j=0..3, k=0..3
    # In the full V (28, 97), B is 4x7, so V[j*7+k, c] for j=0..3, k=0..6
    # B1 occupies columns k=0..3, so V1[j*4+k, c] -> V[j*7+k, c] for k=0..3
    for j in range(4):
        for k in range(4):
            V[j * 7 + k, 0:49] = V1[j * 4 + k, :]
    
    # W1: (16, 49) corresponds to C1 which is 4x4, so W1[k*4+i, c] for k=0..3, i=0..3
    # In the full W (28, 97), C is 4x7, so W[k*4+i, c] for k=0..6, i=0..3
    # C1 occupies columns k=0..3
    for k in range(4):
        for i in range(4):
            W[k * 4 + i, 0:49] = W1[k * 4 + i, :]
    
    # Part 2: columns 49..96
    # U2: (16, 48) -> U[:, 49:97]
    U[:, 49:97] = U2
    
    # V2: (12, 48) corresponds to B2 which is 4x3, so V2[j*3+k, c] for j=0..3, k=0..2
    # In the full V, B2 occupies columns k=4..6, so V2[j*3+k, c] -> V[j*7+(k+4), c]
    for j in range(4):
        for k in range(3):
            V[j * 7 + (k + 4), 49:97] = V2[j * 3 + k, :]
    
    # W2: (12, 48) corresponds to C2 which is 4x3, so W2[k*4+i, c] for k=0..2, i=0..3
    # In the full W, C2 occupies columns k=4..6
    for k in range(3):
        for i in range(4):
            W[k * 4 + i, 49:97] = W2[k * 4 + i, :]
    
    if time.time() > deadline:
        pass  # We've already computed everything
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }

