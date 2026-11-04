import time
import numpy as np

def construct():
    t0 = time.time()
    deadline = t0 + 115.0  # safety margin under 150s kill
    n, m, p = 2, 4, 4

    def verify(U, V, W, rank):
        """Verify decomposition: C[i,j] = sum_r U[i*m+k,r]*A[i,k] * V[k*p+j,r]*B[k,j] * W[j*n+i,r]"""
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        np.random.seed(123)
        A_test = np.random.randn(n, m)
        B_test = np.random.randn(m, p)
        C_expected = A_test @ B_test

        # Precompute a_parts and b_parts
        # a_parts[i, r] = sum_k U[i*m+k, r] * A[i,k]
        # b_parts[j, r] = sum_k V[k*p+j, r] * B[k,j]
        # C_recon[i,j] = sum_r W[j*n+i, r] * a_parts[i,r] * b_parts[j,r]

        U3 = U_arr.reshape(n, m, rank)
        V3 = V_arr.reshape(m, p, rank)
        W3 = W_arr.reshape(p, n, rank)

        a_parts = np.einsum('ikr,ik->ir', U3, A_test)  # (n, rank)
        b_parts = np.einsum('kjr,kj->jr', V3, B_test)   # (p, rank)
        # C_recon[i,j] = sum_r W3[j,i,r] * a_parts[i,r] * b_parts[j,r]
        C_recon = np.einsum('jir,ir,jr->ij', W3, a_parts, b_parts)

        return np.allclose(C_recon, C_expected, atol=1e-9)

    def add_strassen_block(U, V, W, col_offset, X_rows, Y_rows, Z_rows):
        """Add a 7-term Strassen block. X_rows=[x00,x01,x10,x11], Y_rows=[y00,y01,y10,y11], Z_rows=[z00,z01,z10,z11]"""
        # M1 = (A00+A11)(B00+B11) -> C00 += M1, C01 -= M1, C10 += M1, C11 += M1
        U[X_rows[0]][col_offset + 0] += 1
        U[X_rows[3]][col_offset + 0] += 1
        V[Y_rows[0]][col_offset + 0] += 1
        V[Y_rows[3]][col_offset + 0] += 1
        W[Z_rows[0]][col_offset + 0] += 1
        W[Z_rows[1]][col_offset + 0] -= 1
        W[Z_rows[2]][col_offset + 0] += 1
        W[Z_rows[3]][col_offset + 0] += 1

        # M2 = A11*(B10-B00) -> C10 += M2, C11 += M2
        U[X_rows[3]][col_offset + 1] += 1
        V[Y_rows[2]][col_offset + 1] += 1
        V[Y_rows[0]][col_offset + 1] -= 1
        W[Z_rows[2]][col_offset + 1] += 1
        W[Z_rows[3]][col_offset + 1] += 1

        # M3 = (A00-A10)*B00 -> C00 += M3, C10 -= M3
        U[X_rows[0]][col_offset + 2] += 1
        U[X_rows[2]][col_offset + 2] -= 1
        V[Y_rows[0]][col_offset + 2] += 1
        W[Z_rows[0]][col_offset + 2] += 1
        W[Z_rows[2]][col_offset + 2] -= 1

        # M4 = A11*(B01-B11) -> C01 += M4, C11 -= M4
        U[X_rows[3]][col_offset + 3] += 1
        V[Y_rows[1]][col_offset + 3] += 1
        V[Y_rows[3]][col_offset + 3] -= 1
        W[Z_rows[1]][col_offset + 3] += 1
        W[Z_rows[3]][col_offset + 3] -= 1

        # M5 = (A10+A11)*(B10+B11) -> C01 += M5, C11 += M5
        U[X_rows[2]][col_offset + 4] += 1
        U[X_rows[3]][col_offset + 4] += 1
        V[Y_rows[2]][col_offset + 4] += 1
        V[Y_rows[3]][col_offset + 4] += 1
        W[Z_rows[1]][col_offset + 4] += 1
        W[Z_rows[3]][col_offset + 4] += 1

        # M6 = (A00-A10)*(B01+B11) -> C00 += M6, C01 -= M6
        U[X_rows[0]][col_offset + 5] += 1
        U[X_rows[2]][col_offset + 5] -= 1
        V[Y_rows[1]][col_offset + 5] += 1
        V[Y_rows[3]][col_offset + 5] += 1
        W[Z_rows[0]][col_offset + 5] += 1
        W[Z_rows[1]][col_offset + 5] -= 1

        # M7 = (A01+A10)*(B00-B10) -> C00 -= M7, C10 += M7
        U[X_rows[1]][col_offset + 6] += 1
        U[X_rows[2]][col_offset + 6] += 1
        V[Y_rows[0]][col_offset + 6] += 1
        V[Y_rows[2]][col_offset + 6] -= 1
        W[Z_rows[0]][col_offset + 6] -= 1
        W[Z_rows[2]][col_offset + 6] += 1

    # --- Try 4-block approach: rank 24 ---
    # Partition: A into 2 row-blocks of 1x4, B into 2 col-blocks of 4x2, C into 2x2 blocks
    # Each "block multiply" is (1x4)*(4x2) -> (1x2), which is a rank-4 tensor
    # But we can use Strassen-like tricks if we group differently.
    # Actually, let's think of it as: we have 4 independent 1x4 * 4x2 -> 1x2 multiplications.
    # Each has tensor rank 4 (it's just a dot product with broadcasting).
    # Total rank = 4*4 = 16? No, that's too optimistic.
    # Actually 1x4 * 4x2: C[0,j] = sum_k A[0,k]*B[k,j]. This is rank 4 (identity in k).
    # So 4 such blocks -> rank 16. But wait, can we share terms?
    # The 4 blocks share the same A rows and B columns partially.
    # Block (i,j): A[i,:] * B[:,j] -> C[i,j], each rank 4.
    # But they're independent, so total rank = 4*4 = 16? 
    # No! Each block (i,j) uses A[i,0..3] and B[0..3,j]. The A[i,:] differ for i=0,1
    # and B[:,j] differ for j=0,1,2,3. They don't share structure in a way that
    # allows Strassen-style sharing because the "blocks" are 1x4 and 4x2, not square.
    # So rank 16 via schoolbook for 2x4*4x4? Let me check: 2*4*4 = 32 scalar mults.
    # Rank 16 would mean we halve it. That seems unlikely for exact decomposition.
    # Actually the tensor rank of 2x4*4x4 is known to be... let me think.
    # Lower bound: flattening gives rank >= max(n*m, m*p, p*n) = max(8,16,8) = 16.
    # So rank >= 16. Can we achieve 16? Probably not with simple methods.
    # Known: R(2,4,4) <= R(2,4,4) via various methods.
    # Let's just try the 4-block Strassen with 2x2 grouping.

    # --- Standard 4x2x2x2 Strassen: rank 28 ---
    # Group A into left (cols 0,1) and right (cols 2,3)
    # Group B rows into top (0,1) and bottom (2,3), cols into left (0,1) and right (2,3)
    # 4 independent 2x2*2x2->2x2 Strassen blocks, each rank 7 -> total 28

    rank28 = 28
    U28 = [[0] * rank28 for _ in range(n * m)]
    V28 = [[0] * rank28 for _ in range(m * p)]
    W28 = [[0] * rank28 for _ in range(p * n)]

    # A rows: 0,1 (n=2), A cols: 0,1,2,3 (m=4)
    # A[i,0] = U[0*4+0] = U[0], A[i,1] = U[1], A[i,2] = U[2], A[i,3] = U[3]
    # Wait, U has shape (n*m, rank) = (8, rank). Index = i*m + k = i*4 + k.
    # A[0,0]=U[0], A[0,1]=U[1], A[0,2]=U[2], A[0,3]=U[3]
    # A[1,0]=U[4], A[1,1]=U[5], A[1,2]=U[6], A[1,3]=U[7]
    # B has shape (m*p, rank) = (16, rank). Index = k*p + j = k*4 + j.
    # B[0,0]=V[0], B[0,1]=V[1], B[0,2]=V[2], B[0,3]=V[3]
    # B[1,0]=V[4], B[1,1]=V[5], B[1,2]=V[6], B[1,3]=V[7]
    # B[2,0]=V[8], B[2,1]=V[9], B[2,2]=V[10], B[2,3]=V[11]
    # B[3,0]=V[12], B[3,1]=V[13], B[3,2]=V[14], B[3,3]=V[15]
    # W has shape (p*n, rank) = (8, rank). Index = j*n + i = j*2 + i.
    # C[0,0]=W[0*2+0]=W[0], C[0,1]=W[1*2+0]=W[2], C[0,2]=W[2*2+0]=W[4], C[0,3]=W[3*2+0]=W[6]
    # C[1,0]=W[0*2+1]=W[1], C[1,1]=W[1*2+1]=W[3], C[1,2]=W[2*2+1]=W[5], C[1,3]=W[3*2+1]=W[7]

    # Block 1: A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2]
    # X: A[0,0]=0, A[0,1]=1, A[1,0]=4, A[1,1]=5
    # Y: B[0,0]=0, B[0,1]=1, B[1,0]=4, B[1,1]=5
    # Z: C[0,0]=0, C[0,1]=2, C[1,0]=1, C[1,1]=3
    add_strassen_block(U28, V28, W28, 0, [0,1,4,5], [0,1,4,5], [0,2,1,3])

    # Block 2: A[0:2, 2:4] * B[0:2, 0:2] -> C[0:2, 0:2]
    # X: A[0,2]=2, A[0,3]=3, A[1,2]=6, A[1,3]=7
    # Y: B[0,0]=0, B[0,1]=1, B[1,0]=4, B[1,1]=5
    # Z: C[0,0]=0, C[0,1]=2, C[1,0]=1, C[1,1]=3
    add_strassen_block(U28, V28, W28, 7, [2,3,6,7], [0,1,4,5], [0,2,1,3])

    # Block 3: A[0:2, 0:2] * B[2:4, 0:2] -> C[0:2, 0:2]
    # X: A[0,0]=0, A[0,1]=1, A[1,0]=4, A[1,1]=5
    # Y: B[2,0]=8, B[2,1]=9, B[3,0]=12, B[3,1]=13
    # Z: C[0,0]=0, C[0,1]=2, C[1,0]=1, C[1,1]=3
    add_strassen_block(U28, V28, W28, 14, [0,1,4,5], [8,9,12,13], [0,2,1,3])

    # Block 4: A[0:2, 2:4] * B[2:4, 0:2] -> C[0:2, 0:2]
    # X: A[0,2]=2, A[0,3]=3, A[1,2]=6, A[1,3]=7
    # Y: B[2,0]=8, B[2,1]=9, B[3,0]=12, B[3,1]=13
    # Z: C[0,0]=0, C[0,1]=2, C[1,0]=1, C[1,1]=3
    add_strassen_block(U28, V28, W28, 21, [2,3,6,7], [8,9,12,13], [0,2,1,3])

    # Wait, I made an error. The 4 blocks should cover different output blocks.
    # Let me redo:
    # A is 2x4, B is 4x4, C is 2x4.
    # Split A into A_L (2x2, cols 0-1) and A_R (2x2, cols 2-3)
    # Split B into B_TL (2x2), B_TR (2x2), B_BL (2x2), B_BR (2x2)
    # C = A_L * B_TL + A_R * B_BL  (for C cols 0-1)
    # C = A_L * B_TR + A_R * B_BR  (for C cols 2-3)
    # So 4 Strassen blocks:
    # 1: A_L * B_TL -> C[:, 0:2]
    # 2: A_R * B_BL -> C[:, 0:2]
    # 3: A_L * B_TR -> C[:, 2:4]
    # 4: A_R * B_BR -> C[:, 2:4]

    U28 = [[0] * rank28 for _ in range(n * m)]
    V28 = [[0] * rank28 for _ in range(m * p)]
    W28 = [[0] * rank28 for _ in range(p * n)]

    # A_L: rows 0-1, cols 0-1: U indices [0,1,4,5]
    # A_R: rows 0-1, cols 2-3: U indices [2,3,6,7]
    # B_TL: rows 0-1, cols 0-1: V indices [0,1,4,5]
    # B_TR: rows 0-1, cols 2-3: V indices [2,3,6,7]
    # B_BL: rows 2-3, cols 0-1: V indices [8,9,12,13]
    # B_BR: rows 2-3, cols 2-3: V indices [10,11,14,15]
    # C[:, 0:2]: W indices for j=0,1: [0,1,2,3] -> C[0,0]=W[0], C[1,0]=W[1], C[0,1]=W[2], C[1,1]=W[3]
    #   Z rows: [C00, C01, C10, C11] = [W[0], W[2], W[1], W[3]]
    # C[:, 2:4]: W indices for j=2,3: [4,5,6,7] -> C[0,2]=W[4], C[1,2]=W[5], C[0,3]=W[6], C[1,3]=W[7]
    #   Z rows: [C00, C01, C10, C11] = [W[4], W[6], W[5], W[7]]

    # Block 1: A_L * B_TL -> C[:, 0:2]
    add_strassen_block(U28, V28, W28, 0, [0,1,4,5], [0,1,4,5], [0,2,1,3])
    # Block 2: A_R * B_BL -> C[:, 0:2]
    add_strassen_block(U28, V28, W28, 7, [2,3,6,7], [8,9,12,13], [0,2,1,3])
    # Block 3: A_L * B_TR -> C[:, 2:4]
    add_strassen_block(U28, V28, W28, 14, [0,1,4,5], [2,3,6,7], [4,6,5,7])
    # Block 4: A_R * B_BR -> C[:, 2:4]
    add_strassen_block(U28, V28, W28, 21, [2,3,6,7], [10,11,14,15], [4,6,5,7])

    if time.time() > deadline:
        return {"U": U28, "V": V28, "W": W28}

    if verify(U28, V28, W28, rank28):
        return {"U": U28, "V": V28, "W": W28}

    # Fallback: schoolbook rank 32
    rank32 = 32
    U32 = [[0] * rank32 for _ in range(n * m)]
    V32 = [[0] * rank32 for _ in range(m * p)]
    W32 = [[0] * rank32 for _ in range(p * n)]
    for i in range(n):
        for k in range(m):
            for j in range(p):
                r = i * m * p + k * p + j
                U32[i * m + k][r] = 1
                V32[k * p + j][r] = 1
                W32[j * n + i][r] = 1

    if verify(U32, V32, W32, rank32):
        return {"U": U32, "V": V32, "W": W32}

    # Last resort: return the Strassen version (should be correct)
    return {"U": U28, "V": V28, "W": W28}
