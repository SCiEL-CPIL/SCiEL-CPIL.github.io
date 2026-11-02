"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _strassen_2x2():
    """Return (U, V, W) for 2x2 * 2x2 = 2x2 using Strassen's 7-term algorithm.
    U: (4, 7), V: (4, 7), W: (4, 7)
    Indexing: U[i*2+j], V[j*2+k], W[k*2+i]
    """
    U = np.zeros((4, 7), dtype=int)
    V = np.zeros((4, 7), dtype=int)
    W = np.zeros((4, 7), dtype=int)
    
    # Term 1: (A00+A11)(B00+B11) -> C00, C11
    U[0,0]=1; U[3,0]=1
    V[0,0]=1; V[3,0]=1
    W[0,0]=1; W[3,0]=1
    
    # Term 2: (A10+A11)B00 -> C10, C11
    U[2,1]=1; U[3,1]=1
    V[0,1]=1
    W[2,1]=1; W[3,1]=1
    
    # Term 3: A00(B01-B11) -> C01, C11
    U[0,2]=1
    V[1,2]=1; V[3,2]=-1
    W[1,2]=1; W[3,2]=1
    
    # Term 4: A11(B10-B00) -> C00, C10
    U[3,3]=1
    V[2,3]=1; V[0,3]=-1
    W[0,3]=1; W[2,3]=1
    
    # Term 5: (A00+A01)B11 -> C00, C01
    U[0,4]=1; U[1,4]=1
    V[3,4]=1
    W[0,4]=1; W[1,4]=1
    
    # Term 6: (A10-A00)(B00+B01) -> C11
    U[2,5]=1; U[0,5]=-1
    V[0,5]=1; V[1,5]=1
    W[3,5]=1
    
    # Term 7: (A01-A11)(B10+B11) -> C00
    U[1,6]=1; U[3,6]=-1
    V[2,6]=1; V[3,6]=1
    W[0,6]=1
    
    return U, V, W

def _schoolbook_block(n1, m1, p1):
    """Schoolbook decomposition for n1 x m1 * m1 x p1 -> n1 x p1.
    Returns (U, V, W) as numpy arrays.
    U: (n1*m1, rank), V: (m1*p1, rank), W: (p1*n1, rank)
    """
    rank = n1 * m1 * p1
    U = np.zeros((n1*m1, rank), dtype=int)
    V = np.zeros((m1*p1, rank), dtype=int)
    W = np.zeros((p1*n1, rank), dtype=int)
    col = 0
    for i in range(n1):
        for j in range(m1):
            for k in range(p1):
                U[i*m1+j, col] = 1
                V[j*p1+k, col] = 1
                W[k*n1+i, col] = 1
                col += 1
    return U, V, W

def _embed_block(U_block, V_block, W_block, n1, m1, p1, 
                 i_off, j_off, k_off, n, m, p):
    """Embed a block decomposition into the full 4x5x6 tensor.
    
    U_block: (n1*m1, r) for A[i_off:i_off+n1, j_off:j_off+m1]
    V_block: (m1*p1, r) for B[j_off:j_off+m1, k_off:k_off+p1]
    W_block: (p1*n1, r) for C[i_off:i_off+n1, k_off:k_off+p1]
    
    Returns full-sized (U, V, W) contributions of shape (n*m, r), (m*p, r), (p*n, r)
    """
    r = U_block.shape[1]
    U_full = np.zeros((n*m, r), dtype=int)
    V_full = np.zeros((m*p, r), dtype=int)
    W_full = np.zeros((p*n, r), dtype=int)
    
    # Embed U: rows i*m+j for i in [i_off, i_off+n1), j in [j_off, j_off+m1)
    for ii in range(n1):
        for jj in range(m1):
            U_full[(i_off+ii)*m + (j_off+jj), :] = U_block[ii*m1+jj, :]
    
    # Embed V: rows j*p+k for j in [j_off, j_off+m1), k in [k_off, k_off+p1)
    for jj in range(m1):
        for kk in range(p1):
            V_full[(j_off+jj)*p + (k_off+kk), :] = V_block[jj*p1+kk, :]
    
    # Embed W: rows k*n+i for k in [k_off, k_off+p1), i in [i_off, i_off+n1)
    for kk in range(p1):
        for ii in range(n1):
            W_full[(k_off+kk)*n + (i_off+ii), :] = W_block[kk*n1+ii, :]
    
    return U_full, V_full, W_full

def construct():
    n, m, p = DIMENSIONS  # 4, 5, 6
    t0 = time.time()
    
    # Block structure:
    # A (4x5): rows split 2+2, cols split 2+3
    # B (5x6): rows split 2+3, cols split 3+3
    # C (4x6): rows split 2+2, cols split 3+3
    #
    # A = [[A00 (2x2), A01 (2x3)],
    #      [A10 (2x2), A11 (2x3)]]
    # B = [[B00 (2x3), B01 (2x3)],
    #      [B10 (3x3), B11 (3x3)]]
    # C = [[C00 (2x3), C01 (2x3)],
    #      [C10 (2x3), C11 (2x3)]]
    #
    # C00 = A00*B00 + A01*B10
    # C01 = A00*B01 + A01*B11
    # C10 = A10*B00 + A11*B10
    # C11 = A10*B01 + A11*B11
    #
    # Block multiplications needed:
    # 1. A00 (2x2) * B00 (2x3) -> C00 (2x3): schoolbook rank 12
    # 2. A01 (2x3) * B10 (3x3) -> C00 (2x3): schoolbook rank 18
    # 3. A00 (2x2) * B01 (2x3) -> C01 (2x3): schoolbook rank 12
    # 4. A01 (2x3) * B11 (3x3) -> C01 (2x3): schoolbook rank 18
    # 5. A10 (2x2) * B00 (2x3) -> C10 (2x3): schoolbook rank 12
    # 6. A11 (2x3) * B10 (3x3) -> C10 (2x3): schoolbook rank 18
    # 7. A10 (2x2) * B01 (2x3) -> C11 (2x3): schoolbook rank 12
    # 8. A11 (2x3) * B11 (3x3) -> C11 (2x3): schoolbook rank 18
    #
    # Total schoolbook: 12+18+12+18+12+18+12+18 = 120
    #
    # Now, for the 2x2 * 2x3 blocks, we can use Strassen on the 2x2 part
    # of A and expand to 2x3.
    #
    # Actually, let me think differently. For 2x2 * 2x3:
    # A is 2x2, B is 2x3. We can view B as [B_left (2x1), B_mid (2x1), B_right (2x1)]
    # Or better: use Strassen-like approach.
    #
    # For 2x2 * 2x3, the minimal rank is known to be 11 (vs schoolbook 12).
    # But let me first try a simpler structured approach.
    #
    # Alternative: Use the fact that 2x2*2x3 can be done with rank 11
    # by combining Strassen ideas.
    #
    # Actually, let me try a different blocking:
    # A (4x5): rows 2+2, cols 3+2
    # B (5x6): rows 3+2, cols 2+2+2
    # C (4x6): rows 2+2, cols 2+2+2
    #
    # This gives blocks:
    # A00 (2x3), A01 (2x2)
    # A10 (2x3), A11 (2x2)
    # B00 (3x2), B01 (3x2), B02 (3x2)
    # B10 (2x2), B11 (2x2), B12 (2x2)
    #
    # C00 = A00*B00 + A01*B10  (2x3*3x2 + 2x2*2x2 -> 2x2)
    # C01 = A00*B01 + A01*B11  (2x3*3x2 + 2x2*2x2 -> 2x2)
    # C02 = A00*B02 + A01*B12  (2x3*3x2 + 2x2*2x2 -> 2x2)
    # C10 = A10*B00 + A11*B10  (2x3*3x2 + 2x2*2x2 -> 2x2)
    # C11 = A10*B01 + A11*B11  (2x3*3x2 + 2x2*2x2 -> 2x2)
    # C12 = A10*B02 + A11*B12  (2x3*3x2 + 2x2*2x2 -> 2x2)
    #
    # For 2x3*3x2: schoolbook rank 18
    # For 2x2*2x2: can use Strassen rank 7
    #
    # Total: 6*(18+7) = 6*25 = 150. That's worse!
    #
    # Let me try yet another approach. 
    #
    # Key insight: For 2x2 * 2x3, we can use rank 11.
    # Here's how: Split B (2x3) into B1 (2x2) + B2 (2x1).
    # A*B1 uses Strassen (rank 7) but B1 is 2x2, result is 2x2.
    # A*B2 uses schoolbook (rank 4), result is 2x1.
    # Total: 7+4 = 11.
    #
    # But we need to be careful about the W indexing.
    #
    # Actually, let me just carefully construct the 2x2*2x3 rank-11 decomposition.
    #
    # For A (2x2) * B (2x3) -> C (2x3):
    # C[i,k] = sum_j A[i,j] * B[j,k]
    #
    # Split B into B[:,0:2] (2x2) and B[:,2] (2x1).
    # C[:,0:2] = A * B[:,0:2]  (2x2 * 2x2 -> 2x2, use Strassen rank 7)
    # C[:,2]   = A * B[:,2]    (2x2 * 2x1 -> 2x1, schoolbook rank 4)
    #
    # Total rank: 7 + 4 = 11
    #
    # Similarly for 2x3 * 3x3:
    # Split A (2x3) into A[:,0:2] (2x2) and A[:,2] (2x1).
    # C[:,k] = A[:,0:2]*B[0:2,k] + A[:,2]*B[2,k]
    # This doesn't decompose as nicely.
    #
    # Alternative for 2x3*3x3: Split B (3x3) into B[0:2,:] (2x3) and B[2,:] (1x3).
    # A * B[0:2,:] is 2x3 * 2x3 -> 2x3, schoolbook rank 12
    # A * B[2,:] is 2x3 * 1x3 -> 2x3, rank 2*3*1 = 6... no, it's 2*1*3=6
    # Total: 12+6 = 18. Same as schoolbook.
    #
    # Hmm. Let me try: Split A (2x3) into rows... no, A is 2x3.
    # Split B (3x3) into B[0:2,0:2] (2x2), B[0:2,2] (2x1), B[2,0:2] (1x2), B[2,2] (1x1).
    # This gets complicated.
    #
    # Let me just use the first blocking and accept rank 120 for now,
    # but implement it cleanly with the Strassen 2x2*2x2 sub-blocks.
    #
    # Actually, the simplest improvement: use the first blocking (2+2, 2+3, 3+3)
    # and for the 2x2*2x3 blocks, use rank 11 (Strassen on 2x2 part + schoolbook on 2x1 part).
    #
    # Let me implement this carefully.
    
    # I'll build the full decomposition by summing contributions from each block multiplication.
    
    U_total = np.zeros((n*m, 0), dtype=int)
    V_total = np.zeros((m*p, 0), dtype=int)
    W_total = np.zeros((p*n, 0), dtype=int)
    
    def add_contribution(U_c, V_c, W_c):
        nonlocal U_total, V_total, W_total
        r = U_c.shape[1]
        U_total = np.hstack([U_total, U_c])
        V_total = np.hstack([V_total, V_c])
        W_total = np.hstack([W_total, W_c])
    
    # Block definitions:
    # A row blocks: [0,2), [2,4)
    # A col blocks: [0,2), [2,5)
    # B row blocks: [0,2), [2,5)
    # B col blocks: [0,3), [3,6)
    
    # C00 = A[0:2,0:2]*B[0:2,0:3] + A[0:2,2:5]*B[2:5,0:3]
    # C01 = A[0:2,0:2]*B[0:2,3:6] + A[0:2,2:5]*B[2:5,3:6]
    # C10 = A[2:4,0:2]*B[0:2,0:3] + A[2:4,2:5]*B[2:5,0:3]
    # C11 = A[2:4,0:2]*B[0:2,3:6] + A[2:4,2:5]*B[2:5,3:6]
    
    # For each term, we need a decomposition of the block multiplication.
    
    # Helper: 2x2 * 2x3 -> 2x3 with rank 11
    def _mm_2x2_2x3():
        """Decomposition of 2x2 * 2x3 -> 2x3 with rank 11.
        U: (4, 11), V: (6, 11), W: (6, 11)
        U indexing: i*2+j
        V indexing: j*3+k  (j in 0,1; k in 0,1,2)
        W indexing: k*2+i  (k in 0,1,2; i in 0,1)
        """
        # Part 1: A * B[:,0:2] using Strassen (rank 7)
        # B[:,0:2] is 2x2, indexed as B[j, k] for k=0,1
        # In V, this corresponds to rows j*3+k for k=0,1
        U_s, V_s, W_s = _strassen_2x2()
        # V_s has shape (4,7) with indexing j*2+k for k=0,1
        # We need to map to V with indexing j*3+k for k=0,1
        V_s_mapped = np.zeros((6, 7), dtype=int)
        for j in range(2):
            for k in range(2):
                V_s_mapped[j*3+k, :] = V_s[j*2+k, :]
        # W_s has shape (4,7) with indexing k*2+i for k=0,1
        # We need to map to W with indexing k*2+i for k=0,1,2 (only k=0,1 used)
        W_s_mapped = np.zeros((6, 7), dtype=int)
        for k in range(2):
            for i in range(2):
                W_s_mapped[k*2+i, :] = W_s[k*2+i, :]
        
        # Part 2: A * B[:,2] using schoolbook (rank 4)
        # B[:,2] is 2x1, indexed as B[j, 2]
        # In V, row j*3+2
        # C[:,2] is 2x1, indexed as C[i, 2]
        # In W, row 2*2+i = 4+i
        U_p2 = np.zeros((4, 4), dtype=int)
        V_p2 = np.zeros((6, 4), dtype=int)
        W_p2 = np.zeros((6, 4), dtype=int)
        col = 0
        for i in range(2):
            for j in range(2):
                U_p2[i*2+j, col] = 1
                V_p2[j*3+2, col] = 1
                W_p2[2*2+i, col] = 1
                col += 1
        
        U_11 = np.hstack([U_s, U_p2])
        V_11 = np.hstack([V_s_mapped, V_p2])
        W_11 = np.hstack([W_s_mapped, W_p2])
        return U_11, V_11, W_11
    
    # Helper: 2x3 * 3x3 -> 2x3 with rank 18 (schoolbook)
    def _mm_2x3_3x3():
        return _schoolbook_block(2, 3, 3)
    
    # Now build all 8 block contributions:
    
    # 1. A[0:2,0:2] (2x2) * B[0:2,0:3] (2x3) -> C[0:2,0:3] (2x3)
    U1, V1, W1 = _mm_2x2_2x3()
    U1f, V1f, W1f = _embed_block(U1, V1, W1, 2, 2, 3, 0, 0, 0, n, m, p)
    add_contribution(U1f, V1f, W1f)
    
    # 2. A[0:2,2:5] (2x3) * B[2:5,0:3] (3x3) -> C[0:2,0:3] (2x3)
    U2, V2, W2 = _mm_2x3_3x3()
    U2f, V2f, W2f = _embed_block(U2, V2, W2, 2, 3, 3, 0, 2, 0, n, m, p)
    add_contribution(U2f, V2f, W2f)
    
    # 3. A[0:2,0:2] (2x2) * B[0:2,3:6] (2x3) -> C[0:2,3:6] (2x3)
    U3, V3, W3 = _mm_2x2_2x3()
    U3f, V3f, W3f = _embed_block(U3, V3, W3, 2, 2, 3, 0, 0, 3, n, m, p)
    add_contribution(U3f, V3f, W3f)
    
    # 4. A[0:2,2:5] (2x3) * B[2:5,3:6] (3x3) -> C[0:2,3:6] (2x3)
    U4, V4, W4 = _mm_2x3_3x3()
    U4f, V4f, W4f = _embed_block(U4, V4, W4, 2, 3, 3, 0, 2, 3, n, m, p)
    add_contribution(U4f, V4f, W4f)
    
    # 5. A[2:4,0:2] (2x2) * B[0:2,0:3] (2x3) -> C[2:4,0:3] (2x3)
    U5, V5, W5 = _mm_2x2_2x3()
    U5f, V5f, W5f = _embed_block(U5, V5, W5, 2, 2, 3, 2, 0, 0, n, m, p)
    add_contribution(U5f, V5f, W5f)
    
    # 6. A[2:4,2:5] (2x3) * B[2:5,0:3] (3x3) -> C[2:4,0:3] (2x3)
    U6, V6, W6 = _mm_2x3_3x3()
    U6f, V6f, W6f = _embed_block(U6, V6, W6, 2, 3, 3, 2, 2, 0, n, m, p)
    add_contribution(U6f, V6f, W6f)
    
    # 7. A[2:4,0:2] (2x2) * B[0:2,3:6] (2x3) -> C[2:4,3:6] (2x3)
    U7, V7, W7 = _mm_2x2_2x3()
    U7f, V7f, W7f = _embed_block(U7, V7, W7, 2, 2, 3, 2, 0, 3, n, m, p)
    add_contribution(U7f, V7f, W7f)
    
    # 8. A[2:4,2:5] (2x3) * B[2:5,3:6] (3x3) -> C[2:4,3:6] (2x3)
    U8, V8, W8 = _mm_2x3_3x3()
    U8f, V8f, W8f = _embed_block(U8, V8, W8, 2, 3, 3, 2, 2, 3, n, m, p)
    add_contribution(U8f, V8f, W8f)
    
    # Convert to lists
    U_list = U_total.tolist()
    V_list = V_total.tolist()
    W_list = W_total.tolist()
    
    rank = U_total.shape[1]
    
    # Verify correctness with random test
    if time.time() - t0 < 10:
        np.random.seed(42)
        A_test = np.random.randint(-5, 6, size=(n, m))
        B_test = np.random.randint(-5, 6, size=(m, p))
        C_test = A_test @ B_test
        
        # Compute using decomposition
        C_decomp = np.zeros((n, p))
        for r in range(rank):
            u_col = U_total[:, r]
            v_col = V_total[:, r]
            w_col = W_total[:, r]
            # C[i,k] += sum_j u[i*m+j] * v[j*p+k] * w[k*n+i]
            for i in range(n):
                for k in range(p):
                    s = 0
                    for j in range(m):
                        s += u_col[i*m+j] * v_col[j*p+k]
                    C_decomp[i,k] += s * w_col[k*n+i]
        
        if not np.array_equal(C_test, C_decomp):
            # Fallback to schoolbook
            rank_fb = n * m * p
            U_fb = np.zeros((n*m, rank_fb), dtype=int)
            V_fb = np.zeros((m*p, rank_fb), dtype=int)
            W_fb = np.zeros((p*n, rank_fb), dtype=int)
            col = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        U_fb[i*m+j, col] = 1
                        V_fb[j*p+k, col] = 1
                        W_fb[k*n+i, col] = 1
                        col += 1
            return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
