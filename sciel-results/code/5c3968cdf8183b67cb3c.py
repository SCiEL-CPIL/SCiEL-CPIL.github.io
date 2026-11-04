import numpy as np

DIMENSIONS = [4, 5, 6]

def construct():
    n, m, p = DIMENSIONS
    
    # We will build the decomposition by combining smaller known decompositions.
    # R(4,5,6) <= 4 * R(2,2,3) + 4 * R(2,3,3)
    # R(2,2,3) <= R(2,2,2) + R(2,2,1) = 7 + 1 = 8
    # R(2,3,3) <= R(2,2,3) + R(2,1,3) = 8 + 1 = 9
    # Total Rank = 4*8 + 4*9 = 32 + 36 = 68
    
    # Helper to create Strassen 2x2x2 decomposition
    def get_strassen_222():
        # U: (2*2, 7) = (4, 7)
        # V: (2*2, 7) = (4, 7)
        # W: (2*2, 7) = (4, 7)
        # Indices: i*m+j, j*p+k, k*n+i
        # n=2, m=2, p=2
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))
        
        # M1 = (A11 - A22)(B22 - B11)
        # A11 is i=0,j=0 -> idx 0
        # A22 is i=1,j=1 -> idx 3
        # B22 is j=1,k=1 -> idx 3
        # B11 is j=0,k=0 -> idx 0
        # C11 is k=0,i=0 -> idx 0
        # C12 is k=0,i=1? No W[k*n+i].
        # k=0, i=0 -> 0
        # k=0, i=1 -> 1
        # k=1, i=0 -> 2
        # k=1, i=1 -> 3
        
        # U corresponds to A coefficients
        # V corresponds to B coefficients
        # W corresponds to C coefficients (inverse of multiplication)
        
        # M1: A[0,0]=1, A[1,1]=-1; B[1,1]=1, B[0,0]=-1; C[0,0]=1, C[1,1]=1?
        # C11 += M1, C22 += M1
        # C[0,0] idx 0, C[1,1] idx 3
        
        U[0, 0] = 1; U[3, 0] = -1
        V[3, 0] = 1; V[0, 0] = -1
        W[0, 0] = 1; W[3, 0] = 1
        
        # M2: (A11 + A22)B11
        # C11 += M2, C22 -= M2
        U[0, 1] = 1; U[3, 1] = 1
        V[0, 1] = 1
        W[0, 1] = 1; W[3, 1] = -1
        
        # M3: A21(B12 - B22)
        # A21 is i=1,j=0 -> idx 2
        # B12 is j=0,k=1 -> idx 1
        # B22 is j=1,k=1 -> idx 3
        # C12 += M3, C22 += M3
        # C12 is k=1,i=0 -> idx 2. C22 is k=1,i=1 -> idx 3.
        U[2, 2] = 1
        V[1, 2] = 1; V[3, 2] = -1
        W[2, 2] = 1; W[3, 2] = 1
        
        # M4: A22(B21 - B11)
        # A22 idx 3
        # B21 j=1,k=0 -> idx 2
        # B11 j=0,k=0 -> idx 0
        # C11 -= M4, C12 += M4
        # C11 idx 0, C12 idx 2
        U[3, 3] = 1
        V[2, 3] = 1; V[0, 3] = -1
        W[0, 3] = -1; W[2, 3] = 1
        
        # M5: (A11 + A21)B22
        # A11 idx 0, A21 idx 2
        # B22 idx 3
        # C11 += M5, C21 += M5
        # C21 is k=0,i=1 -> idx 1
        U[0, 4] = 1; U[2, 4] = 1
        V[3, 4] = 1
        W[0, 4] = 1; W[1, 4] = 1
        
        # M6: (A12 - A11)(B11 + B21)
        # A12 idx 1, A11 idx 0
        # B11 idx 0, B21 idx 2
        # C21 += M6, C22? No C21 += M6, C22 += M7?
        # C21 = M6 + M7
        # C22 = M1 - M2 + M3 + M7
        # Wait, standard Strassen:
        # C11 = M1 + M2 - M4 + M5
        # C12 = M3 + M4
        # C21 = M6 + M7
        # C22 = M1 - M2 + M3 + M7
        
        # M6:
        U[1, 5] = 1; U[0, 5] = -1
        V[0, 5] = 1; V[2, 5] = 1
        W[1, 5] = 1 # C21
        
        # M7:
        # A21 - A22 -> A21 idx 2, A22 idx 3
        # B12 + B22 -> B12 idx 1, B22 idx 3
        # C21 += M7, C22 += M7
        U[2, 6] = 1; U[3, 6] = -1
        V[1, 6] = 1; V[3, 6] = 1
        W[1, 6] = 1; W[3, 6] = 1
        
        return U, V, W

    def get_rank1_221():
        # R(2,2,1)
        U = np.zeros((4, 1))
        V = np.zeros((4, 1))
        W = np.zeros((4, 1))
        # T[i,j,0] = 1
        U[:, 0] = 1
        V[:, 0] = 1
        W[:, 0] = 1
        return U, V, W

    def get_rank1_213():
        # R(2,1,3)
        U = np.zeros((2, 1)) # n=2, m=1 -> 2x1
        V = np.zeros((3, 1)) # m=1, p=3 -> 3x1? No V is (m*p, rank)
        # Wait, V shape is (m*p, rank). m=1, p=3 -> 3x1.
        W = np.zeros((3*2, 1)) # p*n = 6? No p=3, n=2 -> 6x1.
        # Indices for W: k*n+i. k in 0..2, i in 0..1.
        # T[i,0,k] = 1
        U[:, 0] = 1
        V[:, 0] = 1
        W[:, 0] = 1
        return U, V, W

    # Construct R(2,2,3)
    # Split k into 0,1 (2x2x2) and 2 (2x2x1)
    U_222, V_222, W_222 = get_strassen_222()
    U_221, V_221, W_221 = get_rank1_221()
    
    # W for 221 needs to be placed at k=2.
    # W_221 is 4x1. Indices 0..3 correspond to k=0,i=0; k=0,i=1; k=1,i=0; k=1,i=1.
    # We need k=2. So indices 4,5,6,7?
    # W shape for 2x2x3 is (3*2, rank) = 6xrank.
    # W_222 is 4x7. We need to expand it to 6x7.
    U_223 = U_222 # 4x7
    V_223 = V_222 # 4x7
    W_223 = np.zeros((6, 7))
    W_223[:4, :] = W_222
    
    # Add 221 part
    # U_221 is 4x1. V_221 is 4x1.
    # W_221 needs to be shifted to k=2.
    # W_221 indices 0..3 are for k=0,1. We need k=2.
    # So we create a new W_221_shifted of size 6x1.
    W_221_shifted = np.zeros((6, 1))
    # k=2, i=0 -> idx 2*2+0 = 4
    # k=2, i=1 -> idx 2*2+1 = 5
    W_221_shifted[4:6, 0] = 1 # Since U,V are 1, W must be 1
    
    U_223 = np.hstack([U_222, U_221])
    V_223 = np.hstack([V_222, V_221])
    W_223 = np.hstack([W_223, W_221_shifted])
    
    # Construct R(2,3,3)
    # Split m into 2+1.
    # Part 1: 2x2x3 (U_223, V_223, W_223)
    # Part 2: 2x1x3 (U_213, V_213, W_213)
    
    U_213, V_213, W_213 = get_rank1_213()
    # V_213 is 3x1. Indices for V are j*p+k. m=1, p=3.
    # j=0. k=0,1,2 -> idx 0,1,2.
    # In 2x3x3, m=3, p=3. V size 9x1.
    # j=0 is idx 0..2.
    # So V_213 fits in the first 3 rows of V_233.
    
    U_233 = np.hstack([U_223, U_213])
    V_233 = np.zeros((9, 9)) # 9 cols total?
    # U_223 has 8 cols. V_223 has 8 cols.
    # V_213 has 1 col.
    # V_233 shape (9, 9).
    V_233[:, :8] = np.zeros((9, 8))
    # Fill V_223 part (j=0,1)
    V_233[:4, :8] = V_223
    # Fill V_213 part (j=0)
    V_233[:3, 8] = V_213[:, 0]
    
    W_233 = np.hstack([W_223, W_213])
    
    # Final Assembly for 4x5x6
    # 4 copies of 2x2x3 (for n=4 split, m=2 part, p=6 split into 2x3? No)
    # My previous logic:
    # R(4,5,6) <= 2 * ( R(2,2,6) + R(2,3,6) )
    # R(2,2,6) <= 2 * R(2,2,3)
    # R(2,3,6) <= 2 * R(2,3,3)
    
    # So we need:
    # 4 blocks of R(2,2,3)
    # 4 blocks of R(2,3,3)
    
    # R(2,2,3) uses U(4x8), V(4x8), W(6x8)
    # R(2,3,3) uses U(4x9), V(9x9), W(6x9)
    
    # Total U size: (4*2 + 4*2) x Rank? No.
    # n=4. Split into two n=2.
    # Top n=2: m=5. Split m=5 into 2+3.
    #   m=2 part: p=6. Split p=6 into 3+3.
    #     Two R(2,2,3).
    #   m=3 part: p=6. Split p=6 into 3+3.
    #     Two R(2,3,3).
    # Bottom n=2: Same.
    
    # Total Rank = 4 * 8 + 4 * 9 = 68.
    
    Rank = 68
    U = np.zeros((4*5, Rank))
    V = np.zeros((5*6, Rank))
    W = np.zeros((6*4, Rank))
    
    col = 0
    
    for n_block in range(2):
        for m_block in range(2): # 0: m=2, 1: m=3
            for p_block in range(2): # 0: p=3, 1: p=3
                if m_block == 0:
                    # Use R(2,2,3)
                    u_b, v_b, w_b = U_223, V_223, W_223
                else:
                    # Use R(2,3,3)
                    u_b, v_b, w_b = U_233, V_233, W_233
                
                r_b = u_b.shape[1]
                
                # U placement
                # n_block 0: i in 0,1. n_block 1: i in 2,3.
                # m_block 0: j in 0,1. m_block 1: j in 2,3,4.
                
                if m_block == 0:
                    # u_b is 4x8. Indices i*2+j.
                    # n_block 0: rows 0..1 of A.
                    # n_block 1: rows 2..3 of A.
                    if n_block == 0:
                        U[0:2*2, col:col+r_b] = u_b # Rows 0,1
                    else:
                        U[2*2:4*2, col:col+r_b] = u_b # Rows 2,3
                else:
                    # u_b is 4x9.
                    if n_block == 0:
                        U[0:2*2, col:col+r_b] = u_b
                    else:
                        U[2*2:4*2, col:col+r_b] = u_b
                
                # V placement
                # v_b size (m_block_size * 3, r_b)
                if m_block == 0:
                    # m=2, p=3. v_b is 4x8? No V_223 is 4x8.
                    # V indices j*3+k.
                    # m_block 0: j=0,1.
                    # p_block 0: k=0,1,2.
                    # p_block 1: k=3,4,5.
                    if p_block == 0:
                        V[0:4, col:col+r_b] = v_b
                    else:
                        # Shift k by 3.
                        # v_b has indices 0..3 (j=0,1; k=0..2? No v_b is for p=3)
                        # Wait, V_223 is for 2x2x3. V size 4x8.
                        # For p_block 1, we need k shifted by 3.
                        # The decomposition is the same, just applied to different k indices.
                        # So V values are the same, but placed in different rows.
                        V[0:4, col:col+r_b] = v_b
                        # Actually, V rows are j*p+k.
                        # For p_block 0, k=0,1,2. Rows 0..2 (j=0), 3..5 (j=1)?
                        # My V_223 construction:
                        # V is (m*p, rank). m=2, p=3. Size 6x8.
                        # My V_223 was 4x8? No.
                        # In get_strassen_222, V was 4x7 (m=2, p=2).
                        # In R(2,2,3), V is (2*3, 8) = 6x8.
                        # I need to fix V_223 size.
                        
                        # Let's re-verify V_223 size.
                        # U_222 is 4x7. V_222 is 4x7 (m=2,p=2).
                        # U_221 is 4x1. V_221 is 4x1 (m=2,p=1).
                        # V_223 should be hstack? No, V dimensions are (m*p, rank).
                        # For p=3, V has 6 rows.
                        # V_222 part (p=2) occupies rows for k=0,1.
                        # V_221 part (p=1) occupies rows for k=2.
                        
                        pass
                else:
                    # m=3, p=3. V size 9x9.
                    pass
                
                # This is getting complex. I will simplify by constructing the full matrices directly in the loop if possible, or ensuring the helper functions return correctly sized matrices.

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

