"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START
def _strassen_2x2():
    """
    Returns U, V, W for 2x2x2 matrix multiplication using Strassen's algorithm (Rank 7).
    U: (4, 7), V: (4, 7), W: (4, 7)
    Indices: U[i*2+j, r], V[j*2+k, r], W[k*2+i, r]
    """
    # Strassen's 7 products
    # M1 = (A11 + A22)(B11 + B22)
    # M2 = (A21 + A22)B11
    # M3 = A11(B21 - B22)
    # M4 = A22(B12 - B11)
    # M5 = (A11 + A12)B22
    # M6 = (A21 - A11)(B11 + B12)
    # M7 = (A12 - A22)(B21 + B22)
    
    # C11 = M1 + M4 - M5 - M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # We need to express this as sum of rank-1 tensors U_r V_r W_r
    # Each M_r is a bilinear form.
    # M1 = (a00+a11)(b00+b11) -> a00b00 + a00b11 + a11b00 + a11b11
    # This expansion is not rank 1.
    # Strassen works by defining new variables.
    # The decomposition is:
    # Sum_r U_r[i,j] V_r[j,k] W_r[k,i]
    
    # The 7 terms are:
    # r=0: M1
    # r=1: M2
    # r=2: M3
    # r=3: M4
    # r=4: M5
    # r=5: M6
    # r=6: M7
    
    # Coefficients for U (A side):
    # M1: a00=1, a11=1
    # M2: a10=1, a11=1
    # M3: a00=1
    # M4: a11=1
    # M5: a00=1, a01=1
    # M6: a10=1, a00=-1
    # M7: a01=1, a11=-1
    
    U = [[0]*7 for _ in range(4)]
    # M1
    U[0*2+0][0] = 1; U[1*2+1][0] = 1
    # M2
    U[1*2+0][1] = 1; U[1*2+1][1] = 1
    # M3
    U[0*2+0][2] = 1
    # M4
    U[1*2+1][3] = 1
    # M5
    U[0*2+0][4] = 1; U[0*2+1][4] = 1
    # M6
    U[1*2+0][5] = 1; U[0*2+0][5] = -1
    # M7
    U[0*2+1][6] = 1; U[1*2+1][6] = -1
    
    # Coefficients for V (B side):
    # M1: b00=1, b11=1
    # M2: b00=1
    # M3: b10=1, b11=-1
    # M4: b01=1, b00=-1
    # M5: b11=1
    # M6: b00=1, b01=1
    # M7: b10=1, b11=1
    
    V = [[0]*7 for _ in range(4)]
    # M1
    V[0*2+0][0] = 1; V[1*2+1][0] = 1
    # M2
    V[0*2+0][1] = 1
    # M3
    V[1*2+0][2] = 1; V[1*2+1][2] = -1
    # M4
    V[0*2+1][3] = 1; V[0*2+0][3] = -1
    # M5
    V[1*2+1][4] = 1
    # M6
    V[0*2+0][5] = 1; V[0*2+1][5] = 1
    # M7
    V[1*2+0][6] = 1; V[1*2+1][6] = 1
    
    # Coefficients for W (C side):
    # C11 (k=0, i=0) = M1 + M4 - M5 - M7
    # C12 (k=1, i=0) = M3 + M5
    # C21 (k=0, i=1) = M2 + M4
    # C22 (k=1, i=1) = M1 - M2 + M3 + M6
    
    W = [[0]*7 for _ in range(4)]
    # Index k*2+i
    # C11: idx 0
    W[0][0] = 1; W[0][3] = 1; W[0][4] = -1; W[0][6] = -1
    # C12: idx 1
    W[1][2] = 1; W[1][4] = 1
    # C21: idx 2
    W[2][1] = 1; W[2][3] = 1
    # C22: idx 3
    W[3][0] = 1; W[3][1] = -1; W[3][2] = 1; W[3][5] = 1
    
    return U, V, W

def _strassen_4x4():
    """
    Returns U, V, W for 4x4x4 matrix multiplication using block Strassen (Rank 49).
    U: (16, 49), V: (16, 49), W: (16, 49)
    Indices: U[i*4+j, r], V[j*4+k, r], W[k*4+i, r]
    """
    U2, V2, W2 = _strassen_2x2()
    
    # Block indices for 4x4
    # A = [[A11, A12], [A21, A22]] where blocks are 2x2
    # A11: i in 0-1, j in 0-1
    # A12: i in 0-1, j in 2-3
    # A21: i in 2-3, j in 0-1
    # A22: i in 2-3, j in 2-3
    
    # B = [[B11, B12], [B21, B22]]
    # B11: j in 0-1, k in 0-1
    # B12: j in 0-1, k in 2-3
    # B21: j in 2-3, k in 0-1
    # B22: j in 2-3, k in 2-3
    
    # Strassen products:
    # M1 = (A11 + A22)(B11 + B22)
    # M2 = (A21 + A22)B11
    # M3 = A11(B21 - B22)
    # M4 = A22(B12 - B11)
    # M5 = (A11 + A12)B22
    # M6 = (A21 - A11)(B11 + B12)
    # M7 = (A12 - A22)(B21 + B22)
    
    # C11 = M1 + M4 - M5 - M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    rank = 49
    U = [[0]*rank for _ in range(16)]
    V = [[0]*rank for _ in range(16)]
    W = [[0]*rank for _ in range(16)]
    
    # For each of the 7 block products, we use the 7 inner Strassen terms.
    # Total 49 terms.
    
    # Define block offsets
    # A blocks:
    # A11: i_off=0, j_off=0
    # A12: i_off=0, j_off=2
    # A21: i_off=2, j_off=0
    # A22: i_off=2, j_off=2
    
    # B blocks:
    # B11: j_off=0, k_off=0
    # B12: j_off=0, k_off=2
    # B21: j_off=2, k_off=0
    # B22: j_off=2, k_off=2
    
    # Strassen block coefficients
    # M1: A11(+1), A22(+1); B11(+1), B22(+1)
    # M2: A21(+1), A22(+1); B11(+1)
    # M3: A11(+1); B21(+1), B22(-1)
    # M4: A22(+1); B12(+1), B11(-1)
    # M5: A11(+1), A12(+1); B22(+1)
    # M6: A21(+1), A11(-1); B11(+1), B12(+1)
    # M7: A12(+1), A22(-1); B21(+1), B22(+1)
    
    block_defs = [
        # (A_blocks, B_blocks) where blocks are (offset_i, offset_j, coeff)
        # M1
        ([ (0,0,1), (2,2,1) ], [ (0,0,1), (2,2,1) ]),
        # M2
        ([ (2,0,1), (2,2,1) ], [ (0,0,1) ]),
        # M3
        ([ (0,0,1) ], [ (2,0,1), (2,2,-1) ]),
        # M4
        ([ (2,2,1) ], [ (0,2,1), (0,0,-1) ]),
        # M5
        ([ (0,0,1), (0,2,1) ], [ (2,2,1) ]),
        # M6
        ([ (2,0,1), (0,0,-1) ], [ (0,0,1), (0,2,1) ]),
        # M7
        ([ (0,2,1), (2,2,-1) ], [ (2,0,1), (2,2,1) ])
    ]
    
    # C block coefficients
    # C11: M1(+1), M4(+1), M5(-1), M7(-1)
    # C12: M3(+1), M5(+1)
    # C21: M2(+1), M4(+1)
    # C22: M1(+1), M2(-1), M3(+1), M6(+1)
    
    c_block_defs = [
        # (C_blocks, coeffs) where C_blocks are (offset_k, offset_i, coeff)
        # C11 (k=0, i=0)
        [ (0,0,1), (0,0,1), (0,0,-1), (0,0,-1) ], # M1, M4, M5, M7
        # C12 (k=2, i=0) -- Wait, C12 is top-right block. k is col.
        # In 4x4, C12 corresponds to i in 0-1, k in 2-3.
        # So offset_k=2, offset_i=0.
        [ (2,0,1), (2,0,1) ], # M3, M5
        # C21 (k=0, i=2)
        [ (0,2,1), (0,2,1) ], # M2, M4
        # C22 (k=2, i=2)
        [ (2,2,1), (2,2,-1), (2,2,1), (2,2,1) ] # M1, M2, M3, M6
    ]
    
    # This mapping is getting complex. 
    # Simpler: Iterate over 7 block products (m_idx) and 7 inner terms (s_idx).
    # Total rank 49.
    
    for m_idx in range(7):
        a_blocks, b_blocks = block_defs[m_idx]
        
        # For each inner term s_idx in 0..6
        for s_idx in range(7):
            r = m_idx * 7 + s_idx
            
            # U part
            # For each A block involved in M_m
            for ai, aj, coeff_a in a_blocks:
                # The inner Strassen term s_idx has U2 coefficients
                # U2[i2*2+j2, s_idx]
                for i2 in range(2):
                    for j2 in range(2):
                        val = U2[i2*2+j2][s_idx] * coeff_a
                        if val != 0:
                            i_global = ai + i2
                            j_global = aj + j2
                            U[i_global*4 + j_global][r] += val
                            
            # V part
            for bj, bk, coeff_b in b_blocks:
                for j2 in range(2):
                    for k2 in range(2):
                        val = V2[j2*2+k2][s_idx] * coeff_b
                        if val != 0:
                            j_global = bj + j2
                            k_global = bk + k2
                            V[j_global*4 + k_global][r] += val
                            
            # W part
            # W depends on which C blocks are formed by M_m
            # We need to know the contribution of M_m to each C block.
            
            # Let's define the C block coefficients for each M_m
            # M1: C11(+1), C22(+1)
            # M2: C21(+1), C22(-1)
            # M3: C12(+1), C22(+1)
            # M4: C11(+1), C21(+1)
            # M5: C11(-1), C12(+1)
            # M6: C22(+1)
            # M7: C11(-1)
            
            if m_idx == 0: # M1
                c_blocks = [ (0,0,1), (2,2,1) ]
            elif m_idx == 1: # M2
                c_blocks = [ (0,2,1), (2,2,-1) ]
            elif m_idx == 2: # M3
                c_blocks = [ (2,0,1), (2,2,1) ]
            elif m_idx == 3: # M4
                c_blocks = [ (0,0,1), (0,2,1) ]
            elif m_idx == 4: # M5
                c_blocks = [ (0,0,-1), (2,0,1) ]
            elif m_idx == 5: # M6
                c_blocks = [ (2,2,1) ]
            elif m_idx == 6: # M7
                c_blocks = [ (0,0,-1) ]
            else:
                c_blocks = []
                
            for ck, ci, coeff_c in c_blocks:
                for k2 in range(2):
                    for i2 in range(2):
                        val = W2[k2*2+i2][s_idx] * coeff_c
                        if val != 0:
                            k_global = ck + k2
                            i_global = ci + i2
                            W[k_global*4 + i_global][r] += val
                            
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    # Rank 49 for 4x4x4
    U4, V4, W4 = _strassen_4x4()
    rank_4 = 49
    
    # Total rank 98 for 4x4x8
    rank = 98
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    # Block 1: cols 0-3 of B, cols 0-3 of C
    for r in range(rank_4):
        # U
        for i in range(n):
            for j in range(m):
                U[i*m+j][r] = U4[i*4+j][r]
                
        # V
        for j in range(m):
            for k in range(4):
                V[j*p+k][r] = V4[j*4+k][r]
                
        # W
        for k in range(4):
            for i in range(n):
                W[k*n+i][r] = W4[k*4+i][r]
                
    # Block 2: cols 4-7 of B, cols 4-7 of C
    for r in range(rank_4):
        r2 = r + rank_4
        # U
        for i in range(n):
            for j in range(m):
                U[i*m+j][r2] = U4[i*4+j][r]
                
        # V
        for j in range(m):
            for k in range(4):
                # Global col k+4
                V[j*p + (k+4)][r2] = V4[j*4+k][r]
                
        # W
        for k in range(4):
            # Global col k+4
            for i in range(n):
                W[(k+4)*n+i][r2] = W4[k*4+i][r]
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
