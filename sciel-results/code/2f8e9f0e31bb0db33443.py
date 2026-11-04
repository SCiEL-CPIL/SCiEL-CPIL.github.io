"""Matrix multiplication 2x4x6 rank-minimal exact tensor decomposition."""
import time

DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 4, 6
    
    # Strassen 2x2x2 decomposition: 7 multiplications
    # For C = A*B (2x2 each):
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B10)
    # M7 = (A01-A11)(B01+B11)
    # C00 = M1+M4-M5+M7
    # C01 = M3+M5
    # C10 = M2+M5
    # C11 = M1-M2+M3+M6
    
    # Strassen factor matrices for one 2x2x2 block:
    # U factors (for A): 7 vectors in R^4 (A has 4 entries: A00,A01,A10,A11)
    # V factors (for B): 7 vectors in R^4 (B has 4 entries: B00,B01,B10,B11)
    # W factors (for C): 7 vectors in R^4 (C has 4 entries: C00,C01,C10,C11)
    
    # A indices: 0=A00, 1=A01, 2=A10, 3=A11
    # B indices: 0=B00, 1=B01, 2=B10, 3=B11
    # C indices: 0=C00, 1=C01, 2=C10, 3=C11
    
    U_s = [
        [1, 0, 0, 1],   # M1: A00+A11
        [0, 0, 1, 1],   # M2: A10+A11
        [1, 0, 0, 0],   # M3: A00
        [0, 0, 0, 1],   # M4: A11
        [1, 1, 0, 0],   # M5: A00+A01
        [-1, 0, 1, 0],  # M6: A10-A00
        [0, 1, 0, -1],  # M7: A01-A11
    ]
    V_s = [
        [1, 0, 0, 1],   # M1: B00+B11
        [1, 0, 0, 0],   # M2: B00
        [0, -1, 1, 0],  # M3: B10-B11
        [1, 1, 0, -1],  # M4: B01-B00
        [0, 0, 0, 1],   # M5: B11
        [1, 0, 1, 0],   # M6: B00+B10
        [0, 1, 0, 1],   # M7: B01+B11
    ]
    W_s = [
        [1, 0, 0, -1],  # M1: C00 - C10... wait let me redo
        # C00 = M1+M4-M5+M7 => W for M1: C00=1, C11=-1
        # C01 = M3+M5 => 
        # C10 = M2+M5 =>
        # C11 = M1-M2+M3+M6 =>
        # Let me re-derive W factors:
        # W_s[r][c] gives coefficient of M_c in C_r
        # C00 = 1*M1 + 0*M2 + 0*M3 + 1*M4 + (-1)*M5 + 0*M6 + 1*M7
        # C01 = 0*M1 + 0*M2 + 1*M3 + 0*M4 + 1*M5 + 0*M6 + 0*M7
        # C10 = 0*M1 + 1*M2 + 0*M3 + 0*M4 + 1*M5 + 0*M6 + 0*M7
        # C11 = 1*M1 + (-1)*M2 + 1*M3 + 0*M4 + 0*M5 + 1*M6 + 0*M7
    ]
    W_s = [
        [1, 0, 0, 1, -1, 0, 1],   # C00
        [0, 0, 1, 0, 1, 0, 0],    # C01
        [0, 1, 0, 0, 1, 0, 0],    # C10
        [1, -1, 1, 0, 0, 1, 0],   # C11
    ]
    
    # Now build full decomposition
    # 6 blocks: a in {0,1}, b in {0,1,2}
    # Each block uses 7 columns
    rank = 6 * 7
    U = [[0] * rank for _ in range(n * m)]  # 8 x rank
    V = [[0] * rank for _ in range(m * p)]  # 24 x rank
    W = [[0] * rank for _ in range(p * n)]  # 12 x rank
    
    col_offset = 0
    for a in range(2):  # A column block / B row block
        for b in range(3):  # B column block
            for s in range(7):  # Strassen factor index
                c = col_offset + s
                # A entries for block a: A[i, 2a+j'] for i,j' in {0,1}
                # A index in U: i*m + j = i*4 + (2a + j')
                for i in range(2):
                    for jp in range(2):  # j' = 0,1
                        U[i * m + (2 * a + jp)][c] += U_s[s][i * 2 + jp]
                
                # B entries for block (a,b): B[2a+j', 2b+k'] for j',k' in {0,1}
                # B index in V: j*p + k = (2a + jp) * 6 + (2b + kp)
                for jp in range(2):
                    for kp in range(2):
                        V[(2 * a + jp) * p + (2 * b + kp)][c] += V_s[s][jp * 2 + kp]
                
                # C entries for block (a,b): C[i, 2b+k'] for i,k' in {0,1}
                # C index in W: k*n + i = (2b + kp) * 2 + i
                for i in range(2):
                    for kp in range(2):
                        W[(2 * b + kp) * n + i][c] += W_s[i * 2 + kp][s]
            
            col_offset += 7
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
