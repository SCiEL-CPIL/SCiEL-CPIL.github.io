"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    deadline = time.time() + 140
    n, m, p = 2, 4, 6
    
    # Strassen decomposition for 2x2x2
    # U: 4x7, V: 4x7, W: 4x7
    # Indices: U[i*2+j], V[j*2+k], W[k*2+i]
    U_s = np.zeros((4, 7))
    V_s = np.zeros((4, 7))
    W_s = np.zeros((4, 7))
    
    # M1 = (A00)*(B01 - B11)
    U_s[0, 0] = 1; V_s[1, 0] = 1; V_s[3, 0] = -1
    W_s[1, 0] = 1; W_s[3, 0] = 1 # C01 += M1, C11 += M1? 
    # Strassen:
    # C00 = M1 + M4 - M5 + M7
    # C01 = M2 + M3
    # C10 = M5 + M6
    # C11 = M1 - M2 + M3 + M4
    # Let's use standard Strassen formulas:
    # M1 = A00(B01 - B11)
    # M2 = (A10 - A00)B00
    # M3 = (A00 + A11)B11
    # M4 = (A11 - A01)(B00 + B10)
    # M5 = (A01 + A11)(B10 - B00)
    # M6 = (A00 - A10)(B01 + B11)
    # M7 = (A10 + A01)(B11 - B10)
    
    # C00 = M1 + M4 - M5 + M7
    # C01 = M2 + M3
    # C10 = M5 + M6
    # C11 = M1 - M2 + M3 + M4
    
    # Reset W_s
    W_s[:] = 0
    # M1 contributes to C00, C11
    W_s[0, 0] = 1 # C00 (k=0,i=0) -> idx 0
    W_s[3, 0] = 1 # C11 (k=1,i=1) -> idx 1*2+1=3
    
    # M2 contributes to C01
    U_s[2, 1] = 1; U_s[0, 1] = -1 # A10, A00
    V_s[0, 1] = 1 # B00
    W_s[1, 1] = 1 # C01 (k=1,i=0) -> idx 1*2+0=1
    
    # M3 contributes to C01, C11
    U_s[0, 2] = 1; U_s[3, 2] = 1 # A00, A11
    V_s[3, 2] = 1 # B11
    W_s[1, 2] = 1 # C01
    W_s[3, 2] = 1 # C11
    
    # M4 contributes to C00, C11
    U_s[3, 3] = 1; U_s[1, 3] = -1 # A11, A01
    V_s[0, 3] = 1; V_s[2, 3] = 1 # B00, B10
    W_s[0, 3] = 1 # C00
    W_s[3, 3] = 1 # C11
    
    # M5 contributes to C00, C10
    U_s[1, 4] = 1; U_s[3, 4] = 1 # A01, A11
    V_s[2, 4] = 1; V_s[0, 4] = -1 # B10, B00
    W_s[0, 4] = -1 # C00
    W_s[2, 4] = 1 # C10 (k=0,i=1) -> idx 0*2+1=1? No.
    # W index is k*n+i.
    # C10 is i=1, k=0. idx = 0*2+1 = 1.
    # Wait. C is 2x2. C[i,k].
    # My W index is k*n+i.
    # C00: k=0, i=0 -> 0.
    # C01: k=1, i=0 -> 2.
    # C10: k=0, i=1 -> 1.
    # C11: k=1, i=1 -> 3.
    
    # Let's re-map W_s indices carefully.
    # W_s is 4x7. Row index is k*2+i.
    # M1: C00(0), C11(3).
    W_s[0, 0] = 1; W_s[3, 0] = 1
    # M2: C01(2).
    W_s[2, 1] = 1
    # M3: C01(2), C11(3).
    W_s[2, 2] = 1; W_s[3, 2] = 1
    # M4: C00(0), C11(3).
    W_s[0, 3] = 1; W_s[3, 3] = 1
    # M5: C00(0), C10(1).
    W_s[0, 4] = -1; W_s[1, 4] = 1
    # M6: C10(1).
    U_s[0, 5] = 1; U_s[2, 5] = -1 # A00, A10
    V_s[1, 5] = 1; V_s[3, 5] = 1 # B01, B11
    W_s[1, 5] = 1
    # M7: C00(0).
    U_s[2, 6] = 1; U_s[1, 6] = 1 # A10, A01
    V_s[3, 6] = 1; V_s[2, 6] = -1 # B11, B10
    W_s[0, 6] = 1
    
    # Verify Strassen locally
    # C00 = M1 + M4 - M5 + M7
    # C01 = M2 + M3
    # C10 = M5 + M6
    # C11 = M1 - M2 + M3 + M4
    
    # Build full 2x4x6 decomposition
    # We have 3 blocks l=0,1,2 for C columns.
    # Each block l requires computing C_l (2x2) from A (2x4) and B_l (4x2).
    # C_l = A_L B_{l,T} + A_R B_{l,B}
    # This is 2 Strassen decompositions per l.
    # Total 3 * 2 * 7 = 42 rank.
    
    R = 42
    U_full = np.zeros((8, R))
    V_full = np.zeros((24, R))
    W_full = np.zeros((12, R))
    
    col = 0
    for l in range(3):
        # Sub-block L: A_L (cols 0,1 of A), B_{l,T} (rows 0,1 of B, cols 2l, 2l+1)
        # A_L indices in A: i=0,1; j=0,1.
        # B_{l,T} indices in B: j=0,1; k=2l, 2l+1.
        # C_l indices in C: i=0,1; k=2l, 2l+1.
        
        for r_s in range(7):
            # U part
            # A_L is A[:, 0:2].
            # U_s rows correspond to A[0,0], A[0,1], A[1,0], A[1,1].
            # Global U rows: i*4+j.
            # A[0,0] -> 0. A[0,1] -> 1. A[1,0] -> 4. A[1,1] -> 5.
            U_full[0, col] = U_s[0, r_s]
            U_full[1, col] = U_s[1, r_s]
            U_full[4, col] = U_s[2, r_s]
            U_full[5, col] = U_s[3, r_s]
            
            # V part
            # B_{l,T} is B[0:2, 2l:2l+2].
            # V_s rows correspond to B[0,0], B[0,1], B[1,0], B[1,1] relative to B_{l,T}.
            # Global V rows: j*6+k.
            # B[0, 2l] -> 0*6 + 2l.
            # B[0, 2l+1] -> 0*6 + 2l+1.
            # B[1, 2l] -> 1*6 + 2l.
            # B[1, 2l+1] -> 1*6 + 2l+1.
            V_full[2*l, col] = V_s[0, r_s]
            V_full[2*l+1, col] = V_s[1, r_s]
            V_full[6+2*l, col] = V_s[2, r_s]
            V_full[6+2*l+1, col] = V_s[3, r_s]
            
            # W part
            # C_l is C[:, 2l:2l+2].
            # W_s rows correspond to C[0,0], C[0,1], C[1,0], C[1,1] relative to C_l.
            # Global W rows: k*2+i.
            # C[0, 2l] -> k=2l, i=0 -> 2l*2+0 = 4l.
            # C[0, 2l+1] -> k=2l+1, i=0 -> 2l+1*2+0 = 4l+2.
            # C[1, 2l] -> k=2l, i=1 -> 2l*2+1 = 4l+1.
            # C[1, 2l+1] -> k=2l+1, i=1 -> 2l+1*2+1 = 4l+3.
            
            W_full[4*l, col] = W_s[0, r_s]
            W_full[4*l+2, col] = W_s[1, r_s]
            W_full[4*l+1, col] = W_s[2, r_s]
            W_full[4*l+3, col] = W_s[3, r_s]
            
            col += 1
        
        # Sub-block R: A_R (cols 2,3 of A), B_{l,B} (rows 2,3 of B, cols 2l, 2l+1)
        for r_s in range(7):
            # U part
            # A_R is A[:, 2:4].
            # A[0,2] -> 2. A[0,3] -> 3. A[1,2] -> 6. A[1,3] -> 7.
            U_full[2, col] = U_s[0, r_s]
            U_full[3, col] = U_s[1, r_s]
            U_full[6, col] = U_s[2, r_s]
            U_full[7, col] = U_s[3, r_s]
            
            # V part
            # B_{l,B} is B[2:4, 2l:2l+2].
            # B[2, 2l] -> 2*6 + 2l = 12+2l.
            # B[2, 2l+1] -> 12+2l+1.
            # B[3, 2l] -> 3*6 + 2l = 18+2l.
            # B[3, 2l+1] -> 18+2l+1.
            V_full[12+2*l, col] = V_s[0, r_s]
            V_full[12+2*l+1, col] = V_s[1, r_s]
            V_full[18+2*l, col] = V_s[2, r_s]
            V_full[18+2*l+1, col] = V_s[3, r_s]
            
            # W part
            # Same C_l mapping.
            W_full[4*l, col] = W_s[0, r_s]
            W_full[4*l+2, col] = W_s[1, r_s]
            W_full[4*l+1, col] = W_s[2, r_s]
            W_full[4*l+3, col] = W_s[3, r_s]
            
            col += 1
    
    # Verify
    T_comp = np.zeros((8, 24, 12))
    for r in range(R):
        T_comp += np.einsum('i,j,k->ijk', U_full[:,r], V_full[:,r], W_full[:,r])
    
    T_target = np.zeros((8, 24, 12))
    for i in range(2):
        for j in range(4):
            for k in range(6):
                T_target[i*4+j, j*6+k, k*2+i] = 1.0
    
    if not np.allclose(T_comp, T_target, atol=1e-9):
        # Fallback to schoolbook
        rank = 48
        U = [[0] * rank for _ in range(8)]
        V = [[0] * rank for _ in range(24)]
        W = [[0] * rank for _ in range(12)]
        col = 0
        for i in range(2):
            for j in range(4):
                for k in range(6):
                    U[i*4+j][col] = 1
                    V[j*6+k][col] = 1
                    W[k*2+i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    U_out = [[float(x) for x in row] for row in U_full]
    V_out = [[float(x) for x in row] for row in V_full]
    W_out = [[float(x) for x in row] for row in W_full]
    
    return {"U": U_out, "V": V_out, "W": W_out}
# EVOLVE-BLOCK-END
