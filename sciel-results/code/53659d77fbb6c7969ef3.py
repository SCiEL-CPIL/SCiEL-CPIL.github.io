"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    n, m, p = 3, 4, 8
    start_time = time.time()
    deadline = start_time + 115

    # We seek a low-rank exact decomposition of the matrix multiplication tensor.
    # Schoolbook rank = n*m*p = 96.
    # Strassen on 2x2 blocks gives rank 7 per 2x2x2 block.
    
    # Strategy: Try to combine Strassen blocks cleverly.
    # Partition i into {0,1} and {2}.
    # For i in {0,1}, use Strassen on 2x2 subblocks of j and k.
    # For i=2, use schoolbook.
    
    # j-groups: {0,1}, {2,3}
    # k-groups: {0,1}, {2,3}, {4,5}, {6,7}
    
    # For i-pair {0,1} with each j-pair and k-pair: 7 terms each.
    # Number of (j-pair, k-pair) combos: 2 * 4 = 8
    # Strassen terms: 8 * 7 = 56
    # For i=2 with each j-pair and k-pair: 4 terms each (schoolbook for 1x2x2)
    # Schoolbook terms: 8 * 4 = 32
    # Total rank: 56 + 32 = 88
    
    rank = 88
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Strassen coefficients for 2x2 matrix multiplication
    # M1 = (A11 + A22) * (B11 + B22)
    # M2 = (A21 + A22) * B11
    # M3 = A11 * (B21 - B22)
    # M4 = A22 * (B12 - B11)
    # M5 = (A11 + A12) * B22
    # M6 = (A21 - A11) * (B11 + B12)
    # M7 = (A12 - A22) * (B21 + B22)
    
    # C11 = M1 + M4 - M5 - M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M7
    
    # A is 2x2: A[di, dj] where di, dj in {0,1}
    # B is 2x2: B[dj, dk] where dj, dk in {0,1}
    # C is 2x2: C[di, dk] where di, dk in {0,1}
    
    strassen_terms = [
        # (U_coeffs, V_coeffs, W_coeffs)
        # Each is a 4-element list indexed by [di*2+dj] for U, [dj*2+dk] for V, [dk*2+di] for W
        # Wait, W index is k*n+i, so for 2x2 block starting at k0, i0:
        # W[(k0+dk)*n + (i0+di)]
        
        # M1: A11+A22 times B11+B22, contributes to C11(+), C22(+)
        # U: [1,0,0,1] for A11,A12,A21,A22
        # V: [1,0,0,1] for B11,B12,B21,B22
        # W: [1,0,0,1] for C11,C12,C21,C22 (di=0,dk=0 -> idx 0; di=0,dk=1 -> idx 1; di=1,dk=0 -> idx 2; di=1,dk=1 -> idx 3)
        # C11 is di=0,dk=0 -> idx 0; C22 is di=1,dk=1 -> idx 3
        ([1,0,0,1], [1,0,0,1], [1,0,0,1]),
        
        # M2: (A21+A22) * B11, contributes to C21(+)
        ([0,0,1,1], [1,0,0,0], [0,0,1,0]),
        
        # M3: A11 * (B21-B22), contributes to C12(+)
        ([1,0,0,0], [0,0,1,-1], [0,1,0,0]),
        
        # M4: A22 * (B12-B11), contributes to C11(-), C21(+)
        ([0,0,0,1], [0,1,-1,0], [-1,0,1,0]),
        
        # M5: (A11+A12) * B22, contributes to C11(-), C12(+)
        ([1,1,0,0], [0,0,0,1], [-1,1,0,0]),
        
        # M6: (A21-A11) * (B11+B12), contributes to ... let me recheck
        # Actually standard Strassen:
        # M1 = (A11 + A22)(B11 + B22)
        # M2 = (A21 + A22)B11
        # M3 = A11(B21 - B22)
        # M4 = A22(B12 - B11)
        # M5 = (A11 + A12)B22
        # M6 = (A21 - A11)(B11 + B12)
        # M7 = (A12 - A22)(B21 + B22)
        # C11 = M1 - M5 + M4 - M7
        # C12 = M3 + M5
        # C21 = M2 - M4 + M6
        # C22 = M1 - M2 - M3 + M7
        
        # Let me redo with correct formulas.
    ]
    
    # Let me use the correct Strassen coefficients:
    # C11 = M1 - M5 + M4 - M7
    # C12 = M3 + M5
    # C21 = M2 - M4 + M6
    # C22 = M1 - M2 - M3 + M7
    
    # M1: U=[1,0,0,1], V=[1,0,0,1], W=[1,0,0,1] (C11 and C22 both get +M1)
    # M2: U=[0,0,1,1], V=[1,0,0,0], W=[0,0,1,0] (C21 gets +M2, C22 gets -M2)
    # Wait, W coefficients: for C21 (di=1,dk=0) idx=2, for C22 (di=1,dk=1) idx=3
    # So W for M2: [0,0,1,-1]
    
    # M3: U=[1,0,0,0], V=[0,0,1,-1], W=[0,1,0,0] (C12 gets +M3, C22 gets -M3)
    # W for M3: C12 (di=0,dk=1) idx=1, C22 (di=1,dk=1) idx=3 -> [0,1,0,-1]
    
    # M4: U=[0,0,0,1], V=[0,1,-1,0], W=[-1,0,1,0] (C11 gets +M4, C21 gets -M4)
    # W for M4: C11 (di=0,dk=0) idx=0, C21 (di=1,dk=0) idx=2 -> [1,0,-1,0]
    # Wait C11 = ... + M4, C21 = ... - M4. So W=[1,0,-1,0]
    
    # M5: U=[1,1,0,0], V=[0,0,0,1], W=[-1,1,0,0] (C11 gets -M5, C12 gets +M5)
    # W for M5: C11 idx=0, C12 idx=1 -> [-1,1,0,0]
    
    # M6: U=[0,-1,1,0], V=[1,1,0,0], W=[0,0,1,0] (C21 gets +M6)
    # W for M6: C21 idx=2 -> [0,0,1,0]
    
    # M7: U=[0,1,0,-1], V=[0,0,1,1], W=[-1,0,0,1] (C11 gets -M7, C22 gets +M7)
    # W for M7: C11 idx=0, C22 idx=3 -> [-1,0,0,1]
    
    strassen_terms = [
        ([1,0,0,1], [1,0,0,1], [1,0,0,1]),   # M1
        ([0,0,1,1], [1,0,0,0], [0,0,1,-1]),   # M2
        ([1,0,0,0], [0,0,1,-1], [0,1,0,-1]),  # M3
        ([0,0,0,1], [0,1,-1,0], [1,0,-1,0]),  # M4
        ([1,1,0,0], [0,0,0,1], [-1,1,0,0]),   # M5
        ([0,-1,1,0], [1,1,0,0], [0,0,1,0]),   # M6
        ([0,1,0,-1], [0,0,1,1], [-1,0,0,1]),  # M7
    ]
    
    def add_strassen_block(i0, j0, k0, col_start):
        col = col_start
        for t in range(7):
            u_c = strassen_terms[t][0]
            v_c = strassen_terms[t][1]
            w_c = strassen_terms[t][2]
            
            for di in range(2):
                for dj in range(2):
                    U[(i0+di)*m + (j0+dj)][col] = u_c[di*2+dj]
            
            for dj in range(2):
                for dk in range(2):
                    V[(j0+dj)*p + (k0+dk)][col] = v_c[dj*2+dk]
            
            for di in range(2):
                for dk in range(2):
                    W[(k0+dk)*n + (i0+di)][col] = w_c[dk*2+di]
            
            col += 1
        return col
    
    def add_schoolbook_1x2x2(i0, j0, k0, col_start):
        col = col_start
        for dj in range(2):
            for dk in range(2):
                U[i0*m + (j0+dj)][col] = 1
                V[(j0+dj)*p + (k0+dk)][col] = 1
                W[(k0+dk)*n + i0][col] = 1
                col += 1
        return col
    
    col = 0
    for j0 in [0, 2]:
        for k0 in [0, 2, 4, 6]:
            col = add_strassen_block(0, j0, k0, col)
            col = add_schoolbook_1x2x2(2, j0, k0, col)
    
    # Verify
    def verify():
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        r = U_arr.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(r):
                        s += U_arr[i*m+j, c] * V_arr[j*p+k, c] * W_arr[k*n+i, c]
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True
    
    if verify():
        return {"U": U, "V": V, "W": W}
    
    # Fallback: schoolbook
    rank_fb = n * m * p
    U_fb = [[0] * rank_fb for _ in range(n * m)]
    V_fb = [[0] * rank_fb for _ in range(m * p)]
    W_fb = [[0] * rank_fb for _ in range(p * n)]
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_fb[i*m+j][c] = 1
                V_fb[j*p+k][c] = 1
                W_fb[k*n+i][c] = 1
                c += 1
    return {"U": U_fb, "V": V_fb, "W": W_fb}
# EVOLVE-BLOCK-END
