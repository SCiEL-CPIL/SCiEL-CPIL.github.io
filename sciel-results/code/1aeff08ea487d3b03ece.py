import numpy as np

DIMENSIONS = [2, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    rank = 28
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Helper to add a term
    # u_idx, v_idx, w_idx, coeff
    col = 0
    
    # Group 1: C1 = A * B1 (B1 is cols 0,1)
    # A_L (cols 0,1) * B_T (rows 0,1, cols 0,1) -> Strassen
    # A indices: 0, 1, 3, 4
    # B_T indices: 
    # y11: V[0], y12: V[1]
    # y21: V[5], y22: V[6]
    
    # Strassen terms
    # M1 = (x11+x22)(y11+y22)
    U[0, col] += 1; U[4, col] += 1
    V[0, col] += 1; V[6, col] += 1
    W[0, col] += 1; W[3, col] += 1 # C11, C22
    col += 1
    
    # M2 = (x21+x22)y11
    U[3, col] += 1; U[4, col] += 1
    V[0, col] += 1
    W[1, col] += 1; W[3, col] -= 1 # C21, C22
    col += 1
    
    # M3 = x11(y12-y22)
    U[0, col] += 1
    V[1, col] += 1; V[6, col] -= 1
    W[2, col] += 1; W[3, col] += 1 # C12, C22
    col += 1
    
    # M4 = x22(y21-y11)
    U[4, col] += 1
    V[5, col] += 1; V[0, col] -= 1
    W[0, col] += 1; W[1, col] += 1 # C11, C21
    col += 1
    
    # M5 = (x11+x12)y22
    U[0, col] += 1; U[1, col] += 1
    V[6, col] += 1
    W[0, col] -= 1; W[2, col] += 1 # C11, C12
    col += 1
    
    # M6 = (x21-x11)(y11+y12)
    U[3, col] += 1; U[0, col] -= 1
    V[0, col] += 1; V[1, col] += 1
    W[3, col] += 1 # C22
    col += 1
    
    # M7 = (x12-x22)(y21+y22)
    U[1, col] += 1; U[4, col] -= 1
    V[5, col] += 1; V[6, col] += 1
    W[0, col] += 1 # C11
    col += 1
    
    # A_R (col 2) * B_B (row 2, cols 0,1)
    # A indices: 2, 5
    # B_B indices: V[10], V[11]
    
    # T1 = A13 B31 -> C11
    U[2, col] += 1
    V[10, col] += 1
    W[0, col] += 1
    col += 1
    
    # T2 = A13 B32 -> C12
    U[2, col] += 1
    V[11, col] += 1
    W[2, col] += 1
    col += 1
    
    # T3 = A23 B31 -> C21
    U[5, col] += 1
    V[10, col] += 1
    W[1, col] += 1
    col += 1
    
    # T4 = A23 B32 -> C22
    U[5, col] += 1
    V[11, col] += 1
    W[3, col] += 1
    col += 1
    
    # Group 2.1: C2a = A * B2a (B2a is cols 2,3)
    # A_L (cols 0,1) * B_T (rows 0,1, cols 2,3)
    # B_T indices:
    # y11: V[2], y12: V[3]
    # y21: V[7], y22: V[8]
    
    # Strassen terms
    # M1
    U[0, col] += 1; U[4, col] += 1
    V[2, col] += 1; V[8, col] += 1
    W[4, col] += 1; W[7, col] += 1 # C13, C23
    col += 1
    
    # M2
    U[3, col] += 1; U[4, col] += 1
    V[2, col] += 1
    W[5, col] += 1; W[7, col] -= 1 # C23, C24? No, C23 is W[5], C24 is W[7]?
    # W indices for C2 (cols 2,3):
    # C02: 2*2+0=4
    # C03: 3*2+0=6
    # C12: 2*2+1=5
    # C13: 3*2+1=7
    # So W[4]=C02, W[6]=C03, W[5]=C12, W[7]=C13.
    # My previous mapping:
    # C11->0, C12->2, C21->1, C22->3.
    # Here C02->4, C03->6, C12->5, C13->7.
    # Strassen outputs:
    # C11 (top-left of block) -> W[4]
    # C12 (top-right of block) -> W[6]
    # C21 (bottom-left of block) -> W[5]
    # C22 (bottom-right of block) -> W[7]
    
    # Let's re-verify W indices.
    # W[k*n+i]. n=2.
    # k=2 (col 2 of C, 0-indexed), i=0 (row 0) -> 2*2+0=4. Correct.
    # k=3 (col 3 of C), i=0 -> 3*2+0=6. Correct.
    # k=2, i=1 -> 2*2+1=5. Correct.
    # k=3, i=1 -> 3*2+1=7. Correct.
    
    # So for Group 2.1:
    # C11_block -> W[4]
    # C12_block -> W[6]
    # C21_block -> W[5]
    # C22_block -> W[7]
    
    # Redo Group 2.1 Strassen mappings:
    # M1: C11, C22 -> W[4], W[7]
    # M2: C21, C22 -> W[5], W[7]
    # M3: C12, C22 -> W[6], W[7]
    # M4: C11, C21 -> W[4], W[5]
    # M5: C11, C12 -> W[4], W[6]
    # M6: C22 -> W[7]
    # M7: C11 -> W[4]
    
    # I already wrote M1 above with W[4], W[7]. Correct.
    # M2:
    U[3, col] += 1; U[4, col] += 1
    V[2, col] += 1
    W[5, col] += 1; W[7, col] -= 1
    col += 1
    
    # M3
    U[0, col] += 1
    V[3, col] += 1; V[8, col] -= 1
    W[6, col] += 1; W[7, col] += 1
    col += 1
    
    # M4
    U[4, col] += 1
    V[7, col] += 1; V[2, col] -= 1
    W[4, col] += 1; W[5, col] += 1
    col += 1
    
    # M5
    U[0, col] += 1; U[1, col] += 1
    V[8, col] += 1
    W[4, col] -= 1; W[6, col] += 1
    col += 1
    
    # M6
    U[3, col] += 1; U[0, col] -= 1
    V[2, col] += 1; V[3, col] += 1
    W[7, col] += 1
    col += 1
    
    # M7
    U[1, col] += 1; U[4, col] -= 1
    V[7, col] += 1; V[8, col] += 1
    W[4, col] += 1
    col += 1
    
    # A_R (col 2) * B_B (row 2, cols 2,3)
    # B_B indices: V[12], V[13]
    
    # T1 = A13 B33 -> C03 (W[6])? No, B33 is row 2, col 2 of B.
    # B is 3x5.
    # B2a is cols 2,3.
    # B_T is rows 0,1 of B2a.
    # B_B is row 2 of B2a.
    # B_B[0] is B[2,2] -> V[12].
    # B_B[1] is B[2,3] -> V[13].
    
    # C02 (W[4]) gets A13 * B33?
    # C = A B.
    # C[0,2] = A[0,:] dot B[:,2].
    # B[:,2] is col 2 of B.
    # B_T[0] is B[0,2]. B_B[0] is B[2,2].
    # So A_R[0] * B_B[0] contributes to C[0,2].
    # A_R[0] is A[0,2]. U[2].
    
    U[2, col] += 1
    V[12, col] += 1
    W[4, col] += 1
    col += 1
    
    # T2 = A13 B34 -> C03 (W[6])
    U[2, col] += 1
    V[13, col] += 1
    W[6, col] += 1
    col += 1
    
    # T3 = A23 B33 -> C12 (W[5])
    U[5, col] += 1
    V[12, col] += 1
    W[5, col] += 1
    col += 1
    
    # T4 = A23 B34 -> C13 (W[7])
    U[5, col] += 1
    V[13, col] += 1
    W[7, col] += 1
    col += 1
    
    # Group 2.2: C2b = A * B2b (B2b is col 4)
    # 6 terms.
    # B2b indices:
    # B[0,4] -> V[4]
    # B[1,4] -> V[9]
    # B[2,4] -> V[14]
    
    # T1 = A11 B15 -> C04 (W[8])
    U[0, col] += 1
    V[4, col] += 1
    W[8, col] += 1
    col += 1
    
    # T2 = A12 B25 -> C04 (W[8])
    U[1, col] += 1
    V[9, col] += 1
    W[8, col] += 1
    col += 1
    
    # T3 = A13 B35 -> C04 (W[8])
    U[2, col] += 1
    V[14, col] += 1
    W[8, col] += 1
    col += 1
    
    # T4 = A21 B15 -> C14 (W[9])
    U[3, col] += 1
    V[4, col] += 1
    W[9, col] += 1
    col += 1
    
    # T5 = A22 B25 -> C14 (W[9])
    U[4, col] += 1
    V[9, col] += 1
    W[9, col] += 1
    col += 1
    
    # T6 = A23 B35 -> C14 (W[9])
    U[5, col] += 1
    V[14, col] += 1
    W[9, col] += 1
    col += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
