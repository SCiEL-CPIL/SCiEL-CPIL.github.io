"""Matrix multiplication 4,4,5 - Strassen block decomposition."""
import time

DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 140
    n, m, p = DIMENSIONS
    nm, mp, pn = n*m, m*p, p*n
    
    # We'll build the decomposition using Strassen for 2x2x2 blocks
    # A is 4x4, B is 4x5, C = A*B is 4x5
    # 
    # Block decomposition:
    # A = [[A11, A12], [A21, A22]] where each Ai j is 2x2
    # B = [[B11, B12, B13], [B21, B22, B23]] where B11,B21,B12,B22 are 2x2, B13,B23 are 2x1
    # C = [[C11, C12, C13], [C21, C22, C23]] where C11,C21,C12,C22 are 2x2, C13,C23 are 2x1
    #
    # C11 = A11*B11 + A12*B21  (2x2 = 2x2*2x2 + 2x2*2x2)
    # C12 = A11*B12 + A12*B22  (2x2)
    # C13 = A11*B13 + A12*B23  (2x1)
    # C21 = A21*B11 + A22*B21  (2x2)
    # C22 = A21*B12 + A22*B22  (2x2)
    # C23 = A21*B13 + A22*B23  (2x1)
    #
    # For each 2x2*2x2=2x2, use Strassen (7 mults)
    # For each 2x2*2x1=2x1, use schoolbook (4 mults)
    #
    # Total terms: 4*7 + 2*4 = 28 + 8 = 36
    #
    # But Strassen requires linear combinations, so we need to track the coefficients.
    
    # Let's build U, V, W with the Strassen structure
    # Each "multiplication" in Strassen is a term with specific U,V,W entries
    
    # Strassen for X(2x2) * Y(2x2) = Z(2x2):
    # X = [[x11, x12], [x21, x22]], Y = [[y11, y12], [y21, y22]], Z = [[z11, z12], [z21, z22]]
    #
    # M1 = (x11+x22)*(y11+y22)  -> contributes to z11, z22
    # M2 = (x21+x22)*y11        -> contributes to z21, z22
    # M3 = x11*(y12-y22)        -> contributes to z12, z22
    # M4 = x22*(y21-y11)        -> contributes to z11, z21
    # M5 = (x11+x12)*y22        -> contributes to z12, z11
    # M6 = (x21-x11)*(y11+y12)  -> contributes to z22
    # M7 = (x12-x22)*(y21+y22)  -> contributes to z11, z22
    #
    # z11 = M1 + M4 - M5 + M7
    # z12 = M3 + M5
    # z21 = M2 + M4
    # z22 = M1 - M2 + M3 + M6
    
    # For the tensor decomposition, each Strassen "multiplication" Mi becomes a term
    # where U, V, W have the appropriate linear combination coefficients.
    
    # Let me define the index mappings:
    # A is 4x4: A[i,j] where i,j in 0..3
    #   A11: A[0,0],A[0,1],A[1,0],A[1,1] -> U rows: 0,1,4,5
    #   A12: A[0,2],A[0,3],A[1,2],A[1,3] -> U rows: 2,3,6,7
    #   A21: A[2,0],A[2,1],A[3,0],A[3,1] -> U rows: 8,9,12,13
    #   A22: A[2,2],A[2,3],A[3,2],A[3,3] -> U rows: 10,11,14,15
    #
    # B is 4x5: B[j,k] where j in 0..3, k in 0..4
    #   B11: B[0,0],B[0,1],B[1,0],B[1,1] -> V rows: 0,1,5,6
    #   B12: B[0,2],B[0,3],B[1,2],B[1,3] -> V rows: 2,3,7,8
    #   B13: B[0,4],B[1,4]                -> V rows: 4,9
    #   B21: B[2,0],B[2,1],B[3,0],B[3,1] -> V rows: 10,11,15,16
    #   B22: B[2,2],B[2,3],B[3,2],B[3,3] -> V rows: 12,13,17,18
    #   B23: B[2,4],B[3,4]                -> V rows: 14,19
    #
    # C is 4x5: C[i,k] where i in 0..3, k in 0..4
    #   C11: C[0,0],C[0,1],C[1,0],C[1,1] -> W rows: 0,4,1,5  (W[k*n+i])
    #   C12: C[0,2],C[0,3],C[1,2],C[1,3] -> W rows: 8,12,9,13
    #   C13: C[0,4],C[1,4]                -> W rows: 16,17
    #   C21: C[2,0],C[2,1],C[3,0],C[3,1] -> W rows: 2,6,3,7
    #   C22: C[2,2],C[2,3],C[3,2],C[3,3] -> W rows: 10,14,11,15
    #   C23: C[2,4],C[3,4]                -> W rows: 18,19
    
    # W index: W[k*n+i] where n=4, so W[k*4+i]
    # C[i,k] -> W[k*4+i]
    # C[0,0] -> W[0], C[0,1] -> W[4], C[0,2] -> W[8], C[0,3] -> W[12], C[0,4] -> W[16]
    # C[1,0] -> W[1], C[1,1] -> W[5], C[1,2] -> W[9],  C[1,3] -> W[13], C[1,4] -> W[17]
    # C[2,0] -> W[2], C[2,1] -> W[6], C[2,2] -> W[10], C[2,3] -> W[14], C[2,4] -> W[18]
    # C[3,0] -> W[3], C[3,1] -> W[7], C[3,2] -> W[11], C[3,3] -> W[15], C[3,4] -> W[19]
    
    # U index: U[i*m+j] where m=4, so U[i*4+j]
    # A[i,j] -> U[i*4+j]
    # A[0,0] -> U[0], A[0,1] -> U[1], A[0,2] -> U[2], A[0,3] -> U[3]
    # A[1,0] -> U[4], A[1,1] -> U[5], A[1,2] -> U[6], A[1,3] -> U[7]
    # A[2,0] -> U[8], A[2,1] -> U[9], A[2,2] -> U[10], A[2,3] -> U[11]
    # A[3,0] -> U[12], A[3,1] -> U[13], A[3,2] -> U[14], A[3,3] -> U[15]
    
    # V index: V[j*p+k] where p=5, so V[j*5+k]
    # B[j,k] -> V[j*5+k]
    # B[0,0] -> V[0], B[0,1] -> V[1], B[0,2] -> V[2], B[0,3] -> V[3], B[0,4] -> V[4]
    # B[1,0] -> V[5], B[1,1] -> V[6], B[1,2] -> V[7], B[1,3] -> V[8], B[1,4] -> V[9]
    # B[2,0] -> V[10], B[2,1] -> V[11], B[2,2] -> V[12], B[2,3] -> V[13], B[2,4] -> V[14]
    # B[3,0] -> V[15], B[3,1] -> V[16], B[3,2] -> V[17], B[3,3] -> V[18], B[3,4] -> V[19]
    
    # Now let's build the Strassen decomposition
    # We'll have a list of terms, each term is (U_entries, V_entries, W_entries)
    # where each is a dict from row_index to coefficient
    
    terms = []
    
    def add_strassen_2x2(A_block, B_block, C_block):
        """
        Add Strassen terms for 2x2*2x2=2x2.
        A_block: dict mapping (local_i, local_j) to U_row
        B_block: dict mapping (local_j, local_k) to V_row
        C_block: dict mapping (local_i, local_k) to W_row
        """
        # Local indices: A: (0,0),(0,1),(1,0),(1,1) -> x11,x12,x21,x22
        # B: (0,0),(0,1),(1,0),(1,1) -> y11,y12,y21,y22
        # C: (0,0),(0,1),(1,0),(1,1) -> z11,z12,z21,z22
        
        x11 = A_block[(0,0)]
        x12 = A_block[(0,1)]
        x21 = A_block[(1,0)]
        x22 = A_block[(1,1)]
        
        y11 = B_block[(0,0)]
        y12 = B_block[(0,1)]
        y21 = B_block[(1,0)]
        y22 = B_block[(1,1)]
        
        z11 = C_block[(0,0)]
        z12 = C_block[(0,1)]
        z21 = C_block[(1,0)]
        z22 = C_block[(1,1)]
        
        # M1 = (x11+x22)*(y11+y22)
        U_m1 = {x11: 1, x22: 1}
        V_m1 = {y11: 1, y22: 1}
        W_m1 = {z11: 1, z22: 1}
        terms.append((U_m1, V_m1, W_m1))
        
        # M2 = (x21+x22)*y11
        U_m2 = {x21: 1, x22: 1}
        V_m2 = {y11: 1}
        W_m2 = {z21: 1, z22: -1}
        terms.append((U_m2, V_m2, W_m2))
        
        # M3 = x11*(y12-y22)
        U_m3 = {x11: 1}
        V_m3 = {y12: 1, y22: -1}
        W_m3 = {z12: 1, z22: 1}
        terms.append((U_m3, V_m3, W_m3))
        
        # M4 = x22*(y21-y11)
        U_m4 = {x22: 1}
        V_m4 = {y21: 1, y11: -1}
        W_m4 = {z11: 1, z21: 1}
        terms.append((U_m4, V_m4, W_m4))
        
        # M5 = (x11+x12)*y22
        U_m5 = {x11: 1, x12: 1}
        V_m5 = {y22: 1}
        W_m5 = {z12: 1, z11: -1}
        terms.append((U_m5, V_m5, W_m5))
        
        # M6 = (x21-x11)*(y11+y12)
        U_m6 = {x21: 1, x11: -1}
        V_m6 = {y11: 1, y12: 1}
        W_m6 = {z22: 1}
        terms.append((U_m6, V_m6, W_m6))
        
        # M7 = (x12-x22)*(y21+y22)
        U_m7 = {x12: 1, x22: -1}
        V_m7 = {y21: 1, y22: 1}
        W_m7 = {z11: 1, z22: 1}
        terms.append((U_m7, V_m7, W_m7))
    
    def add_schoolbook_2x1(A_block, B_block, C_block):
        """
        Add schoolbook terms for 2x2*2x1=2x1.
        A_block: dict mapping (i,j) to U_row
        B_block: dict mapping (j,k) to V_row  (only k=0)
        C_block: dict mapping (i,k) to W_row  (only k=0)
        """
        # C[i,0] = sum_j A[i,j]*B[j,0]
        for i in [0, 1]:
            for j in [0, 1]:
                U_t = {A_block[(i,j)]: 1}
                V_t = {B_block[(j,0)]: 1}
                W_t = {C_block[(i,0)]: 1}
                terms.append((U_t, V_t, W_t))
    
    # Define block index mappings
    # A11: A[0,0],A[0,1],A[1,0],A[1,1]
    A11 = {(0,0): 0, (0,1): 1, (1,0): 4, (1,1): 5}
    A12 = {(0,0): 2, (0,1): 3, (1,0): 6, (1,1): 7}
    A21 = {(0,0): 8, (0,1): 9, (1,0): 12, (1,1): 13}
    A22 = {(0,0): 10, (0,1): 11, (1,0): 14, (1,1): 15}
    
    # B11: B[0,0],B[0,1],B[1,0],B[1,1]
    B11 = {(0,0): 0, (0,1): 1, (1,0): 5, (1,1): 6}
    B12 = {(0,0): 2, (0,1): 3, (1,0): 7, (1,1): 8}
    B13 = {(0,0): 4, (1,0): 9}
    B21 = {(0,0): 10, (0,1): 11, (1,0): 15, (1,1): 16}
    B22 = {(0,0): 12, (0,1): 13, (1,0): 17, (1,1): 18}
    B23 = {(0,0): 14, (1,0): 19}
    
    # C11: C[0,0],C[0,1],C[1,0],C[1,1] -> W[0],W[4],W[1],W[5]
    C11 = {(0,0): 0, (0,1): 4, (1,0): 1, (1,1): 5}
    C12 = {(0,0): 8, (0,1): 12, (1,0): 9, (1,1): 13}
    C13 = {(0,0): 16, (1,0): 17}
    C21 = {(0,0): 2, (0,1): 6, (1,0): 3, (1,1): 7}
    C22 = {(0,0): 10, (0,1): 14, (1,0): 11, (1,1): 15}
    C23 = {(0,0): 18, (1,0): 19}
    
    # Add Strassen terms for the 2x2*2x2 blocks
    add_strassen_2x2(A11, B11, C11)  # C11 = A11*B11 + ... wait, this is only one product
    # Actually C11 = A11*B11 + A12*B21, so we need two Strassen decompositions
    
    # Hmm, I need to handle the sum of two products. Let me redo this.
    
    # For C11 = A11*B11 + A12*B21:
    # Use Strassen for A11*B11 (7 terms) and Strassen for A12*B21 (7 terms)
    # Total 14 terms for C11
    
    # This gives: 4 blocks * 14 + 2 blocks * 4 = 56 + 8 = 64 terms
    
    # Let me clear and redo
    terms = []
    
    # C11 = A11*B11 + A12*B21
    add_strassen_2x2(A11, B11, C11)
    add_strassen_2x2(A12, B21, C11)
    
    # C12 = A11*B12 + A12*B22
    add_strassen_2x2(A11, B12, C12)
    add_strassen_2x2(A12, B22, C12)
    
    # C13 = A11*B13 + A12*B23 (2x1)
    add_schoolbook_2x1(A11, B13, C13)
    add_schoolbook_2x1(A12, B23, C13)
    
    # C21 = A21*B11 + A22*B21
    add_strassen_2x2(A21, B11, C21)
    add_strassen_2x2(A22, B21, C21)
    
    # C22 = A21*B12 + A22*B22
    add_strassen_2x2(A21, B12, C22)
    add_strassen_2x2(A22, B22, C22)
    
    # C23 = A21*B13 + A22*B23 (2x1)
    add_schoolbook_2x1(A21, B13, C23)
    add_schoolbook_2x1(A22, B23, C23)
    
    rank = len(terms)
    
    # Build U, V, W
    U = [[0.0]*rank for _ in range(nm)]
    V = [[0.0]*rank for _ in range(mp)]
    W = [[0.0]*rank for _ in range(pn)]
    
    for c, (U_t, V_t, W_t) in enumerate(terms):
        for row, coeff in U_t.items():
            U[row][c] = coeff
        for row, coeff in V_t.items():
            V[row][c] = coeff
        for row, coeff in W_t.items():
            W[row][c] = coeff
    
    # Verify
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = sum(U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c] for c in range(rank))
                if abs(s - 1.0) > 1e-9:
                    # Fall back to schoolbook
                    rank_sb = n*m*p
                    U = [[0.0]*rank_sb for _ in range(nm)]
                    V = [[0.0]*rank_sb for _ in range(mp)]
                    W = [[0.0]*rank_sb for _ in range(pn)]
                    col = 0
                    for ii in range(n):
                        for jj in range(m):
                            for kk in range(p):
                                U[ii*m+jj][col] = 1.0
                                V[jj*p+kk][col] = 1.0
                                W[kk*n+ii][col] = 1.0
                                col += 1
                    return {"U": U, "V": V, "W": W}
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
