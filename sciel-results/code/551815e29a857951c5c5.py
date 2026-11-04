import numpy as np
import time

DIMENSIONS = [4, 4, 7]

def construct():
    n, m, p = DIMENSIONS
    # n=4, m=4, p=7
    
    # Part 1: 4x4 * 4x4 -> 4x4 (Strassen rank 49)
    # Part 2: 4x4 * 4x3 -> 4x3 (Schoolbook rank 48)
    
    rank = 97
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # --- Helper: 2x2 Strassen Decomposition ---
    # Returns U2, V2, W2 for 2x2x2 with rank 7
    # U2 indices: i*2+j (0..3)
    # V2 indices: j*2+k (0..3)
    # W2 indices: k*2+i (0..3)
    def get_strassen_2x2():
        U2 = [[0]*7 for _ in range(4)]
        V2 = [[0]*7 for _ in range(4)]
        W2 = [[0]*7 for _ in range(4)]
        
        # P1 = (A11+A22)(B11+B22)
        # A11=U[0], A22=U[3] -> U2[0][0]=1, U2[3][0]=1
        # B11=V[0], B22=V[3] -> V2[0][0]=1, V2[3][0]=1
        # C11 += P1, C22 += P1 -> W2[0][0]=1 (C11), W2[3][0]=1 (C22)
        # Note: W index is k*n+i. For 2x2, n=2.
        # C11 is i=0, k=0 -> idx 0. C22 is i=1, k=1 -> idx 3.
        
        # P2 = (A21+A22)B11
        # A21=U[2], A22=U[3] -> U2[2][1]=1, U2[3][1]=1
        # B11=V[0] -> V2[0][1]=1
        # C21 += P2 -> W2[2][1]=1 (i=1, k=0 -> idx 2)
        
        # P3 = A11(B12-B22)
        # A11=U[0] -> U2[0][2]=1
        # B12=V[1], B22=V[3] -> V2[1][2]=1, V2[3][2]=-1
        # C12 += P3, C22 += P3 -> W2[1][2]=1 (i=0, k=1 -> idx 1), W2[3][2]=1 (i=1, k=1 -> idx 3)
        
        # P4 = A22(B21-B11)
        # A22=U[3] -> U2[3][3]=1
        # B21=V[2], B11=V[0] -> V2[2][3]=1, V2[0][3]=-1
        # C11 += P4, C21 += P4 -> W2[0][3]=1 (i=0, k=0 -> idx 0), W2[2][3]=1 (i=1, k=0 -> idx 2)
        
        # P5 = (A11+A12)B22
        # A11=U[0], A12=U[1] -> U2[0][4]=1, U2[1][4]=1
        # B22=V[3] -> V2[3][4]=1
        # C11 -= P5, C12 += P5 -> W2[0][4]=-1 (i=0, k=0 -> idx 0), W2[1][4]=1 (i=0, k=1 -> idx 1)
        
        # P6 = (A21-A11)(B11+B12)
        # A21=U[2], A11=U[0] -> U2[2][5]=1, U2[0][5]=-1
        # B11=V[0], B12=V[1] -> V2[0][5]=1, V2[1][5]=1
        # C22 += P6 -> W2[3][5]=1 (i=1, k=1 -> idx 3)
        
        # P7 = (A12-A22)(B21+B22)
        # A12=U[1], A22=U[3] -> U2[1][6]=1, U2[3][6]=-1
        # B21=V[2], B22=V[3] -> V2[2][6]=1, V2[3][6]=1
        # C11 += P7, C12 += P7 -> W2[0][6]=1 (i=0, k=0 -> idx 0), W2[1][6]=1 (i=0, k=1 -> idx 1)
        
        # Fill in the matrices
        U2[0][0] = 1; U2[3][0] = 1
        V2[0][0] = 1; V2[3][0] = 1
        W2[0][0] = 1; W2[3][0] = 1
        
        U2[2][1] = 1; U2[3][1] = 1
        V2[0][1] = 1
        W2[2][1] = 1
        
        U2[0][2] = 1
        V2[1][2] = 1; V2[3][2] = -1
        W2[1][2] = 1; W2[3][2] = 1
        
        U2[3][3] = 1
        V2[2][3] = 1; V2[0][3] = -1
        W2[0][3] = 1; W2[2][3] = 1
        
        U2[0][4] = 1; U2[1][4] = 1
        V2[3][4] = 1
        W2[0][4] = -1; W2[1][4] = 1
        
        U2[2][5] = 1; U2[0][5] = -1
        V2[0][5] = 1; V2[1][5] = 1
        W2[3][5] = 1
        
        U2[1][6] = 1; U2[3][6] = -1
        V2[2][6] = 1; V2[3][6] = 1
        W2[0][6] = 1; W2[1][6] = 1
        
        return U2, V2, W2

    U2, V2, W2 = get_strassen_2x2()
    
    # Define the 7 Strassen steps for 4x4
    # Each step l has:
    # - Linear map LA: A (4x4) -> A'_l (2x2)
    # - Linear map LB: B (4x4) -> B'_l (2x2)
    # - Linear map LC: C'_l (2x2) -> C (4x4) (with signs)
    
    # We represent LA, LB, LC as 4x16 matrices (flattened 2x2 to flattened 4x4)
    # Actually, it's easier to just define the coefficients for each entry of A'_l, B'_l, C'_l
    # in terms of A, B, C entries.
    
    # A indices: 0..15 (i*4+j)
    # B indices: 0..15 (j*4+k)
    # C indices: 0..15 (k*4+i)
    
    # Blocks of A:
    # A11: 0,1,4,5
    # A12: 2,3,6,7
    # A21: 8,9,12,13
    # A22: 10,11,14,15
    
    # Blocks of B:
    # B11: 0,4,8,12 (j=0,1; k=0,1) -> j*4+k. 
    # Wait, B is 4x4. Index j*4+k.
    # B11 (j=0,1; k=0,1): 0,4,1,5? No.
    # j=0: k=0,1 -> 0, 1
    # j=1: k=0,1 -> 4, 5
    # So B11 indices: 0, 1, 4, 5.
    # B12 (j=0,1; k=2,3):
    # j=0: k=2,3 -> 2, 3
    # j=1: k=2,3 -> 6, 7
    # So B12 indices: 2, 3, 6, 7.
    # B21 (j=2,3; k=0,1):
    # j=2: k=0,1 -> 8, 9
    # j=3: k=0,1 -> 12, 13
    # So B21 indices: 8, 9, 12, 13.
    # B22 (j=2,3; k=2,3):
    # j=2: k=2,3 -> 10, 11
    # j=3: k=2,3 -> 14, 15
    # So B22 indices: 10, 11, 14, 15.
    
    # C indices: k*4+i.
    # C11 (k=0,1; i=0,1):
    # k=0: i=0,1 -> 0, 1
    # k=1: i=0,1 -> 4, 5
    # So C11 indices: 0, 1, 4, 5.
    # C12 (k=0,1; i=2,3):
    # k=0: i=2,3 -> 2, 3
    # k=1: i=2,3 -> 6, 7
    # So C12 indices: 2, 3, 6, 7.
    # C21 (k=2,3; i=0,1):
    # k=2: i=0,1 -> 8, 9
    # k=3: i=0,1 -> 12, 13
    # So C21 indices: 8, 9, 12, 13.
    # C22 (k=2,3; i=2,3):
    # k=2: i=2,3 -> 10, 11
    # k=3: i=2,3 -> 14, 15
    # So C22 indices: 10, 11, 14, 15.
    
    # The 7 Strassen steps for 4x4:
    # l=0: M1 = (A11+A22)(B11+B22)
    # l=1: M2 = (A21+A22)B11
    # l=2: M3 = A11(B12-B22)
    # l=3: M4 = A22(B21-B11)
    # l=4: M5 = (A11+A12)B22
    # l=5: M6 = (A21-A11)(B11+B12)
    # l=6: M7 = (A12-A22)(B21+B22)
    
    # For each l, we need to map the 2x2 Strassen decomposition to the 4x4 context.
    # The 2x2 Strassen decomposition uses U2, V2, W2.
    # U2 maps A'_l (2x2) to 7 products.
    # V2 maps B'_l (2x2) to 7 products.
    # W2 maps C'_l (2x2) to 7 products.
    
    # We need to define the linear maps:
    # LA_l: A (16) -> A'_l (4)
    # LB_l: B (16) -> B'_l (4)
    # LC_l: C'_l (4) -> C (16)
    
    # Let's define these as 4x16 matrices (rows correspond to A'_l entries 0..3).
    
    def make_identity_block(idx_list):
        # Returns a 4x16 matrix that picks out the entries in idx_list
        # The 4 entries of the 2x2 block are in row-major order: 0,1,2,3
        # Corresponding to (0,0), (0,1), (1,0), (1,1)
        mat = np.zeros((4, 16))
        for r, idx in enumerate(idx_list):
            mat[r, idx] = 1
        return mat

    A11_idx = [0, 1, 4, 5]
    A12_idx = [2, 3, 6, 7]
    A21_idx = [8, 9, 12, 13]
    A22_idx = [10, 11, 14, 15]
    
    B11_idx = [0, 1, 4, 5]
    B12_idx = [2, 3, 6, 7]
    B21_idx = [8, 9, 12, 13]
    B22_idx = [10, 11, 14, 15]
    
    C11_idx = [0, 1, 4, 5]
    C12_idx = [2, 3, 6, 7]
    C21_idx = [8, 9, 12, 13]
    C22_idx = [10, 11, 14, 15]
    
    LA = []
    LB = []
    LC = []
    
    # l=0: M1 = (A11+A22)(B11+B22)
    LA0 = make_identity_block(A11_idx) + make_identity_block(A22_idx)
    LB0 = make_identity_block(B11_idx) + make_identity_block(B22_idx)
    LC0 = make_identity_block(C11_idx) + make_identity_block(C22_idx)
    # C11 += M1, C22 += M1
    # So LC0 should map C'_l to C.
    # C'_l is 2x2. C11 gets C'_l[0:2, 0:2], C22 gets C'_l[2:4, 2:4]?
    # No, C'_l is the product of two 2x2 matrices.
    # M1 is a 2x2 matrix.
    # C11 (top-left 2x2 of C) gets M1.
    # C22 (bottom-right 2x2 of C) gets M1.
    # So LC0 maps the 4 entries of M1 to C11 and C22.
    # M1[0] -> C11[0], M1[1] -> C11[1], M1[2] -> C11[2], M1[3] -> C11[3]
    # M1[0] -> C22[0], M1[1] -> C22[1], M1[2] -> C22[2], M1[3] -> C22[3]
    LC0 = np.zeros((4, 16))
    for i in range(4):
        LC0[i, C11_idx[i]] = 1
        LC0[i, C22_idx[i]] = 1
    LA.append(LA0); LB.append(LB0); LC.append(LC0)
    
    # l=1: M2 = (A21+A22)B11
    LA1 = make_identity_block(A21_idx) + make_identity_block(A22_idx)
    LB1 = make_identity_block(B11_idx)
    LC1 = np.zeros((4, 16))
    for i in range(4):
        LC1[i, C21_idx[i]] = 1
    LA.append(LA1); LB.append(LB1); LC.append(LC1)
    
    # l=2: M3 = A11(B12-B22)
    LA2 = make_identity_block(A11_idx)
    LB2 = make_identity_block(B12_idx) - make_identity_block(B22_idx)
    LC2 = np.zeros((4, 16))
    for i in range(4):
        LC2[i, C12_idx[i]] = 1
        LC2[i, C22_idx[i]] = 1
    LA.append(LA2); LB.append(LB2); LC.append(LC2)
    
    # l=3: M4 = A22(B21-B11)
    LA3 = make_identity_block(A22_idx)
    LB3 = make_identity_block(B21_idx) - make_identity_block(B11_idx)
    LC3 = np.zeros((4, 16))
    for i in range(4):
        LC3[i, C11_idx[i]] = 1
        LC3[i, C21_idx[i]] = 1
    LA.append(LA3); LB.append(LB3); LC.append(LC3)
    
    # l=4: M5 = (A11+A12)B22
    LA4 = make_identity_block(A11_idx) + make_identity_block(A12_idx)
    LB4 = make_identity_block(B22_idx)
    LC4 = np.zeros((4, 16))
    for i in range(4):
        LC4[i, C11_idx[i]] = -1
        LC4[i, C12_idx[i]] = 1
    LA.append(LA4); LB.append(LB4); LC.append(LC4)
    
    # l=5: M6 = (A21-A11)(B11+B12)
    LA5 = make_identity_block(A21_idx) - make_identity_block(A11_idx)
    LB5 = make_identity_block(B11_idx) + make_identity_block(B12_idx)
    LC5 = np.zeros((4, 16))
    for i in range(4):
        LC5[i, C22_idx[i]] = 1
    LA.append(LA5); LB.append(LB5); LC.append(LC5)
    
    # l=6: M7 = (A12-A22)(B21+B22)
    LA6 = make_identity_block(A12_idx) - make_identity_block(A22_idx)
    LB6 = make_identity_block(B21_idx) + make_identity_block(B22_idx)
    LC6 = np.zeros((4, 16))
    for i in range(4):
        LC6[i, C11_idx[i]] = 1
        LC6[i, C12_idx[i]] = 1
    LA.append(LA6); LB.append(LB6); LC.append(LC6)
    
    # Now, for each l in 0..6, we add the 7 Strassen sub-multiplications.
    # Total 49 terms.
    col = 0
    for l in range(7):
        # LA[l] is 4x16, LB[l] is 4x16, LC[l] is 4x16
        # U2 is 4x7, V2 is 4x7, W2 is 4x7
        # The term for sub-step k (0..6) is:
        # u = LA[l] @ U2[:, k]  (16-vector)
        # v = LB[l] @ V2[:, k]  (16-vector)
        # w = LC[l] @ W2[:, k]  (16-vector)
        
        for k in range(7):
            u_vec = LA[l] @ U2[:, k]
            v_vec = LB[l] @ V2[:, k]
            w_vec = LC[l] @ W2[:, k]
            
            for i in range(16):
                if u_vec[i] != 0:
                    U[i][col] = u_vec[i]
                if v_vec[i] != 0:
                    V[i][col] = v_vec[i]
                if w_vec[i] != 0:
                    W[i][col] = w_vec[i]
            col += 1
            
    # Part 2: 4x4 * 4x3 -> 4x3 (Schoolbook rank 48)
    # B columns 4,5,6 (indices 4,5,6 in p)
    # A is 4x4.
    # C is 4x3.
    # C[i, k] = sum_j A[i, j] B[j, k]
    # i in 0..3, j in 0..3, k in 4..6
    for i in range(4):
        for j in range(4):
            for k in range(4, 7):
                U[i * 4 + j][col] = 1
                V[j * 7 + k][col] = 1
                W[k * 4 + i][col] = 1
                col += 1
                
    return {"U": U, "V": V, "W": W}

if __name__ == "__main__":
    res = construct()
    print("Rank:", len(res["U"][0]))
