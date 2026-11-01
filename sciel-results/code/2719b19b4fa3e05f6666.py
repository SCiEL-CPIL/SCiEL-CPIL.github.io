"""
Strategy: 
1. Recognize that the 2x4x4 matrix multiplication tensor is isomorphic to two disjoint 2x2x2 matrix multiplication tensors.
2. Specifically, let A be 2x4, B be 4x4, C be 2x4.
   We can partition A into two 2x2 matrices A_top and A_bottom? No, A is 2x4.
   Let's look at the indices: C[i, k] = sum_j A[i, j] B[j, k].
   i in {0, 1}, j in {0, 1, 2, 3}, k in {0, 1, 2, 3}.
   
   This is not two independent 2x2 multiplications in the standard sense because the j index (inner dimension) is shared.
   However, we can split the inner dimension j into two sets: j in {0, 1} and j in {2, 3}.
   Let A_1 be A[:, 0:2] (2x2) and A_2 be A[:, 2:4] (2x2).
   Let B_1 be B[0:2, :] (2x4) and B_2 be B[2:4, :] (2x4).
   Then C = A_1 B_1 + A_2 B_2.
   
   This is a sum of two products of (2x2) * (2x4).
   This doesn't immediately reduce to 2x2x2 Strassen directly because B is 2x4.
   
   Let's re-evaluate the "Block Strassen" hint.
   "Strassen block decomposition reduces rank from 32 to 28 by exploiting 2x2 sub-block structure... A can be split into two 2x2 blocks that each multiply with corresponding 2x2 blocks of B"
   
   Wait, if A is 2x4 and B is 4x4, we can't split A into 2x2 blocks that multiply with 2x2 blocks of B to give a 2x4 output directly in a block-diagonal way unless we split the output C as well.
   
   Let's split C into two 2x2 blocks: C_1 = C[:, 0:2] and C_2 = C[:, 2:4].
   C_1 = A B_1 (where B_1 is 4x2)
   C_2 = A B_2 (where B_2 is 4x2)
   
   This is two independent multiplications of (2x4) * (4x2).
   The rank of (2x4)*(4x2) multiplication?
   The tensor is T_{i, j, k} where i in 0..1, j in 0..3, k in 0..1.
   This is a 2x4x2 tensor.
   The schoolbook rank is 2*4*2 = 16.
   Can we do better?
   The border rank of 2x4x2 is at least max(2*4, 4*2, 2*2) = 8? No, that's for square.
   For non-square, the rank is at least the max of the dimensions of the factors?
   Actually, the rank of the multiplication tensor for m x n * n x p is at least max(mn, np, mp) if we consider the flattening? No.
   
   Let's look at the specific case 2x4 * 4x2.
   This is equivalent to computing 4 entries of a 2x2 matrix, each being a dot product of a row of A (length 4) and a column of B (length 4).
   Wait, A is 2x4, B is 4x2.
   C_1 (2x2) = A (2x4) * B_1 (4x2).
   This is 4 dot products of length 4.
   Schoolbook rank for this sub-problem is 4 (one per entry? No, schoolbook is sum of products of scalars).
   Each entry C_ik = sum_j A_ij B_jk.
   There are 2*2=4 entries. Each entry has 4 terms. Total 16 terms.
   
   Can we use Strassen here?
   Strassen works for 2x2 * 2x2.
   Here we have 2x4 * 4x2.
   We can split A into two 2x2 blocks: A = [A_1, A_2] where A_1, A_2 are 2x2.
   We can split B_1 into two 2x2 blocks: B_1 = [B_{11}; B_{21}] where B_{11}, B_{21} are 2x2.
   Then C_1 = A_1 B_{11} + A_2 B_{21}.
   
   This is the sum of two 2x2 * 2x2 products.
   Using Strassen for each:
   C_1 = Strassen(A_1, B_{11}) + Strassen(A_2, B_{21}).
   Rank = 7 + 7 = 14.
   
   Similarly for C_2:
   C_2 = A B_2.
   B_2 = [B_{12}; B_{22}] where B_{12}, B_{22} are 2x2.
   C_2 = A_1 B_{12} + A_2 B_{22}.
   Rank = 7 + 7 = 14.
   
   Total rank = 14 + 14 = 28.
   
   This matches the "Measured: 1 -> 1.143" hint (32/28 = 1.1428...).
   So the target is Rank 28.
   
   How to construct the matrices U, V, W for the full 2x4x4 problem?
   The problem asks for U (n*m, rank) = (8, R), V (m*p, rank) = (16, R), W (p*n, rank) = (8, R).
   Wait, n=2, m=4, p=4.
   U shape: (2*4, R) = (8, R).
   V shape: (4*4, R) = (16, R).
   W shape: (4*2, R) = (8, R).
   
   The tensor T[i, j, k] corresponds to C[i, k] += A[i, j] B[j, k].
   Indices:
   i in 0..1 (rows of A, cols of C)
   j in 0..3 (cols of A, rows of B)
   k in 0..3 (cols of B, cols of C)
   
   We decompose into 4 Strassen blocks:
   Block 1: A_1 (rows 0,1; cols 0,1 of A) * B_{11} (rows 0,1 of B; cols 0,1 of B) -> C_{11} (rows 0,1 of C; cols 0,1 of C)
   Block 2: A_2 (rows 0,1; cols 2,3 of A) * B_{21} (rows 2,3 of B; cols 0,1 of B) -> C_{11}
   Block 3: A_1 * B_{12} (rows 0,1 of B; cols 2,3 of B) -> C_{12}
   Block 4: A_2 * B_{22} (rows 2,3 of B; cols 2,3 of B) -> C_{12}
   
   Each block is a 2x2 * 2x2 multiplication.
   We use the standard Strassen decomposition for each.
   
   Let's define the Strassen factors for a single 2x2 * 2x2 multiplication.
   Let X be 2x2, Y be 2x2, Z = XY.
   Strassen computes Z using 7 products M1..M7.
   M1 = (X00 + X11) * (Y00 + Y11)
   M2 = (X10 + X11) * Y00
   M3 = X00 * (Y10 - Y11)
   M4 = X11 * (Y01 - Y00)
   M5 = (X00 + X01) * Y11
   M6 = (X10 - X00) * (Y00 + Y10)
   M7 = (X11 - X01) * (Y01 + Y11)
   
   Z00 = M1 - M4 + M5 + M7
   Z01 = M3 + M5
   Z10 = M2 + M4
   Z11 = M1 - M2 + M3 + M6
   
   Wait, let's verify these formulas.
   Standard Strassen:
   m1 = (a11 + a22)(b11 + b22)
   m2 = (a21 + a22)b11
   m3 = a11(b21 - b22)
   m4 = a22(b12 - b11)
   m5 = (a11 + a12)b22
   m6 = (a21 - a11)(b11 + b21)
   m7 = (a22 - a12)(b12 + b22)
   
   c11 = m1 - m4 + m5 + m7
   c12 = m3 + m5
   c21 = m2 + m4
   c22 = m1 - m2 + m3 + m6
   
   Let's check c12 = m3 + m5.
   m3 = a11(b21 - b22) = a11 b21 - a11 b22
   m5 = (a11 + a12)b22 = a11 b22 + a12 b22
   sum = a11 b21 + a12 b22. Correct.
   
   So for each of the 4 blocks, we add 7 columns to U, V, W.
   Total rank = 28.
   
   We need to map the local indices of the 2x2 blocks to the global indices of the 2x4x4 problem.
   
   Global Indices:
   A index: i * 4 + j. i in {0,1}, j in {0,1,2,3}.
   B index: j * 4 + k. j in {0,1,2,3}, k in {0,1,2,3}.
   C index: k * 2 + i. k in {0,1,2,3}, i in {0,1}.
   Wait, the task says: "output factor follows the official trace convention W[k*n+i,r]".
   n=2. So W index is k*2 + i.
   This matches C[k, i]? No, C is usually row i, col k.
   Let's check the tensor definition.
   T[i, j, k] = 1 if A[i, j] and B[j, k] contribute to C[i, k].
   The flattened index for U is i*m + j = i*4 + j.
   The flattened index for V is j*p + k = j*4 + k.
   The flattened index for W is k*n + i = k*2 + i.
   
   So we need to fill U, V, W such that:
   sum_r U[i*4+j, r] V[j*4+k, r] W[k*2+i, r] = 1 for all i,j,k.
   Wait, the tensor T is 1 for all valid i,j,k?
   Yes, because every term A[i,j]B[j,k] contributes to C[i,k].
   
   So we just need to construct the decomposition.
   
   Let's define the 4 blocks.
   Block 1: A_1 (i in 0,1; j in 0,1), B_11 (j in 0,1; k in 0,1), C_11 (i in 0,1; k in 0,1)
   Block 2: A_2 (i in 0,1; j in 2,3), B_21 (j in 2,3; k in 0,1), C_11 (i in 0,1; k in 0,1)
   Block 3: A_1 (i in 0,1; j in 0,1), B_12 (j in 0,1; k in 2,3), C_12 (i in 0,1; k in 2,3)
   Block 4: A_2 (i in 0,1; j in 2,3), B_22 (j in 2,3; k in 2,3), C_12 (i in 0,1; k in 2,3)
   
   For each block, we use Strassen.
   
   Let's write a helper to add a Strassen block to U, V, W.
   Inputs:
   - i_offset: 0 or 1 (row index of A/C) -- actually i is 0..1 always, but we map local 0,1 to global i.
   - j_offset_A: 0 or 2 (col offset of A)
   - j_offset_B: 0 or 2 (row offset of B)
   - k_offset: 0 or 2 (col offset of B/C)
   
   Local indices:
   i_loc in {0, 1}
   j_loc in {0, 1}
   k_loc in {0, 1}
   
   Global A index: i_loc * 4 + (j_offset_A + j_loc)
   Global B index: (j_offset_B + j_loc) * 4 + (k_offset + k_loc)
   Global C index: (k_offset + k_loc) * 2 + i_loc
   
   Strassen factors (U, V, W) for the 7 products:
   We need to define the linear forms for U (coefficients of A), V (coefficients of B), W (coefficients of C).
   
   M1: U = (1, 0, 0, 1) on A_11, A_22? No, A is 2x2.
   A = [[a11, a12], [a21, a22]].
   M1 uses (a11 + a22). So U vector for this product has 1 at a11 and 1 at a22.
   M1 uses (b11 + b22). So V vector has 1 at b11 and 1 at b22.
   W vector for M1 contributes to c11 and c22.
   c11 += M1. c22 += M1.
   So W has 1 at c11 and 1 at c22.
   
   Let's list the 7 products and their U, V, W vectors (local 2x2 indices).
   Indices for A: 00, 01, 10, 11.
   Indices for B: 00, 01, 10, 11.
   Indices for C: 00, 01, 10, 11.
   
   M1:
   U: [1, 0, 0, 1] (a11, a22)
   V: [1, 0, 0, 1] (b11, b22)
   W: [1, 0, 0, 1] (c11, c22)
   
   M2:
   U: [0, 0, 1, 1] (a21, a22)
   V: [1, 0, 0, 0] (b11)
   W: [0, 0, 1, 0] (c21)
   
   M3:
   U: [1, 0, 0, 0] (a11)
   V: [0, 0, 1, -1] (b21, -b22)
   W: [0, 1, 0, 0] (c12)
   
   M4:
   U: [0, 0, 0, 1] (a22)
   V: [0, 1, -1, 0] (b12, -b11)
   W: [0, 0, 1, 0] (c21) -- Wait, c21 = m2 + m4. Yes.
   
   M5:
   U: [1, 1, 0, 0] (a11, a12)
   V: [0, 0, 0, 1] (b22)
   W: [1, 1, 0, 0] (c11, c12)
   
   M6:
   U: [0, 0, 1, -1] (a21, -a11)
   V: [1, 0, 1, 0] (b11, b21)
   W: [0, 0, 0, 1] (c22)
   
   M7:
   U: [0, 0, 0, 1] (a22) -- Wait, M7 = (a22 - a12)(b12 + b22).
   U: [0, -1, 0, 1] (-a12, a22)
   V: [0, 1, 0, 1] (b12, b22)
   W: [1, 0, 0, 1] (c11, c22)
   
   Let's double check M7 U vector.
   a22 - a12. Indices: a12 is 01, a22 is 11.
   So U = [0, -1, 0, 1]. Correct.
   
   Now we just iterate over the 4 blocks and add these 7 vectors to the global U, V, W matrices.
   
   Total Rank = 28.
"""
import time
import numpy as np

DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    
    n, m, p = DIMENSIONS
    # n=2, m=4, p=4
    # U: (n*m, R) = (8, R)
    # V: (m*p, R) = (16, R)
    # W: (p*n, R) = (8, R)
    
    R = 28
    U = np.zeros((n * m, R))
    V = np.zeros((m * p, R))
    W = np.zeros((p * n, R))
    
    # Strassen factors for 2x2 * 2x2
    # Local indices for A (2x2): 00, 01, 10, 11
    # Local indices for B (2x2): 00, 01, 10, 11
    # Local indices for C (2x2): 00, 01, 10, 11
    
    # U_vecs: list of 7 vectors of length 4
    # V_vecs: list of 7 vectors of length 4
    # W_vecs: list of 7 vectors of length 4
    
    # M1
    U_M1 = [1, 0, 0, 1]
    V_M1 = [1, 0, 0, 1]
    W_M1 = [1, 0, 0, 1]
    
    # M2
    U_M2 = [0, 0, 1, 1]
    V_M2 = [1, 0, 0, 0]
    W_M2 = [0, 0, 1, 0]
    
    # M3
    U_M3 = [1, 0, 0, 0]
    V_M3 = [0, 0, 1, -1]
    W_M3 = [0, 1, 0, 0]
    
    # M4
    U_M4 = [0, 0, 0, 1]
    V_M4 = [0, 1, -1, 0]
    W_M4 = [0, 0, 1, 0]
    
    # M5
    U_M5 = [1, 1, 0, 0]
    V_M5 = [0, 0, 0, 1]
    W_M5 = [1, 1, 0, 0]
    
    # M6
    U_M6 = [0, 0, 1, -1]
    V_M6 = [1, 0, 1, 0]
    W_M6 = [0, 0, 0, 1]
    
    # M7
    U_M7 = [0, -1, 0, 1]
    V_M7 = [0, 1, 0, 1]
    W_M7 = [1, 0, 0, 1]
    
    strassen_U = [U_M1, U_M2, U_M3, U_M4, U_M5, U_M6, U_M7]
    strassen_V = [V_M1, V_M2, V_M3, V_M4, V_M5, V_M6, V_M7]
    strassen_W = [W_M1, W_M2, W_M3, W_M4, W_M5, W_M6, W_M7]
    
    # Define the 4 blocks
    # Block (j_offset_A, j_offset_B, k_offset)
    # j_offset_A: 0 or 2 (cols of A)
    # j_offset_B: 0 or 2 (rows of B)
    # k_offset: 0 or 2 (cols of B/C)
    
    blocks = [
        (0, 0, 0), # A_1 * B_11 -> C_11
        (2, 2, 0), # A_2 * B_21 -> C_11
        (0, 0, 2), # A_1 * B_12 -> C_12
        (2, 2, 2), # A_2 * B_22 -> C_12
    ]
    
    col = 0
    for j_off_A, j_off_B, k_off in blocks:
        for s in range(7):
            u_vec = strassen_U[s]
            v_vec = strassen_V[s]
            w_vec = strassen_W[s]
            
            # Map local indices to global indices
            # A local (i_loc, j_loc) -> global i*4 + (j_off_A + j_loc)
            # i_loc is 0 or 1. j_loc is 0 or 1.
            # A global index = i_loc * 4 + (j_off_A + j_loc)
            
            # B local (j_loc, k_loc) -> global (j_off_B + j_loc)*4 + (k_off + k_loc)
            # j_loc is 0 or 1. k_loc is 0 or 1.
            # B global index = (j_off_B + j_loc) * 4 + (k_off + k_loc)
            
            # C local (i_loc, k_loc) -> global (k_off + k_loc)*2 + i_loc
            # i_loc is 0 or 1. k_loc is 0 or 1.
            # C global index = (k_off + k_loc) * 2 + i_loc
            
            # u_vec indices: 0->00, 1->01, 2->10, 3->11
            # So index l in u_vec corresponds to:
            # if l==0: i_loc=0, j_loc=0
            # if l==1: i_loc=0, j_loc=1
            # if l==2: i_loc=1, j_loc=0
            # if l==3: i_loc=1, j_loc=1
            
            for l in range(4):
                val_u = u_vec[l]
                if val_u != 0:
                    i_loc = l // 2
                    j_loc = l % 2
                    global_u_idx = i_loc * 4 + (j_off_A + j_loc)
                    U[global_u_idx, col] += val_u
                    
                val_v = v_vec[l]
                if val_v != 0:
                    j_loc = l // 2
                    k_loc = l % 2
                    global_v_idx = (j_off_B + j_loc) * 4 + (k_off + k_loc)
                    V[global_v_idx, col] += val_v
                    
                val_w = w_vec[l]
                if val_w != 0:
                    i_loc = l // 2
                    k_loc = l % 2
                    global_w_idx = (k_off + k_loc) * 2 + i_loc
                    W[global_w_idx, col] += val_w
            
            col += 1
            
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
