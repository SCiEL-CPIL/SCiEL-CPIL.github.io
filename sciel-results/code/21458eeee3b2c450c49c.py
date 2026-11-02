"""Program for matrix multiplication tensor decomposition (4x4 * 4x7)."""
import time
import numpy as np

DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START
def _get_2_2_2_decomp():
    """
    Returns U, V, W for 2x2x2 matrix multiplication with rank 7.
    U: (4, 7), V: (4, 7), W: (4, 7)
    Indices: U[i*2+j, r], V[j*2+k, r], W[k*2+i, r]
    """
    # Strassen's 7 multiplications
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # Coefficients for A (u), B (v), C (w) for each of the 7 terms
    # u, v, w are 2x2 matrices (flattened to length 4)
    
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # Helper to set a 2x2 matrix into the flattened vector
    def set_mat(vec, mat):
        vec[0] = mat[0][0]
        vec[1] = mat[0][1]
        vec[2] = mat[1][0]
        vec[3] = mat[1][1]
        
    # Term 1: (A00+A11)(B00+B11) -> C00+=1, C01+=1, C10+=1, C11+=1 ? No.
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    # Wait, standard Strassen for 2x2 blocks:
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    # Here C11 is top-left 2x2 block.
    
    # Let's define the contributions to the 4 output elements (C00, C01, C10, C11)
    # for each of the 7 M's.
    
    # M1: (A00+A11)(B00+B11)
    # A part: A00 + A11 => [[1,0],[0,1]]
    # B part: B00 + B11 => [[1,0],[0,1]]
    # C part: Contributes to C00(1), C01(1), C10(1), C11(1)?
    # Actually, the product of two 2x2 matrices is a 2x2 matrix.
    # The "C part" in the tensor decomposition represents the coefficients of the resulting 2x2 matrix.
    # For M1, it contributes +1 to all 4 entries of the resulting 2x2 block?
    # No. M1 is a 2x2 matrix.
    # In the formula C11 = M1 + ..., M1 is added to the top-left 2x2 block of C.
    # So the "C vector" for M1 is all 1s.
    
    # Let's list the A, B, C matrices for each M.
    
    # M1
    A1 = np.array([[1, 0], [0, 1]])
    B1 = np.array([[1, 0], [0, 1]])
    C1 = np.array([[1, 1], [1, 1]]) # Adds to all 4 positions? 
    # Wait. C11 (top-left block) gets M1.
    # C12 (top-right block) gets M3+M5.
    # C21 (bottom-left block) gets M2+M4.
    # C22 (bottom-right block) gets M1-M2+M3+M6.
    
    # So for M1:
    # It contributes to C11 with +1, C22 with +1.
    # C11 is the top-left 2x2 block of the 4x4 result? No, this is the 2x2x2 decomposition.
    # The "C" here is the 2x2 result matrix of the 2x2x2 multiplication.
    # The formula above is for combining 2x2 blocks into 4x4.
    # Here we are decomposing the 2x2x2 multiplication itself.
    # The result C is a 2x2 matrix.
    # The Strassen formulas give C as a linear combination of M1..M7.
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # So for each M, we have a coefficient vector for C00, C01, C10, C11.
    
    coeffs_C = np.zeros((7, 4))
    # M1
    coeffs_C[0, 0] = 1 # C00
    coeffs_C[0, 1] = 0 # C01
    coeffs_C[0, 2] = 0 # C10
    coeffs_C[0, 3] = 1 # C11
    
    # M2
    coeffs_C[1, 0] = 0
    coeffs_C[1, 1] = 0
    coeffs_C[1, 2] = 1 # C10
    coeffs_C[1, 3] = -1 # C11
    
    # M3
    coeffs_C[2, 0] = 0
    coeffs_C[2, 1] = 1 # C01
    coeffs_C[2, 2] = 0
    coeffs_C[2, 3] = 1 # C11
    
    # M4
    coeffs_C[3, 0] = 1 # C00
    coeffs_C[3, 1] = 0
    coeffs_C[3, 2] = 1 # C10
    coeffs_C[3, 3] = 0
    
    # M5
    coeffs_C[4, 0] = -1 # C00
    coeffs_C[4, 1] = 1 # C01
    coeffs_C[4, 2] = 0
    coeffs_C[4, 3] = 0
    
    # M6
    coeffs_C[5, 0] = 0
    coeffs_C[5, 1] = 0
    coeffs_C[5, 2] = 0
    coeffs_C[5, 3] = 1 # C11
    
    # M7
    coeffs_C[6, 0] = 1 # C00
    coeffs_C[6, 1] = 0
    coeffs_C[6, 2] = 0
    coeffs_C[6, 3] = 0
    
    # Now A and B matrices for each M
    # M1: (A00+A11)(B00+B11)
    A_mat[0] = np.array([[1, 0], [0, 1]])
    B_mat[0] = np.array([[1, 0], [0, 1]])
    
    # M2: (A10+A11)B00
    A_mat[1] = np.array([[0, 0], [1, 1]])
    B_mat[1] = np.array([[1, 0], [0, 0]])
    
    # M3: A00(B01-B11)
    A_mat[2] = np.array([[1, 0], [0, 0]])
    B_mat[2] = np.array([[0, 1], [0, -1]])
    
    # M4: A11(B10-B00)
    A_mat[3] = np.array([[0, 0], [0, 1]])
    B_mat[3] = np.array([[-1, 0], [1, 0]])
    
    # M5: (A00+A01)B11
    A_mat[4] = np.array([[1, 1], [0, 0]])
    B_mat[4] = np.array([[0, 0], [0, 1]])
    
    # M6: (A10-A00)(B00+B01)
    A_mat[5] = np.array([[-1, 0], [1, 0]])
    B_mat[5] = np.array([[1, 1], [0, 0]])
    
    # M7: (A01-A11)(B10+B11)
    A_mat[6] = np.array([[0, 1], [0, -1]])
    B_mat[6] = np.array([[0, 0], [1, 1]])
    
    # Fill U, V, W
    for r in range(7):
        set_mat(U[:, r], A_mat[r])
        set_mat(V[:, r], B_mat[r])
        # W is just the coefficients for C00..C11
        W[:, r] = coeffs_C[r]
        
    return U, V, W

def _get_2_2_1_decomp():
    """
    Returns U, V, W for 2x2x1 matrix multiplication with rank 4.
    U: (4, 4), V: (2, 4), W: (2, 4)
    """
    U = np.zeros((4, 4))
    V = np.zeros((2, 4))
    W = np.zeros((2, 4))
    
    # Terms:
    # 0: A00 B0 -> C0
    U[0, 0] = 1 # A00
    V[0, 0] = 1 # B0
    W[0, 0] = 1 # C0
    
    # 1: A01 B1 -> C0
    U[1, 1] = 1 # A01
    V[1, 1] = 1 # B1
    W[0, 1] = 1 # C0
    
    # 2: A10 B0 -> C1
    U[2, 2] = 1 # A10
    V[0, 2] = 1 # B0
    W[1, 2] = 1 # C1
    
    # 3: A11 B1 -> C1
    U[3, 3] = 1 # A11
    V[1, 3] = 1 # B1
    W[1, 3] = 1 # C1
    
    return U, V, W

def construct():
    n, m, p = DIMENSIONS # 4, 4, 7
    
    # Target shapes
    # U: (16, rank)
    # V: (28, rank)
    # W: (28, rank)
    
    # Part 1: 4x4x4 (Rank 49)
    # Part 2: 4x4x3 (Rank 44)
    
    U_parts = []
    V_parts = []
    W_parts = []
    
    # --- Construct 4x4x4 ---
    U_222, V_222, W_222 = _get_2_2_2_decomp()
    
    # We need to "inflate" the 2x2x2 decomp to 4x4x4 using Strassen blocks.
    # There are 7 blocks.
    # For each block, we have A_mat, B_mat, C_coeffs from the Strassen logic.
    # We reuse the A_mat, B_mat from _get_2_2_2_decomp logic.
    
    # Let's redefine the Strassen block coefficients explicitly for clarity
    A_blocks = [
        np.array([[1, 0], [0, 1]]), # M1
        np.array([[0, 0], [1, 1]]), # M2
        np.array([[1, 0], [0, 0]]), # M3
        np.array([[0, 0], [0, 1]]), # M4
        np.array([[1, 1], [0, 0]]), # M5
        np.array([[-1, 0], [1, 0]]),# M6
        np.array([[0, 1], [0, -1]]) # M7
    ]
    B_blocks = [
        np.array([[1, 0], [0, 1]]), # M1
        np.array([[1, 0], [0, 0]]), # M2
        np.array([[0, 1], [0, -1]]),# M3
        np.array([[-1, 0], [1, 0]]),# M4
        np.array([[0, 0], [0, 1]]), # M5
        np.array([[1, 1], [0, 0]]), # M6
        np.array([[0, 0], [1, 1]])  # M7
    ]
    C_coeffs = [
        np.array([1, 0, 0, 1]), # M1: C00, C11
        np.array([0, 0, 1, -1]),# M2: C10, C11
        np.array([0, 1, 0, 1]), # M3: C01, C11
        np.array([1, 0, 1, 0]), # M4: C00, C10
        np.array([-1, 1, 0, 0]),# M5: C00, C01
        np.array([0, 0, 0, 1]), # M6: C11
        np.array([1, 0, 0, 0])  # M7: C00
    ]
    
    U_444 = np.zeros((16, 49))
    V_444 = np.zeros((16, 49))
    W_444 = np.zeros((16, 49))
    
    col_idx = 0
    for b in range(7):
        for r in range(7):
            # Expand A_block[b] with U_222[:, r]
            # A_block[b] is 2x2. U_222[:, r] is 4x1 (coeffs for local A elements).
            # The global A is 4x4.
            # The local A is a 2x2 submatrix.
            # The coefficient for global A[i, j] is:
            # If i, j belong to block (bi, bj), then coeff = A_block[b][bi, bj] * U_222[i_local, j_local, r]
            
            for bi in range(2):
                for bj in range(2):
                    c_block = A_blocks[b][bi, bj]
                    if c_block != 0:
                        for li in range(2):
                            for lj in range(2):
                                global_i = bi * 2 + li
                                global_j = bj * 2 + lj
                                local_u = U_222[li * 2 + lj, r]
                                U_444[global_i * 4 + global_j, col_idx] = c_block * local_u
                                
            # Same for V
            for bj in range(2):
                for bk in range(2):
                    c_block = B_blocks[b][bj, bk]
                    if c_block != 0:
                        for lj in range(2):
                            for lk in range(2):
                                global_j = bj * 2 + lj
                                global_k = bk * 2 + lk
                                local_v = V_222[lj * 2 + lk, r]
                                V_444[global_j * 4 + global_k, col_idx] = c_block * local_v
                                
            # Same for W
            # W_222[:, r] gives coeffs for local C elements (C00, C01, C10, C11)
            # C_coeffs[b] gives coeffs for global C blocks (C00, C01, C10, C11)
            for bc in range(2): # Block row
                for bk in range(2): # Block col
                    c_block = C_coeffs[b][bc * 2 + bk]
                    if c_block != 0:
                        for li in range(2):
                            for lk in range(2):
                                global_i = bc * 2 + li
                                global_k = bk * 2 + lk
                                local_w = W_222[li * 2 + lk, r]
                                W_444[global_k * 4 + global_i, col_idx] = c_block * local_w
                                
            col_idx += 1
            
    U_parts.append(U_444)
    V_parts.append(V_444)
    W_parts.append(W_444)
    
    # --- Construct 4x4x3 ---
    # Split into two 2x4x3
    # Each 2x4x3 splits into two 2x2x3
    # Each 2x2x3 is 2x2x2 (rank 7) + 2x2x1 (rank 4) = rank 11
    
    U_222, V_222, W_222 = _get_2_2_2_decomp()
    U_221, V_221, W_221 = _get_2_2_1_decomp()
    
    # Construct one 2x2x3 (n=2, m=2, p=3)
    # U: (4, 11), V: (6, 11), W: (6, 11)
    U_223 = np.zeros((4, 11))
    V_223 = np.zeros((6, 11))
    W_223 = np.zeros((6, 11))
    
    # Part 1: 2x2x2 (k=0,1)
    # U_222 is (4,7). V_222 is (4,7). W_222 is (4,7).
    # V index: j*p+k. p=2.
    # W index: k*n+i. n=2.
    U_223[:, 0:7] = U_222
    for r in range(7):
        for j in range(2):
            for k in range(2):
                V_223[j*2+k, r] = V_222[j*2+k, r]
        for k in range(2):
            for i in range(2):
                W_223[k*2+i, r] = W_222[k*2+i, r]
                
    # Part 2: 2x2x1 (k=2)
    # U_221 is (4,4). V_221 is (2,4). W_221 is (2,4).
    # U index: i*m+j. m=2.
    # V index: j*p+k. p=1. k is fixed to 2 in the 2x2x3 context?
    # In 2x2x3, the "1" part of 2x2x1 corresponds to the 3rd column (k=2).
    # So V index for the 2x2x1 part is j*3+2.
    # W index for the 2x2x1 part is 2*2+i = 4+i.
    
    U_223[:, 7:11] = U_221
    for r in range(4):
        for j in range(2):
            V_223[j*3+2, 7+r] = V_221[j, r]
        for i in range(2):
            W_223[4+i, 7+r] = W_221[i, r]
            
    # Now we have one 2x2x3 decomp.
    # We need two of them for 2x4x3 (splitting m=4 into m1=2, m2=2).
    # 2x4x3: n=2, m=4, p=3.
    # U: (8, 22), V: (12, 22), W: (6, 22)
    
    U_243 = np.zeros((8, 22))
    V_243 = np.zeros((12, 22))
    W_243 = np.zeros((6, 22))
    
    # Part 1: m=0,1 (first 2x2x3)
    # U rows 0-3 correspond to j=0,1.
    # V rows 0-5 correspond to j=0,1.
    # W rows 0-5 are shared.
    
    U_243[0:4, 0:11] = U_223
    for r in range(11):
        for j in range(2):
            for k in range(3):
                V_243[j*3+k, r] = V_223[j*3+k, r]
        W_243[:, r] = W_223[:, r]
        
    # Part 2: m=2,3 (second 2x2x3)
    # U rows 4-7 correspond to j=2,3.
    # V rows 6-11 correspond to j=2,3.
    # W rows 0-5 are shared (added to).
    
    U_243[4:8, 11:22] = U_223
    for r in range(11):
        for j in range(2):
            for k in range(3):
                V_243[(j+2)*3+k, 11+r] = V_223[j*3+k, r]
        W_243[:, 11+r] = W_223[:, r]
        
    # Now we have one 2x4x3 decomp.
    # We need two of them for 4x4x3 (splitting n=4 into n1=2, n2=2).
    # 4x4x3: n=4, m=4, p=3.
    # U: (16, 44), V: (12, 44), W: (12, 44)
    
    U_443 = np.zeros((16, 44))
    V_443 = np.zeros((12, 44))
    W_443 = np.zeros((12, 44))
    
    # Part 1: n=0,1
    # U rows 0-7 (i=0,1)
    # W rows 0-5 (k=0,1,2, i=0,1)
    
    U_443[0:8, 0:22] = U_243
    for r in range(22):
        V_443[:, r] = V_243[:, r]
        for k in range(3):
            for i in range(2):
                W_443[k*4+i, r] = W_243[k*2+i, r]
                
    # Part 2: n=2,3
    # U rows 8-15 (i=2,3)
    # W rows 6-11 (k=0,1,2, i=2,3)
    
    U_443[8:16, 22:44] = U_243
    for r in range(22):
        V_443[:, 22+r] = V_243[:, r]
        for k in range(3):
            for i in range(2):
                W_443[k*4+(i+2), 22+r] = W_243[k*2+i, r]
                
    U_parts.append(U_443)
    V_parts.append(V_443)
    W_parts.append(W_443)
    
    # Combine all parts
    # Total Rank = 49 + 44 = 93
    
    U_final = np.hstack(U_parts)
    V_final = np.hstack(V_parts)
    W_final = np.hstack(W_parts)
    
    return {"U": U_final.tolist(), "V": V_final.tolist(), "W": W_final.tolist()}
# EVOLVE-BLOCK-END
