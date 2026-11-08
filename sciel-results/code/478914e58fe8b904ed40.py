"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [4, 4, 9]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 4, 4, 9
    rank_school = n * m * p  # 144

    # Strassen for 4x4x4: rank 112 (2^7)
    # We need to handle the 4x4x9 case.
    # The matrix multiplication is A (4x4) * B (4x9) = C (4x9).
    # We can split B into B1 (4x4) and B2 (4x4) and B3 (4x1).
    # C = A*B1 + A*B2 + A*B3.
    # If we use Strassen for A*B1 and A*B2, and schoolbook for A*B3, we get:
    # Strassen rank for 4x4x4 is 2^7 = 128? No, Strassen for 2x2 is 7.
    # For 4x4, it's 7^2 = 49 if we just do 2x2 blocks?
    # Wait, Strassen's algorithm on a 4x4 matrix (treating it as 2x2 blocks of 2x2)
    # uses 7 multiplications of 2x2 matrices.
    # Each 2x2 multiplication is rank 7 (if we use Strassen again) or 8 (schoolbook).
    # If we use Strassen recursively:
    # Level 0: 2x2, rank 7
    # Level 1: 4x4, rank 7^2 = 49?
    # Let's verify the tensor rank of 4x4x4 multiplication.
    # Known result: R(M_{4,4,4}) <= 49. (Landsberg, 2010 or similar).
    # Actually, the tensor rank of 4x4x4 is exactly 49?
    # Strassen's algorithm gives a decomposition of rank 7^2 = 49.
    # So we can use rank 49 for the 4x4x4 part.
    
    # But we have 4x4x9.
    # B is 4x9. We can split it into two 4x4 matrices and one 4x1 matrix.
    # C = A * [B1 B2 B3] = [A*B1 A*B2 A*B3].
    # A*B1 and A*B2 are 4x4x4 multiplications.
    # A*B3 is 4x4x1.
    
    # Rank of 4x4x4 is 49 (using Strassen).
    # Rank of 4x4x1 is 16 (schoolbook, or maybe lower? 4*4*1 = 16).
    # Total rank = 49 + 49 + 16 = 114.
    
    # This is better than 144.
    
    # Let's construct the Strassen decomposition for 4x4x4.
    # Strassen for 2x2:
    # M1 = (A11 + A22) * (B11 + B22)
    # M2 = (A21 + A22) * B11
    # M3 = A11 * (B12 - B22)
    # M4 = A22 * (B21 - B11)
    # M5 = (A11 + A12) * B22
    # M6 = (A21 - A11) * (B11 + B12)
    # M7 = (A12 - A22) * (B21 + B22)
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # For 4x4, we treat A, B as 2x2 blocks of 2x2 matrices.
    # A = [[A11, A12], [A21, A22]] where each Ai is 2x2.
    # B = [[B11, B12], [B21, B22]] where each Bj is 2x2.
    # The 7 multiplications are tensor products of the 2x2 Strassen steps.
    
    # Let's build the U, V, W matrices for the 4x4x4 case using the recursive structure.
    # We will define a function to get the U, V, W for Strassen 2x2.
    # Then we will use it to build the 4x4 case.
    
    # Actually, it's easier to just hardcode the 49 terms for 4x4x4.
    # Or better, we can use the fact that the 4x4x4 multiplication tensor is the Kronecker product of two 2x2x2 multiplication tensors.
    # The rank of the Kronecker product is the product of the ranks.
    # Rank(2x2x2) = 7 (Strassen).
    # Rank(4x4x4) <= 7 * 7 = 49.
    
    # Let's construct the 49 terms.
    # Each term is a product of a 2x2 Strassen term and another 2x2 Strassen term.
    # The U vector for a term (i, j) is U_2x2(i) tensor U_2x2(j).
    # Similarly for V and W.
    
    # Let's define the U, V, W for the 2x2 Strassen case.
    # U is 4x7, V is 4x7, W is 4x7.
    # U maps A to the 7 intermediate values.
    # V maps B to the 7 intermediate values.
    # W maps the 7 intermediate values to C.
    
    # Let's build these matrices.
    U2 = np.zeros((4, 7), dtype=int)
    V2 = np.zeros((4, 7), dtype=int)
    W2 = np.zeros((4, 7), dtype=int)
    
    # Term 1: M1 = (A11 + A22) * (B11 + B22)
    # A11 is index 0, A22 is index 3 (in 0-based, row-major: 00,01,10,11 -> 0,1,2,3)
    # B11 is index 0, B22 is index 3.
    # M1 = (A0 + A3) * (B0 + B3)
    U2[0, 0] = 1; U2[3, 0] = 1
    V2[0, 0] = 1; V2[3, 0] = 1
    # C11 (0) = M1 - M2 + M3 + M6? No, let's use the standard Strassen formula.
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # Let's re-verify the Strassen formula.
    # M1 = (A11 + A22)(B11 + B22)
    # M2 = (A21 + A22)B11
    # M3 = A11(B12 - B22)
    # M4 = A22(B21 - B11)
    # M5 = (A11 + A12)B22
    # M6 = (A21 - A11)(B11 + B12)
    # M7 = (A12 - A22)(B21 + B22)
    
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # Let's fill U2, V2, W2.
    # Term 0 (M1)
    U2[0, 0] = 1; U2[3, 0] = 1
    V2[0, 0] = 1; V2[3, 0] = 1
    W2[0, 0] = 1 # C11
    W2[3, 0] = 1 # C22
    
    # Term 1 (M2)
    U2[2, 1] = 1; U2[3, 1] = 1
    V2[0, 1] = 1
    W2[2, 1] = 1 # C21
    W2[3, 1] = -1 # C22
    
    # Term 2 (M3)
    U2[0, 2] = 1
    V2[1, 2] = 1; V2[3, 2] = -1
    W2[1, 2] = 1 # C12
    W2[3, 2] = 1 # C22
    
    # Term 3 (M4)
    U2[3, 3] = 1
    V2[2, 3] = 1; V2[0, 3] = -1
    W2[0, 3] = 1 # C11
    W2[2, 3] = 1 # C21
    
    # Term 4 (M5)
    U2[0, 4] = 1; U2[1, 4] = 1
    V2[3, 4] = 1
    W2[0, 4] = -1 # C11
    W2[1, 4] = 1 # C12
    
    # Term 5 (M6)
    U2[2, 5] = 1; U2[0, 5] = -1
    V2[0, 5] = 1; V2[1, 5] = 1
    W2[3, 5] = 1 # C22
    
    # Term 6 (M7)
    U2[1, 6] = 1; U2[3, 6] = -1
    V2[2, 6] = 1; V2[3, 6] = 1
    W2[0, 6] = 1 # C11
    
    # Now, for 4x4, we have 16 elements in A, B, C.
    # We can map the 4x4 case to 49 terms.
    # The U, V, W for 4x4 are 16x49 matrices.
    # Each term (a, b) where a, b in 0..6 corresponds to a Kronecker product of U2[:, a] and U2[:, b].
    # But we need to be careful with the indexing.
    
    # Let's build U4, V4, W4.
    U4 = np.zeros((16, 49), dtype=int)
    V4 = np.zeros((16, 49), dtype=int)
    W4 = np.zeros((16, 49), dtype=int)
    
    for a in range(7):
        for b in range(7):
            col = a * 7 + b
            # U4[:, col] = np.kron(U2[:, a], U2[:, b])
            # But we need to check the ordering.
            # A is 4x4, so A[i, j] is at index i*4 + j.
            # U2 maps a 2x2 matrix (indices 0..3) to 7 values.
            # The Kronecker product will map the 4x4 matrix to 49 values.
            # We need to ensure the ordering is consistent.
            
            U4[:, col] = np.kron(U2[:, a], U2[:, b])
            V4[:, col] = np.kron(V2[:, a], V2[:, b])
            W4[:, col] = np.kron(W2[:, a], W2[:, b])
            
    # Let's verify the 4x4x4 decomposition.
    # We can test it with random matrices.
    # A = np.random.rand(4, 4)
    # B = np.random.rand(4, 4)
    # C = A @ B
    # U4_mat = U4.T @ A.flatten()
    # V4_mat = V4.T @ B.flatten()
    # W4_mat = W4.T @ (U4_mat * V4_mat)
    # C_check = W4_mat.reshape(4, 4)
    # print(np.allclose(C, C_check))
    
    # Now, we need to handle the 4x4x9 case.
    # B is 4x9. We can split it into B1 (4x4), B2 (4x4), B3 (4x1).
    # C = A*B1 + A*B2 + A*B3.
    # A*B1 and A*B2 are 4x4x4 multiplications.
    # A*B3 is 4x4x1.
    
    # For A*B1, we use the 49 terms from U4, V4, W4.
    # For A*B2, we use the same 49 terms, but with B2 instead of B1.
    # For A*B3, we use schoolbook, which is 16 terms.
    
    # Total rank = 49 + 49 + 16 = 114.
    
    # Let's construct the final U, V, W.
    # U is 16x114, V is 36x114, W is 36x114.
    # U has 16 rows (A is 4x4).
    # V has 36 rows (B is 4x9, so 36 elements).
    # W has 36 rows (C is 4x9, so 36 elements).
    
    rank = 114
    U = np.zeros((16, rank), dtype=int)
    V = np.zeros((36, rank), dtype=int)
    W = np.zeros((36, rank), dtype=int)
    
    # Part 1: A*B1 (4x4x4)
    # U[:, 0:49] = U4
    # V[0:16, 0:49] = V4 (B1 is the first 4 columns of B, so indices 0..15 in B)
    # W[0:16, 0:49] = W4 (C1 is the first 4 columns of C, so indices 0..15 in C)
    U[:, 0:49] = U4
    V[0:16, 0:49] = V4
    W[0:16, 0:49] = W4
    
    # Part 2: A*B2 (4x4x4)
    # U[:, 49:98] = U4
    # V[16:32, 49:98] = V4 (B2 is the next 4 columns of B, so indices 16..31 in B)
    # W[16:32, 49:98] = W4 (C2 is the next 4 columns of C, so indices 16..31 in C)
    U[:, 49:98] = U4
    V[16:32, 49:98] = V4
    W[16:32, 49:98] = W4
    
    # Part 3: A*B3 (4x4x1)
    # Schoolbook for 4x4x1.
    # A is 4x4, B3 is 4x1, C3 is 4x1.
    # C3[i] = sum_j A[i, j] * B3[j].
    # So we have 16 terms, one for each (i, j).
    for i in range(4):
        for j in range(4):
            col = 98 + i * 4 + j
            U[i * 4 + j, col] = 1
            V[32 + j, col] = 1 # B3 is the last 1 column of B, so index 32 + j
            W[32 + i, col] = 1 # C3 is the last 1 column of C, so index 32 + i
    
    # Convert to lists
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
