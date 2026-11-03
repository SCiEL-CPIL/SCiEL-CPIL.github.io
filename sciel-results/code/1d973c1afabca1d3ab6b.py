import time
import random
import numpy as np

DIMENSIONS = [3, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for 3x3 * 3x8 matrix multiplication.
    Returns {'U', 'V', 'W'} with integer/half-integer entries.
    """
    n, m, p = DIMENSIONS
    deadline = time.time() + 145  # 5s safety margin
    
    # 1. Schoolbook decomposition as fallback (Rank 72)
    rank_school = n * m * p
    U_school = [[0] * rank_school for _ in range(n * m)]
    V_school = [[0] * rank_school for _ in range(m * p)]
    W_school = [[0] * rank_school for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1
    
    best_rank = rank_school
    best_U, best_V, best_W = U_school, V_school, W_school

    # Validation function
    def validate_decomposition(U_mat, V_mat, W_mat, rank_val):
        U_arr = np.array(U_mat, dtype=float)
        V_arr = np.array(V_mat, dtype=float)
        W_arr = np.array(W_mat, dtype=float)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U_arr[i * m + j, :], V_arr[j * p + k, :] * W_arr[k * n + i, :])
                    if abs(val - 1.0) > 1e-9:
                        return False
        return True

    # Verify schoolbook
    if not validate_decomposition(U_school, V_school, W_school, rank_school):
        # Should not happen
        return {"U": U_school, "V": U_school, "W": W_school}

    # 2. Block Strassen Decomposition (Rank 31)
    # Partition:
    # A: 2x2 (top-left), 2x1 (top-right), 1x2 (bottom-left), 1x1 (bottom-right)
    # B: 2x8 (top), 1x8 (bottom)
    # C: 2x8 (top), 1x8 (bottom)
    
    # Blocks:
    # 1. i in {0,1}, j in {0,1}: 2x2x8 -> 4 * Strassen(2x2x2) -> Rank 28
    # 2. i = 2, j in {0,1}: 1x2x8 -> Rank 1
    # 3. i in {0,1}, j = 2: 2x1x8 -> Rank 1
    # 4. i = 2, j = 2: 1x1x8 -> Rank 1
    
    rank_block1 = 28
    rank_block2 = 1
    rank_block3 = 1
    rank_block4 = 1
    total_rank = rank_block1 + rank_block2 + rank_block3 + rank_block4
    
    U_new = [[0] * total_rank for _ in range(n * m)]
    V_new = [[0] * total_rank for _ in range(m * p)]
    W_new = [[0] * total_rank for _ in range(p * n)]
    
    # Strassen 2x2x2 coefficients
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    # C11 = M1+M4-M5+M7
    # C12 = M3+M5
    # C21 = M2+M4
    # C22 = M1-M2+M3+M6
    
    # We need to fill U, V, W for the 2x2x2 blocks.
    # Indices for 2x2x2:
    # i in {0,1}, j in {0,1}, k in {0,1} (relative to block)
    
    # Strassen Terms (u, v, w) where u is index in A (0-3), v in B (0-3), w in C (0-3)
    # A indices: 00, 01, 10, 11 -> 0, 1, 2, 3
    # B indices: 00, 01, 10, 11 -> 0, 1, 2, 3
    # C indices: 00, 01, 10, 11 -> 0, 1, 2, 3
    
    strassen_terms = [
        # M1
        [(0, 1), (0, 1), (0, 1)], # A00, B00, C00
        [(0, 1), (1, 1), (0, 1)], # A00, B01, C00 -- Wait, Strassen uses sums.
        # It is easier to define the linear forms.
    ]
    
    # Let's use the explicit rank 1 tensors for Strassen.
    # There are 7 terms.
    # Term 1: (A00+A11)(B00+B11) -> C00, C11
    # This is not rank 1.
    # Strassen decomposition is a sum of 7 rank 1 tensors?
    # No, Strassen is a decomposition of the tensor into 7 terms.
    # Each term is a product of 3 linear forms.
    # L1_A = A00+A11, L1_B = B00+B11, L1_C = C00-C11? No.
    
    # Standard Strassen Rank 1 Tensors:
    # 1. (A00+A11)(B00+B11) -> contributes to C00, C11
    # Actually, the decomposition is:
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # The 7 M's are:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # We can represent each M as a sum of rank 1 tensors?
    # No, M1 is a product of two linear forms.
    # The tensor for M1 is L_A \otimes L_B \otimes I?
    # No, it contributes to multiple C indices.
    # M1 contributes to C00 and C11.
    # So the tensor for M1 is:
    # L_A \otimes L_B \otimes (E_{00} + E_{11})?
    # This is rank 2?
    # Actually, Strassen rank 7 means the tensor is sum of 7 rank 1 tensors.
    # The 7 tensors are:
    # T1 = (A00+A11) \otimes (B00+B11) \otimes (C00 - C11) ? No.
    
    # Let's just use the known rank 7 decomposition vectors.
    # U, V, W for 2x2x2.
    # Rows of U: A00, A01, A10, A11.
    # Rows of V: B00, B01, B10, B11.
    # Rows of W: C00, C01, C10, C11.
    
    U_str = [
        [1, 1, 1, 0, 1, 1, 0], # A00
        [0, 0, 0, 1, 1, 0, 1], # A01
        [1, 1, 0, 1, 0, 1, 1], # A10
        [1, 1, 0, 1, 1, 0, 1]  # A11
    ]
    V_str = [
        [1, 1, 0, 1, 0, 1, 1], # B00
        [0, 0, 1, 0, 1, 1, 0], # B01
        [1, 0, 1, 1, 1, 0, 1], # B10
        [1, 0, 1, 0, 1, 0, 1]  # B11
    ]
    W_str = [
        [1, 0, 0, 1, -1, 0, 1], # C00
        [0, 0, 1, 0, 1, 0, 0],  # C01
        [0, 1, 0, 1, 0, 1, 0],  # C10
        [1, -1, 1, 0, 0, 0, 1]  # C11
    ]
    
    # Verify Strassen
    def validate_strassen():
        for i in range(4):
            for j in range(4):
                for k in range(4):
                    val = np.dot(np.array(U_str[i]), np.array(V_str[j]) * np.array(W_str[k]))
                    # We need val to be 1 if i,j,k correspond to valid multiplication, 0 otherwise?
                    # No, the tensor T_{i,j,k} = 1 for all i,j,k in 2x2x2.
                    # So val should be 1 for all i,j,k.
                    if abs(val - 1.0) > 1e-9:
                        return False
        return True
    
    if not validate_strassen():
        print("Strassen validation failed")
        return {"U": best_U, "V": best_V, "W": best_W}

    # Fill Block 1: i in {0,1}, j in {0,1}, k in {0..7}
    # 4 blocks of 2x2x2.
    for block_k in range(4):
        for r in range(7):
            col_idx = block_k * 7 + r
            for i in range(2):
                for j in range(2):
                    U_new[i * m + j][col_idx] = U_str[i * 2 + j][r]
            for j in range(2):
                for k_rel in range(2):
                    k_abs = block_k * 2 + k_rel
                    V_new[j * p + k_abs][col_idx] = V_str[j * 2 + k_rel][r]
            for k_rel in range(2):
                for i in range(2):
                    k_abs = block_k * 2 + k_rel
                    W_new[k_abs * n + i][col_idx] = W_str[i * 2 + k_rel][r]

    # Block 2: i=2, j in {0,1}, k in {0..7}
    # Rank 1
    col_idx = rank_block1
    for j in range(2):
        U_new[2 * m + j][col_idx] = 1
    for j in range(2):
        for k in range(p):
            V_new[j * p + k][col_idx] = 1
    for k in range(p):
        W_new[k * n + 2][col_idx] = 1

    # Block 3: i in {0,1}, j=2, k in {0..7}
    # Rank 1
    col_idx = rank_block1 + rank_block2
    for i in range(2):
        U_new[i * m + 2][col_idx] = 1
    for k in range(p):
        V_new[2 * p + k][col_idx] = 1
    for k in range(p):
        for i in range(2):
            W_new[k * n + i][col_idx] = 1

    # Block 4: i=2, j=2, k in {0..7}
    # Rank 1
    col_idx = rank_block1 + rank_block2 + rank_block3
    U_new[2 * m + 2][col_idx] = 1
    for k in range(p):
        V_new[2 * p + k][col_idx] = 1
    for k in range(p):
        W_new[k * n + 2][col_idx] = 1

    if validate_decomposition(U_new, V_new, W_new, total_rank):
        best_rank = total_rank
        best_U, best_V, best_W = U_new, V_new, W_new

    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
