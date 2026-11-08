import numpy as np
import time

DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    n, m, p = DIMENSIONS
    
    # We use the recursive Strassen algorithm.
    # Rank of 2x2x2 is 7.
    # Rank of 4x4x4 is 49 (7 * 7).
    # We need to multiply 4x4 by 4x8.
    # This is equivalent to two 4x4x4 multiplications (one for the left half of B, one for the right).
    # Total rank = 49 + 49 = 98.
    
    R = 98
    U = np.zeros((n * m, R), dtype=float)
    V = np.zeros((m * p, R), dtype=float)
    W = np.zeros((p * n, R), dtype=float)
    
    # Define the rank 7 decomposition for 2x2x2 (Strassen)
    # Indices: U[i*2+j], V[j*2+k], W[k*2+i]
    S_U = np.zeros((4, 7))
    S_V = np.zeros((4, 7))
    S_W = np.zeros((4, 7))
    
    # Strassen terms for 2x2:
    # M1 = (A00+A11)(B00+B11)
    S_U[0, 0] = 1; S_U[3, 0] = 1
    S_V[0, 0] = 1; S_V[3, 0] = 1
    S_W[0, 0] = 1; S_W[3, 0] = 1
    
    # M2 = (A10+A11)B00
    S_U[2, 1] = 1; S_U[3, 1] = 1
    S_V[0, 1] = 1
    S_W[2, 1] = 1; S_W[3, 1] = 1
    
    # M3 = A00(B01-B11)
    S_U[0, 2] = 1
    S_V[1, 2] = 1; S_V[3, 2] = -1
    S_W[1, 2] = 1; S_W[3, 2] = 1
    
    # M4 = A11(B10-B00)
    S_U[3, 3] = 1
    S_V[2, 3] = 1; S_V[0, 3] = -1
    S_W[0, 3] = 1; S_W[2, 3] = 1
    
    # M5 = (A00+A01)B11
    S_U[0, 4] = 1; S_U[1, 4] = 1
    S_V[3, 4] = 1
    S_W[0, 4] = -1; S_W[1, 4] = 1
    
    # M6 = (A10-A00)(B00+B01)
    S_U[2, 5] = 1; S_U[0, 5] = -1
    S_V[0, 5] = 1; S_V[1, 5] = 1
    S_W[3, 5] = 1
    
    # M7 = (A01-A11)(B10+B11)
    S_U[1, 6] = 1; S_U[3, 6] = -1
    S_V[2, 6] = 1; S_V[3, 6] = 1
    S_W[0, 6] = 1; S_W[3, 6] = 1
    
    # Define the 7 "super-terms" for 4x4x4 Strassen.
    # Each term t is defined by linear combinations of the 2x2 blocks of A and B.
    # We represent the linear combination coefficients for A blocks (2x2) and B blocks (2x2).
    # A_coefs[t] is a 2x2 matrix where A_coefs[t][I, J] is the coefficient of block A[I, J].
    # B_coefs[t] is a 2x2 matrix where B_coefs[t][I, J] is the coefficient of block B[I, J].
    # C_coefs[t] is a 2x2 matrix where C_coefs[t][I, J] is the coefficient of the result block C[I, J].
    
    # Standard Strassen for 4x4 (blocks 00, 01, 10, 11):
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # Reconstruction:
    # C00 = M1 + M4 - M5 + M6
    # C01 = M4 + M5
    # C10 = M2 + M3
    # C11 = M1 - M2 + M3 + M5
    
    A_coefs = [
        [[1, 0], [0, 1]], # M1: A00 + A11
        [[0, 0], [1, 1]], # M2: A10 + A11
        [[1, 0], [0, 0]], # M3: A00
        [[0, 0], [0, 1]], # M4: A11
        [[1, 1], [0, 0]], # M5: A00 + A01
        [[-1, 0], [1, 0]],# M6: A10 - A00
        [[0, 1], [0, -1]] # M7: A01 - A11
    ]
    
    B_coefs = [
        [[1, 0], [0, 1]], # M1: B00 + B11
        [[1, 0], [0, 0]], # M2: B00
        [[0, 1], [0, -1]],# M3: B01 - B11
        [[0, -1], [1, 0]],# M4: B10 - B00
        [[0, 0], [0, 1]], # M5: B11
        [[1, 1], [0, 0]], # M6: B00 + B01
        [[0, 1], [0, 1]]  # M7: B10 + B11
    ]
    
    C_coefs = [
        [[1, 0], [0, 1]], # M1: C00 + C11
        [[0, 0], [1, 1]], # M2: C10 + C11
        [[0, 1], [0, 1]], # M3: C01 + C11
        [[1, 0], [0, 1]], # M4: C00 + C10
        [[-1, 1], [0, 1]],# M5: -C00 + C01 + C11
        [[1, 0], [0, 0]], # M6: C00
        [[0, 0], [0, 1]]  # M7: C11
    ]
    
    # We need to fill U, V, W for the full 4x4x8 multiplication.
    # We split B into two 4x4 halves: B_left (cols 0-3) and B_right (cols 4-7).
    # We will add 49 terms for the left half and 49 for the right half.
    
    def add_strassen_4x4(offset, B_col_offset):
        # offset: starting column index in U, V, W
        # B_col_offset: 0 for left half, 4 for right half
        
        # For each super-term t (0..6):
        for t in range(7):
            # For each inner Strassen term s (0..6):
            for s in range(7):
                col = offset + t * 7 + s
                
                # The U vector for this term is a combination of the 16 entries of A.
                # The V vector is a combination of the 16 entries of the relevant B half.
                # The W vector is a combination of the 16 entries of the relevant C half.
                
                # We need to compute the effective coefficients for each entry.
                # Let's trace the indices.
                # Global A index: i, j (0..3). Block I = i//2, J = j//2. Local i_l = i%2, j_l = j%2.
                # Global B index: j, k (0..3 for left, 4..7 for right). Block J = j//2, K = (k - B_col_offset)//2. Local j_l = j%2, k_l = (k-B_col_offset)%2.
                # Global C index: i, k (0..3 for left, 4..7 for right). Block I = i//2, K = (k - B_col_offset)//2. Local i_l = i%2, k_l = (k-B_col_offset)%2.
                
                # The term (t, s) contributes:
                # U[i, j] += sum_{I', J'} A_coefs[t][I', J'] * (S_U[i_l, j_l, s] if (I', J') == (I, J) else 0) ?
                # No, that's not quite right.
                
                # Let's look at the structure of the recursive Strassen.
                # The "scalar" multiplication in the outer Strassen is actually a 2x2 matrix multiplication.
                # The outer Strassen computes P_t = L_A^t L_B^t.
                # L_A^t is a 2x2 matrix of 2x2 blocks.
                # L_A^t[I, J] = sum_{I', J'} A_coefs[t][I', J'] A[I', J'].
                # This is a 2x2 matrix.
                # Similarly L_B^t[I, J] is a 2x2 matrix.
                # P_t = L_A^t L_B^t is a 2x2 matrix of 2x2 matrices.
                # P_t[I, J] = sum_K L_A^t[I, K] L_B^t[K, J].
                # Each entry P_t[I, J][i_l, k_l] is a scalar (linear combination of A and B entries).
                # We add P_t[I, J][i_l, k_l] to C[I, J][i_l, k_l] with coefficient C_coefs[t][I, J].
                
                # So the total contribution to C[I, J][i_l, k_l] is:
                # sum_t C_coefs[t][I, J] * sum_K L_A^t[I, K][i_l, k_l] L_B^t[K, J][k_l, j_l]?
                # Wait, the product of two 2x2 matrices X and Y is Z[i, j] = sum_k X[i, k] Y[k, j].
                # Here X = L_A^t[I, K] and Y = L_B^t[K, J].
                # X and Y are 2x2 matrices.
                # X[i_l, k_l] = sum_{I', J'} A_coefs[t][I', J'] A[I', J'][i_l, k_l] ? No.
                # L_A^t is a 2x2 matrix where entry (I, K) is a 2x2 matrix.
                # Entry (I, K) is sum_{I', J'} A_coefs[t][I', J'] A[I', J'].
                # So L_A^t[I, K] is a 2x2 matrix.
                # Its entry (i_l, k_l) is sum_{I', J'} A_coefs[t][I', J'] A[I', J'][i_l, k_l].
                # Similarly for L_B^t[K, J].
                
                # So the scalar multiplication in the inner Strassen is not just a single product.
                # The inner Strassen decomposition S_U, S_V, S_W is for the 2x2 multiplication.
                # We substitute the 2x2 matrices L_A^t[I, K] and L_B^t[K, J] into the inner Strassen.
                
                # This means that for a fixed t, we have 4 products of 2x2 matrices (for I, J in 0..1).
                # Wait, P_t[I, J] = sum_K L_A^t[I, K] L_B^t[K, J].
                # This is a sum of 2 products of 2x2 matrices.
                # So for each t, we have 4 (I, J) * 2 (K) = 8 products of 2x2 matrices?
                # No, the Strassen algorithm for 4x4 computes 7 products of 2x2 matrices.
                # P_t is a 2x2 matrix.
                # The entries of P_t are 2x2 matrices.
                # The rank of the 4x4x4 tensor is 49.
                # This implies that the 7 outer terms are not independent products of 2x2 matrices in the simple sense.
                
                # Let's go back to the explicit expansion.
                # The rank 49 decomposition is the Kronecker product of the rank 7 decomposition with itself?
                # No, the Kronecker product of two rank 7 decompositions has rank 49.
                # Does it decompose the 4x4x4 tensor?
                # T_4 = sum_{i, j, k} T_{ijk} e_i \otimes e_j \otimes e_k.
                # If we write i = 2i_1 + i_2, j = 2j_1 + j_2, k = 2k_1 + k_2.
                # T_{ijk} = delta_{i_1, j_1} delta_{j_1, k_1} ? No, that's for identity.
                # For matrix multiplication, T_{ijk} is 1 if j is the "inner" index.
                # Wait, the tensor for matrix multiplication C = AB is T_{i, j, k} = delta_{i, i'} delta_{j, j'} delta_{k, k'}?
                # No. C[i, k] = sum_j A[i, j] B[j, k].
                # The tensor T is defined by T_{i, j, k} = 1 if the term A[i, j] B[j, k] contributes to C[i, k].
                # In the decomposition, we sum over r: U[i, j, r] V[j, k, r] W[i, k, r] = C[i, k].
                # So T_{i, j, k} is the coefficient of A[i, j] B[j, k] in C[i, k].
                # This is 1 if the indices match, 0 otherwise.
                # Specifically, T is a 3-tensor of size 4x4x4? No, 4x4 (for A) x 4x4 (for B) -> 4x4 (for C).
                # The tensor has dimensions (4*4) x (4*4) x (4*4) = 16 x 16 x 16.
                # Entry (u, v, w) is 1 if u corresponds to (i, j), v to (j, k), w to (i, k).
                # So u = i*4 + j, v = j*4 + k, w = i*4 + k.
                
                # The recursive Strassen decomposition works by splitting the indices.
                # i = 2I + i_l, j = 2J + j_l, k = 2K + k_l.
                # u = (2I + i_l)*4 + (2J + j_l) = 8I + 4i_l + 2J + j_l.
                # v = (2J + j_l)*4 + (2K + k_l) = 8J + 4j_l + 2K + k_l.
                # w = (2I + i_l)*4 + (2K + k_l) = 8I + 4i_l + 2K + k_l.
                
                # The tensor T_4 can be written as a sum of 8 tensors T_2 \otimes T_2 \otimes T_2?
                # No.
                # T_4 = sum_{I, J, K} T_{I, J, K}^{block} \otimes e_I \otimes e_J \otimes e_K.
                # Where T_{I, J, K}^{block} is the tensor for the block multiplication.
                # The block multiplication is 2x2 x 2x2 -> 2x2.
                # So T_{I, J, K}^{block} is T_2 if the block indices are consistent, 0 otherwise.
                # Specifically, T_4[u, v, w] = 1 if u=(i, j), v=(j, k), w=(i, k).
                # u=(2I+i_l, 2J+j_l), v=(2J+j_l, 2K+k_l), w=(2I+i_l, 2K+k_l).
                # This requires J (from u) == J (from v) and I (from u) == I (from w) and K (from v) == K (from w).
                # So T_4 is non-zero only when the block indices match.
                # T_4 = sum_{I, J, K} T_2[i_l, j_l, k_l] e_{I, J} \otimes e_{J, K} \otimes e_{I, K}.
                # This is a sum of 8 terms (I, J, K in 0..1).
                # Each term is T_2 \otimes (e_I e_J \otimes e_J e_K \otimes e_I e_K).
                # The second factor is a rank 1 tensor in the block space.
                # So T_4 = sum_{I, J, K} T_2 \otimes T_{I, J, K}^{block_structure}.
                # The rank of T_4 is at most rank(T_2) * rank(sum_{I, J, K} T_{I, J, K}^{block_structure}).
                # The block structure tensor is the tensor for 2x2x2 matrix multiplication?
                # No, it's the tensor for the "skeleton" of the multiplication.
                # The skeleton is: if block (I, J) of A and (J, K) of B are non-zero, then (I, K) of C is non-zero.
                # This is exactly the 2x2x2 matrix multiplication tensor T_2!
                # So T_4 = T_2 \otimes T_2?
                # Let's check the dimensions.
                # T_2 has size 4x4x4 (flattened 2x2).
                # T_2 \otimes T_2 has size 16x16x16.
                # Rank(T_2 \otimes T_2) = Rank(T_2) * Rank(T_2) = 7 * 7 = 49.
                # So the rank 49 decomposition is simply the Kronecker product of the rank 7 decomposition with itself!
                
                # Let's verify this.
                # T_2 = sum_{s=1}^7 u_s \otimes v_s \otimes w_s.
                # T_2 \otimes T_2 = sum_{s=1}^7 sum_{t=1}^7 (u_s \otimes u_t) \otimes (v_s \otimes v_t) \otimes (w_s \otimes w_t).
                # This is a rank 49 decomposition.
                # Does it correspond to the 4x4x4 multiplication?
                # Yes, because the 4x4x4 multiplication tensor is T_2 \otimes T_2.
                
                # So we can construct the rank 49 decomposition by taking the Kronecker product of the S_U, S_V, S_W matrices.
                
                # S_U is 4x7. S_U \otimes S_U is 16x49.
                # S_V is 4x7. S_V \otimes S_V is 16x49.
                # S_W is 4x7. S_W \otimes S_W is 16x49.
                
                # We need to map these to the 4x4x8 problem.
                # For the left half of B (cols 0-3), we use S_V \otimes S_V for the first 16 columns of B?
                # No, B is 4x8.
                # We split B into B_left (4x4) and B_right (4x4).
                # For B_left, we use the first 16 entries of V.
                # For B_right, we use the next 16 entries of V.
                # The U matrix is the same for both (it's the 4x4 A matrix).
                # The W matrix is split into W_left (4x4) and W_right (4x4).
                
                # So for the left half:
                # U_left = S_U \otimes S_U (16x49)
                # V_left = S_V \otimes S_V (16x49)
                # W_left = S_W \otimes S_W (16x49)
                
                # For the right half:
                # U_right = S_U \otimes S_U (16x49)
                # V_right = S_V \otimes S_V (16x49)
                # W_right = S_W \otimes S_W (16x49)
                
                # We concatenate these.
                # U = [U_left | U_right] (16x98)
                # V = [V_left | V_right] (16x98) but V_right is shifted by 16 in the row index?
                # No, V is (m*p, rank) = (4*8, 98) = (32, 98).
                # V_left occupies rows 0-15.
                # V_right occupies rows 16-31.
                # W is (p*n, rank) = (8*4, 98) = (32, 98).
                # W_left occupies rows 0-15.
                # W_right occupies rows 16-31.
                
                # Let's construct the matrices.
                
                U_49 = np.kron(S_U, S_U)
                V_49 = np.kron(S_V, S_V)
                W_49 = np.kron(S_W, S_W)
                
                # Left half
                U[:, offset:offset+49] = U_49
                V[0:16, offset:offset+49] = V_49
                W[0:16, offset:offset+49] = W_49
                
                # Right half
                U[:, offset+49:offset+98] = U_49
                V[16:32, offset+49:offset+98] = V_49
                W[16:32, offset+49:offset+98] = W_49
                
                break # Only need to do this once, not inside the t, s loop.
        
    # Wait, the add_strassen_4x4 function was defined to add 49 terms.
    # But I realized the rank 49 decomposition is just the Kronecker product.
    # So I don't need the complex loop.
    
    U_49 = np.kron(S_U, S_U)
    V_49 = np.kron(S_V, S_V)
    W_49 = np.kron(S_W, S_W)
    
    # Left half (cols 0-3 of B, 0-3 of C)
    U[:, 0:49] = U_49
    V[0:16, 0:49] = V_49
    W[0:16, 0:49] = W_49
    
    # Right half (cols 4-7 of B, 4-7 of C)
    U[:, 49:98] = U_49
    V[16:32, 49:98] = V_49
    W[16:32, 49:98] = W_49
    
    # Convert to lists for the return interface
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
