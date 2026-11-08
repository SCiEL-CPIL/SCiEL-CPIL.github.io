import numpy as np
import time

DIMENSIONS = [4, 4, 9]

def _strassen_2x2():
    """
    Returns U, V, W for 2x2x2 Strassen decomposition.
    U shape (4, 7), V shape (4, 7), W shape (4, 7).
    Indices:
    U: i*2 + j
    V: j*2 + k
    W: k*2 + i
    """
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # M0: (a00+a11)(b00+b11)
    # U: a00, a11
    U[0, 0] = 1; U[3, 0] = 1
    # V: b00, b11
    V[0, 0] = 1; V[3, 0] = 1
    # W: c00, c11
    W[0, 0] = 1; W[3, 0] = 1
    
    # M1: (a10+a11)b00
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    # W: c10, c11?
    # c10 = M1 + M3? No.
    # c00 = M0 + M3 - M4 + M6
    # c01 = M2 + M4
    # c10 = M1 + M3
    # c11 = M0 - M1 + M2 + M5
    # Wait, standard Strassen:
    # M1 = (a11+a22)(b11+b22)
    # M2 = (a21+a22)b11
    # M3 = a11(b12-b22)
    # M4 = a22(b21-b11)
    # M5 = (a11+a12)b22
    # M6 = (a21-a11)(b11+b12)
    # M7 = (a12-a22)(b21+b22)
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # Let's stick to 0-indexed:
    # M0 (M1): (a00+a11)(b00+b11)
    # M1 (M2): (a10+a11)b00
    # M2 (M3): a00(b01-b11)
    # M3 (M4): a11(b10-b00)
    # M4 (M5): (a00+a01)b11
    # M5 (M6): (a10-a00)(b00+b01)
    # M6 (M7): (a01-a11)(b10+b11)
    
    # C00 = M0 + M3 - M4 + M6
    # C01 = M2 + M4
    # C10 = M1 + M3
    # C11 = M0 - M1 + M2 + M5
    
    # Reset W for M0
    W[0, 0] = 1 # C00
    W[3, 0] = 1 # C11
    
    # M1: (a10+a11)b00
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1 # C10
    W[3, 1] = -1 # C11
    
    # M2: a00(b01-b11)
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1 # C01
    W[3, 2] = 1 # C11
    
    # M3: a11(b10-b00)
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1 # C00
    W[2, 3] = 1 # C10
    
    # M4: (a00+a01)b11
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1 # C00
    W[1, 4] = 1 # C01
    
    # M5: (a10-a00)(b00+b01)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1 # C11
    
    # M6: (a01-a11)(b10+b11)
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1 # C00
    
    return U, V, W

def _build_4x4_4():
    """
    Builds 4x4x4 tensor using Strassen on 2x2 blocks.
    Rank 49.
    """
    U2, V2, W2 = _strassen_2x2()
    # U2 is 4x7, V2 is 4x7, W2 is 4x7
    # We need to expand this to 4x4x4.
    # The "scalar" in Strassen is a 2x2 matrix.
    # So each entry in U2, V2, W2 is actually a 2x2 matrix of coefficients?
    # No, U2[i,j] is coefficient of A_ij.
    # In the block version, A is 4x4.
    # A_block[0,0] is A[0:2, 0:2].
    # The Strassen formula operates on blocks.
    # M_S = (Sum A_block * coeff) * (Sum B_block * coeff)
    # This product is a 2x2 matrix multiplication.
    # So we need to embed the 2x2x2 tensor into the 4x4x4 tensor.
    
    # U4 shape (16, 49), V4 shape (16, 49), W4 shape (16, 49)
    U4 = np.zeros((16, 49))
    V4 = np.zeros((16, 49))
    W4 = np.zeros((16, 49))
    
    for s in range(7):
        # For each Strassen term s, we have a 2x2x2 multiplication.
        # The rank-1 terms of the 2x2x2 are indexed by r in 0..6.
        for r in range(7):
            col = s * 7 + r
            # U2[alpha, s] is coeff for A_block_alpha in L_A_s.
            # But A_block_alpha is a 2x2 matrix.
            # So we need to expand U2[alpha, s] into 4 entries in U4.
            # alpha in U2 is 0..3 (i*2+j for 2x2).
            # In U4, alpha4 is i4*4+j4.
            # The block A_block[alpha] corresponds to A4[2*i_a + i4, 2*j_a + j4]?
            # No, alpha in U2 is the index of the block.
            # Block 0 is A[0:2, 0:2]. Block 1 is A[0:2, 2:4].
            # Block 2 is A[2:4, 0:2]. Block 3 is A[2:4, 2:4].
            # So for U2[alpha, s] = val, we add val to U4[indices of block alpha, col].
            
            # Wait, U2 is for the "outer" Strassen.
            # The "inner" multiplication is the 2x2x2 one.
            # So the coefficient for A4[alpha4, col] is U2[alpha_block, s] * U2_inner[alpha_inner, r]?
            # No.
            # L_A_s is a sum of blocks.
            # L_A_s = sum_alpha c_alpha A_block_alpha.
            # Then M_s = L_A_s L_B_s.
            # M_s is a 2x2 matrix.
            # M_s = sum_{r} (U_inner[alpha_inner, r] L_A_s[alpha_inner]?) No.
            
            # Let's trace:
            # C = sum_s w_s M_s.
            # M_s = (sum_alpha u_s_alpha A_alpha) (sum_beta v_s_beta B_beta).
            # A_alpha is 2x2. B_beta is 2x2.
            # Product is 2x2.
            # So M_s[ki] = sum_{jk} A_alpha[jk] B_beta[jk]? No, standard matrix mult.
            # M_s = A_s B_s.
            # A_s = sum_alpha u_s_alpha A_alpha.
            # B_s = sum_beta v_s_beta B_beta.
            # So A_s[jk] = sum_alpha u_s_alpha A_alpha[jk].
            # This means the coefficient of A_alpha[jk] in A_s[jk] is u_s_alpha.
            # So U4[alpha4, col] should be u_s_alpha * U_inner[jk, r]?
            # No, the inner multiplication is rank 7.
            # M_s[ki] = sum_r U_inner[jk, r] V_inner[jk, r] W_inner[ki, r]?
            # No, M_s is the result of multiplying two 2x2 matrices.
            # The tensor for 2x2x2 is T.
            # M_s[ki] = sum_{jk} A_s[jk] B_s[jk].
            # This is exactly the 2x2x2 multiplication.
            # So we can use the U2, V2, W2 we already have!
            
            # So for a fixed s and r:
            # The contribution to U4 is:
            # For each alpha (0..3) in U2:
            #   If U2[alpha, s] != 0:
            #     For each (i, j) in the 2x2 block alpha:
            #       alpha4 = i*4 + j? No.
            #       Block alpha corresponds to rows 2*(alpha//2) + i, cols 2*(alpha%2) + j?
            #       Block 0: rows 0,1 cols 0,1.
            #       Block 1: rows 0,1 cols 2,3.
            #       Block 2: rows 2,3 cols 0,1.
            #       Block 3: rows 2,3 cols 2,3.
            
            # Wait, U2 index is i*2+j.
            # Block index alpha = i*2+j.
            # So row block i, col block j.
            # Actual row = 2*i + i_inner.
            # Actual col = 2*j + j_inner.
            
            # So for U2[alpha, s] = val:
            #   i_block = alpha // 2
            #   j_block = alpha % 2
            #   For i_inner in 0..1:
            #     For j_inner in 0..1:
            #       alpha4 = (2*i_block + i_inner)*4 + (2*j_block + j_inner)
            #       U4[alpha4, col] += val * U2[i_inner*2 + j_inner, r]
            
            # Similarly for V.
            # V2 index beta = j*2+k.
            # Block beta corresponds to row block j, col block k.
            # Actual row = 2*j + j_inner.
            # Actual col = 2*k + k_inner.
            # beta4 = (2*j + j_inner)*4 + (2*k + k_inner)?
            # No, V4 index is j4*4 + k4.
            # j4 is the "row" of B (which is col of A).
            # k4 is the "col" of B.
            
            # And W.
            # W2 index gamma = k*2+i.
            # Block gamma corresponds to row block k, col block i.
            # Actual row = 2*k + k_inner.
            # Actual col = 2*i + i_inner.
            # gamma4 = (2*k + k_inner)*4 + (2*i + i_inner)?
            # No, W4 index is k4*4 + i4.
            
            # Let's implement this mapping.
            
            for alpha in range(4):
                val_u = U2[alpha, s]
                if val_u != 0:
                    i_b = alpha // 2
                    j_b = alpha % 2
                    for i_in in range(2):
                        for j_in in range(2):
                            a4 = (2*i_b + i_in)*4 + (2*j_b + j_in)
                            u_in = U2[i_in*2 + j_in, r]
                            U4[a4, col] += val_u * u_in
            
            for beta in range(4):
                val_v = V2[beta, s]
                if val_v != 0:
                    j_b = beta // 2
                    k_b = beta % 2
                    for j_in in range(2):
                        for k_in in range(2):
                            b4 = (2*j_b + j_in)*4 + (2*k_b + k_in)
                            v_in = V2[j_in*2 + k_in, r]
                            V4[b4, col] += val_v * v_in
            
            for gamma in range(4):
                val_w = W2[gamma, s]
                if val_w != 0:
                    k_b = gamma // 2
                    i_b = gamma % 2
                    for k_in in range(2):
                        for i_in in range(2):
                            g4 = (2*k_b + k_in)*4 + (2*i_b + i_in)
                            w_in = W2[k_in*2 + i_in, r]
                            W4[g4, col] += val_w * w_in
                            
    return U4, V4, W4

def _build_4x4_1():
    """
    Builds 4x4x1 tensor. Rank 16.
    """
    U = np.zeros((16, 16))
    V = np.zeros((16, 16))
    W = np.zeros((16, 16))
    for i in range(4):
        for j in range(4):
            col = i*4 + j
            U[i*4 + j, col] = 1
            V[j, col] = 1
            W[i, col] = 1
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    # p=9. Split into 8 + 1.
    # M_4,4,8 = M_4,4,4 + M_4,4,4.
    # M_4,4,1 = M_4,4,1.
    
    U1, V1, W1 = _build_4x4_4() # 16x49
    U2, V2, W2 = _build_4x4_4() # 16x49
    U3, V3, W3 = _build_4x4_1() # 16x16
    
    # Concatenate U
    U_total = np.hstack([U1, U2, U3]) # 16 x 114
    
    # V1 is for B[0:4, 0:4] (k=0..3)
    # V2 is for B[0:4, 4:8] (k=4..7)
    # V3 is for B[0:4, 8:9] (k=8)
    # V_total shape (36, 114)
    V_total = np.zeros((36, 114))
    V_total[0:16, 0:49] = V1
    V_total[16:32, 49:98] = V2
    V_total[32:36, 98:114] = V3 # V3 is 16x16, but only first 4 rows used?
    # V3 index is j*1 + 0 = j. So rows 0..3.
    # V_total rows 32..35 correspond to j=0..3, k=8.
    # So V_total[32:36, 98:114] = V3[0:4, :].
    
    # W1 is for C[0:4, 0:4] (k=0..3, i=0..3)
    # W2 is for C[4:8, 0:4] (k=4..7, i=0..3)
    # W3 is for C[8:9, 0:4] (k=8, i=0..3)
    # W_total shape (36, 114)
    W_total = np.zeros((36, 114))
    W_total[0:16, 0:49] = W1
    W_total[16:32, 49:98] = W2
    W_total[32:36, 98:114] = W3[0:4, :]
    
    return {"U": U_total.tolist(), "V": V_total.tolist(), "W": W_total.tolist()}

if __name__ == "__main__":
    # Quick verification
    res = construct()
    U = np.array(res["U"])
    V = np.array(res["V"])
    W = np.array(res["W"])
    
    # Test with random matrices
    A = np.random.rand(4, 4)
    B = np.random.rand(4, 9)
    
    a_flat = A.flatten() # i*4+j
    b_flat = B.flatten() # j*9+k
    
    # U[alpha, r] V[beta, r] W[gamma, r]
    # Sum over r
    # C_flat[gamma] = sum_r sum_alpha sum_beta U[alpha,r] V[beta,r] W[gamma,r] a[alpha] b[beta]
    # This is (a^T U) (V^T b) W^T?
    # Let u_r = sum_alpha U[alpha,r] a[alpha]
    # Let v_r = sum_beta V[beta,r] b[beta]
    # c_gamma = sum_r W[gamma,r] u_r v_r
    
    u_vec = U.T @ a_flat
    v_vec = V.T @ b_flat
    c_flat = (W * (u_vec * v_vec).T).sum(axis=1)
    
    C_expected = A @ B
    c_expected = C_expected.flatten() # k*4+i
    
    if np.allclose(c_flat, c_expected):
        print("Verification Passed")
    else:
        print("Verification Failed")
        print(np.max(np.abs(c_flat - c_expected)))
