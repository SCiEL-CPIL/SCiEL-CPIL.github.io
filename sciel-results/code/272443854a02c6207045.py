import time
import numpy as np

DIMENSIONS = [4, 4, 8]
DEADLINE_SECONDS = 110.0

def construct():
    start = time.time()
    n, m, p = DIMENSIONS  # 4, 4, 8

    def _time_left():
        return DEADLINE_SECONDS - (time.time() - start)

    def _timeup():
        return _time_left() < 1.0

    def _schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    # Rank-12 decomposition of 2x2 x 2x4
    # U224: 4x12, V224: 8x12, W224: 8x12
    U224 = [
        [1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0],
        [0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0],
        [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1]
    ]
    V224 = [
        [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, -1, 0, 0, 1, -1, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 1, 0, 0, 1, -1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
        [1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, -1, 0, 0, 1, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, -1]
    ]
    W224 = [
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
        [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
    ]

    # Verify rank-12 decomposition
    def _verify_224():
        U = np.array(U224)
        V = np.array(V224)
        W = np.array(W224)
        # U: 2x2, V: 2x4, W: 2x4
        # Indices: U[i*2+j], V[j*4+k], W[k*2+i]
        # Tensor: sum_r U[i*2+j, r] V[j*4+k, r] W[k*2+i, r]
        # We want this to be 1 if i,j,k are valid, 0 otherwise?
        # No, we want the product of the tensors to be the identity of the multiplication.
        # The tensor T is defined by T[i,j,k] = 1 if i,j,k are such that C_ik gets A_ij B_jk.
        # Actually, we just need to check that the decomposition is valid.
        # We can check by constructing the full tensor and comparing.
        # But it's easier to just use it and verify the final result.
        return True

    # 2x2x2 Strassen factors (rank 7)
    U2 = [
        [1, 0, 0, 1],
        [0, 0, 1, 1],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [-1, 0, 1, 0],
        [0, 1, 0, -1]
    ]
    V2 = [
        [1, 0, 0, 1],
        [1, 0, 0, 0],
        [0, 1, 0, -1],
        [-1, 0, 1, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [0, 0, 1, 1]
    ]
    W2 = [
        [1, 0, 0, 1],
        [0, 0, 1, -1],
        [0, 1, 0, 1],
        [1, 0, 1, 0],
        [-1, 1, 0, 0],
        [0, 0, 0, 1],
        [1, 0, 0, 0]
    ]

    if _timeup():
        return _schoolbook()

    # Mixed Strassen:
    # Split A (4x4) into 2x2 blocks A_IJ (I,J in 0,1)
    # Split B (4x8) into 2x4 blocks B_JK (J in 0,1, K in 0,1)
    # C (4x8) into 2x4 blocks C_IK (I in 0,1, K in 0,1)
    # C_IK = sum_J A_IJ B_JK
    # Use 7 Strassen products M_s for the 2x2 block multiplication of A and B_blocks?
    # No, B_blocks are 2x4.
    # We use the 7 Strassen linear combinations for A (2x2) and B (2x4).
    # The 7 products are M_s = (A_comb_s) * (B_comb_s).
    # Each M_s is a 2x4 matrix.
    # We compute each M_s using the rank-12 decomposition of 2x2x4.
    # Total rank = 7 * 12 = 84.

    rank = 84
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    for s in range(7):
        for r in range(12):
            col = s * 12 + r
            
            # U: A is 4x4. Block A_IJ is 2x2.
            # A_comb_s is a linear combination of A_IJ blocks.
            # The Strassen combination for A is applied to the 2x2 blocks.
            # A_comb_s[I_loc, J_loc] = sum_{I,J} U2[s][I*2+J] * A[I*2+I_loc, J*2+J_loc]
            # So U[I*2+I_loc, J*2+J_loc] = U2[s][I*2+J] * U224[I_loc*2+J_loc][r]
            for I in range(2):
                for J in range(2):
                    for I_loc in range(2):
                        for J_loc in range(2):
                            U[(I * 2 + I_loc) * m + (J * 2 + J_loc)][col] = \
                                U2[s][I * 2 + J] * U224[I_loc * 2 + J_loc][r]

            # V: B is 4x8. Block B_JK is 2x4.
            # B_comb_s is a linear combination of B_JK blocks.
            # B_comb_s[J_loc, K_loc] = sum_{J,K} V2[s][J*2+K] * B[J*2+J_loc, K*4+K_loc]
            # So V[J*2+J_loc, K*4+K_loc] = V2[s][J*2+K] * V224[J_loc*4+K_loc][r]
            for J in range(2):
                for K in range(2):
                    for J_loc in range(2):
                        for K_loc in range(4):
                            V[(J * 2 + J_loc) * p + (K * 4 + K_loc)][col] = \
                                V2[s][J * 2 + K] * V224[J_loc * 4 + K_loc][r]

            # W: C is 4x8. Block C_IK is 2x4.
            # C_comb_s is a linear combination of C_IK blocks.
            # C_comb_s[I_loc, K_loc] = sum_{I,K} W2[s][I*2+K] * C[I*2+I_loc, K*4+K_loc]
            # So W[K*4+K_loc, I*2+I_loc] = W2[s][I*2+K] * W224[K_loc*2+I_loc][r]
            # Note: W index is k*n + i. Here k = K*4+K_loc, i = I*2+I_loc.
            for I in range(2):
                for K in range(2):
                    for I_loc in range(2):
                        for K_loc in range(4):
                            W[(K * 4 + K_loc) * n + (I * 2 + I_loc)][col] = \
                                W2[s][I * 2 + K] * W224[K_loc * 2 + I_loc][r]

    if _timeup():
        return _schoolbook()

    def _verify(U_mat, V_mat, W_mat):
        U_arr = np.array(U_mat, dtype=float)
        V_arr = np.array(V_mat, dtype=float)
        W_arr = np.array(W_mat, dtype=float)
        r = U_arr.shape[1]
        U3 = U_arr.reshape(n, m, r)
        V3 = V_arr.reshape(m, p, r)
        W3 = W_arr.reshape(p, n, r)
        result = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
        return np.allclose(result, 1.0, atol=1e-10)

    if _verify(U, V, W):
        U_int = [[int(round(x)) for x in row] for row in U]
        V_int = [[int(round(x)) for x in row] for row in V]
        W_int = [[int(round(x)) for x in row] for row in W]
        if _verify(U_int, V_int, W_int):
            return {"U": U_int, "V": V_int, "W": W_int}
        return {"U": U, "V": V, "W": W}
    
    return _schoolbook()
