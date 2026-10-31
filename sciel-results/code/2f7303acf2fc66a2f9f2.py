import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 10.0

    n, m, p = 4, 4, 4

    # Coefficients for 2x2 Strassen algorithm
    # Shape: (7, 2, 2)
    # Indices: [mult_idx, row, col]

    # A coefficients: M_s = sum a_s[i][j] A[i][j]
    A = np.array([
        [[1, 0], [0, 1]],  # M0: A00 + A11
        [[0, 0], [1, 1]],  # M1: A10 + A11
        [[1, 0], [0, 0]],  # M2: A00
        [[0, 0], [0, 1]],  # M3: A11
        [[1, 1], [0, 0]],  # M4: A00 + A01
        [[-1, 0], [1, 0]], # M5: A10 - A00
        [[0, 1], [0, -1]]  # M6: A01 - A11
    ])

    # B coefficients: M_s = sum b_s[j][k] B[j][k]
    B = np.array([
        [[1, 0], [0, 1]],  # M0: B00 + B11
        [[1, 0], [0, 0]],  # M1: B00
        [[0, 1], [0, -1]], # M2: B01 - B11
        [[-1, 0], [1, 0]], # M3: B10 - B00
        [[0, 0], [0, 1]],  # M4: B11
        [[1, 1], [0, 0]],  # M5: B00 + B01
        [[0, 0], [1, 1]]   # M6: B10 + B11
    ])

    # C coefficients: C[i][k] = sum c_s[i][k] M_s
    C = np.array([
        [[1, 0], [0, 1]],  # M0: C00+=M0, C11+=M0
        [[0, 0], [1, -1]], # M1: C10+=M1, C11-=M1
        [[0, 1], [0, 1]],  # M2: C01+=M2, C11+=M2
        [[1, 0], [1, 0]],  # M3: C00+=M3, C10+=M3
        [[-1, 1], [0, 0]], # M4: C00-=M4, C01+=M4
        [[0, 0], [0, 1]],  # M5: C11+=M5
        [[1, 0], [0, 0]]   # M6: C00+=M6
    ])

    # Construct 4x4 coefficients using Kronecker product logic
    # The 4x4 matrix is viewed as 2x2 blocks of 2x2 matrices.
    # Let A_4 be the 4x4 matrix. A_4[i,j] where i=2*S+s, j=2*T+t.
    # The rank-49 decomposition is formed by 7 outer multiplications (S,T blocks) 
    # and 7 inner multiplications (s,t elements).
    # Total rank = 7 * 7 = 49.
    # Index for the 49 multiplications: idx = s_outer * 7 + t_inner

    # U shape: (16, 49) -> (i*4+j, s_outer*7+t_inner)
    # V shape: (16, 49) -> (j*4+k, s_outer*7+t_inner)
    # W shape: (16, 49) -> (k*4+i, s_outer*7+t_inner)

    U = np.zeros((16, 49), dtype=np.int64)
    V = np.zeros((16, 49), dtype=np.int64)
    W = np.zeros((16, 49), dtype=np.int64)

    for s_out in range(7):
        for t_in in range(7):
            col_idx = s_out * 7 + t_in

            # U contributions:
            # U[i*4+j, col_idx] = A[s_out, S, T] * A[t_in, s, t]
            # where i = 2S+s, j = 2T+t
            for S in range(2):
                for T in range(2):
                    a_outer = A[s_out, S, T]
                    if a_outer == 0: continue
                    for s in range(2):
                        for t in range(2):
                            a_inner = A[t_in, s, t]
                            if a_inner == 0: continue
                            i = 2 * S + s
                            j = 2 * T + t
                            U[i * 4 + j, col_idx] = a_outer * a_inner

            # V contributions:
            # V[j*4+k, col_idx] = B[s_out, T, K] * B[t_in, t, k2]
            # where j = 2T+t, k = 2K+k2
            # Note: The 't' from A's inner index matches the 't' in B's inner index?
            # Yes, because the inner multiplication is M_t = (sum A_inner) * (sum B_inner).
            # The index t_in determines which inner linear form is used for both A and B.
            for T in range(2):
                for K in range(2):
                    b_outer = B[s_out, T, K]
                    if b_outer == 0: continue
                    for t in range(2):
                        for k2 in range(2):
                            b_inner = B[t_in, t, k2]
                            if b_inner == 0: continue
                            j = 2 * T + t
                            k = 2 * K + k2
                            V[j * 4 + k, col_idx] = b_outer * b_inner

            # W contributions:
            # W[k*4+i, col_idx] = C[s_out, S, K] * C[t_in, s, k2]
            # where i = 2S+s, k = 2K+k2
            # C[i,k] gets contribution from M_s_out * M_t_in?
            # The output C is sum_s_out sum_t_in (C_outer * C_inner) * M_s_out * M_t_in
            # So the coefficient in W is C[s_out, S, K] * C[t_in, s, k2]
            for S in range(2):
                for K in range(2):
                    c_outer = C[s_out, S, K]
                    if c_outer == 0: continue
                    for s in range(2):
                        for k2 in range(2):
                            c_inner = C[t_in, s, k2]
                            if c_inner == 0: continue
                            i = 2 * S + s
                            k = 2 * K + k2
                            W[k * 4 + i, col_idx] = c_outer * c_inner

    if time.time() > deadline:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
