import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit

    n, m, p = 4, 4, 4

    # Build the target tensor T[a_idx, b_idx, c_idx]
    # a_idx = i*m + k  (A[i,k])
    # b_idx = k*p + j  (B[k,j])
    # c_idx = j*n + i  (C[i,j] stored as W[j*n+i])
    target = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for k in range(m):
            for j in range(p):
                a_idx = i * m + k
                b_idx = k * p + j
                c_idx = j * n + i
                target[a_idx, b_idx, c_idx] = 1.0

    def verify(U, V, W, rank):
        """Verify exact decomposition: sum_r U[:,r] * V[:,r] * W[:,r] == target."""
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        recon = np.einsum('ir,jr,kr->ijk', U_arr, V_arr, W_arr, optimize=True)
        return np.allclose(recon, target, atol=1e-12)

    def time_ok():
        return time.time() < deadline - 5.0

    # ============================================================
    # Strassen 2x2x2 decomposition (rank 7)
    # For 2x2: U row = i*2+k, V row = k*2+j, W row = j*2+i
    # ============================================================
    def strassen_2x2():
        U2 = np.zeros((4, 7), dtype=float)
        V2 = np.zeros((4, 7), dtype=float)
        W2 = np.zeros((4, 7), dtype=float)

        # M1 = (A00+A11)*(B00+B11) -> contributes to C00, C11
        # A00=row0, A11=row3; B00=row0, B11=row3; C00=col0, C11=col3
        U2[0, 0] = 1; U2[3, 0] = 1
        V2[0, 0] = 1; V2[3, 0] = 1
        W2[0, 0] = 1; W2[3, 0] = 1

        # M2 = (A10+A11)*(B00) -> contributes to C10, C11
        # A10=row2, A11=row3; B00=row0; C10=col1, C11=col3
        U2[2, 1] = 1; U2[3, 1] = 1
        V2[0, 1] = 1
        W2[1, 1] = 1; W2[3, 1] = -1

        # M3 = A00*(B01-B11) -> contributes to C01, C11
        # A00=row0; B01=row1, B11=row3; C01=col2, C11=col3
        U2[0, 2] = 1
        V2[1, 2] = 1; V2[3, 2] = -1
        W2[2, 2] = 1; W2[3, 2] = 1

        # M4 = A11*(B10-B00) -> contributes to C00, C10
        # A11=row3; B10=row2, B00=row0; C00=col0, C10=col1
        U2[3, 3] = 1
        V2[2, 3] = 1; V2[0, 3] = -1
        W2[0, 3] = 1; W2[1, 3] = 1

        # M5 = (A00+A01)*B11 -> contributes to C00, C01
        # A00=row0, A01=row1; B11=row3; C00=col0, C01=col2
        U2[0, 4] = 1; U2[1, 4] = 1
        V2[3, 4] = 1
        W2[0, 4] = -1; W2[2, 4] = 1

        # M6 = (A10-A00)*(B00+B01) -> contributes to C11
        # A10=row2, A00=row0; B00=row0, B01=row1; C11=col3
        U2[2, 5] = 1; U2[0, 5] = -1
        V2[0, 5] = 1; V2[1, 5] = 1
        W2[3, 5] = 1

        # M7 = (A01-A11)*(B10+B11) -> contributes to C00
        # A01=row1, A11=row3; B10=row2, B11=row3; C00=col0
        U2[1, 6] = 1; U2[3, 6] = -1
        V2[2, 6] = 1; V2[3, 6] = 1
        W2[0, 6] = 1

        return U2, V2, W2

    # ============================================================
    # Kronecker product: 4x4x4 from 2x2x2 Strassen (rank 49)
    # ============================================================
    def strassen_kronecker_4x4():
        U2, V2, W2 = strassen_2x2()

        rank = 49
        U49 = np.zeros((16, rank), dtype=float)
        V49 = np.zeros((16, rank), dtype=float)
        W49 = np.zeros((16, rank), dtype=float)

        # A[i,j]: i = I*2+ii, j = J*2+jj
        # U row = i*4 + j = (I*2+ii)*4 + (J*2+jj)
        # Block Strassen: U2[I*2+J, s], Inner Strassen: U2[ii*2+jj, t]
        # Combined rank index: r = s*7 + t
        for I in range(2):
            for J in range(2):
                for ii in range(2):
                    for jj in range(2):
                        a_row = (I * 2 + ii) * 4 + (J * 2 + jj)
                        for s in range(7):
                            for t in range(7):
                                coeff = U2[I * 2 + J, s] * U2[ii * 2 + jj, t]
                                if coeff != 0:
                                    U49[a_row, s * 7 + t] = coeff

        # B[k,l]: k = K*2+kk, l = L*2+ll
        # V row = k*4 + l = (K*2+kk)*4 + (L*2+ll)
        for K in range(2):
            for L in range(2):
                for kk in range(2):
                    for ll in range(2):
                        b_row = (K * 2 + kk) * 4 + (L * 2 + ll)
                        for s in range(7):
                            for t in range(7):
                                coeff = V2[K * 2 + L, s] * V2[kk * 2 + ll, t]
                                if coeff != 0:
                                    V49[b_row, s * 7 + t] = coeff

        # C[i,j]: i = I*2+ii, j = J*2+jj
        # W row = j*4 + i = (J*2+jj)*4 + (I*2+ii)
        for I in range(2):
            for J in range(2):
                for ii in range(2):
                    for jj in range(2):
                        w_row = (J * 2 + jj) * 4 + (I * 2 + ii)
                        for s in range(7):
                            for t in range(7):
                                coeff = W2[I * 2 + J, s] * W2[ii * 2 + jj, t]
                                if coeff != 0:
                                    W49[w_row, s * 7 + t] = coeff

        return U49, V49, W49, rank

    # ============================================================
    # Naive decomposition (rank 64) as fallback
    # ============================================================
    def naive():
        rank = 64
        U = np.zeros((16, rank), dtype=float)
        V = np.zeros((16, rank), dtype=float)
        W = np.zeros((16, rank), dtype=float)
        for i in range(4):
            for k in range(4):
                for j in range(4):
                    r = i * 16 + k * 4 + j
                    U[i * 4 + k, r] = 1.0
                    V[k * 4 + j, r] = 1.0
                    W[j * 4 + i, r] = 1.0
        return U, V, W, rank

    # ============================================================
    # Execute: verify Strassen Kronecker, fall back to naive
    # ============================================================
    U, V, W, rank = strassen_kronecker_4x4()

    if verify(U.tolist(), V.tolist(), W.tolist(), rank):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Fallback to naive
    if time_ok():
        U_n, V_n, W_n, r_n = naive()
        if verify(U_n.tolist(), V_n.tolist(), W_n.tolist(), r_n):
            return {"U": U_n.tolist(), "V": V_n.tolist(), "W": W_n.tolist()}

    # Ultimate fallback: return naive without verification
    U_f, V_f, W_f, _ = naive()
    return {"U": U_f.tolist(), "V": V_f.tolist(), "W": W_f.tolist()}
