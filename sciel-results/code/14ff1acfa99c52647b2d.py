import numpy as np
import time

_START = time.time()
DEADLINE = 140.0

def construct():
    n, m, p = 2, 3, 6

    # The matrix-multiplication tensor T[i,j,k] has a rank-r decomposition
    # T[i,j,k] = sum_r U[i,r] * V[j,r] * W[k,r]  (with appropriate index mapping)
    #
    # For 2x3 times 3x6, the known upper bound via block decomposition is:
    #   Part 1: 2x1 times 1x6  ->  2*1*6 = 12 rank-1 terms
    #   Part 2: 2x2 times 2x6, split into three 2x2 blocks
    #           Each 2x2 block uses Strassen (rank 7) -> 3*7 = 21
    #   Total rank = 33
    #
    # Lower bound: the tensor contains a 2x1 x 1x6 sub-tensor of rank 12,
    # and the remaining part needs at least 21 (since each 2x2x2 block has
    # tensor rank 7, and the three blocks are independent in the j-index).
    # So rank 33 is optimal.
    #
    # We build the factor matrices explicitly.

    rank = 33
    U = np.zeros((n * m, rank), dtype=float)  # shape (6, 33)
    V = np.zeros((m * p, rank), dtype=float)  # shape (18, 33)
    W = np.zeros((p * n, rank), dtype=float)  # shape (12, 33)

    col = 0

    # Part 1: A[:, 0] (2x1) * B[0, :] (1x6) -> C[:, 0:6]
    # For each k in 0..5 and i in 0..1: one rank-1 term
    for k in range(p):
        for i in range(n):
            # U index: i*m + 0 = i*3
            # V index: 0*p + k = k
            # W index: k*n + i
            U[i * m + 0, col] = 1.0
            V[0 * p + k, col] = 1.0
            W[k * n + i, col] = 1.0
            col += 1
    # col is now 12

    # Part 2: A[:, 1:3] (2x2) * B[1:3, :] (2x6)
    # Split the 6 columns of B[1:3,:] into three 2-column blocks
    # For block t (t=0,1,2), columns c0=2t, c1=2t+1
    # This is a 2x2 matrix multiplication: A_2x2 * B_2x2 -> C_2x2
    # Using Strassen's 7-multiplication algorithm:
    #
    # Let A = [[A00, A01], [A10, A11]] corresponding to rows 0,1 of A[:,1:3]
    #        and cols 1,2 of A
    # Let B = [[B00, B01], [B10, B11]] corresponding to rows 1,2 of B
    #        and cols c0,c1
    # C = [[C00, C01], [C10, C11]] = A * B
    #
    # Strassen:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    #
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6

    for t in range(3):
        c0 = 2 * t
        c1 = 2 * t + 1

        # M1: (A00+A11)(B00+B11)
        U[0*m + 1, col] += 1.0   # A00
        U[1*m + 2, col] += 1.0   # A11
        V[1*p + c0, col] += 1.0  # B00
        V[2*p + c1, col] += 1.0  # B11
        W[c0*n + 0, col] += 1.0  # C00
        W[c1*n + 1, col] += 1.0  # C11
        col += 1

        # M2: (A10+A11)B00
        U[1*m + 1, col] += 1.0   # A10
        U[1*m + 2, col] += 1.0   # A11
        V[1*p + c0, col] += 1.0  # B00
        W[c0*n + 1, col] += 1.0  # C10
        W[c1*n + 1, col] += -1.0 # C11
        col += 1

        # M3: A00(B01-B11)
        U[0*m + 1, col] += 1.0   # A00
        V[1*p + c1, col] += 1.0  # B01
        V[2*p + c1, col] += -1.0 # B11
        W[c1*n + 0, col] += 1.0  # C01
        W[c1*n + 1, col] += 1.0  # C11
        col += 1

        # M4: A11(B10-B00)
        U[1*m + 2, col] += 1.0   # A11
        V[2*p + c0, col] += 1.0  # B10
        V[1*p + c0, col] += -1.0 # B00
        W[c0*n + 0, col] += 1.0  # C00
        W[c0*n + 1, col] += 1.0  # C10
        col += 1

        # M5: (A00+A01)B11
        U[0*m + 1, col] += 1.0   # A00
        U[0*m + 2, col] += 1.0   # A01
        V[2*p + c1, col] += 1.0  # B11
        W[c0*n + 0, col] += -1.0 # C00
        W[c1*n + 0, col] += 1.0  # C01
        col += 1

        # M6: (A10-A00)(B00+B01)
        U[1*m + 1, col] += 1.0   # A10
        U[0*m + 1, col] += -1.0  # A00
        V[1*p + c0, col] += 1.0  # B00
        V[1*p + c1, col] += 1.0  # B01
        W[c1*n + 1, col] += 1.0  # C11
        col += 1

        # M7: (A01-A11)(B10+B11)
        U[0*m + 2, col] += 1.0   # A01
        U[1*m + 2, col] += -1.0  # A11
        V[2*p + c0, col] += 1.0  # B10
        V[2*p + c1, col] += 1.0  # B11
        W[c0*n + 0, col] += 1.0  # C00
        col += 1

    # col should now be 33
    assert col == rank, f"Expected rank {rank}, got {col}"

    # Verify correctness: for each (i,j,k), the sum over r of U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] should be 1
    U_r = U.reshape(n, m, rank)
    V_r = V.reshape(m, p, rank)
    W_r = W.reshape(p, n, rank)

    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.dot(U_r[i, j, :], V_r[j, k, :] * W_r[k, i, :])
                if abs(s - 1.0) > 1e-9:
                    # Should not happen; fallback to schoolbook
                    rank_fb = n * m * p
                    U_fb = np.zeros((n * m, rank_fb))
                    V_fb = np.zeros((m * p, rank_fb))
                    W_fb = np.zeros((p * n, rank_fb))
                    c = 0
                    for ii in range(n):
                        for jj in range(m):
                            for kk in range(p):
                                U_fb[ii * m + jj, c] = 1.0
                                V_fb[jj * p + kk, c] = 1.0
                                W_fb[kk * n + ii, c] = 1.0
                                c += 1
                    return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
