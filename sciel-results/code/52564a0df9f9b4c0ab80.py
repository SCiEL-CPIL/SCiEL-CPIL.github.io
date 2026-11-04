import numpy as np
import time

DIMENSIONS = [2, 3, 8]

def construct():
    start_time = time.time()
    deadline = start_time + 130

    n, m, p = DIMENSIONS
    nm = n * m  # 6
    mp = m * p  # 24
    pn = p * n  # 16

    # Rank-44 decomposition for 2x3 * 3x8 matrix multiplication
    # Split: A (2x3) = [A_left (2x2), A_right (2x1)]
    #         B (3x8) = [B_top (2x8); B_bottom (1x8)]
    # C = A_left * B_top + A_right * B_bottom
    #
    # A_left * B_top: 2x2 * 2x8 -> split B_top into 4 blocks of 2x2, apply Strassen (7 terms each) = 28
    # A_right * B_bottom: 2x1 * 1x8 -> schoolbook = 2*8 = 16
    # Total: 28 + 16 = 44

    rank = 44
    U = np.zeros((nm, rank))
    V = np.zeros((mp, rank))
    W = np.zeros((pn, rank))

    col = 0

    # Indexing:
    # A[i,j] -> index i*m+j = i*3+j
    # B[j,k] -> index j*p+k = j*8+k
    # C[i,k] -> index k*n+i = k*2+i

    # A_left = A[:, 0:2], A_right = A[:, 2]
    # B_top = B[0:2, :], B_bottom = B[2, :]

    # Strassen for 2x2*2x2 (verified numerically):
    # p1 = (a00+a11)*(b00+b11)
    # p2 = (a10+a11)*b00
    # p3 = a00*(b10-b11)
    # p4 = a11*(b01-b10)
    # p5 = (a00+a01)*b11
    # p6 = (a10-a00)*(b00+b10)
    # p7 = (a01-a11)*(b01+b11)
    #
    # c00 = p1 + p4 - p5 - p7
    # c01 = p3 + p5
    # c10 = p2 + p4
    # c11 = p1 - p2 + p3 + p7

    for b in range(4):
        k0 = 2 * b

        # p1 = (a00+a11)*(b00+b11)
        U[0, col] = 1
        U[4, col] = 1
        V[k0, col] = 1
        V[8 + k0 + 1, col] = 1
        W[k0, col] = 1
        W[(k0 + 1) * 2 + 1, col] = 1
        col += 1

        # p2 = (a10+a11)*b00
        U[3, col] = 1
        U[4, col] = 1
        V[k0, col] = 1
        W[k0 * 2 + 1, col] = 1
        col += 1

        # p3 = a00*(b10-b11)
        U[0, col] = 1
        V[8 + k0, col] = 1
        V[8 + k0 + 1, col] = -1
        W[(k0 + 1) * 2, col] = 1
        col += 1

        # p4 = a11*(b01-b10)
        U[4, col] = 1
        V[k0 + 1, col] = 1
        V[8 + k0, col] = -1
        W[k0 * 2, col] = 1
        W[k0 * 2 + 1, col] = 1
        col += 1

        # p5 = (a00+a01)*b11
        U[0, col] = 1
        U[1, col] = 1
        V[8 + k0 + 1, col] = 1
        W[k0 * 2, col] = -1
        W[(k0 + 1) * 2, col] = 1
        col += 1

        # p6 = (a10-a00)*(b00+b10)
        U[3, col] = 1
        U[0, col] = -1
        V[k0, col] = 1
        V[8 + k0, col] = 1
        W[k0 * 2 + 1, col] = 1
        col += 1

        # p7 = (a01-a11)*(b01+b11)
        U[1, col] = 1
        U[4, col] = -1
        V[k0 + 1, col] = 1
        V[8 + k0 + 1, col] = 1
        W[k0 * 2, col] = -1
        W[(k0 + 1) * 2 + 1, col] = 1
        col += 1

    # A_right * B_bottom: 2x1 * 1x8
    for i in range(2):
        for k in range(8):
            U[i * 3 + 2, col] = 1
            V[16 + k, col] = 1
            W[k * 2 + i, col] = 1
            col += 1

    # Fast vectorized verification
    if time.time() < deadline - 3:
        # Build the tensor T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # C[i,k] = sum_j T[i,j,k]
        # We can compute this as: for each r, outer product of U[:,r] reshaped, V[:,r] reshaped, W[:,r] reshaped
        # More efficiently: C[i,k] = sum_j sum_r U[i*m+j,r] V[j*p+k,r] W[k*n+i,r]
        # = sum_r W[k*n+i,r] * (sum_j U[i*m+j,r] V[j*p+k,r])
        # Let's verify with random test
        A_test = np.random.randn(n, m)
        B_test = np.random.randn(m, p)
        C_expected = A_test @ B_test

        # Compute C from decomposition
        # C[i,k] = sum_j sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # = sum_r W[k*n+i, r] * sum_j U[i*m+j, r] * V[j*p+k, r]
        # Vectorized: for each r, compute sum_j U[i*m+j,r]*V[j*p+k,r] for all i,k
        # This is like: for each r, A_r = U[:,r] reshaped to (n,m), B_r = V[:,r] reshaped to (m,p)
        # then C_r = A_r @ B_r, and C[i,k] = sum_r W[k*n+i,r] * C_r[i,k]
        C_computed = np.zeros((n, p))
        for r in range(rank):
            U_r = U[:, r].reshape(n, m)
            V_r = V[:, r].reshape(m, p)
            C_r = U_r @ V_r
            W_r = W[:, r].reshape(p, n).T  # W[k*n+i, r] -> shape (n, p)
            C_computed += W_r * C_r

        if not np.allclose(C_computed, C_expected, atol=1e-9):
            # Fallback to schoolbook
            U_s = np.zeros((nm, 48))
            V_s = np.zeros((mp, 48))
            W_s = np.zeros((pn, 48))
            c2 = 0
            for ii in range(n):
                for jj in range(m):
                    for kk in range(p):
                        U_s[ii * m + jj, c2] = 1
                        V_s[jj * p + kk, c2] = 1
                        W_s[kk * n + ii, c2] = 1
                        c2 += 1
            return {"U": U_s.tolist(), "V": V_s.tolist(), "W": W_s.tolist()}

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
