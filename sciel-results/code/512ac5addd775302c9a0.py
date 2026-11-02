import time
import numpy as np

DIMENSIONS = [2, 2, 12]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 2, 2, 12  # C[n,p] = A[n,m] @ B[m,p]
    num_pairs = 6  # 6 column pairs in B
    rank = 42  # 6 * 7 (Strassen's rank for 2x2x2)

    U = np.zeros((4, 42), dtype=int)   # A coefficients
    V = np.zeros((24, 42), dtype=int)  # B coefficients
    W = np.zeros((24, 42), dtype=int)  # C coefficients

    # Strassen's 7 intermediate products:
    # S1 = (A00+A11)(B00+B11)  -> C00 += S1, C11 += S1
    # S2 = (A10+A11)(B00)      -> C10 += S2, C11 -= S2
    # S3 = (A00)(B01-B11)      -> C01 += S3, C11 += S3
    # S4 = (A11)(B10-B00)      -> C00 += S4, C10 += S4
    # S5 = (A00+A01)(B11)      -> C00 -= S5, C01 += S5
    # S6 = (A10-A00)(B00+B01)  -> C11 += S6
    # S7 = (A01-A11)(B10+B11)  -> C00 += S7
    #
    # C00 = S1 + S4 - S5 + S7
    # C01 = S3 + S5
    # C10 = S2 + S4
    # C11 = S1 - S2 + S3 + S6

    for t in range(num_pairs):
        b = 7 * t
        k0 = 2 * t
        k1 = 2 * t + 1

        # S1 = (A00+A11)(B[k0,0]+B[k1,1])
        U[0][b] = 1
        U[3][b] = 1
        V[k0][b] = 1
        V[12+k1][b] = 1
        W[2*k0][b] = 1
        W[2*k1+1][b] = 1

        # S2 = (A10+A11)(B[k0,0])
        U[2][b+1] = 1
        U[3][b+1] = 1
        V[k0][b+1] = 1
        W[2*k0+1][b+1] = 1
        W[2*k1+1][b+1] = -1

        # S3 = A00 * (B[k1,0]-B[k1,1])
        U[0][b+2] = 1
        V[k1][b+2] = 1
        V[12+k1][b+2] = -1
        W[2*k1][b+2] = 1
        W[2*k1+1][b+2] = 1

        # S4 = A11 * (B[k0,1]-B[k0,0])
        U[3][b+3] = 1
        V[12+k0][b+3] = 1
        V[k0][b+3] = -1
        W[2*k0][b+3] = 1
        W[2*k0+1][b+3] = 1

        # S5 = (A00+A01) * B[k1,1]
        U[0][b+4] = 1
        U[1][b+4] = 1
        V[12+k1][b+4] = 1
        W[2*k0][b+4] = -1
        W[2*k1][b+4] = 1

        # S6 = (A10-A00) * (B[k0,0]+B[k0,1])
        U[2][b+5] = 1
        U[0][b+5] = -1
        V[k0][b+5] = 1
        V[k0+1][b+5] = 1
        W[2*k1+1][b+5] = 1

        # S7 = (A01-A11) * (B[k0,1]+B[k1,1])
        U[1][b+6] = 1
        U[3][b+6] = -1
        V[12+k0][b+6] = 1
        V[12+k1][b+6] = 1
        W[2*k0][b+6] = 1

    # Verify correctness on random integer test cases
    for trial in range(10):
        A = np.random.randint(-5, 6, (2, 2))
        B = np.random.randint(-5, 6, (2, 12))
        C_expected = A @ B
        # Compute via tensor decomposition
        C_check = np.zeros((2, 12), dtype=int)
        for r in range(rank):
            a_r = U[:, r] @ A.T  # (n,) = U[n, r] * A[:, ?]
            # Actually: a_r[i] = sum_j U[i*2+j, r] * A[i,j]
            a_vals = np.array([sum(U[i*2+j, r] * A[i, j] for j in range(2)) for i in range(2)])
            b_vals = np.array([sum(V[i*2+j, r] * B[i, j] for i in range(2)) for j in range(12)])
            c_vals = np.array([sum(W[j*2+i, r] for i in range(2) for j in range(12) if False)])
            # Simpler: C[i,k] += sum_r W[i*2+k, r] * (sum_j U[i*2+j,r]*A[i,j]) * (sum_j2 V[j2*12+k,r]*B[j2,k])
            pass
        # More direct verification
        C_from_decomp = np.zeros((2, 12), dtype=int)
        for r in range(rank):
            for i in range(2):
                a_contrib = sum(U[i*2+j, r] * A[i, j] for j in range(2))
                for k in range(12):
                    b_contrib = sum(V[j*12+k, r] * B[j, k] for j in range(2))
                    C_from_decomp[i, k] += W[i*2+k, r] * a_contrib * b_contrib
        if not np.array_equal(C_from_decomp, C_expected):
            raise ValueError("Tensor decomposition failed verification")

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
