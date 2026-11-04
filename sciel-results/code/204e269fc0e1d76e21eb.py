"""
Rank-33 exact decomposition for 2x3 by 3x6 matrix multiplication.
Structure: A1(2x1)*B1(1x6) schoolbook (rank 12) + A2(2x2)*B2(2x6) as 3 Strassen blocks (rank 21).
Total rank 33. Verified deterministically with fallback to schoolbook rank 36.
"""
import time
import numpy as np

DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140

    n, m, p = DIMENSIONS
    dim_a = n * m  # 6
    dim_b = m * p  # 18
    dim_c = p * n  # 12

    # Build target tensor for verification
    T = np.zeros((dim_a, dim_b, dim_c), dtype=np.int64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1

    def verify(U, V, W):
        rank = U.shape[1]
        result = np.zeros((dim_a, dim_b, dim_c), dtype=np.int64)
        for c in range(rank):
            result += np.einsum('i,j,k->ijk', U[:, c], V[:, c], W[:, c])
        return np.array_equal(result, T)

    def schoolbook():
        rank = n * m * p
        U = np.zeros((dim_a, rank), dtype=np.int64)
        V = np.zeros((dim_b, rank), dtype=np.int64)
        W = np.zeros((dim_c, rank), dtype=np.int64)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    U[a, col] = 1
                    V[b, col] = 1
                    W[c, col] = 1
                    col += 1
        return U, V, W

    def structural():
        rank = 33
        U = np.zeros((dim_a, rank), dtype=np.int64)
        V = np.zeros((dim_b, rank), dtype=np.int64)
        W = np.zeros((dim_c, rank), dtype=np.int64)
        col = 0

        # Part 1: A1(2x1) * B1(1x6), schoolbook rank 12
        for k in range(p):
            for i in range(n):
                U[i * m + 0, col] = 1
                V[0 * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1

        # Part 2: A2(2x2) * B2(2x6), 3 Strassen blocks rank 21
        # A2: rows i=0,1; cols j=1,2
        #   U[1]=A00, U[2]=A01, U[4]=A10, U[5]=A11
        # B2 block t: rows j=1,2; cols k=2t,2t+1
        #   V[6+2t]=B00, V[7+2t]=B01, V[12+2t]=B10, V[13+2t]=B11
        # C block t: rows i=0,1; cols k=2t,2t+1
        #   W[4t]=C00, W[4t+1]=C10, W[4t+2]=C01, W[4t+3]=C11

        for t in range(3):
            # M1 = (A00+A11)(B00+B11) -> C00 + C11
            U[1, col] = 1; U[5, col] = 1
            V[6 + 2*t, col] = 1; V[13 + 2*t, col] = 1
            W[4*t, col] = 1; W[4*t + 3, col] = 1
            col += 1

            # M2 = (A10+A11)B00 -> C10 - C11
            U[4, col] = 1; U[5, col] = 1
            V[6 + 2*t, col] = 1
            W[4*t + 1, col] = 1; W[4*t + 3, col] = -1
            col += 1

            # M3 = A00(B01-B11) -> C01 + C11
            U[1, col] = 1
            V[7 + 2*t, col] = 1; V[13 + 2*t, col] = -1
            W[4*t + 2, col] = 1; W[4*t + 3, col] = 1
            col += 1

            # M4 = A11(B10-B00) -> C00 + C10
            U[5, col] = 1
            V[12 + 2*t, col] = 1; V[6 + 2*t, col] = -1
            W[4*t, col] = 1; W[4*t + 1, col] = 1
            col += 1

            # M5 = (A00+A01)B11 -> -C00 + C01
            U[1, col] = 1; U[2, col] = 1
            V[13 + 2*t, col] = 1
            W[4*t, col] = -1; W[4*t + 2, col] = 1
            col += 1

            # M6 = (A10-A00)(B00+B01) -> C11
            U[4, col] = 1; U[1, col] = -1
            V[6 + 2*t, col] = 1; V[7 + 2*t, col] = 1
            W[4*t + 3, col] = 1
            col += 1

            # M7 = (A01-A11)(B10+B11) -> C00
            U[2, col] = 1; U[5, col] = -1
            V[12 + 2*t, col] = 1; V[13 + 2*t, col] = 1
            W[4*t, col] = 1
            col += 1

        return U, V, W

    U, V, W = structural()
    if not verify(U, V, W):
        U, V, W = schoolbook()

    if time.time() > deadline - 10:
        pass  # safety, we're deterministic so this won't trigger

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
