import time
import numpy as np

DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 140
    n, m, p = DIMENSIONS  # 4, 4, 6

    def verify_full(U, V, W, n, m, p):
        Ua = np.array(U, dtype=float); Va = np.array(V, dtype=float); Wa = np.array(W, dtype=float)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.dot(Ua[i * m + j], Va[j * p + k] * Wa[k * n + i])
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True

    # Strassen 2x2x2 decomposition (Rank 7)
    # U: (4, 7), V: (4, 7), W: (4, 7)
    # Indices: U[i*2+j], V[j*2+k], W[k*2+i]
    U2 = np.zeros((4, 7))
    V2 = np.zeros((4, 7))
    W2 = np.zeros((4, 7))
    
    # M1 = (a00+a11)(b00+b11) -> c00, c11
    U2[0, 0] = 1; U2[3, 0] = 1
    V2[0, 0] = 1; V2[3, 0] = 1
    W2[0, 0] = 1; W2[3, 0] = 1
    
    # M2 = (a10+a11)b00 -> c10, c11
    U2[2, 1] = 1; U2[3, 1] = 1
    V2[0, 1] = 1
    W2[2, 1] = 1; W2[3, 1] = -1
    
    # M3 = a00(b01-b11) -> c01, c11
    U2[0, 2] = 1
    V2[1, 2] = 1; V2[3, 2] = -1
    W2[1, 2] = 1; W2[3, 2] = 1
    
    # M4 = a11(b10-b00) -> c00, c10
    U2[3, 3] = 1
    V2[2, 3] = 1; V2[0, 3] = -1
    W2[0, 3] = 1; W2[2, 3] = 1
    
    # M5 = (a00+a01)b11 -> c00, c01
    U2[0, 4] = 1; U2[1, 4] = 1
    V2[3, 4] = 1
    W2[0, 4] = -1; W2[1, 4] = 1
    
    # M6 = (a10-a00)(b00+b01) -> c11
    U2[2, 5] = 1; U2[0, 5] = -1
    V2[0, 5] = 1; V2[1, 5] = 1
    W2[3, 5] = 1
    
    # M7 = (a01-a11)(b10+b11) -> c00, c11
    U2[1, 6] = 1; U2[3, 6] = -1
    V2[2, 6] = 1; V2[3, 6] = 1
    W2[0, 6] = 1; W2[3, 6] = 1

    # Schoolbook 2x2x1 decomposition (Rank 4)
    # A (2x2) * B (2x1) = C (2x1)
    # U: (4, 4), V: (2, 4), W: (2, 4)
    # Indices: U[i*2+j], V[j*1+k] -> V[j], W[k*2+i]
    U1 = np.zeros((4, 4))
    V1 = np.zeros((2, 4))
    W1 = np.zeros((2, 4))
    
    # Term 0: a00 b0 -> c0
    U1[0, 0] = 1; V1[0, 0] = 1; W1[0, 0] = 1
    # Term 1: a01 b1 -> c0
    U1[1, 1] = 1; V1[1, 1] = 1; W1[0, 1] = 1
    # Term 2: a10 b0 -> c1
    U1[2, 2] = 1; V1[0, 2] = 1; W1[1, 2] = 1
    # Term 3: a11 b1 -> c1
    U1[3, 3] = 1; V1[1, 3] = 1; W1[1, 3] = 1

    # Construct 4x4x4 part (Rank 49)
    # U444 = kron(U2, U2)
    # Indices: (i1*2+i2, j1*2+j2)
    # Kronecker product of U2 (4x7) and U2 (4x7) gives 16x49.
    # The index mapping in np.kron(A, B) for A[i,j] B[k,l] is (i*rows_B + k, j*cols_B + l).
    # We want i = i1*2 + i2, j = j1*2 + j2.
    # So i1 corresponds to i in U2, i2 corresponds to k in U2.
    # This matches np.kron(U2, U2).
    U444 = np.kron(U2, U2)
    V444 = np.kron(V2, V2)
    W444 = np.kron(W2, W2)
    
    # Construct 4x4x2 part (Rank 28)
    # U442 = kron(U2, U1)
    # U2 (4x7), U1 (4x4) -> 16x28.
    U442 = np.kron(U2, U1)
    # V442 = kron(V2, V1)
    # V2 (4x7), V1 (2x4) -> 8x28.
    # V indices: j, k.
    # j = j1*2 + j2, k = k2 (since p=1 in subproblem, k is 0 or 1? No, p=1 means k=0 only).
    # Wait, V1 is for B (2x1). V1 indices are j (0,1).
    # So V442 indices are j (0..3) and k (0..1).
    # j = j1*2 + j2.
    # k = k2 (0 or 1).
    # np.kron(V2, V1) gives indices (j1*2+j2, k1*1+k2)?
    # V2 has 4 rows (j). V1 has 2 rows (j).
    # np.kron(V2, V1) row index = j1*2 + j2.
    # Col index = r1*4 + r2.
    # This matches.
    V442 = np.kron(V2, V1)
    # W442 = kron(W2, W1)
    # W2 (4x7), W1 (2x4) -> 8x28.
    # W indices: k, i.
    # k = k2 (0,1). i = i1*2 + i2.
    # np.kron(W2, W1) row index = k1*2 + k2?
    # W2 has 4 rows (k). W1 has 2 rows (k).
    # np.kron(W2, W1) row index = k1*2 + k2.
    # This matches.
    W442 = np.kron(W2, W1)

    # Assemble 4x4x6
    rank_total = U444.shape[1] + U442.shape[1] # 49 + 28 = 77
    U = np.zeros((16, rank_total))
    V = np.zeros((24, rank_total))
    W = np.zeros((24, rank_total))
    
    # Fill U
    U[:, :49] = U444
    U[:, 49:] = U442
    
    # Fill V
    # V444 is 16x49. V indices j (0..3), k (0..3).
    # V is 24x77. V indices j (0..3), k (0..5).
    # For k in 0..3, V[j, k] = V444[j, k].
    for j in range(4):
        for k in range(4):
            V[j*6 + k, :49] = V444[j*4 + k, :]
            
    # V442 is 8x28. V indices j (0..3), k (0..1).
    # For k in 4..5, V[j, k] = V442[j, k-4].
    for j in range(4):
        for k in range(2):
            V[j*6 + (k+4), 49:] = V442[j*2 + k, :]
            
    # Fill W
    # W444 is 16x49. W indices k (0..3), i (0..3).
    # W is 24x77. W indices k (0..5), i (0..3).
    # For k in 0..3, W[k, i] = W444[k, i].
    for k in range(4):
        for i in range(4):
            W[k*4 + i, :49] = W444[k*4 + i, :]
            
    # W442 is 8x28. W indices k (0..1), i (0..3).
    # For k in 4..5, W[k, i] = W442[k-4, i].
    for k in range(2):
        for i in range(4):
            W[(k+4)*4 + i, 49:] = W442[k*4 + i, :]
            
    if verify_full(U, V, W, 4, 4, 6):
        return {"U": U, "V": V, "W": W}
        
    # Fallback
    rank = n * m * p
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1.0
                V[j * p + k][col] = 1.0
                W[k * n + i][col] = 1.0
                col += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
