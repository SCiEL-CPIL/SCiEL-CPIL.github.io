"""Matrix multiplication 4x4 by 4x8 - rank-minimal decomposition.

Strategy:
  1. Try to find a rank-64 decomposition via Kronecker product of two
     rank-8 2x2x2x2 "Strassen-like" blocks (if a rank-8 algorithm exists
     with integer entries, we can achieve 8*8=64).
  2. If that fails, fall back to the Kronecker product of Strassen (rank-7)
     and schoolbook 2x2x4 (rank-16) = rank 112.
  3. Final fallback: full schoolbook rank 128.

The primary construction uses Strassen 2x2x2 (rank 7) ⊗ schoolbook 2x2x4
(rank 16) = rank 112, which is the best guaranteed construction.

We also attempt a direct rank-96 construction using two Strassen blocks
for the 4x4x8 problem: split into (2x2)⊗(2x4) in a different way.
Actually, let's use the following: 4x4x8 = (2x2x2) ⊗ (2x2x4) where
we use Strassen for the first factor and a rank-8 decomposition for the
second. If no rank-8 exists, we use schoolbook rank-16, giving 112.

Let me think more carefully. The tensor for 4×4×8 has 4·4·8=128 entries.
Schoolbook: rank 128.

Kronecker: 4=2·2, 4=2·2, 8=2·4.
- (2×2×2) ⊗ (2×2×4): rank 7·16 = 112 (Strassen ⊗ schoolbook)
- (2×2×4) ⊗ (2×2×2): rank 16·7 = 112 (same)
- (4×4×2) ⊗ (1×2×4): rank of 4×4×2? Schoolbook = 32. 1×2×4 = 8. Total 256. Worse.

Another option: use a rank-8 algorithm for 2×2×4 if one exists.
The bilinear complexity of 2×2×4 matrix multiplication:
- Schoolbook: 2·2·4 = 16
- Can we do better? 

Actually, for 2×2×4 (multiply 2×2 by 2×4):
A is 2×2, B is 2×4, C = AB is 2×4.
We can write B as [B1 | B2] where B1, B2 are 2×2.
Then C = [A·B1 | A·B2].
Using Strassen for each: 7+7 = 14 multiplications.
But we can share some: A·B1 and A·B2 share the A factors.

Actually, let me just use the 112-rank construction which is clean and correct.

Let me also try: can we do 4×4×8 as (2×4×2) ⊗ (2×2×4)?
2×4×2: multiply 2×4 by 4×2 = 2×2. Schoolbook rank 16.
Not helpful.

Let me stick with the rank-112 construction and make sure it's correct.
"""
import time
import numpy as np

DIMENSIONS = [4, 4, 8]
n, m, p = DIMENSIONS


def verify(U, V, W):
    """Check if the decomposition is exact."""
    n, m, p = DIMENSIONS
    r = len(U[0]) if U else 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(r):
                    s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                if abs(s - 1.0) > 1e-9:
                    return False, (i, j, k), s
    return True, None, None


def make_schoolbook():
    n, m, p = DIMENSIONS
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


def strassen_2x2x2():
    """
    Rank-7 Strassen decomposition for 2x2x2 matrix multiplication.
    U[i*2+j, c], V[j*2+k, c], W[k*2+i, c], shapes (4,7).
    """
    U = [[0] * 7 for _ in range(4)]
    V = [[0] * 7 for _ in range(4)]
    W = [[0] * 7 for _ in range(4)]

    # c=0: M1 = (A11+A22)(B11+B22)
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1

    # c=1: M2 = (A21+A22)B11
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[2][1] = 1; W[3][1] = -1

    # c=2: M3 = A11(B12-B22)
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[1][2] = 1; W[3][2] = 1

    # c=3: M4 = A22(B21-B11)
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[2][3] = 1; W[0][3] = -1

    # c=4: M5 = (A11+A12)B22
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = -1; W[1][4] = 1

    # c=5: M6 = (A21-A11)(B11+B12)
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1

    # c=6: M7 = (A12-A22)(B21+B22)
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = 1; W[1][6] = -1

    return U, V, W


def schoolbook_2x2x4():
    """
    Rank-16 schoolbook decomposition for 2x2x4 matrix multiplication.
    U[i*2+j, c], V[j*4+k, c], W[k*2+i, c], shapes (4,16), (8,16), (8,16).
    """
    U = [[0] * 16 for _ in range(4)]
    V = [[0] * 16 for _ in range(8)]
    W = [[0] * 16 for _ in range(8)]

    col = 0
    for i in range(2):
        for j in range(2):
            for k in range(4):
                U[i * 2 + j][col] = 1
                V[j * 4 + k][col] = 1
                W[k * 2 + i][col] = 1
                col += 1

    return U, V, W


def strassen_2x2x4():
    """
    Rank-14 decomposition for 2x2x4 using two Strassen calls.
    A (2x2) times B (2x4) = [A*B1 | A*B2] where B1, B2 are 2x2.
    Each Strassen call uses 7 multiplications, total 14.
    U[i*2+j, c], V[j*4+k, c], W[k*2+i, c], shapes (4,14), (8,14), (8,14).
    """
    U_str, V_str, W_str = strassen_2x2x2()
    rank = 14
    U = [[0] * rank for _ in range(4)]
    V = [[0] * rank for _ in range(8)]
    W = [[0] * rank for _ in range(8)]

    # First half: A * B1 (columns 0,1 of B)
    for c in range(7):
        for i in range(2):
            for j in range(2):
                U[i * 2 + j][c] = U_str[i * 2 + j][c]
        for j in range(2):
            for k in range(2):
                V[j * 4 + k][c] = V_str[j * 2 + k][c]
        for k in range(2):
            for i in range(2):
                W[k * 2 + i][c] = W_str[k * 2 + i][c]

    # Second half: A * B2 (columns 2,3 of B)
    for c in range(7):
        cc = c + 7
        for i in range(2):
            for j in range(2):
                U[i * 2 + j][cc] = U_str[i * 2 + j][c]
        for j in range(2):
            for k in range(2):
                V[j * 4 + 2 + k][cc] = V_str[j * 2 + k][c]
        for k in range(2):
            for i in range(2):
                W[(2 + k) * 2 + i][cc] = W_str[k * 2 + i][c]

    return U, V, W


def construct():
    start_time = time.time()
    deadline = start_time + 140.0

    n, m, p = DIMENSIONS

    # Construction: Strassen 2x2x2 (rank 7) ⊗ Strassen 2x2x4 (rank 14) = rank 98
    # 4x4x8 = (2x2x2) ⊗ (2x2x4)
    # i = 2*i1 + i2, j = 2*j1 + j2, k = 4*k1 + k2
    # Rank = 7 * 14 = 98

    U1, V1, W1 = strassen_2x2x2()
    U2, V2, W2 = strassen_2x2x4()

    U1_np = np.array(U1, dtype=float)
    V1_np = np.array(V1, dtype=float)
    W1_np = np.array(W1, dtype=float)
    U2_np = np.array(U2, dtype=float)
    V2_np = np.array(V2, dtype=float)
    W2_np = np.array(W2, dtype=float)

    U = np.zeros((16, 98))
    V = np.zeros((32, 98))
    W = np.zeros((32, 98))

    for c1 in range(7):
        for c2 in range(14):
            c = c1 * 14 + c2
            U[:, c] = np.kron(U1_np[:, c1].reshape(2, 2), U2_np[:, c2].reshape(2, 2)).reshape(16)
            V[:, c] = np.kron(V1_np[:, c1].reshape(2, 2), V2_np[:, c2].reshape(2, 4)).reshape(32)
            W[:, c] = np.kron(W1_np[:, c1].reshape(2, 2), W2_np[:, c2].reshape(4, 2)).reshape(32)

    U_list = [[int(round(x)) for x in row] for row in U]
    V_list = [[int(round(x)) for x in row] for row in V]
    W_list = [[int(round(x)) for x in row] for row in W]

    valid, err, val = verify(U_list, V_list, W_list)
    if valid:
        return {"U": U_list, "V": V_list, "W": W_list}

    # Fallback: schoolbook rank 128
    return make_schoolbook()


if __name__ == "__main__":
    result = construct()
    r = len(result["U"][0])
    print(f"Rank: {r}")
    valid, err, val = verify(result["U"], result["V"], result["W"])
    print(f"Valid: {valid}")
