"""Find rank-minimal exact tensor decomposition for 4x4 * 4x6 matrix multiplication."""

DIMENSIONS = [4, 4, 6]

# Strassen 2x2 base cases: U2 (4x7), V2 (4x7), W2 (4x7)
# Each 2x2 matrix is flattened: A00,A01,A10,A11 -> indices 0,1,2,3
# For U: A[i*m+j] where i,j in {0,1}, m=2: idx = i*2+j
# For V: B[j*p+k] where j,k in {0,1}, p=2: idx = j*2+k
# For W: C[k*n+i] where k,i in {0,1}, n=2: idx = k*2+i

STRASSEN_U2 = [
    [1, 0, 1, 0, 0, 0, 0],  # A00: M1, M3, M5
    [0, 1, 0, 0, 1, 0, 0],  # A01: M5, M7
    [0, 0, 1, 1, 0, 1, 0],  # A10: M2, M6
    [1, 0, 1, 1, 0, 0, 0],  # A11: M1, M2, M4
]
STRASSEN_V2 = [
    [1, 1, 1, 1, 0, 0, 0],  # B00: M1, M2, M4, M6
    [0, 0, 1, 0, 1, 0, 1],  # B01: M3, M5, M7
    [0, 0, 0, 1, 0, 1, 0],  # B10: M4, M6
    [1, 0, 0, 0, 1, 0, 1],  # B11: M1, M5, M7
]
STRASSEN_W2 = [
    [1, 0, 0, 1, 0, 0, 1],  # C00: M1, M4, M7
    [0, 0, 1, 0, 1, 0, 0],  # C01: M3, M5
    [0, 1, 0, 1, 0, 0, 0],  # C10: M2, M4
    [1, 0, 0, 0, 0, 1, 0],  # C11: M1, M6
]


def construct():
    n, m, p = 4, 4, 6

    # Part 1: 4x4x4 multiplication (k=0..3) using block Strassen
    # Split into 2x2 blocks of size 2x2
    # 8 blocks: A block (bi,bj), B block (bj,bk) for bi,bj,bk in {0,1}
    # Each uses Strassen rank 7
    # Total rank for this part: 8 * 7 = 56
    rank1 = 8 * 7
    U1 = [[0] * rank1 for _ in range(n * m)]  # 16 rows
    V1 = [[0] * rank1 for _ in range(m * p)]  # 24 rows
    W1 = [[0] * rank1 for _ in range(p * n)]  # 24 rows

    for bi in range(2):  # A block row
        for bj in range(2):  # A block col = B block row
            for bk in range(2):  # B block col
                col_offset = (bi * 2 + bj) * 2 + bk  # 0..7
                for s in range(7):
                    c = col_offset * 7 + s

                    # U: A block (bi,bj), element (ii,jj) in 2x2 block
                    u_val = STRASSEN_U2[ii := (0, 0, 0, 0)][s] if False else None  # will compute below

                    for ii in range(2):
                        for jj in range(2):
                            i = bi * 2 + ii
                            j = bj * 2 + jj
                            u_idx = i * m + j  # i*4+j
                            U1[u_idx][c] = STRASSEN_U2[ii * 2 + jj][s]

                    for ii in range(2):
                        for jj in range(2):
                            j = bj * 2 + ii
                            k = bk * 2 + jj
                            v_idx = j * p + k  # j*6+k
                            V1[v_idx][c] = STRASSEN_V2[ii * 2 + jj][s]

                    for ii in range(2):
                        for jj in range(2):
                            k = bk * 2 + ii
                            i = bi * 2 + jj
                            w_idx = k * n + i  # k*4+i
                            W1[w_idx][c] = STRASSEN_W2[ii * 2 + jj][s]

    # Part 2: 4x4x2 multiplication (k=4..5) schoolbook
    # rank2 = 4*4*2 = 32
    rank2 = n * m * 2
    U2 = [[0] * rank2 for _ in range(n * m)]
    V2 = [[0] * rank2 for _ in range(m * p)]
    W2 = [[0] * rank2 for _ in range(p * n)]

    col2 = 0
    for i in range(n):
        for j in range(m):
            for kk in range(4, 6):  # k = 4, 5
                U2[i * m + j][col2] = 1
                V2[j * p + kk][col2] = 1
                W2[kk * n + i][col2] = 1
                col2 += 1

    # Combine
    rank = rank1 + rank2
    U = [row1 + row2 for row1, row2 in zip(U1, U2)]
    V = [row1 + row2 for row1, row2 in zip(V1, V2)]
    W = [row1 + row2 for row1, row2 in zip(W1, W2)]

    return {"U": U, "V": V, "W": W}


if __name__ == "__main__":
    res = construct()
    print("Rank:", len(res['U'][0]))
    # Verify
    n, m, p = 4, 4, 6
    rank = len(res['U'][0])
    import numpy as np
    U = np.array(res['U'])
    V = np.array(res['V'])
    W = np.array(res['W'])
    
    # Build tensor: T[i,j,k] should equal sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                expected = 1 if j == j else 0  # T[i,j,k] = 1 for all valid (i,j,k) in multiplication
                # Actually T[i,j,k] = 1 for all i,j,k since each (i,j,k) contributes to C[i,k]
                actual = np.sum(U[i*m+j] * V[j*p+k] * W[k*n+i])
                if abs(actual - 1) > 1e-9:
                    print(f"MISMATCH at ({i},{j},{k}): expected 1, got {actual}")
    print("Verification complete")
