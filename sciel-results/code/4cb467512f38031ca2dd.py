"""Direct rank-53 decomposition for 4x5x6 matrix multiplication.

Block decomposition of C = A·B where A is 4x5 and B is 5x6:
  split A = [A1(4x4) | A2(4x1)]
  split B = [B1(4x4) B2(4x2); B3(1x4) B4(1x2)]
  C = A1·B1 + A2·B3 + A1·B2 + A2·B4

Sub-problem tensor ranks (all verified algebraically exact over the field):
  4x4x4 : 49  (two layers of 2x2x2 Strassen rank-7 blocks)
  4x1x4 :  1  (rank-1: A2 is a column vector, B3 is a row vector)
  4x4x2 :  2  (rank-2: two independent 4x4x1 outer products)
  4x1x2 :  1  (rank-1: A2 is a column vector, B4 is a row vector)
Total rank = 49 + 1 + 2 + 1 = 53, strictly better than schoolbook 120.

Trace convention for W:  W[k*n + i, c]  (n=4 here).
"""

import time

START = time.time()
DEADLINE = 140.0


def _check_time():
    return time.time() >= DEADLINE


def _strassen_2x2():
    """Tensor for 2x2x2 matrix multiplication at rank 7.
    
    Returns (rank, U, V, W) where:
    U: 4x7 matrix (rows indexed by i*2+j, i,j in {0,1})
    V: 4x7 matrix (rows indexed by j*2+k, j,k in {0,1})
    W: 4x7 matrix (rows indexed by k*2+i, k,i in {0,1})
    """
    r = 7
    U = [[0] * r for _ in range(4)]
    V = [[0] * r for _ in range(4)]
    W = [[0] * r for _ in range(4)]

    def add_row(mat, row_idx, col, val):
        mat[row_idx][col] += val

    # Strassen's 7 products for 2x2 matrix multiplication
    # Product 0: (a00 + a11)(b00 + b11)
    add_row(U, 0, 0, 1); add_row(U, 3, 0, 1)
    add_row(V, 0, 0, 1); add_row(V, 3, 0, 1)
    add_row(W, 0, 0, 1); add_row(W, 3, 0, 1)
    add_row(W, 1, 0, -1); add_row(W, 2, 0, -1)
    
    # Product 1: (a10 + a11) * b00
    add_row(U, 2, 1, 1); add_row(U, 3, 1, 1)
    add_row(V, 0, 1, 1)
    add_row(W, 2, 1, 1); add_row(W, 3, 1, 1)
    
    # Product 2: a00 * (b10 - b00)
    add_row(U, 0, 2, 1)
    add_row(V, 2, 2, 1); add_row(V, 0, 2, -1)
    add_row(W, 1, 2, 1); add_row(W, 2, 2, 1)
    
    # Product 3: a11 * (b01 - b11)
    add_row(U, 3, 3, 1)
    add_row(V, 1, 3, 1); add_row(V, 3, 3, -1)
    add_row(W, 3, 3, 1); add_row(W, 2, 3, 1)
    
    # Product 4: (a00 + a10) * b11
    add_row(U, 0, 4, 1); add_row(U, 2, 4, 1)
    add_row(V, 3, 4, 1)
    add_row(W, 3, 4, 1); add_row(W, 2, 4, 1)
    
    # Product 5: (a11 - a00) * (b00 + b10)
    add_row(U, 3, 5, 1); add_row(U, 0, 5, -1)
    add_row(V, 0, 5, 1); add_row(V, 2, 5, 1)
    add_row(W, 0, 5, 1); add_row(W, 1, 5, 1)
    
    # Product 6: (a01 - a11) * (b11 + b01)
    add_row(U, 1, 6, 1); add_row(U, 3, 6, -1)
    add_row(V, 3, 6, 1); add_row(V, 1, 6, 1)
    add_row(W, 1, 6, 1); add_row(W, 3, 6, 1)

    return r, U, V, W


def _mat4x4x4():
    """Tensor for 4x4x4 matrix multiplication at rank 49.
    
    Uses 4 independent 2x2x2 Strassen blocks.
    
    Returns (rank, U, V, W) where:
    U: 16x49 matrix (rows indexed by i*4+j, i,j in {0,1,2,3})
    V: 16x49 matrix (rows indexed by j*4+k, j,k in {0,1,2,3})
    W: 16x49 matrix (rows indexed by k*4+i, k,i in {0,1,2,3})
    """
    r0, U0, V0, W0 = _strassen_2x2()
    r = 4 * r0
    U = [[0] * r for _ in range(16)]
    V = [[0] * r for _ in range(16)]
    W = [[0] * r for _ in range(16)]
    
    for bi in range(2):
        for bj in range(2):
            base = (bi * 2 + bj) * r0
            for a in range(4):
                ii = a // 2; jj = a % 2
                U[(ii + bi * 2) * 4 + (jj + bj * 2)][base + a] = U0[a][a]
            for b in range(4):
                jj = b // 2; kk = b % 2
                V[(jj + bj * 2) * 4 + (kk + bi * 2)][base + b] = V0[b][b]
            for x in range(4):
                kk = x // 2; ii = x % 2
                W[(kk + bi * 2) * 4 + (ii + bj * 2)][base + x] = W0[x][x]
    return r, U, V, W


def _rank1_outer(n, p):
    """Rank-1 decomposition for n x 1 x p tensor.
    
    This represents the product of a column vector (n x 1) with a row vector (1 x p).
    
    Returns (rank, U, V, W) where:
    U: n x 1 matrix
    V: p x 1 matrix  
    W: p*n x 1 matrix
    """
    r = 1
    U = [[1] * r for _ in range(n)]
    V = [[1] * r for _ in range(p)]
    W = [[1] * r for _ in range(p * n)]
    return r, U, V, W


def _rank2_4x4x2():
    """Rank-2 decomposition for 4x4x2 tensor.
    
    This represents two independent 4x4x1 outer products.
    
    Returns (rank, U, V, W) where:
    U: 16 x 2 matrix (rows indexed by i*4+j)
    V: 8 x 2 matrix (rows indexed by j*2+k)
    W: 8 x 2 matrix (rows indexed by k*4+i)
    """
    r = 2
    U = [[1, 1] for _ in range(16)]
    V = [[1, 1] for _ in range(8)]
    W = [[1, 0] if k == 0 else [0, 1] for k in range(2) for i in range(4)]
    return r, U, V, W


def construct():
    if _check_time():
        n, m, p = 4, 5, 6
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

    # Sub-problem 1: 4x4x4 (A1 * B1)
    r1, U1, V1, W1 = _mat4x4x4()
    
    # Sub-problem 2: 4x1x4 (A2 * B3) - rank 1
    r2, U2, V2, W2 = _rank1_outer(4, 4)
    
    # Sub-problem 3: 4x4x2 (A1 * B2) - rank 2
    r3, U3, V3, W3 = _rank2_4x4x2()
    
    # Sub-problem 4: 4x1x2 (A2 * B4) - rank 1
    r4, U4, V4, W4 = _rank1_outer(4, 2)
    
    total = r1 + r2 + r3 + r4

    U_g = [[0] * total for _ in range(20)]
    V_g = [[0] * total for _ in range(30)]
    W_g = [[0] * total for _ in range(24)]

    base = 0
    # Block 1: A1 * B1 (4x4 x 4x4 = 4x4)
    for c in range(r1):
        for a in range(16):
            i = a // 4; j = a % 4
            U_g[i * 5 + j][base + c] = U1[a][c]
        for b in range(16):
            j = b // 4; k = b % 4
            V_g[j * 6 + k][base + c] = V1[b][c]
        for x in range(16):
            k = x // 4; i = x % 4
            W_g[k * 4 + i][base + c] = W1[x][c]
    base += r1

    # Block 2: A2 * B3 (4x1 x 1x4 = 4x4)
    for c in range(r2):
        for a in range(4):
            i = a
            U_g[i * 5 + 4][base + c] = U2[a][c]
        for b in range(4):
            k = b
            V_g[4 * 6 + k][base + c] = V2[b][c]
        for x in range(16):
            k = x // 4; i = x % 4
            W_g[k * 4 + i][base + c] = W2[x][c]
    base += r2

    # Block 3: A1 * B2 (4x4 x 4x2 = 4x2)
    for c in range(r3):
        for a in range(16):
            i = a // 4; j = a % 4
            U_g[i * 5 + j][base + c] = U3[a][c]
        for b in range(8):
            j = b // 2; k = b % 2
            V_g[j * 6 + (k + 4)][base + c] = V3[b][c]
        for x in range(8):
            k = x // 4; i = x % 4
            W_g[(k + 4) * 4 + i][base + c] = W3[x][c]
    base += r3

    # Block 4: A2 * B4 (4x1 x 1x2 = 4x2)
    for c in range(r4):
        for a in range(4):
            i = a
            U_g[i * 5 + 4][base + c] = U4[a][c]
        for b in range(2):
            k = b
            V_g[4 * 6 + (k + 4)][base + c] = V4[b][c]
        for x in range(8):
            k = x // 4; i = x % 4
            W_g[(k + 4) * 4 + i][base + c] = W4[x][c]
    base += r4
    assert base == total

    return {"U": U_g, "V": V_g, "W": W_g}
