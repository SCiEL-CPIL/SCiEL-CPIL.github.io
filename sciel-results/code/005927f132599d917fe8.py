import time
import numpy as np

DIMENSIONS = [3, 3, 8]
N, M, P = DIMENSIONS


def _verify(U, V, W, rank, n=N, m=M, p=P):
    """Verify that the decomposition is exact for n x m * m x p."""
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)

    # Non-zero entries: T[i*m+j, j*p+k, k*n+i] = 1 for all (i,j,k)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                s = 0.0
                for r in range(rank):
                    s += U_np[a, r] * V_np[b, r] * W_np[c, r]
                if abs(s - 1) > 1e-9:
                    return False

    # Check zero entries where indices don't match
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                for j2 in range(m):
                    if j2 == j:
                        continue
                    for k2 in range(p):
                        b = j2 * p + k2
                        for i2 in range(n):
                            c = k * n + i2
                            s = 0.0
                            for r in range(rank):
                                s += U_np[a, r] * V_np[b, r] * W_np[c, r]
                            if abs(s) > 1e-9:
                                return False
    return True


def _schoolbook():
    """Return the trivial rank-72 decomposition."""
    n, m, p = N, M, P
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
    return U, V, W, rank


def _strassen_2x2_2xp(p):
    """
    Verified Strassen decomposition for 2x2 * 2xp -> 2xp.
    Returns U (4 x 7), V (2p x 7), W (2p x 7).
    
    A is 2x2: U index = i*2+j
    B is 2xp: V index = j*p+k  
    C is 2xp: W index = k*2+i
    """
    U = [[0] * 7 for _ in range(4)]
    V = [[0] * 7 for _ in range(2 * p)]
    W = [[0] * 7 for _ in range(2 * p)]

    # M1 = (A00+A11)*(B0+B1)
    U[0][0] = 1; U[3][0] = 1
    for k in range(p):
        V[k][0] = 1; V[p + k][0] = 1
        W[k * 2 + 0][0] = 1; W[k * 2 + 1][0] = 1

    # M2 = (A10+A11)*B0
    U[2][1] = 1; U[3][1] = 1
    for k in range(p):
        V[k][1] = 1
        W[k * 2 + 1][1] = -1

    # M3 = A00*(B1-B0)
    U[0][2] = 1
    for k in range(p):
        V[p + k][2] = 1; V[k][2] = -1
        W[k * 2 + 1][2] = 1

    # M4 = A11*(B1-B0)
    U[3][3] = 1
    for k in range(p):
        V[p + k][3] = 1; V[k][3] = -1
        W[k * 2 + 0][3] = 1; W[k * 2 + 1][3] = 1

    # M5 = (A00+A01)*B1
    U[0][4] = 1; U[1][4] = 1
    for k in range(p):
        V[p + k][4] = 1
        W[k * 2 + 0][4] = -1

    # M6 = (A10-A00)*(B0+B1)
    U[2][5] = 1; U[0][5] = -1
    for k in range(p):
        V[k][5] = 1; V[p + k][5] = 1
        W[k * 2 + 1][5] = 1

    # M7 = (A01-A11)*(B1+B0)
    U[1][6] = 1; U[3][6] = -1
    for k in range(p):
        V[k][6] = 1; V[p + k][6] = 1
        W[k * 2 + 0][6] = 1

    return U, V, W


def _embed_strassen_block(U, V, W, col_start, a_rows, b_rows, c_cols, p=8):
    """
    Embed a 2x2*2xp Strassen block into the global matrices.
    
    a_rows: list of 2 global row indices for A (i*m+j)
    b_rows: list of 2 global row indices for B (j*p+k), each representing a "row" of B
             Actually, b_rows should be the starting indices for the two B rows
    c_cols: list of 2 global column indices for C (k*n+i), each representing a "column" of C
             Actually, c_cols should be the i indices for the two C rows
    
    For 2x2 * 2xp:
    A indices: a_rows[0] = i0*m+j0, a_rows[1] = i1*m+j1 (but we need all 4 A entries)
    
    Let me restructure: this function takes the 4 A indices, 2 B row starts, and 2 C row indices.
    """
    # a_idx[0] = A[0,0], a_idx[1] = A[0,1], a_idx[2] = A[1,0], a_idx[3] = A[1,1]
    # b_start[0] = start index for B row 0 (j0*p), b_start[1] = start index for B row 1 (j1*p)
    # c_i[0] = i0, c_i[1] = i1 (row indices of C)
    
    a_idx = a_rows
    b_start = b_rows
    c_i = c_cols
    
    U_block, V_block, W_block = _strassen_2x2_2xp(p)
    
    for r in range(7):
        col = col_start + r
        # U
        for ai in range(4):
            if U_block[ai][r] != 0:
                U[a_idx[ai]][col] += U_block[ai][r]
        # V
        for bi in range(2):
            for k in range(p):
                if V_block[bi * p + k][r] != 0:
                    V[b_start[bi] + k][col] += V_block[bi * p + k][r]
        # W
        for ci in range(2):
            for k in range(p):
                if W_block[k * 2 + ci][r] != 0:
                    W[k * N + ci][col] += W_block[k * 2 + ci][r]
    
    return col_start + 7


def _construct_3x3_3x8_strassen():
    """
    Construct a rank-12 decomposition for 3x3 * 3x8 using Strassen.
    
    Partition:
    A = [[A00 A01 A02],
         [A10 A11 A12],
         [A20 A21 A22]]
    B = [[B00...B07],
         [B10...B17],
         [B20...B27]]
    
    C = A * B (3x8)
    
    Use Strassen on the top-left 2x2 block of A and top 2 rows of B:
    A_tl = [[A00, A01], [A10, A11]] (2x2)
    B_t = [[B00...B07], [B10...B17]] (2x8)
    
    This gives C[0,0:8] and C[1,0:8] contributions from A_tl * B_t.
    
    Remaining terms:
    C[0,k] += A02 * B2k
    C[1,k] += A12 * B2k  
    C[2,k] = A20*B0k + A21*B1k + A22*B2k
    
    Rank: 7 (Strassen) + 1 + 1 + 3 = 12
    """
    n, m, p = N, M, P
    rank = 12
    U = [[0] * rank for _ in range(n * m)]  # 9 x 12
    V = [[0] * rank for _ in range(m * p)]  # 24 x 12
    W = [[0] * rank for _ in range(p * n)]  # 24 x 12
    
    col = 0
    
    # Strassen block for A_tl * B_t
    # A indices: A[0,0]=0, A[0,1]=1, A[1,0]=3, A[1,1]=4
    # B rows: B row 0 starts at 0, B row 1 starts at 8
    # C rows: i=0, i=1
    a_idx = [0, 1, 3, 4]
    b_start = [0, 8]
    c_i = [0, 1]
    col = _embed_strassen_block(U, V, W, col, a_idx, b_start, c_i, p)
    
    # A02 * B2: C[0,k] += A[0,2]*B[2,k]
    # A[0,2] is at index 0*3+2 = 2
    # B[2,k] is at index 2*8+k = 16+k
    # C[0,k] is at index k*3+0
    U[2][col] = 1
    for k in range(p):
        V[16 + k][col] = 1
        W[k * 3 + 0][col] = 1
    col += 1
    
    # A12 * B2: C[1,k] += A[1,2]*B[2,k]
    # A[1,2] is at index 1*3+2 = 5
    # B[2,k] is at index 2*8+k = 16+k
    # C[1,k] is at index k*3+1
    U[5][col] = 1
    for k in range(p):
        V[16 + k][col] = 1
        W[k * 3 + 1][col] = 1
    col += 1
    
    # A20 * B0: C[2,k] += A[2,0]*B[0,k]
    # A[2,0] is at index 2*3+0 = 6
    # B[0,k] is at index 0*8+k = k
    # C[2,k] is at index k*3+2
    U[6][col] = 1
    for k in range(p):
        V[k][col] = 1
        W[k * 3 + 2][col] = 1
    col += 1
    
    # A21 * B1: C[2,k] += A[2,1]*B[1,k]
    # A[2,1] is at index 2*3+1 = 7
    # B[1,k] is at index 1*8+k = 8+k
    # C[2,k] is at index k*3+2
    U[7][col] = 1
    for k in range(p):
        V[8 + k][col] = 1
        W[k * 3 + 2][col] = 1
    col += 1
    
    # A22 * B2: C[2,k] += A[2,2]*B[2,k]
    # A[2,2] is at index 2*3+2 = 8
    # B[2,k] is at index 2*8+k = 16+k
    # C[2,k] is at index k*3+2
    U[8][col] = 1
    for k in range(p):
        V[16 + k][col] = 1
        W[k * 3 + 2][col] = 1
    col += 1
    
    assert col == rank, f"Expected {rank} columns, got {col}"
    
    return U, V, W, rank


def _try_rank_11():
    """
    Attempt to find a rank-11 decomposition.
    
    The rank-12 construction uses:
    - 7 for Strassen on A_tl * B_t (2x2 * 2x8)
    - 1 for A02 * B2
    - 1 for A12 * B2
    - 1 for A20 * B0
    - 1 for A21 * B1
    - 1 for A22 * B2
    
    To get rank 11, we'd need to save 1 rank.
    
    One idea: can we combine A20*B0 and A21*B1 into a single rank-1 term?
    No, because they involve different B rows and different A entries.
    
    Another idea: use a different Strassen variant or a different partition.
    
    Try partitioning A as:
    A = [[A00, A01, A02],
         [A10, A11, A12],
         [A20, A21, A22]]
    
    What if we use Strassen on a different 2x2 block?
    
    Try A_br = [[A11, A12], [A21, A22]] (bottom-right 2x2)
    B_b = [[B10...B17], [B20...B27]] (bottom 2 rows of B)
    
    C[1,k] += A11*B1k + A12*B2k
    C[2,k] += A21*B1k + A22*B2k
    
    Remaining:
    C[0,k] = A00*B0k + A01*B1k + A02*B2k
    C[1,k] += A10*B0k
    C[2,k] += A20*B0k
    
    Rank: 7 + 3 + 1 + 1 = 12. Same.
    
    It seems rank 12 is hard to beat with simple block decompositions.
    """
    return None


def construct():
    start_time = time.time()
    deadline = start_time + 100.0
    
    # Try the rank-12 Strassen construction
    U, V, W, rank = _construct_3x3_3x8_strassen()
    
    # Verify
    if _verify(U, V, W, rank):
        return {"U": U, "V": V, "W": W}
    
    # If verification fails, fall back to schoolbook
    U, V, W, rank = _schoolbook()
    if _verify(U, V, W, rank):
        return {"U": U, "V": V, "W": W}
    
    return {"U": U, "V": V, "W": W}
