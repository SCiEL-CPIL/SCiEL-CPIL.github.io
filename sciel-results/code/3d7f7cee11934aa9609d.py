"""Tensor decomposition of 5x5x5 matrix multiplication."""
import time
import random

DIMENSIONS = [5, 5, 5]
# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 100.0  # safety margin

    n, m, p = 5, 5, 5

    U_rows = []
    V_rows = []
    W_rows = []

    def idxU(i, j):
        return i * m + j
    def idxV(j, k):
        return j * p + k
    def idxW(k, i):
        return k * n + i

    # Part 1: Frame terms (i==4 or j==4), k in 0..4 -> 45 terms
    for i in range(n):
        for j in range(m):
            if i == 4 or j == 4:
                for k in range(p):
                    U_rows.append(idxU(i, j))
                    V_rows.append(idxV(j, k))
                    W_rows.append(idxW(k, i))

    # Part 2: 4x4x4 core via Strassen (rank 7)
    # Strassen partition: 4x4 into 2x2 blocks of size 2
    # A blocks: A11=(0-1,0-1), A12=(0-1,2-3), A21=(2-3,0-1), A22=(2-3,2-3)
    # B blocks: B11=(0-1,0-1), B12=(0-1,2-3), B21=(2-3,0-1), B22=(2-3,2-3)
    #
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22) B11
    # M3 = A11 (B12-B22)
    # M4 = A22 (B21-B11)
    # M5 = (A11+A12) B22
    # M6 = (A21-A11) (B11+B12)
    # M7 = (A12-A22) (B21+B22)
    #
    # C11 = M1 + M4 - M5 - M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M7

    # For each M_t, build the coefficient vectors.
    # A product a[i][j] * b[j][k] contributes +sign to U[i*m+j], V[j*p+k], W[k*n+i]
    # For a block product A_block * B_block, we sum over shared index j:
    #   sum_{j in block_cols} a[i][j] * b[j][k]
    # So for each (i, j) in A_block and (j, k) in B_block where the j matches,
    # we add sign to U[i*m+j], V[j*p+k], W[k*n+i]

    def block_positions(bi, bj):
        """Return list of (row, col) positions for block (bi,bj) in 4x4 matrix."""
        positions = []
        for di in range(2):
            for dj in range(2):
                positions.append((bi*2 + di, bj*2 + dj))
        return positions

    # Each M_t is defined as a list of (A_block_id, B_block_id, sign)
    # where block_id is (bi, bj)
    M_defs = [
        # M1 = (A11+A22)(B11+B22) = A11*B11 + A11*B22 + A22*B11 + A22*B22
        [((0,0),(0,0),1), ((0,0),(1,1),1), ((1,1),(0,0),1), ((1,1),(1,1),1)],
        # M2 = (A21+A22) B11 = A21*B11 + A22*B11
        [((1,0),(0,0),1), ((1,1),(0,0),1)],
        # M3 = A11 (B12-B22) = A11*B12 - A11*B22
        [((0,0),(0,1),1), ((0,0),(1,1),-1)],
        # M4 = A22 (B21-B11) = A22*B21 - A22*B11
        [((1,1),(1,0),1), ((1,1),(0,0),-1)],
        # M5 = (A11+A12) B22 = A11*B22 + A12*B22
        [((0,0),(1,1),1), ((0,1),(1,1),1)],
        # M6 = (A21-A11) (B11+B12) = A21*B11 + A21*B12 - A11*B11 - A11*B12
        [((1,0),(0,0),1), ((1,0),(0,1),1), ((0,0),(0,0),-1), ((0,0),(0,1),-1)],
        # M7 = (A12-A22) (B21+B22) = A12*B21 + A12*B22 - A22*B21 - A22*B22
        [((0,1),(1,0),1), ((0,1),(1,1),1), ((1,1),(1,0),-1), ((1,1),(1,1),-1)],
    ]

    for parts in M_defs:
        U_acc = {}
        V_acc = {}
        W_acc = {}
        for (A_bid, B_bid, sign) in parts:
            A_pos = block_positions(*A_bid)
            B_pos = block_positions(*B_bid)
            # For matrix product A_block * B_block:
            # (A*B)[i][k] = sum_j A[i][j] * B[j][k]
            # A_block has rows i in A_bid row range, cols j in A_bid col range
            # B_block has rows j in B_bid row range, cols k in B_bid col range
            # The shared index j must be in both A_block's col range and B_block's row range
            # Since A_block cols = A_bid[1]*2..A_bid[1]*2+1 and B_block rows = B_bid[0]*2..B_bid[0]*2+1
            # For a valid product, A_bid[1] must equal B_bid[0]
            for (i, j) in A_pos:
                for (j2, k) in B_pos:
                    if j == j2:  # shared index must match
                        ui = i * m + j
                        vi = j * p + k
                        wi = k * n + i
                        U_acc[ui] = U_acc.get(ui, 0) + sign
                        V_acc[vi] = V_acc.get(vi, 0) + sign
                        W_acc[wi] = W_acc.get(wi, 0) + sign
        U_rows.append(U_acc)
        V_rows.append(V_acc)
        W_rows.append(W_acc)

    # Assemble the full matrices
    rank = len(U_rows)

    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    for r in range(rank):
        u_row = U_rows[r]
        v_row = V_rows[r]
        w_row = W_rows[r]

        if isinstance(u_row, dict):
            for idx, val in u_row.items():
                U[idx][r] = val
            for idx, val in v_row.items():
                V[idx][r] = val
            for idx, val in w_row.items():
                W[idx][r] = val
        else:
            U[u_row][r] = 1
            V[v_row][r] = 1
            W[w_row][r] = 1

    # Verify correctness
    random.seed(42)
    for trial in range(100):
        A = [[random.randint(0, 5) for _ in range(m)] for _ in range(n)]
        B = [[random.randint(0, 5) for _ in range(p)] for _ in range(m)]
        C = [[sum(A[i][jj] * B[jj][k] for jj in range(m)) for k in range(p)] for i in range(n)]
        C_tensor = [[0] * p for _ in range(n)]
        for i in range(n):
            for k in range(p):
                s = 0
                for j in range(m):
                    for r in range(rank):
                        s += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                C_tensor[i][k] = s
        if C_tensor != C:
            return _schoolbook()
        if time.time() > deadline:
            break

    return {"U": U, "V": V, "W": W}

def _schoolbook():
    """Fallback schoolbook decomposition, rank 125."""
    n, m, p = 5, 5, 5
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
