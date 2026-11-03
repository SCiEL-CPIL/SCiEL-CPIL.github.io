import time
import numpy as np

DIMENSIONS = [4, 5, 5]

# EVOLVE-BLOCK-START

def _strassen_blocks():
    """
    Return 4x4 Strassen factor matrices (S, T, R) in row-major order:
      S: for A (4x4, rows = (i, j) with i in {0,1}, j in {0,1})
      T: for B (4x4, rows = (j, k))
      R: for C (4x4, rows = (i, k), in trace convention W[k*n+i] -> here just (i,k) order)
    Each column c corresponds to one Strassen product.
    """
    # Standard Strassen for 2x2 * 2x2 -> 2x2
    # A = [[a00, a01], [a10, a11]], B = [[b00, b01], [b10, b11]], C = [[c00, c01], [c10, c11]]
    #
    # M1 = (a00 + a11)(b00 + b11)
    # M2 = (a10 + a11)b00
    # M3 = a00(b10 - b11)
    # M4 = a11(b01 - b00)
    # M5 = (a00 + a10)b11
    # M6 = (a10 - a00)(b00 + b01)
    # M7 = (a01 - a11)(b10 + b11)
    #
    # c00 = M1 - M2 + M3 + M4
    # c01 = M5 + M7
    # c10 = M6 + M2
    # c11 = M1 - M3 + M5 - M7

    # Index mapping: row index = i*2 + j for A, i*2 + k for C, j*2 + k for B
    # A rows: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    # B rows: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    # C rows: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)

    S = np.zeros((4, 7), dtype=np.int64)
    T = np.zeros((4, 7), dtype=np.int64)
    R = np.zeros((4, 7), dtype=np.int64)

    # M1 = (a00 + a11)(b00 + b11)
    S[0, 0] = 1  # a00
    S[3, 0] = 1  # a11
    T[0, 0] = 1  # b00
    T[3, 0] = 1  # b11
    R[0, 0] = 1  # c00
    R[3, 0] = 1  # c11

    # M2 = (a10 + a11)b00
    S[2, 1] = 1  # a10
    S[3, 1] = 1  # a11
    T[0, 1] = 1  # b00
    R[0, 1] = -1  # c00
    R[2, 1] = 1   # c10

    # M3 = a00(b10 - b11)
    S[0, 2] = 1  # a00
    T[2, 2] = 1  # b10
    T[3, 2] = -1 # b11
    R[0, 2] = 1  # c00
    R[3, 2] = -1 # c11

    # M4 = a11(b01 - b00)
    S[3, 3] = 1  # a11
    T[1, 3] = 1  # b01
    T[0, 3] = -1 # b00
    R[0, 3] = 1  # c00

    # M5 = (a00 + a10)b11
    S[0, 4] = 1  # a00
    S[2, 4] = 1  # a10
    T[3, 4] = 1  # b11
    R[1, 4] = 1  # c01
    R[3, 4] = 1  # c11

    # M6 = (a10 - a00)(b00 + b01)
    S[2, 5] = 1  # a10
    S[0, 5] = -1 # a00
    T[0, 5] = 1  # b00
    T[1, 5] = 1  # b01
    R[2, 5] = 1  # c10

    # M7 = (a01 - a11)(b10 + b11)
    S[1, 6] = 1  # a01
    S[3, 6] = -1 # a11
    T[2, 6] = 1  # b10
    T[3, 6] = 1  # b11
    R[1, 6] = 1  # c01
    R[3, 6] = -1 # c11

    return S, T, R


def _schoolbook_block(n_i, n_j, n_k):
    """
    Return schoolbook factor matrices for n_i x n_j * n_j x n_k.
    Shapes: (n_i*n_j, n_i*n_j*n_k), (n_j*n_k, n_i*n_j*n_k), (n_k*n_i, n_i*n_j*n_k)
    """
    rank = n_i * n_j * n_k
    S = np.zeros((n_i * n_j, rank), dtype=np.int64)
    T = np.zeros((n_j * n_k, rank), dtype=np.int64)
    R = np.zeros((n_k * n_i, rank), dtype=np.int64)
    c = 0
    for i in range(n_i):
        for j in range(n_j):
            for k in range(n_k):
                S[i * n_j + j, c] = 1
                T[j * n_k + k, c] = 1
                R[k * n_i + i, c] = 1
                c += 1
    return S, T, R


def _build_decomposition(n, m, p):
    """
    Build the full decomposition for n x m * m x p.
    Split into 2x2 blocks in the (i, k) space with j split into {0,1} and {2,...,m-1}.
    """
    m1 = 2  # size of first j-group
    m2 = m - m1  # size of second j-group
    assert m2 == 3

    # We'll build U, V, W as lists of (local_row, local_col, value) tuples
    # then map to global indices.
    # Global U row: i*m + j
    # Global V row: j*p + k
    # Global W row: k*n + i

    U_entries = []  # (global_row, col, val)
    V_entries = []
    W_entries = []
    col = 0

    S_st, T_st, R_st = _strassen_blocks()  # 4x7 each

    def add_strassen(i_base, j_base, k_base):
        """Add Strassen block for A[i_base:i_base+2, j_base:j_base+2] * B[j_base:j_base+2, k_base:k_base+2]"""
        nonlocal col
        for c in range(7):
            # S entries (A factors)
            for local_i in range(2):
                for local_j in range(2):
                    val = S_st[local_i * 2 + local_j, c]
                    if val != 0:
                        global_i = i_base + local_i
                        global_j = j_base + local_j
                        U_entries.append((global_i * m + global_j, col + c, val))
            # T entries (B factors)
            for local_j in range(2):
                for local_k in range(2):
                    val = T_st[local_j * 2 + local_k, c]
                    if val != 0:
                        global_j = j_base + local_j
                        global_k = k_base + local_k
                        V_entries.append((global_j * p + global_k, col + c, val))
            # R entries (C factors)
            for local_i in range(2):
                for local_k in range(2):
                    val = R_st[local_i * 2 + local_k, c]
                    if val != 0:
                        global_i = i_base + local_i
                        global_k = k_base + local_k
                        W_entries.append((global_k * n + global_i, col + c, val))
        col += 7

    def add_schoolbook(i_list, j_list, k_list):
        """Add schoolbook for A[i_list, j_list] * B[j_list, k_list]"""
        nonlocal col
        for i in i_list:
            for j in j_list:
                for k in k_list:
                    U_entries.append((i * m + j, col, 1))
                    V_entries.append((j * p + k, col, 1))
                    W_entries.append((k * n + i, col, 1))
                    col += 1

    # Block structure:
    # i-groups: {0,1} and {2,3}
    # j-groups: {0,1} and {2,3,4}
    # k-groups: {0,1} and {2,3,4}

    # 4 blocks in (i-group, k-group) space:
    blocks = [
        # (i_list, k_list, j1_list, j2_list, use_strassen_for_j1)
        ([0, 1], [0, 1], [0, 1], [2, 3, 4], True),   # top-left: Strassen + schoolbook
        ([0, 1], [2, 3, 4], [0, 1], [2, 3, 4], False), # top-right: schoolbook + schoolbook
        ([2, 3], [0, 1], [0, 1], [2, 3, 4], True),   # bottom-left: Strassen + schoolbook
        ([2, 3], [2, 3, 4], [0, 1], [2, 3, 4], False), # bottom-right: schoolbook + schoolbook
    ]

    for i_list, k_list, j1_list, j2_list, use_strassen in blocks:
        i_base = i_list[0]
        k_base = k_list[0]
        j1_base = j1_list[0]

        if use_strassen:
            # j in {0,1}: Strassen for 2x2 * 2x2
            add_strassen(i_base, j1_base, k_base)
            # j in {2,3,4}: schoolbook for 2x3 * 3x2
            add_schoolbook(i_list, j2_list, k_list)
        else:
            # j in {0,1}: schoolbook for 2x2 * 2x3
            add_schoolbook(i_list, j1_list, k_list)
            # j in {2,3,4}: schoolbook for 2x3 * 3x3
            add_schoolbook(i_list, j2_list, k_list)

    rank = col
    n_U = n * m
    n_V = m * p
    n_W = p * n

    U = np.zeros((n_U, rank), dtype=np.int64)
    V = np.zeros((n_V, rank), dtype=np.int64)
    W = np.zeros((n_W, rank), dtype=np.int64)

    for (r, c, v) in U_entries:
        U[r, c] = v
    for (r, c, v) in V_entries:
        V[r, c] = v
    for (r, c, v) in W_entries:
        W[r, c] = v

    return U, V, W


def _verify(U, V, W, n, m, p):
    """Verify decomposition by checking that the tensor product equals the identity tensor."""
    rank = U.shape[1]
    U3 = U.reshape(n, m, rank)
    V3 = V.reshape(m, p, rank)
    W3 = W.reshape(p, n, rank)

    # Compute sum_c U[i,j,c] * V[j,k,c] * W[k,i,c] for all (i,j,k)
    # This should equal 1 for all (i,j,k)
    result = np.zeros((n, m, p), dtype=np.float64)
    for c in range(rank):
        # U[:, :, c] is n x m
        # V[:, :, c] is m x p
        # W[:, :, c] is p x n
        # We need to compute the outer product structure
        # Actually, the tensor T[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
        # This is not a standard matrix multiplication.
        # Let's do it element-wise.
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    result[i, j, k] += U3[i, j, c] * V3[j, k, c] * W3[k, i, c]

    return np.allclose(result, 1.0, atol=1e-9)


def construct():
    start = time.time()
    n, m, p = DIMENSIONS  # 4, 5, 5

    # Build the block-Strassen decomposition
    U, V, W = _build_decomposition(n, m, p)

    # Verify (this is exact for integer entries, but we use float for speed)
    # The verification is O(n*m*p*rank) which is 4*5*5*98 = 9800 operations, very fast.
    assert _verify(U, V, W, n, m, p), "Decomposition verification failed"

    # Convert to lists as required by the interface
    U_list = [list(row) for row in U]
    V_list = [list(row) for row in V]
    W_list = [list(row) for row in W]

    return {"U": U_list, "V": V_list, "W": W_list}

# EVOLVE-BLOCK-END
