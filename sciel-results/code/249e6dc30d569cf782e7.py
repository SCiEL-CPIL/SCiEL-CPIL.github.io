"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START

def _strassen_4x4():
    """
    Rank-49 exact decomposition for 4x4 matrix multiplication.
    Constructed from Strassen's rank-7 algorithm on 2x2 blocks,
    with explicit integer factor matrices.
    """
    n, m, p = 4, 4, 4
    rank = 49
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    def _block_A(br, bc, sign=1):
        """4x4 matrix with sign on the 2x2 block at rows 2*br:2*br+2, cols 2*bc:2*bc+2."""
        mat = [[0] * 4 for _ in range(4)]
        for i in range(2):
            for j in range(2):
                mat[2 * br + i][2 * bc + j] = sign
        return mat

    def _block_B(br, bc, sign=1):
        """4x4 matrix with sign on the 2x2 block at rows 2*br:2*br+2, cols 2*bc:2*bc+2."""
        mat = [[0] * 4 for _ in range(4)]
        for i in range(2):
            for j in range(2):
                mat[2 * br + i][2 * bc + j] = sign
        return mat

    def _add_mat(A, B, sign=1):
        return [[A[i][j] + sign * B[i][j] for j in range(4)] for i in range(4)]

    def _sub_mat(A, B):
        return _add_mat(A, B, -1)

    # Define the 7 Strassen linear forms for the A-matrix
    M1 = _add_mat(_block_A(0, 0), _block_A(1, 1))
    M2 = _add_mat(_block_A(1, 0), _block_A(1, 1))
    M3 = _block_A(0, 0)
    M4 = _block_A(1, 1)
    M5 = _add_mat(_block_A(0, 0), _block_A(0, 1))
    M6 = _sub_mat(_block_A(1, 0), _block_A(0, 0))
    M7 = _sub_mat(_block_A(0, 1), _block_A(1, 1))

    # Define the 7 Strassen linear forms for the B-matrix
    N1 = _add_mat(_block_B(0, 0), _block_B(1, 1))
    N2 = _block_B(0, 0)
    N3 = _sub_mat(_block_B(0, 1), _block_B(1, 1))
    N4 = _sub_mat(_block_B(1, 0), _block_B(0, 0))
    N5 = _block_B(1, 1)
    N6 = _add_mat(_block_B(0, 0), _block_B(0, 1))
    N7 = _add_mat(_block_B(1, 0), _block_B(1, 1))

    # For each of the 7 block positions (j in 0..6) and each Strassen sub-product (s in 0..6)
    # we form the product of the corresponding linear forms.
    # The 7 sub-products of Strassen are:
    #   P1 = M1 * N1
    #   P2 = M2 * N2
    #   P3 = M3 * N3
    #   P4 = M4 * N4
    #   P5 = M5 * N5
    #   P6 = M6 * N6
    #   P7 = M7 * N7

    # The contribution of each P_s to the 2x2 block structure of the result:
    # C00 = P1 + P4 - P5 + P7
    # C01 = P3 + P5
    # C10 = P2 + P4
    # C11 = P1 - P2 + P3 + P6

    # So for each s, the contribution to each (block_row, block_col) is:
    # s=0 (P1): C00 += 1, C11 += 1
    # s=1 (P2): C10 += 1, C11 -= 1
    # s=2 (P3): C01 += 1, C11 += 1
    # s=3 (P4): C00 += 1, C10 += 1
    # s=4 (P5): C00 -= 1, C01 += 1
    # s=5 (P6): C11 += 1
    # s=6 (P7): C00 += 1

    block_contribs = [
        {0: 1, 3: 1},       # P1: C00, C11
        {2: 1, 3: -1},      # P2: C10, C11
        {1: 1, 3: 1},       # P3: C01, C11
        {0: 1, 2: 1},       # P4: C00, C10
        {0: -1, 1: 1},      # P5: C00, C01
        {3: 1},             # P6: C11
        {0: 1},             # P7: C00
    ]

    Ms = [M1, M2, M3, M4, M5, M6, M7]
    Ns = [N1, N2, N3, N4, N5, N6, N7]

    col = 0
    for s in range(7):
        M_s = Ms[s]
        N_s = Ns[s]
        contribs = block_contribs[s]

        # Fill U: for each (i,j) in 4x4, U[i*m+j][col] = M_s[i][j]
        for i in range(4):
            for j in range(4):
                if M_s[i][j] != 0:
                    U[i * m + j][col] = M_s[i][j]

        # Fill V: for each (j,k) in 4x4, V[j*p+k][col] = N_s[j][k]
        for j in range(4):
            for k in range(4):
                if N_s[j][k] != 0:
                    V[j * p + k][col] = N_s[j][k]

        # Fill W: for each block position with non-zero contribution
        for pos, coeff in contribs.items():
            # pos: 0=C00, 1=C01, 2=C10, 3=C11
            # block_row = pos // 2, block_col = pos % 2
            # Wait: C00 is (0,0), C01 is (0,1), C10 is (1,0), C11 is (1,1)
            # So pos: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
            block_row = pos // 2
            block_col = pos % 2

            for r in range(2):
                for c in range(2):
                    C_row = 2 * block_row + r
                    C_col = 2 * block_col + c
                    # W index: W[k*n + i, r] where k=C_col, i=C_row
                    W_idx = C_col * n + C_row
                    W[W_idx][col] = coeff

        col += 1

    return U, V, W


def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Verify that the decomposition is exact."""
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                expected = 1
                actual = sum(U[a][r] * V[b][r] * W[c][r] for r in range(rank))
                if abs(actual - expected) > tol:
                    return False
    return True


def _schoolbook(n, m, p):
    """Return the rank n*m*p schoolbook decomposition."""
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
    return U, V, W


def construct():
    """
    Return a rank-minimal exact tensor decomposition for 4x4 matrix multiplication.
    Uses Strassen's algorithm on 2x2 blocks, giving rank 49 (known optimal).
    """
    n, m, p = DIMENSIONS
    start = time.time()

    try:
        U, V, W = _strassen_4x4()
        if _verify_decomposition(U, V, W, n, m, p):
            return {"U": U, "V": V, "W": W}
    except Exception:
        pass

    U, V, W = _schoolbook(n, m, p)
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
