"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 3, 10

    U_rows = n * m  # 6
    V_rows = m * p  # 30
    W_rows = p * n  # 20

    U = [[0] * 60 for _ in range(U_rows)]
    V = [[0] * 60 for _ in range(V_rows)]
    W = [[0] * 60 for _ in range(W_rows)]

    col = 0

    # Helper to add a single rank-1 term with coefficients
    def add_term(u_coefs, v_coefs, w_coefs):
        nonlocal col
        for (u_idx, c) in u_coefs:
            U[u_idx][col] += c
        for (v_idx, c) in v_coefs:
            V[v_idx][col] += c
        for (w_idx, c) in w_coefs:
            W[w_idx][col] += c
        col += 1

    # Index helpers
    def u_idx(i, j):
        return i * m + j

    def v_idx(j, k):
        return j * p + k

    def w_idx(k, i):
        return k * n + i

    # Block 1: First 2 columns of B (k=0,1)
    # A is 2x3, B_block is 3x2
    # Partition A into A1 (2x2, cols 0,1) and A2 (2x1, col 2)
    # Partition B_block into B11 (2x2, rows 0,1) and B12 (1x2, row 2)
    # C_block = A1*B11 + A2*B12
    # Use Strassen for A1*B11 (rank 7) and schoolbook for A2*B12 (rank 4)
    # Total: 11 terms

    # Strassen for A1 (2x2) * B11 (2x2)
    # A1 = [[a00, a01], [a10, a11]]
    # B11 = [[b00, b01], [b10, b11]]
    # U indices for A1: u_idx(0,0)=0, u_idx(0,1)=1, u_idx(1,0)=2, u_idx(1,1)=3
    # V indices for B11 (k=0,1): v_idx(0,0)=0, v_idx(0,1)=1, v_idx(1,0)=10, v_idx(1,1)=11
    # W indices for C (k=0,1, i=0,1): w_idx(0,0)=0, w_idx(0,1)=1, w_idx(1,0)=2, w_idx(1,1)=3

    # M1 = (a00+a11)*(b00+b11)
    add_term(
        [(0, 1), (3, 1)],
        [(0, 1), (11, 1)],
        [(0, 1), (3, -1)]
    )

    # M2 = (a10+a11)*b00
    add_term(
        [(2, 1), (3, 1)],
        [(0, 1)],
        [(2, 1)]
    )

    # M3 = a00*(b01-b11)
    add_term(
        [(0, 1)],
        [(1, 1), (11, -1)],
        [(1, 1)]
    )

    # M4 = a11*(b10-b00)
    add_term(
        [(3, 1)],
        [(10, 1), (0, -1)],
        [(2, 1)]
    )

    # M5 = (a00+a01)*b11
    add_term(
        [(0, 1), (1, 1)],
        [(11, 1)],
        [(1, 1), (0, -1)]
    )

    # M6 = (a10-a00)*(b00+b01)
    add_term(
        [(2, 1), (0, -1)],
        [(0, 1), (1, 1)],
        [(3, 1)]
    )

    # M7 = (a01-a11)*(b10+b11)
    add_term(
        [(1, 1), (3, -1)],
        [(10, 1), (11, 1)],
        [(0, 1)]
    )

    # Schoolbook for A2 * B12
    # A2 = [[a02], [a12]] -> U indices: u_idx(0,2)=4, u_idx(1,2)=5
    # B12 = [[b20, b21]] -> V indices: v_idx(2,0)=20, v_idx(2,1)=21
    # C contributions: w_idx(0,0)=0, w_idx(0,1)=1, w_idx(1,0)=2, w_idx(1,1)=3

    # a02*b20 -> C[0,0]
    add_term([(4, 1)], [(20, 1)], [(0, 1)])
    # a02*b21 -> C[0,1]
    add_term([(4, 1)], [(21, 1)], [(1, 1)])
    # a12*b20 -> C[1,0]
    add_term([(5, 1)], [(20, 1)], [(2, 1)])
    # a12*b21 -> C[1,1]
    add_term([(5, 1)], [(21, 1)], [(3, 1)])

    # Blocks 2-5: Remaining 8 columns of B (k=2..9), 4 blocks of 2 columns each
    # For each block, use schoolbook: rank 2*3*2 = 12 per block
    # 4 blocks * 12 = 48 terms
    # Total so far: 11 + 48 = 59

    for block in range(4):
        k_start = 2 + block * 2
        k1 = k_start
        k2 = k_start + 1

        # Schoolbook for all 12 terms in this block
        for i in range(n):
            for j in range(m):
                for k in [k1, k2]:
                    add_term(
                        [(u_idx(i, j), 1)],
                        [(v_idx(j, k), 1)],
                        [(w_idx(k, i), 1)]
                    )

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
