"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 3, 10

    # Block-Kronecker structure:
    # - First 2 columns of B: rank-11 (Strassen for 2x2 + schoolbook for 2x1x2)
    # - Remaining 8 columns: schoolbook, rank 2*3*8 = 48
    # Total rank: 11 + 48 = 59

    U_list = []
    V_list = []
    W_list = []

    def add_term(u, v, w):
        U_list.append(u)
        V_list.append(v)
        W_list.append(w)

    # --- Block 1: A (2x3) * B1 (3x2), columns k=0,1 of B ---
    # A = [[a00, a01, a02], [a10, a11, a12]]
    # B1 = [[b00, b01], [b10, b11], [b20, b21]]
    # C1 = A*B1 (2x2)

    # Strassen for A1 (2x2) * B11 (2x2):
    # A1 = [[a00, a01], [a10, a11]], B11 = [[b00, b01], [b10, b11]]
    # M1 = (a00+a11)*(b00+b11)
    add_term(
        [1, 0, 0, 0, 1, 0],   # a00 + a11
        [1, 0, 0, 0, 0, 1, 0, 0, 0, 0],  # b00 + b11
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]   # W[0,0] = C[0,0]
    )
    # M2 = (a10+a11)*b00
    add_term(
        [0, 0, 0, 1, 1, 0],   # a10 + a11
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # b00
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]   # placeholder, will fix
    )
    # Let me redo this more carefully.

    U_list = []
    V_list = []
    W_list = []

    # Strassen for 2x2: A1*B11
    # A1 = [[a00, a01], [a10, a11]]
    # B11 = [[b00, b01], [b10, b11]]
    # C11 = M1 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M6

    # M1 = (a00+a11)*(b00+b11)
    add_term(
        [1, 0, 0, 0, 1, 0],   # a00 + a11
        [1, 0, 0, 0, 0, 1, 0, 0, 0, 0],  # b00 + b11
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]   # contributes to C[0,0]
    )

    # M2 = (a10+a11)*b00
    add_term(
        [0, 0, 0, 1, 1, 0],   # a10 + a11
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # b00
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]   # placeholder
    )

    # M3 = a00*(b01-b11)
    add_term(
        [1, 0, 0, 0, 0, 0],   # a00
        [0, 1, 0, 0, 0, -1, 0, 0, 0, 0],  # b01 - b11
        [0, 1, 0, 0, 0, 0, 0, 0, 0, 0]   # contributes to C[0,1]
    )

    # M4 = a11*(b10-b00)
    add_term(
        [0, 0, 0, 0, 1, 0],   # a11
        [-1, 0, 0, 0, 1, 0, 0, 0, 0, 0],  # b10 - b00
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]   # placeholder
    )

    # M5 = (a00+a01)*b11
    add_term(
        [1, 1, 0, 0, 0, 0],   # a00 + a01
        [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],  # b11
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]   # placeholder
    )

    # M6 = (a10-a00)*(b00+b01)
    add_term(
        [-1, 0, 0, 1, 0, 0],  # a10 - a00
        [1, 1, 0, 0, 0, 0, 0, 0, 0, 0],  # b00 + b01
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]   # placeholder
    )

    # M7 = (a01-a11)*(b10+b11)
    add_term(
        [0, 1, 0, 0, -1, 0],  # a01 - a11
        [0, 0, 0, 0, 1, 1, 0, 0, 0, 0],  # b10 + b11
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]   # placeholder
    )

    # Now I need to fix the W vectors. Let me think about this differently.
    # Each term contributes to specific C entries.
    # C[0,0] = M1 - M5 + M7
    # C[0,1] = M3 + M5
    # C[1,0] = M2 + M4
    # C[1,1] = M1 - M2 + M6

    # Let me redo with proper W assignments.
    U_list = []
    V_list = []
    W_list = []

    # Term 1: M1 = (a00+a11)*(b00+b11)
    # Contributes +1 to C[0,0] and +1 to C[1,1]
    add_term(
        [1, 0, 0, 0, 1, 0],   # a00 + a11
        [1, 0, 0, 0, 0, 1, 0, 0, 0, 0],  # b00 + b11
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]   # W[0,0] = 1, W[1,1] = 1
    )
    # But W is a single vector per term, so I can only assign one W entry per term.
    # This means I need separate terms for each contribution.

    # Actually, in tensor decomposition, each term is a rank-1 tensor u ⊗ v ⊗ w.
    # So each term has exactly one non-zero entry in W (or rather, W is a vector that
    # gets multiplied element-wise).

    # Let me restart with the correct approach.
    U_list = []
    V_list = []
    W_list = []

    # For each term, U, V, W are vectors such that:
    # C[i,k] = sum_t U[i*m+j_t] * V[j_t*p+k] * W[k*n+i]
    # where j_t is implicit in the structure.

    # Actually, the tensor is T[i,j,k] = 1 if i,j,k are valid indices.
    # A decomposition is: T[i,j,k] = sum_t U[i,j]_t * V[j,k]_t * W[k,i]_t
    # where U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank).

    # So for each term t, U_t is a vector of length n*m, V_t is a vector of length m*p,
    # and W_t is a vector of length p*n.

    # For schoolbook term (i,j,k):
    # U_t[i*m+j] = 1, all other entries 0
    # V_t[j*p+k] = 1, all other entries 0
    # W_t[k*n+i] = 1, all other entries 0

    # For Strassen, we combine multiple schoolbook terms.

    # Let me just do schoolbook for the first 2 columns (rank 12) and schoolbook for
    # the remaining 8 columns (rank 48), total rank 60. This is simple and correct.

    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = [0] * (n * m)
                v = [0] * (m * p)
                w = [0] * (p * n)
                u[i * m + j] = 1
                v[j * p + k] = 1
                w[k * n + i] = 1
                add_term(u, v, w)

    rank = len(U_list)
    # U_list is list of vectors of length 6, each vector is one column of U
    # We need U to be (6, rank), so transpose
    U = [[U_list[t][i] for t in range(rank)] for i in range(n * m)]
    V = [[V_list[t][j] for t in range(rank)] for j in range(m * p)]
    W = [[W_list[t][k] for t in range(rank)] for k in range(p * n)]

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
