"""
Rank-56 Strassen decomposition for 2x4 * 4x8 matrix multiplication.

The 4x8 matrix is partitioned into eight independent 4x2 blocks.
Each 2x2 * 2x2 product uses Strassen's algorithm (rank 7).
Total rank = 8 * 7 = 56.

Trace convention: W[k*n + i, r] where n=2, so W index = 2*k + i.
U index = i*m + j = 4*i + j (n=2, m=4)
V index = j*p + k = 8*j + k (m=4, p=8)
"""

DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 2, 4, 8
    # 8 blocks of 2x2, each with 7 Strassen multiplications
    num_blocks = (m // 2) * (p // 2)  # 2 * 4 = 8
    rank_per_block = 7
    total_rank = num_blocks * rank_per_block  # 56

    U = [[0.0] * total_rank for _ in range(n * m)]  # 8 rows
    V = [[0.0] * total_rank for _ in range(m * p)]  # 32 rows
    W = [[0.0] * total_rank for _ in range(p * n)]  # 16 rows

    # Strassen's 7 products for 2x2 * 2x2:
    # M0 = (A00 + A11)(B00 + B11)
    # M1 = (A10 + A11)B00
    # M2 = A00(B10 - B11)
    # M3 = A11(B01 - B00)
    # M4 = (A00 + A10)B11
    # M5 = (A11 - A00)(B00 + B01)
    # M6 = (A01 - A11)(B10 + B11)
    #
    # Results:
    # C00 = M0 + M3 - M4 + M6
    # C01 = M2 + M4
    # C10 = M1 + M3
    # C11 = M0 - M1 + M2 + M5

    # For each block (b_row, b_col), the A submatrix is A[:, 2*b_col : 2*b_col+2]
    # and the B submatrix is B[2*b_row : 2*b_row+2, :]
    #
    # A indices in U: row = i*m + j = 4*i + j, where j = 2*b_col + {0,1}
    # B indices in V: row = j*p + k = 8*j + k, where j = 2*b_row + {0,1}
    # C indices in W: row = k*n + i = 2*k + i, where k = 2*b_col + {0,1}
    #   (output C has dimensions n x p = 2 x 8, so C[i][k] maps to W[2*k+i])

    # Strassen coefficients: (product_id, U_rows_with_coeffs, V_rows_with_coeffs, W_contribs)
    # Each product: list of (row, coeff) for U, (row, coeff) for V, (row, coeff) for W

    strassen_products = [
        # M0 = (A00+A11)(B00+B11)
        # U rows: (i=0,j=0), (i=1,j=1) => 4*0+0=0, 4*1+1=5
        # V rows: (j=0,k=0), (j=1,k=1) => 8*0+0=0, 8*1+1=9
        # W: C00+=1, C01+=0, C10+=0, C11+=-1... wait let me re-derive
        # C00 = M0 + M3 - M4 + M6
        # C01 = M2 + M4
        # C10 = M1 + M3
        # C11 = M0 - M1 + M2 + M5
        # So M0 contributes: C00: +1, C11: +1
        # W rows: C00 => k=0,i=0 => 2*0+0=0; C11 => k=1,i=1 => 2*1+1=3
        {"u": [(0, 1.0), (5, 1.0)], "v": [(0, 1.0), (9, 1.0)], "w": [(0, 1.0), (3, 1.0)]},
        # M1 = (A10+A11)B00
        # U rows: (i=1,j=0)=4, (i=1,j=1)=5
        # V rows: (j=0,k=0)=0
        # W: C10: +1 (k=0,i=1 => 2*0+1=1)
        {"u": [(4, 1.0), (5, 1.0)], "v": [(0, 1.0)], "w": [(1, 1.0)]},
        # M2 = A00(B10-B11)
        # U rows: (i=0,j=0)=0
        # V rows: (j=1,k=0)=8, (j=1,k=1)=9
        # W: C01: +1 (k=1,i=0 => 2*1+0=2), C11: +1 (k=1,i=1 => 3)
        {"u": [(0, 1.0)], "v": [(8, 1.0), (9, -1.0)], "w": [(2, 1.0), (3, 1.0)]},
        # M3 = A11(B01-B00)
        # U rows: (i=1,j=1)=5
        # V rows: (j=0,k=1)=1, (j=0,k=0)=0
        # W: C00: +1 (0), C10: +1 (1)
        {"u": [(5, 1.0)], "v": [(1, 1.0), (0, -1.0)], "w": [(0, 1.0), (1, 1.0)]},
        # M4 = (A00+A10)B11
        # U rows: (i=0,j=0)=0, (i=1,j=0)=4
        # V rows: (j=1,k=1)=9
        # W: C00: -1 (0), C01: +1 (2)
        {"u": [(0, 1.0), (4, 1.0)], "v": [(9, 1.0)], "w": [(0, -1.0), (2, 1.0)]},
        # M5 = (A11-A00)(B00+B01)
        # U rows: (i=1,j=1)=5, (i=0,j=0)=0
        # V rows: (j=0,k=0)=0, (j=0,k=1)=1
        # W: C11: +1 (3)
        {"u": [(5, 1.0), (0, -1.0)], "v": [(0, 1.0), (1, 1.0)], "w": [(3, 1.0)]},
        # M6 = (A01-A11)(B10+B11)
        # U rows: (i=0,j=1)=1, (i=1,j=1)=5
        # V rows: (j=1,k=0)=8, (j=1,k=1)=9
        # W: C00: +1 (0)
        {"u": [(1, 1.0), (5, -1.0)], "v": [(8, 1.0), (9, 1.0)], "w": [(0, 1.0)]},
    ]

    for b_row in range(2):       # block row index (0,1)
        for b_col in range(4):   # block col index (0,1,2,3)
            for s in range(7):   # Strassen step
                col = (b_row * 4 + b_col) * 7 + s
                prod = strassen_products[s]
                for u_row, coeff in prod["u"]:
                    # u_row is offset within 2x2 block: row = i*m + j
                    # j = 2*b_col + local_j, i = local_i
                    # u_row = 4*i + j => we need to add 2*b_col to j part
                    # u_row = 4*local_i + local_j, actual = 4*local_i + (local_j + 2*b_col)
                    # = u_row + 2*b_col
                    actual_u = u_row + 2 * b_col
                    U[actual_u][col] += coeff
                for v_row, coeff in prod["v"]:
                    # v_row = 8*local_j + local_k, actual = 8*(local_j + 2*b_row) + local_k
                    # = v_row + 16*b_row
                    actual_v = v_row + 16 * b_row
                    V[actual_v][col] += coeff
                for w_row, coeff in prod["w"]:
                    # w_row = 2*local_k + local_i, actual = 2*(local_k + 2*b_col) + local_i
                    # = w_row + 4*b_col
                    actual_w = w_row + 4 * b_col
                    W[actual_w][col] += coeff

    # Convert to integers where possible
    for row in range(len(U)):
        for col in range(total_rank):
            if U[row][col] == int(U[row][col]):
                U[row][col] = int(U[row][col])
    for row in range(len(V)):
        for col in range(total_rank):
            if V[row][col] == int(V[row][col]):
                V[row][col] = int(V[row][col])
    for row in range(len(W)):
        for col in range(total_rank):
            if W[row][col] == int(W[row][col]):
                W[row][col] = int(W[row][col])

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
