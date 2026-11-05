"""Block-decomposition approach for 3x4x6 matrix multiplication tensor.

Strategy: split n=3 into (1,2). The 1x4x6 part has rank 24.
For the 2x4x6 part, split p=6 into three 2-blocks. Each 2x4x2 block is split
into two 2x2x2 Strassen blocks (rank 7 each), giving 14 per p-block, total 42.
Grand total: 24 + 42 = 66.
"""
DIMENSIONS = [3, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = n * m * p  # 72, baseline
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0

    # --- Strassen 2x2x2 ---
    # A2 = [[A00,A01],[A10,A11]], B2 = [[B00,B01],[B10,B11]]
    # M1=(A00+A11)(B00+B11), M2=(A10+A11)B00, M3=A00(B10-B11),
    # M4=A11(B01-B10), M5=(A00+A01)B11, M6=(A10-A00)(B00+B10), M7=(A01-A11)(B11+B10)
    # C00=M1+M4-M5-M7, C01=M3+M5, C10=M2+M4, C11=M1-M2+M3+M7
    strassen_entries = [
        (0, 0, 0, 0), (0, 1, 0, 0), (0, 1, 1, 1), (1, 1, 0, 1),
        (1, 1, 1, 0), (1, 1, 1, 0), (0, 0, 0, 1),
        (1, 0, 0, 0), (0, 0, 0, 1), (1, 0, 0, 1), (1, 0, 1, 0),
        (0, 1, 1, 0), (1, 1, 0, 1), (0, 0, 1, 1),
        (1, 0, 1, 0), (1, 0, 0, 0), (0, 1, 1, 1),
        (1, 1, 1, 1), (0, 0, 1, 1), (1, 0, 0, 1),
    ]

    def add_strassen(i0, j0, k0):
        nonlocal col
        for (ui, uj, vi, vj, wi, wk) in strassen_entries:
            U[(i0 + ui) * m + (j0 + uj)][col] = 1
            V[(j0 + vj) * p + (k0 + wk)][col] = 1
            W[(k0 + wi) * n + (i0 + wi)][col] = 1
            col += 1

    # --- Part 1: rows 0 (n=1) x all m=4 x all p=6, rank = 1*4*6 = 24 ---
    for j in range(m):
        for k in range(p):
            U[0 * m + j][col] = 1   # i=0
            V[j * p + k][col] = 1
            W[k * n + 0][col] = 1
            col += 1

    # --- Part 2: rows 1,2 (n=2) x all m=4 x all p=6 ---
    # Split p=6 into three blocks of 2
    for p_block in range(3):
        k0 = p_block * 2
        # Split m=4 into two blocks of 2
        for m_block in range(2):
            j0 = m_block * 2
            # Strassen 2x2x2 for rows {1,2}, cols {j0, j0+1}, p-rows {k0, k0+1}
            add_strassen(1, j0, k0)

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
