"""Block-diagonal decomposition for 3x4 x 4x8: rank 32 + 56 = 88."""
DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
def _to_half(v):
    """Convert a number to int if integer, else a float half-integer, for clean output."""
    if abs(v - round(v)) < 1e-9:
        return int(round(v))
    if abs(2 * v - round(2 * v)) < 1e-9:
        return float(round(2 * v)) / 2.0
    return v

def construct():
    n, m, p = 3, 4, 8
    # Row-block partition of the 3 rows: block A = row 0 (1x4x8), block B = rows 1,2 (2x4x8)
    rankA = 1 * 4 * 8          # 32, schoolbook
    rankB = 8 * 7              # 56, Strassen on 8 2x2x2 blocks
    rank = rankA + rankB       # 88

    U = [[0] * rank for _ in range(n * m)]   # (12, rank)
    V = [[0] * rank for _ in range(m * p)]   # (32, rank)
    W = [[0] * rank for _ in range(p * n)]   # (24, rank)

    # ---- Block A: 1x4x8, schoolbook (rank 32) ----
    col = 0
    for i in range(1):                # the single row (row 0)
        for j in range(4):
            for k in range(8):
                U[0 * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + 0][col] = 1
                col += 1

    # ---- Block B: 2x4x8, Strassen per 2x2x2 block ----
    # Local index (a,b) maps to global row (1 + a), col (2 * b + j2), row-out (4 * c + k2)
    # Strassen U-factors (indexed 0..6), each (a,b):
    ufac = [
        (0, 0),
        (1, 1),
        (0, 0), (0, 1),
        (0, 1), (1, 0),
        (1, 0), (1, 1),
    ]
    # Strassen W-factors (indexed 0..6), each (k2, i2) with sign:
    wfac = [
        (0, 0,  1),
        (1, 1,  1),
        (0, 0,  1), (1, 0, -1),
        (1, 0,  1), (0, 1, -1),
        (0, 1,  1), (1, 1,  1),
    ]

    col = rankA
    for b in range(2):              # column 2x2 blocks
        for c in range(4):          # output-row 2x2 blocks
            for s in range(7):      # Strassen factors
                # U: possibly two (a,b) entries with signs
                if 2 <= s <= 5:
                    (a0, b0), (a1, b1) = ufac[s]
                    sign = 1 if s in (2, 4) else -1
                    U[(1 + a0) * m + (2 * b + b0)][col] = 1
                    U[(1 + a1) * m + (2 * b + b1)][col] = sign
                else:
                    (a0, b0) = ufac[s]
                    U[(1 + a0) * m + (2 * b + b0)][col] = 1
                # V: one entry (j2, k2)
                j2, k2 = ufac[s]
                V[(2 * b + j2) * p + (4 * c + k2)][col] = 1
                # W: one entry (k2', i2) with sign
                kk, ii, sg = wfac[s]
                W[(4 * c + kk) * n + (1 + ii)][col] = sg
                col += 1

    # Convert entries to int / half-integer as appropriate
    for mat in (U, V, W):
        for r in range(len(mat)):
            for c in range(len(mat[r])):
                mat[r][c] = _to_half(mat[r][c])

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
