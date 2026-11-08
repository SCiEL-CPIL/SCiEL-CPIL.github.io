"""Strassen-based rank-minimal decomposition for M_{2,4,6}."""
import time

DIMENSIONS = [2, 4, 6]

# Deadline guard: keep a safety margin before the 150 s wall-clock limit.
_START = time.time()
DEADLINE = 125.0  # seconds


def _time_left():
    return DEADLINE - (time.time() - _START)


def _schoolbook():
    n, m, p = DIMENSIONS
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


def _strassen_222():
    """
    Return (U, V, W) for 2x2 x 2x2 matrix multiplication, rank 7.
    U shape (4,7), V shape (4,7), W shape (4,7).
    U row index: i*2 + j  (i,j in 0..1)
    V row index: j*2 + k  (j,k in 0..1)
    W row index: k*2 + i  (k,i in 0..1)
    """
    # The 7 bilinear forms for Strassen:
    # m1 = (a00+a11)*(b00+b11)
    # m2 = (a10+a11)*b00
    # m3 = a00*(b01-b11)
    # m4 = a11*(b10-b00)
    # m5 = (a00+a01)*b11
    # m6 = (a10-a00)*(b00+b01)
    # m7 = (a01-a11)*(b10+b11)
    #
    # C00 = m1 + m4 - m5 - m7
    # C01 = m3 + m5
    # C10 = m2 + m4
    # C11 = m1 - m2 + m3 + m7
    #
    # U factors A = [a00 a01; a10 a11] into linear combinations.
    # V factors B = [b00 b01; b10 b11] into linear combinations.
    # W collects the products into C.

    # U: row (i,j), col c -> coefficient of a_{i,j} in m_c
    U = [
        # a00 (i=0,j=0), a01 (i=0,j=1), a10 (i=1,j=0), a11 (i=1,j=1)
        [1, 0, 0, 0],  # m1: a00
        [0, 0, 1, 1],  # m2: a10 + a11
        [1, 0, 0, 0],  # m3: a00
        [0, 0, 0, 1],  # m4: a11
        [1, 1, 0, 0],  # m5: a00 + a01
        [1, 0, -1, 0], # m6: a10 - a00
        [0, 1, 0, -1], # m7: a01 - a11
    ]
    # Transpose so rows = (i,j), cols = c
    U = list(map(list, zip(*U)))

    # V: row (j,k), col c -> coefficient of b_{j,k} in m_c
    V = [
        # b00 (j=0,k=0), b01 (j=0,k=1), b10 (j=1,k=0), b11 (j=1,k=1)
        [1, 0, 0, 0],  # m1: b00
        [1, 0, 0, 0],  # m2: b00
        [0, 1, 0, -1], # m3: b01 - b11
        [0, 0, 1, -1], # m4: b10 - b00
        [0, 0, 0, 1],  # m5: b11
        [1, 1, 0, 0],  # m6: b00 + b01
        [0, 0, 1, 1],  # m7: b10 + b11
    ]
    V = list(map(list, zip(*V)))

    # W: row (k,i), col c -> coefficient of m_c in C_{i,k}
    # C00 = m1 + m4 - m5 - m7
    # C01 = m3 + m5
    # C10 = m2 + m4
    # C11 = m1 - m2 + m3 + m7
    W = [
        # (k=0,i=0): C00
        [1, 0, 0, 0, -1, 0, -1],
        # (k=1,i=0): C01
        [0, 0, 1, 0, 1, 0, 0],
        # (k=0,i=1): C10
        [0, 1, 0, 1, 0, 0, 0],
        # (k=1,i=1): C11
        [1, -1, 1, 0, 0, 0, 1],
    ]

    return U, V, W


def _strassen_246():
    """
    Build decomposition for M_{2,4,6} using Strassen.
    Split B (4x6) into three 4x2 blocks: B0, B1, B2.
    Each block is multiplied by A (2x4) via two Strassen 2x2 decompositions.
    Total rank = 3 * 14 = 42.
    """
    n, m, p = 2, 4, 6
    sU, sV, sW = _strassen_222()

    total_rank = 3 * 2 * 7  # 3 blocks * 2 Strassen per block * 7 rank each
    U = [[0] * total_rank for _ in range(n * m)]  # (8, 42)
    V = [[0] * total_rank for _ in range(m * p)]  # (24, 42)
    W = [[0] * total_rank for _ in range(p * n)]  # (12, 42)

    for block in range(3):
        # B_block is columns [block*2, block*2+2) of B
        # For each of the two Strassen 2x2 decompositions within this block:
        for sub in range(2):
            col_offset = block * 2 + sub
            rank_offset = block * 14 + sub * 7

            # U rows: i in 0..1, j in 0..3
            # For sub=0: j maps to 0,1 (first 2x2 of A)
            # For sub=1: j maps to 2,3 (second 2x2 of A)
            j_base = sub * 2
            for i in range(n):
                for j in range(2):
                    a_row = i * m + (j_base + j)  # row index in U
                    for c in range(7):
                        U[a_row][rank_offset + c] = sU[i * 2 + j][c]

            # V rows: j in 0..3, k in 0..5
            # k maps to col_offset, col_offset+1 within the block
            k_base = block * 2
            for j in range(m):
                for k in range(2):
                    b_row = j * p + (k_base + k)  # row index in V
                    for c in range(7):
                        V[b_row][rank_offset + c] = sV[j * 2 + k][c]

            # W rows: k in 0..5, i in 0..1
            # k maps to col_offset, col_offset+1
            for k in range(2):
                for i in range(n):
                    w_row = (k_base + k) * n + i  # row index in W
                    for c in range(7):
                        W[w_row][rank_offset + c] = sW[k * 2 + i][c]

    return {"U": U, "V": V, "W": W}


def construct():
    if _time_left() < 2.0:
        return _schoolbook()

    try:
        return _strassen_246()
    except Exception:
        if _time_left() < 2.0:
            return _schoolbook()
        raise


if __name__ == "__main__":
    result = construct()
    print(f"Rank: {len(result['U'][0])}")
    print(f"U shape: ({len(result['U'])}, {len(result['U'][0])})")
    print(f"V shape: ({len(result['V'])}, {len(result['V'][0])})")
    print(f"W shape: ({len(result['W'])}, {len(result['W'][0])})")
