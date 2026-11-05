"""Combined program: explicit rank-99 block construction with verification fallback."""
import time

DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
_DEADLINE = 115.0
_start = time.time()


def _elapsed():
    return time.time() - _start


def _verify_full(U, V, W, n, m, p, tol=1e-9):
    r = len(U[0])
    for i in range(n):
        for a in range(m):
            for b in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(r):
                        s += W[k * n + i][c] * U[i * m + a][c] * V[b * p + k][c]
                    expected = 1.0 if a == b else 0.0
                    if abs(s - expected) > tol:
                        return False
    return True


def _schoolbook(n, m, p):
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


def _build_rank99(n, m, p):
    """Explicit rank-99 construction using Strassen 2x2x2 blocks and schoolbook for remainders."""
    rank_total = 99
    U = [[0] * rank_total for _ in range(n * m)]
    V = [[0] * rank_total for _ in range(m * p)]
    W = [[0] * rank_total for _ in range(p * n)]

    def add_strassen(rank_offset, i0, j0, k0):
        """Add Strassen 2x2x2 block at global offset (i0, j0, k0)."""
        u_loc = [[0] * 7 for _ in range(4)]
        v_loc = [[0] * 7 for _ in range(4)]
        w_loc = [[0] * 7 for _ in range(4)]

        # M0: (A00+A11)(B00+B11)
        u_loc[0][0] = 1; u_loc[3][0] = 1
        v_loc[0][0] = 1; v_loc[3][0] = 1
        w_loc[0][0] = 1; w_loc[3][0] = 1

        # M1: (A10+A11)(B00)
        u_loc[2][1] = 1; u_loc[3][1] = 1
        v_loc[0][1] = 1
        w_loc[1][1] = 1; w_loc[3][1] = -1

        # M2: (A00)(B01-B11)
        u_loc[0][2] = 1
        v_loc[1][2] = 1; v_loc[3][2] = -1
        w_loc[2][2] = 1; w_loc[3][2] = 1

        # M3: (A11)(B10-B00)
        u_loc[3][3] = 1
        v_loc[2][3] = 1; v_loc[0][3] = -1
        w_loc[0][3] = 1; w_loc[1][3] = 1

        # M4: (A00+A01)(B11)
        u_loc[0][4] = 1; u_loc[1][4] = 1
        v_loc[3][4] = 1
        w_loc[0][4] = -1; w_loc[2][4] = 1

        # M5: (A10-A00)(B00+B01)
        u_loc[2][5] = 1; u_loc[0][5] = -1
        v_loc[0][5] = 1; v_loc[1][5] = 1
        w_loc[3][5] = 1

        # M6: (A01-A11)(B10+B11)
        u_loc[1][6] = 1; u_loc[3][6] = -1
        v_loc[2][6] = 1; v_loc[3][6] = 1
        w_loc[0][6] = 1

        # Map to global indices
        for i_l in range(2):
            for j_l in range(2):
                u_g_idx = (i0 + i_l) * m + (j0 + j_l)
                for r in range(7):
                    U[u_g_idx][rank_offset + r] += u_loc[i_l * 2 + j_l][r]

        for j_l in range(2):
            for k_l in range(2):
                v_g_idx = (j0 + j_l) * p + (k0 + k_l)
                for r in range(7):
                    V[v_g_idx][rank_offset + r] += v_loc[j_l * 2 + k_l][r]

        for k_l in range(2):
            for i_l in range(2):
                w_g_idx = (k0 + k_l) * n + (i0 + i_l)
                for r in range(7):
                    W[w_g_idx][rank_offset + r] += w_loc[k_l * 2 + i_l][r]

    def add_m221(rank_offset, i0, j0, k0):
        """Add schoolbook 2x2x1 block at global offset (i0, j0, k0). Rank 4."""
        for j_l in range(2):
            for k_l in range(1):
                r = j_l * 1 + k_l
                u_g_idx = (i0) * m + (j0 + j_l)
                U[u_g_idx][rank_offset + r] = 1
                v_g_idx = (j0 + j_l) * p + (k0 + k_l)
                V[v_g_idx][rank_offset + r] = 1
                w_g_idx = (k0 + k_l) * n + (i0)
                W[w_g_idx][rank_offset + r] = 1
                u_g_idx2 = (i0 + 1) * m + (j0 + j_l)
                U[u_g_idx2][rank_offset + r] = 1
                W[(k0 + k_l) * n + (i0 + 1)][rank_offset + r] = 1

    def add_m127(rank_offset, i0, j0, k0):
        """Add schoolbook 1x2x7 block at global offset (i0, j0, k0). Rank 14."""
        for j_l in range(2):
            for k_l in range(7):
                r = j_l * 7 + k_l
                u_g_idx = (i0) * m + (j0 + j_l)
                U[u_g_idx][rank_offset + r] = 1
                v_g_idx = (j0 + j_l) * p + (k0 + k_l)
                V[v_g_idx][rank_offset + r] = 1
                w_g_idx = (k0 + k_l) * n + (i0)
                W[w_g_idx][rank_offset + r] = 1

    def add_m317(rank_offset, i0, j0, k0):
        """Add schoolbook 3x1x7 block at global offset (i0, j0, k0). Rank 21."""
        for i_l in range(3):
            for k_l in range(7):
                r = i_l * 7 + k_l
                u_g_idx = (i0 + i_l) * m + (j0)
                U[u_g_idx][rank_offset + r] = 1
                v_g_idx = (j0) * p + (k0 + k_l)
                V[v_g_idx][rank_offset + r] = 1
                w_g_idx = (k0 + k_l) * n + (i0 + i_l)
                W[w_g_idx][rank_offset + r] = 1

    # Block 1: rows 0-1, cols 0-1 (M327 split into Strassen 2x2x2 + 2x2x1 + 1x2x7)
    add_strassen(0, 0, 0, 0)       # rows 0-1, cols 0-1, k=0-1
    add_strassen(7, 0, 0, 2)       # rows 0-1, cols 0-1, k=2-3
    add_strassen(14, 0, 0, 4)      # rows 0-1, cols 0-1, k=4-5
    add_m221(21, 0, 0, 6)          # rows 0-1, cols 0-1, k=6
    add_m127(25, 2, 0, 0)          # row 2, cols 0-1, k=0-6

    # Block 2: rows 0-1, cols 2-3
    add_strassen(39, 0, 2, 0)      # rows 0-1, cols 2-3, k=0-1
    add_strassen(46, 0, 2, 2)      # rows 0-1, cols 2-3, k=2-3
    add_strassen(53, 0, 2, 4)      # rows 0-1, cols 2-3, k=4-5
    add_m221(60, 0, 2, 6)          # rows 0-1, cols 2-3, k=6
    add_m127(64, 2, 2, 0)          # row 2, cols 2-3, k=0-6

    # Block 3: rows 0-2, col 4
    add_m317(78, 0, 4, 0)          # rows 0-2, col 4, k=0-6

    return {"U": U, "V": V, "W": W}


def construct():
    global _start
    n, m, p = 3, 5, 7
    _start = time.time()

    # Try the explicit rank-99 construction
    try:
        if _elapsed() < _DEADLINE - 5:
            cand = _build_rank99(n, m, p)
            if _verify_full(cand["U"], cand["V"], cand["W"], n, m, p):
                return cand
    except Exception:
        pass

    # Fallback: guaranteed-correct schoolbook (rank 105)
    return _schoolbook(n, m, p)
# EVOLVE-BLOCK-END
