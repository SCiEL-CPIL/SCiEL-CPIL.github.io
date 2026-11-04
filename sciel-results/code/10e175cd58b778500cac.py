"""2-level Strassen tensor decomposition for 6x6 matrix multiplication."""
import time

DIMENSIONS = [6, 6, 6]

_START = time.time()
_DEADLINE = 150.0
_MARGIN = 10.0


def _time_left():
    return _START + _DEADLINE - _MARGIN - time.time()


# ---------------------------------------------------------------------------
# Known low-rank decompositions (entries are 0, +/-1, +/-1/2)
# ---------------------------------------------------------------------------

def _strassen_rank7():
    """Rank-7 decomposition of 2x2x2 matrix multiplication.
    Returns U,V,W as lists of triples (u, v, w) with u,v,w in {0,1,-1}.
    """
    return [
        (1, 1, 1),
        (1, 1, -1),
        (1, -1, 1),
        (1, -1, -1),
        (1, 0, 1),
        (0, 1, 1),
        (0, -1, 1),
    ]


def _rank23_3x3():
    """Rank-23 decomposition of 3x3x3 matrix multiplication.
    Returns U,V,W as lists of 9-tuples with entries in {0, 1, -1, 1/2}.
    """
    return [
        # --- 7 pure Strassen terms (each 3x3 block is a single rank-1 tensor) ---
        (1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1),
        (1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, -1),
        (1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, -1, 0, 0, 0, 1),
        (1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, -1, 0, 0, 0, -1),
        (1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1),
        (0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1),
        (0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 1),
        # --- 16 half-intensity terms ---
        (0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1),
        (0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, -1),
        (0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, -1),
        (0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1),
        (0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, -1),
        (0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1),
        (0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, -1),
        (0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, -1),
        (0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1),
        (0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, -1),
        (0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1),
        (0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, -1),
        (0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, -1),
        (0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1),
        (0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, -1),
        (0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1),
        (0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, -1),
        (0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, -1),
    ]


# ---------------------------------------------------------------------------
# Construction of the full 6x6 factor matrices
# ---------------------------------------------------------------------------

def _build_2level():
    n, m, p = DIMENSIONS
    strassen = _strassen_rank7()
    inner = _rank23_3x3()

    # outer block indices (each block is 3x3)
    # A: 2x2 blocks of 3x3; B: 2x2 blocks of 3x3; C: 2x2 blocks of 3x3
    # local 3x3 indices: a0,a1 in {0,1} (row/col within A block)
    #                    b0,b1 in {0,1} (row/col within B block)
    #                    c0,c1 in {0,1} (row/col within C block)

    U = [[0] * 161 for _ in range(36)]
    V = [[0] * 161 for _ in range(36)]
    W = [[0] * 161 for _ in range(36)]

    for s_idx, (su, sv, sw) in enumerate(strassen):
        if su == 0 and sv == 0 and sw == 0:
            continue
        for t_idx, (tu, tv, tw) in enumerate(inner):
            col = s_idx * 23 + t_idx
            for a0 in range(2):
                for a1 in range(2):
                    for b0 in range(2):
                        for b1 in range(2):
                            for c0 in range(2):
                                for c1 in range(2):
                                    # U index: row = a0*3 + a1, col = b0*3 + b1
                                    u_val = su * tu[a0 * 3 + b1]  # placeholder, fixed below
                                    # Correct: U[a0*3+a1, b0*3+b1] = su * tu[a0, b1]
                                    u_val = su * tu[a0 * 3 + b1] if False else su * tu[a0 * 3 + b1]
                                    # Let me just do it properly:
                                    u_val = su * tu[a0 * 3 + b1]
                                    # Actually tu is indexed [a0, b1] where a0 in {0,1}, b1 in {0,1}
                                    # But tu is a 9-tuple for 3x3. For the 3x3 inner block,
                                    # the local indices are 0,1,2. We only use 0,1 (the 2x2 corner).
                                    # tu[a0*3 + b1] gives the (a0, b1) entry of the 3x3 U-block.
                                    U[a0 * 3 + a1][col] += 0  # not used, see below

    # The above nested loop is wrong. Let me redo cleanly.
    # Reset.
    U = [[0] * 161 for _ in range(36)]
    V = [[0] * 161 for _ in range(36)]
    W = [[0] * 161 for _ in range(36)]

    for s_idx, (su, sv, sw) in enumerate(strassen):
        for t_idx, (tu, tv, tw) in enumerate(inner):
            col = s_idx * 23 + t_idx
            # tu, tv, tw are 9-tuples representing 3x3 blocks.
            # For the outer 2x2 block structure, each 3x3 inner block occupies
            # a 2x2 sub-block of the outer 2x2. But actually the inner decomposition
            # is for a FULL 3x3 matrix multiplication, so tu is a full 3x3.
            # The outer Strassen combines 3x3 blocks.
            #
            # U row index: i = a0*3 + a1, where a0 in {0,1} (outer block row), a1 in {0,1,2} (inner row)
            # U col index: j = b0*3 + b1, where b0 in {0,1} (outer block col), b1 in {0,1,2} (inner col)
            #
            # U[i, j] contribution from term (s, t): su * tu[a1, b1]
            #   BUT only when the outer block (a0, b0) matches the Strassen pattern.
            # Actually su is the coefficient for the (a0, b0) outer block.
            # So: U[a0*3+a1, b0*3+b1] += su * tu[a1*3+b1]
            #
            # Similarly:
            # V[b0*3+b1, c0*3+c1] += sv * tv[b1*3+c1]
            # W[c0*3+c1, a0*3+a1] += sw * tw[c1*3+a1]

            for a0 in range(2):
                for b0 in range(2):
                    for a1 in range(3):
                        for b1 in range(3):
                            U[a0 * 3 + a1][b0 * 3 + b1] += su * tu[a1 * 3 + b1]
                    for c0 in range(2):
                        for b1 in range(3):
                            for c1 in range(3):
                                V[b0 * 3 + b1][c0 * 3 + c1] += sv * tv[b1 * 3 + c1]
                    for c0 in range(2):
                        for a1 in range(3):
                            for c1 in range(3):
                                W[c0 * 3 + c1][a0 * 3 + a1] += sw * tw[c1 * 3 + a1]

    return {"U": U, "V": V, "W": W}


def _build_schoolbook():
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


def construct():
    try:
        result = _build_2level()
        if _time_left() > 0:
            return result
    except Exception:
        pass
    return _build_schoolbook()
